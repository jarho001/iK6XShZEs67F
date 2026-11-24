"""The roster networks (network.arch control and pairwise): shapes and masks, no leak of hidden state
into the policy, value or auxiliary heads, the type chart, the auxiliary labels and losses, and the
truncation bootstrap."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from train_support import TINY_OVERRIDES, league_config, tiny_config, write_pool

from pokejax.baselines import make_policy
from pokejax.engine.actions import bench_slots
from pokejax.engine.observation import observe
from pokejax.env import Env, EnvConfig
from pokejax.train import roster
from pokejax.train.auxiliary import auxiliary_terms, finalize_auxiliary
from pokejax.train.config import load_config
from pokejax.train.league import Layout, League
from pokejax.train.network import PokePolicy, initial_carry, vocabulary
from pokejax.train.ppo import Trainer, merge_leading

ROSTER = (
	"network.width=8",
	"network.pair_width=6",
	"network.interaction_width=3",
	"network.context_width=16",
	"network.core_width=16",
	"network.activation=silu",
	"network.species_embedding=4",
	"network.type_embedding=3",
)
AUXILIARY = (
	"auxiliary.next_action=0.05",
	"auxiliary.item=0.01",
	"auxiliary.ability=0.01",
	"auxiliary.moves=0.03",
)
ARCHS = ("control", "pairwise")
BATTLES = 4
SEAT_ZERO = 0

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


def roster_config(arch, *extra):
	return load_config(None, [*TINY_OVERRIDES, f"network.arch={arch}", *ROSTER, *extra])


def side_zero(timestep):
	return jax.tree.map(lambda x: x[:, SEAT_ZERO], timestep.obs)


def assert_trees_equal(first, second):
	jax.tree.map(
		lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), first, second
	)


@pytest.fixture(scope="module")
def battles(tables, tmp_path_factory, cpu_only):
	"""Four fresh battles: the environment and its reset (both sides' observations included)."""
	pool = write_pool(tmp_path_factory.mktemp("roster"), tables)
	env = Env(EnvConfig(BATTLES, observe="both"), pool, tables=tables)
	es, timestep = jax.jit(env.reset)(jax.random.key(3))
	return env, es, timestep


def league_for(env, tables, config):
	"""A league of max-damage rows with the labels on, and the model it plays."""
	model = PokePolicy(vocabulary(tables), config.network, config.auxiliary)
	policies = {"max-damage": make_policy("max-damage", env.tables, tables)}
	league = League(
		env,
		Layout({"max-damage": BATTLES}),
		model,
		config.network,
		config.pool.size,
		policies,
		labels=config.auxiliary.enabled,
	)
	return model, league


@pytest.mark.parametrize("arch", ARCHS)
def test_roster_shapes_match_the_widths_the_heads_read(tables, battles, arch):
	_, _, timestep = battles
	obs = side_zero(timestep)
	config = roster_config(arch)
	network = config.network
	model = PokePolicy(vocabulary(tables), network)
	hidden = initial_carry(network, (BATTLES,))
	params = model.init(jax.random.key(1), obs, hidden)["params"]
	projected = model.apply({"params": params}, obs, method=PokePolicy.encode)
	pair = network.pair_width if arch == "pairwise" else 0
	move_width = network.width + pair
	switch_width = network.width + 2 * pair
	gates = 3 * network.core_width
	expected = gates + 4 * move_width + 5 * switch_width + network.width
	assert projected.shape == (BATTLES, expected)
	carry, logits, value = model.apply({"params": params}, obs, hidden)
	assert logits.shape == (BATTLES, 9) and value.shape == (BATTLES,)
	assert jax.tree.map(jnp.shape, carry) == jax.tree.map(jnp.shape, hidden)
	features = model.apply({"params": params}, obs, method=PokePolicy.probe_features)
	assert features.shape == (BATTLES, 4, pair if arch == "pairwise" else 2 * network.width)


@pytest.mark.parametrize("arch", ARCHS)
def test_encode_then_step_is_the_whole_call_and_illegal_actions_get_no_mass(tables, battles, arch):
	_, _, timestep = battles
	obs = side_zero(timestep)
	config = roster_config(arch)
	model = PokePolicy(vocabulary(tables), config.network)
	hidden = initial_carry(config.network, (BATTLES,))
	params = model.init(jax.random.key(1), obs, hidden)["params"]
	projected = model.apply({"params": params}, obs, method=PokePolicy.encode)
	_, logits, value = model.apply(
		{"params": params}, hidden, projected, obs.legal_actions, method=PokePolicy.step
	)
	_, whole, whole_value = model.apply({"params": params}, obs, hidden)
	np.testing.assert_allclose(np.asarray(logits), np.asarray(whole), atol=1e-6)
	np.testing.assert_allclose(np.asarray(value), np.asarray(whole_value), atol=1e-6)
	narrowed = obs._replace(legal_actions=jnp.zeros_like(obs.legal_actions).at[:, 5:7].set(True))
	_, narrowed_logits, _ = model.apply({"params": params}, narrowed, hidden)
	probabilities = np.asarray(jax.nn.softmax(narrowed_logits))
	assert np.all(probabilities[:, [0, 1, 2, 3, 4, 7, 8]] == 0.0)
	np.testing.assert_allclose(probabilities[:, 5:7].sum(axis=1), 1.0, atol=1e-6)


def test_hidden_state_changes_neither_the_outputs_nor_the_carry(tables, battles):
	env, es, _ = battles
	config = roster_config("pairwise", *AUXILIARY)
	model, league = league_for(env, tables, config)
	rows = np.arange(BATTLES)
	foe = 1 - SEAT_ZERO
	active = np.asarray(es.state.active)[:, foe]
	# Nothing of the opposing active is revealed at the first decision: its item, ability and move
	# IDs are private, so changing them must leave every observation the same.
	assert not np.asarray(es.state.revealed_moves)[rows, foe, active].any()
	hidden_state = es.state._replace(
		item=es.state.item.at[rows, foe, active].set(7),
		ability=es.state.ability.at[rows, foe, active].set(9),
		moves=es.state.moves.at[rows, foe, active, 0].set(es.state.moves[rows, foe, active, 1]),
	)
	observe_batch = jax.vmap(observe, in_axes=(0, None, None))
	public = observe_batch(es.state, SEAT_ZERO, env.tables)
	assert_trees_equal(public, observe_batch(hidden_state, SEAT_ZERO, env.tables))
	hidden = initial_carry(config.network, (BATTLES,))
	params = model.init(jax.random.key(2), public, hidden, method=PokePolicy.full)["params"]
	first = model.apply({"params": params}, public, hidden, method=PokePolicy.full)
	second = model.apply(
		{"params": params},
		observe_batch(hidden_state, SEAT_ZERO, env.tables),
		hidden,
		method=PokePolicy.full,
	)
	assert_trees_equal(first, second)
	# The auxiliary labels do read the hidden state (so the plumbing is live).
	state = league.reset(jax.random.key(4), jnp.zeros(BATTLES, jnp.int32))
	actions = jnp.zeros((BATTLES, 2), jnp.int32).at[:, foe].set(2)
	labels = league.aux_labels(state, actions)
	changed = league.aux_labels(state._replace(env=state.env._replace(state=hidden_state)), actions)
	assert not np.array_equal(np.asarray(labels.moves), np.asarray(changed.moves))
	assert not np.array_equal(np.asarray(labels.item), np.asarray(changed.item))
	assert not np.array_equal(np.asarray(labels.ability), np.asarray(changed.ability))


def test_changing_the_type_chart_changes_no_input_of_the_roster_networks(
	tables, battles, monkeypatch
):
	_, _, timestep = battles
	chart = dataclasses.replace(
		tables.types,
		exponent=np.zeros_like(tables.types.exponent),
		immune=np.ones_like(tables.types.immune),
	)
	changed_tables = dataclasses.replace(tables, types=chart)
	original = roster.dex_columns()
	rebuilt = roster.build_dex(changed_tables)
	for name in original._fields:
		np.testing.assert_array_equal(getattr(original, name), getattr(rebuilt, name))
	obs = side_zero(timestep)
	for arch in ARCHS:
		config = roster_config(arch)
		model = PokePolicy(vocabulary(tables), config.network)
		hidden = initial_carry(config.network, (BATTLES,))
		params = model.init(jax.random.key(1), obs, hidden)["params"]
		before = model.apply({"params": params}, obs, method=PokePolicy.encode)
		monkeypatch.setattr(roster, "dex_columns", lambda rebuilt=rebuilt: rebuilt)
		after = model.apply({"params": params}, obs, method=PokePolicy.encode)
		monkeypatch.setattr(roster, "dex_columns", lambda original=original: original)
		np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_auxiliary_labels_follow_the_decision_rules(tables, battles):
	env, _, _ = battles
	config = roster_config("pairwise", *AUXILIARY)
	_, league = league_for(env, tables, config)
	learner = jnp.zeros(BATTLES, jnp.int32)
	state = league.reset(jax.random.key(4), learner)
	battle = state.env.state
	rows = np.arange(BATTLES)
	foe = 1 - np.asarray(learner)
	active = np.asarray(battle.active)[rows, foe]
	move_actions = jnp.zeros((BATTLES, 2), jnp.int32).at[:, 1].set(2)
	labels = league.aux_labels(state, move_actions)
	assert np.all(np.asarray(labels.next_kind) == 0)
	np.testing.assert_array_equal(
		np.asarray(labels.next_move), np.asarray(battle.moves)[rows, foe, active, 2]
	)
	switch_actions = jnp.zeros((BATTLES, 2), jnp.int32).at[:, 1].set(5)
	labels = league.aux_labels(state, switch_actions)
	assert np.all(np.asarray(labels.next_kind) == 1)
	slot = np.asarray(bench_slots(jnp.asarray(active)))[:, 1]
	seen = np.asarray(battle.seen_order)[rows, foe, slot]
	np.testing.assert_array_equal(np.asarray(labels.next_rank), np.where(seen > 0, seen - 1, 6))
	outcome = {
		"illegal": jnp.zeros((BATTLES, 2), jnp.bool_),
		"rejected": jnp.zeros((BATTLES, 2), jnp.bool_),
	}
	held = state._replace(
		env=state.env._replace(has_held=state.env.has_held.at[rows, foe].set(True))
	)
	finished = league.finish_labels(labels, held, outcome)
	assert np.all(np.asarray(finished.next_kind) == -1)
	np.testing.assert_array_equal(np.asarray(finished.private), np.asarray(labels.private))


def test_auxiliary_terms_mask_ineligible_rows_and_finalize_to_rates(tables, battles):
	env, _, _ = battles
	config = roster_config("pairwise", *AUXILIARY)
	model, league = league_for(env, tables, config)
	learner = jnp.zeros(BATTLES, jnp.int32)
	state = league.reset(jax.random.key(4), learner)
	actions = jnp.zeros((BATTLES, 2), jnp.int32).at[:, 1].set(2)
	labels = league.aux_labels(state, actions)
	hidden = initial_carry(config.network, (BATTLES,))
	params = model.init(jax.random.key(5), state.obs, hidden, method=PokePolicy.full)["params"]
	_, _, _, heads = model.apply({"params": params}, state.obs, hidden, method=PokePolicy.full)
	total, metrics = auxiliary_terms(heads, labels, state.obs, config.auxiliary)
	assert np.isfinite(float(total))
	rates = finalize_auxiliary(metrics)
	assert "aux_steps" not in rates
	for name, value in rates.items():
		assert np.isfinite(float(value)), name
	assert 0.0 <= float(rates["aux_kind_eligible"]) <= 1.0
	nothing = labels._replace(
		next_kind=jnp.full_like(labels.next_kind, -1),
		private=jnp.zeros_like(labels.private),
		identity=jnp.zeros_like(labels.identity),
	)
	empty_total, empty_metrics = auxiliary_terms(heads, nothing, state.obs, config.auxiliary)
	assert float(empty_total) == pytest.approx(0.0, abs=1e-6)
	assert float(finalize_auxiliary(empty_metrics)["aux_kind_eligible"]) == 0.0


def test_truncation_values_are_read_only_where_a_battle_was_truncated(
	tables, tmp_path_factory, cpu_only
):
	pool = write_pool(tmp_path_factory.mktemp("bootstrap"), tables)
	trainer = Trainer(tiny_config(*ROSTER, *AUXILIARY, "network.arch=pairwise"), pool)
	learner = trainer.init()
	_, traj = trainer.rollout(learner.params, learner.pool, learner.league)
	assert bool(traj.truncated.any())
	values = trainer.final_values(learner.params, traj)
	_, _, direct = trainer.model.apply(
		{"params": learner.params},
		merge_leading(traj.final_obs, 2),
		merge_leading(traj.after, 2),
	)
	np.testing.assert_allclose(
		np.asarray(values), np.asarray(direct).reshape(values.shape), atol=1e-6
	)
	untruncated = traj._replace(truncated=jnp.zeros_like(traj.truncated))
	assert not np.any(np.asarray(trainer.final_values(learner.params, untruncated)))


def test_an_update_with_the_auxiliary_heads_reports_finite_rates(
	tables, tmp_path_factory, cpu_only
):
	pool = write_pool(tmp_path_factory.mktemp("update"), tables)
	config = league_config("network.arch=pairwise", *ROSTER, *AUXILIARY, "run.updates=1")
	trainer = Trainer(config, pool)
	learner = trainer.init()
	_, metrics = trainer.update(learner)
	values = {name: float(value) for name, value in metrics.items()}
	assert all(np.isfinite(value) for value in values.values())
	for name in (
		"aux_kind_eligible",
		"aux_item_eligible",
		"aux_ability_eligible",
		"aux_moves_recall",
	):
		assert 0.0 <= values[name] <= 1.0, name
	assert "aux_next_action_loss" in values and "aux_moves_loss" in values
