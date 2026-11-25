"""Equivalence to Env.advance and the diagnostic rollout's small decision primitives."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.baselines import make_policy
from pokejax.engine.actions import block_switches
from pokejax.engine.randomness import CRIT, HIT, UNFORCED
from pokejax.engine.state import MOVE, NONE, ONGOING, SIDE0, SIDE1, SWITCH, TIE
from pokejax.env import Env, EnvConfig
from pokejax.search import transition
from pokejax.search.matches import choices, start
from pokejax.search.oracle import (
	SearchConfig,
	_compact,
	_prepare,
	_save,
	_successor,
	auxiliary_logits,
	backup,
	decide,
	forward,
)
from pokejax.teams import load_team_pool

BATCH = 32
KEY = jax.random.key(918)


@pytest.fixture(scope="module")
def env(tables, tmp_path_factory):
	fixture = json.loads(Path("tests/fixtures/reset.json").read_text())
	path = tmp_path_factory.mktemp("search") / "teams.json"
	path.write_text(
		json.dumps(
			{
				"format": "gen5randombattle",
				"count": len(fixture["teamSets"]),
				"seed": 0,
				"showdownCommit": fixture["showdownCommit"],
				"teams": fixture["teamSets"],
			}
		)
	)
	return Env(EnvConfig(BATCH), team_pool=load_team_pool(path, tables), tables=tables)


def assert_equal(expected, actual, rows=slice(None)):
	assert jax.tree.structure(expected) == jax.tree.structure(actual)
	for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
		np.testing.assert_array_equal(np.asarray(a)[rows], np.asarray(b)[rows])


def position(es):
	return transition.SearchState(es.state, es.held, es.has_held, es.base, es.blocked)


COMPARE_ENV = jax.jit(lambda env, es, actions: env.advance(es, actions))
SEARCH_BATCH = jax.jit(transition.advance_batch)
OBSERVE_SEARCH = jax.jit(jax.vmap(transition.observe_both, in_axes=(0, None)))
OBSERVE_ENV = jax.jit(lambda env, state: env.observe_both(state))


@jax.jit
def random_actions(es, key):
	legal = transition.observed_mask(position(es))
	return jax.random.categorical(key, jnp.where(legal, 0.0, -jnp.inf)).astype(jnp.int32)


def test_transition_matches_env_on_random_battles_and_held_retries(env, tables):
	state = jax.jit(env.new_battles)(KEY)
	rows = jnp.arange(BATCH)
	ability = state.ability.at[rows, 0, state.active[:, 0]].set(tables.abilities.index["pressure"])
	ability = ability.at[rows, 1, state.active[:, 1]].set(tables.abilities.index["shadowtag"])
	state = state._replace(ability=ability, trap_known=jnp.zeros_like(state.trap_known))
	es = jax.jit(env.init)(state, KEY)
	key = jax.random.key(81)
	refusals = held = one_sided = replacements = finished = 0
	for ply in range(130):
		key, policy = jax.random.split(key)
		actions = random_actions(es, policy)
		if ply == 0:
			actions = actions.at[:, 0].set(4)  # every opening switch from side 0 is refused
		if ply == 1:
			actions = actions.at[:, 1].set(99)  # ignored for the held side
		keys = jax.random.split(jax.random.split(es.key, 3)[1], BATCH)
		next_search, got = SEARCH_BATCH(position(es), actions, keys, None, env.tables)
		next_env, expected = COMPARE_ENV(env, es, actions)
		assert_equal(expected.final_state, next_search.state)
		ongoing = ~np.asarray(expected.info.ended)
		assert_equal(position(next_env), next_search, ongoing)
		assert_equal(expected.info.rejected, got.rejected)
		assert_equal(expected.info.illegal, got.illegal)
		assert_equal(expected.reward, got.reward)
		expected_obs = OBSERVE_ENV(env, expected.final_state)
		got_obs = OBSERVE_SEARCH(next_search, env.tables)
		assert_equal(expected_obs, got_obs)
		assert_equal(expected_obs.legal_actions, transition.observed_mask(next_search))
		if ply == 0:
			assert np.asarray(got.rejected[:, 0]).all()
			np.testing.assert_array_equal(next_search.held[:, 1], actions[:, 1])
		if ply == 1:
			assert not np.asarray(got.illegal[:, 1]).any()
		refusals += int(got.rejected.sum())
		held += int(next_search.has_held.sum())
		req = np.asarray(next_search.state.request)[ongoing]
		one_sided += int((req == NONE).any(axis=-1).sum())
		replacements += int((req == SWITCH).sum())
		finished += int(expected.info.ended.sum())
		es = next_env
	assert refusals >= BATCH and held >= BATCH
	assert one_sided > 0 and replacements > 0 and finished >= BATCH


def test_tiled_carries_and_lane_keys_with_forced_outcomes(env, tables):
	state = jax.tree.map(lambda x: x[0], jax.jit(env.new_battles)(KEY))
	moves = state.moves.at[0, state.active[0], 0].set(tables.moves.index["tackle"])
	moves = moves.at[1, state.active[1], 0].set(tables.moves.index["splash"])
	pp = state.pp.at[:, :, 0].set(20)
	ability = jnp.full_like(state.ability, tables.abilities.index["pressure"])
	state = state._replace(moves=moves, pp=pp, ability=ability, item=jnp.zeros_like(state.item))
	pos = transition.init(state, env.tables)
	carry = (jnp.asarray([[1.0, 2.0], [3.0, 4.0]]), jnp.asarray([[5.0], [6.0]]))
	lanes, carries = transition.tile((pos, carry), BATCH)
	for index in (0, BATCH - 1):
		assert_equal(carry, jax.tree.map(lambda x, index=index: x[index], carries))

	keys = jax.random.split(KEY, BATCH)
	actions = jnp.zeros((BATCH, 2), jnp.int32)
	forced = transition.tile(UNFORCED.at[CRIT, 0, 0].set(0), BATCH)
	forced = forced.at[:, HIT, 0, 0].set(jnp.arange(BATCH, dtype=forced.dtype) % 2)
	next_pos, out = SEARCH_BATCH(lanes, actions, keys, forced, env.tables)
	hp = np.asarray(next_pos.state.hp[:, 1, state.active[1]])
	assert np.unique(hp[::2]).size == 1
	assert (hp[1::2] < hp[::2]).all()
	assert np.unique(hp[1::2]).size > 1  # damage still samples from each lane's native chance key
	assert not np.asarray(out.illegal).any()

	winners = jnp.asarray([SIDE0, SIDE1, TIE, SIDE0] * (BATCH // 4), state.winner.dtype)
	ended = lanes.state._replace(winner=winners)
	done = jax.vmap(transition.init, in_axes=(0, None))(ended, env.tables)
	unchanged, outcome = SEARCH_BATCH(done, actions + 8, keys, None, env.tables)
	assert_equal(done, unchanged)
	np.testing.assert_array_equal(outcome.reward[:3], [[1, -1], [-1, 1], [0, 0]])
	assert not np.asarray(outcome.illegal | outcome.rejected).any()


@dataclass(frozen=True)
class TinyModel:
	"""Deterministic update so a repeated versus a genuine waiting observation is visible."""

	def apply(self, _params, obs, carry):
		advance = 1 + 10 * obs.request[..., NONE]
		after = jax.tree.map(lambda x: x + advance[..., None], carry)
		return after, jnp.where(obs.legal_actions, 0.0, -jnp.inf), advance


def test_held_carry_kept_but_waiting_carry_advances(env):
	state = jax.tree.map(lambda x: x[:2], jax.jit(env.new_battles)(KEY))
	state = state._replace(request=jnp.asarray([[MOVE, MOVE], [SWITCH, NONE]], state.request.dtype))
	positions = jax.vmap(transition.init, in_axes=(0, None))(state, env.tables)
	positions = positions._replace(has_held=jnp.asarray([[False, True], [False, False]]))
	carry = (jnp.zeros((2, 2, 1)), jnp.full((2, 2, 2), 7.0))
	after, logits, _, _ = forward(TinyModel(), {}, positions, carry, env.tables)
	np.testing.assert_array_equal(after[0][..., 0], [[1, 0], [1, 11]])
	np.testing.assert_array_equal(after[1][..., 0], [[8, 7], [8, 18]])
	np.testing.assert_array_equal(np.isfinite(logits[1, 1]), [True] + [False] * 8)


def test_root_batches_pair_opponent_actions_and_keys_and_pad(env):
	state = jax.tree.map(lambda x: x[:1], jax.jit(env.new_battles)(KEY))
	positions = jax.vmap(transition.init, in_axes=(0, None))(state, env.tables)
	config = SearchConfig(worlds=3, draws=2, opponent="mirror", padding=64)
	logits = jnp.where(transition.observed_mask(positions), 0.0, -jnp.inf)
	carry = jnp.asarray([[[8.0], [9.0]]])
	lanes, searched, legal = _prepare(
		config, positions, jnp.asarray([0]), carry, logits, None, jax.random.split(KEY, 1)
	)
	reps = config.replicates
	assert searched[0]
	np.testing.assert_array_equal(lanes.active[: 9 * reps], np.repeat(legal[0], reps))
	assert not np.asarray(lanes.active[9 * reps :]).any()
	assert len(lanes.active) == config.padding
	paired = np.asarray(lanes.actions[: 9 * reps, 1]).reshape(9, reps)
	assert len(set(paired[0])) > 1
	np.testing.assert_array_equal(paired, np.broadcast_to(paired[0], paired.shape))
	keys = np.asarray(jax.random.key_data(lanes.keys)).reshape(-1, 2)
	np.testing.assert_array_equal(keys[:reps], keys[reps : 2 * reps])
	assert np.unique(keys[:reps], axis=0).shape[0] == reps
	np.testing.assert_array_equal(lanes.carry, np.broadcast_to(carry, (64, 2, 1)))


def test_backup_and_depth_skip_held_and_waiting_requests(env):
	state = jax.tree.map(lambda x: x[:1], jax.jit(env.new_battles)(KEY))
	positions = jax.vmap(transition.init, in_axes=(0, None))(state, env.tables)
	conf = SearchConfig(worlds=1, draws=1, opponent="mirror", padding=1)
	carry = jnp.zeros((1, 2, 1))
	logits = jnp.where(transition.observed_mask(positions), 0.0, -jnp.inf)
	lanes, _, _ = _prepare(
		conf, positions, jnp.asarray([0]), carry, logits, None, jax.random.split(KEY, 1)
	)
	battle = lanes.position.state
	win = battle.winner.at[:4].set(jnp.asarray([ONGOING, ONGOING, SIDE0, TIE], battle.winner.dtype))
	req = battle.request.at[:4, 0].set(jnp.asarray([NONE, MOVE, NONE, NONE], battle.request.dtype))
	pos = lanes.position._replace(
		state=lanes.position.state._replace(winner=win, request=req),
		has_held=lanes.position.has_held.at[1, 0].set(True),
	)
	lanes = lanes._replace(active=jnp.arange(lanes.active.size) < 4)
	policy = jnp.zeros((9, 2, 9))
	values = jnp.full((9, 2), 0.5)
	next_lanes = _successor(conf, lanes, pos, lanes.carry, policy, values, 0)
	np.testing.assert_array_equal(next_lanes.active[:4], [True, True, False, False])
	np.testing.assert_allclose(next_lanes.score[:4], [0, 0, 1, 0])
	req = req.at[:2, 0].set(SWITCH)
	pos = pos._replace(state=pos.state._replace(request=req), has_held=lanes.position.has_held)
	leaves = _successor(conf, next_lanes, pos, lanes.carry, policy, values, 1)
	assert not np.asarray(leaves.active).any()
	np.testing.assert_allclose(leaves.score[:4], [0.5 * 0.99**2, 0.5 * 0.99**2, 1, 0])
	np.testing.assert_allclose(
		backup(jnp.asarray([SIDE0, SIDE1, TIE]), 0, 7.0, 2, 0.99), [0.99**2, -(0.99**2), 0.0]
	)


def test_compaction_keeps_original_pair_indices_and_completed_scores(env):
	state = jax.tree.map(lambda x: x[:2], jax.jit(env.new_battles)(KEY))
	positions = jax.vmap(transition.init, in_axes=(0, None))(state, env.tables)
	config = SearchConfig(worlds=1, draws=1, opponent="mirror", padding=1)
	logits = jnp.where(transition.observed_mask(positions), 0.0, -jnp.inf)
	lanes, _, _ = _prepare(
		config,
		positions,
		jnp.asarray([0, 1]),
		jnp.zeros((2, 2, 1)),
		logits,
		None,
		jax.random.split(KEY, 2),
	)
	ids = jnp.arange(lanes.active.size)
	lanes = lanes._replace(active=(ids == 1) | (ids == 12), score=jnp.full(ids.shape, 0.25))
	wave, wave_ids = _compact(lanes, ids, 4, ids.size)
	np.testing.assert_array_equal(wave_ids, [1, 12, ids.size, ids.size])
	np.testing.assert_array_equal(wave.active, [True, True, False, False])
	np.testing.assert_array_equal(wave.actions[:2], lanes.actions[jnp.asarray([1, 12])])
	scores = _save(lanes.score, wave_ids, jnp.asarray([-1.0, 1.0, 7.0, 7.0]))
	expected = np.full(ids.shape, 0.25)
	expected[[1, 12]] = [-1.0, 1.0]
	np.testing.assert_array_equal(scores, expected)


def test_decision_rule_masks_and_applies_a_strict_paired_margin():
	scores = jnp.asarray([[[0.25, -0.25], [0.5, 0.0], [9.0, 9.0]]] * 2)
	logits = jnp.asarray([[1.0, 0.0, 2.0]] * 2)
	legal = jnp.asarray([[True, True, False]] * 2)
	searched = jnp.asarray([True, False])
	noise = jnp.zeros_like(logits)
	got = decide(scores, legal, logits, searched, SearchConfig(), noise)
	np.testing.assert_array_equal(got.action, [1, 0])
	np.testing.assert_allclose(got.best_advantage, [0.25, 0.0])
	margin = decide(scores, legal, logits, searched, SearchConfig(margin=0.25), noise)
	np.testing.assert_array_equal(margin.action, [0, 0])
	sample = decide(
		scores,
		legal,
		logits,
		searched,
		SearchConfig(sample_strength=2),
		jnp.asarray([[0.0, 0.8, 0.0], [0.0, 0.8, 0.0]]),
	)
	np.testing.assert_array_equal(sample.raw, [0, 0])
	np.testing.assert_array_equal(sample.action, [1, 0])


def test_auxiliary_logits_condition_kind_and_split_unseen_mass():
	moves = jnp.asarray([2, 1, 1, 0])
	ranks = jnp.asarray([3, 6, 6, 0, 1])
	mask = jnp.asarray([True, False, False, False, True, True, True, False, False])
	probs = jnp.exp(auxiliary_logits(jnp.zeros(12), moves, ranks, mask))
	np.testing.assert_allclose(probs[:4].sum(), 0.4975, rtol=1e-6)
	np.testing.assert_allclose(probs[5], probs[6], rtol=1e-6)
	assert probs[4] > probs[5]
	np.testing.assert_array_equal(np.asarray(probs)[~np.asarray(mask)], 0)
	forced = jnp.arange(9) == 6
	np.testing.assert_allclose(
		jnp.exp(auxiliary_logits(jnp.zeros(12), moves, ranks, forced)), forced.astype(float)
	)


def test_match_samples_raw_policy_and_accepts_table_backed_baselines(env, tables):
	state = jax.tree.map(lambda x: x[:2], jax.jit(env.new_battles)(KEY))
	match = start(state, env.tables, KEY)
	sides = jnp.asarray([0, 1])
	rows = jnp.arange(2)
	legal = transition.observed_mask(match.position)
	logits = jnp.where(legal, 0.0, -jnp.inf)
	our_key, _ = jax.random.split(jax.random.fold_in(KEY, 0), 2)
	expected = jax.random.categorical(our_key, logits[rows, sides])
	engine_legal = block_switches(match.position.base, match.position.blocked)
	for name in ("max-damage", "heuristics"):
		policy = make_policy(name, env.tables, tables)
		own, foe, _ = choices(match, sides, logits, KEY, 0, policy, False)
		np.testing.assert_array_equal(own, expected)
		assert np.asarray(engine_legal[rows, 1 - sides, foe]).all()
