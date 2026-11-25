"""Player-view reconstruction and information isolation for Method 1."""

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.observation import observe
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import (
	ILLUSION_SLOT,
	NONE,
	ONGOING,
	SLEEP,
	SWITCH,
	TEAM_SLOTS,
	TURN_RESIDUAL,
)
from pokejax.engine.switching import switch_members
from pokejax.env import Env, EnvConfig
from pokejax.search.belief import Belief, Catalogue
from pokejax.search.method1 import Method1, _worlds
from pokejax.search.oracle import SearchConfig
from pokejax.search.rebuild import views
from pokejax.search.transition import advance_batch, init, observed_mask
from pokejax.teams import save_npz, team_pool_from_data

KEY = jax.random.key(4128)
BATCH = 48


@pytest.fixture(scope="module")
def pool(tables, tmp_path_factory):
	recording = json.loads(Path("tests/fixtures/reset.json").read_text())
	teams = recording["teamSets"]
	pool = team_pool_from_data(
		{
			"format": "gen5randombattle",
			"seed": 0,
			"count": len(teams),
			"showdownCommit": recording["showdownCommit"],
			"teams": teams,
		},
		tables,
	)
	path = tmp_path_factory.mktemp("belief") / "pool.npz"
	save_npz(pool, path)
	return pool, path


@pytest.fixture(scope="module")
def catalogue(tables, pool):
	return Catalogue.load(pool[1], tables, cache=pool[1].with_name("catalogue.npz"))


@pytest.fixture(scope="module")
def env(tables, pool):
	return Env(EnvConfig(BATCH), pool[1], tables=tables)


def assert_tree(expected, actual):
	for e, a in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
		np.testing.assert_array_equal(e, a)


def row(tree, index):
	return jax.tree.map(lambda x: x[index], tree)


def stack(rows):
	return jax.tree.map(lambda *x: np.stack(x), *rows)


@dataclass(frozen=True)
class TinyValue:
	num_moves: int

	def full(self):
		pass

	def apply(self, _params, obs, carry, *, method=None):
		after = carry + obs.turn[..., None]
		value = (obs.own.hp.sum(-1) - obs.opponent.hp.sum(-1)) / 6
		logits = jnp.zeros_like(obs.legal_actions, dtype=jnp.float32)
		if method is None:
			return after, logits, value
		head = jnp.zeros((*value.shape, self.num_moves + 9), jnp.float32)
		return after, logits, value, {"next_action": head}


def test_catalogue_cache_and_weighted_joint_sets(tables, pool, catalogue):
	cached = Catalogue.load(pool[1], tables, cache=pool[1].with_name("catalogue.npz"))
	assert_tree(catalogue.sets, cached.sets)
	np.testing.assert_array_equal(catalogue.weight, cached.weight)
	assert catalogue.weight.sum() == int(pool[0].team_size.sum())
	i = int(np.argmax(catalogue.weight))
	member = row(catalogue.sets, i)
	valid = catalogue.candidates(
		member.species, member.level, member.moves, member.item, member.ability
	)
	assert i in valid
	assert catalogue.candidates(member.species, 255, (), 0, 0).size == 0


def test_illusion_keeps_both_identities_and_resolves_duplicate(catalogue):
	c = catalogue
	zoroark = c.zoroark
	assert zoroark in c.by_species
	row = next(i for i, sp in enumerate(c.sets.species) if sp != zoroark)
	appearance = (int(c.sets.species[row]), int(c.sets.level[row]), (), 0, 0)
	belief = Belief(c, np.random.default_rng(18))
	bank, weights = belief._propose((appearance,))
	assert (bank.disguise[:, 0] == -1).any()
	assert (bank.disguise[:, 0] == 1).any()
	np.testing.assert_allclose(weights.sum(), 1)
	bank, _ = belief._propose((appearance, appearance))
	assert ((bank.disguise >= 0).sum(1) == 1).all()
	assert ((c.sets.species[bank.sets] == zoroark).sum(1) == 1).all()
	assert belief._propose((appearance, appearance, appearance)) is None


def _true_compatible(belief, catalogue, initial, state, side):
	evidence = belief.signature
	foe = 1 - side
	ranks = np.asarray(state.seen_order[foe])
	slots = np.argsort(np.where(ranks, ranks, 99))[: len(evidence)]
	true_teams = []
	for slot in slots:
		sp = int(initial.species[foe, slot])
		sets = catalogue.sets
		rows = catalogue.by_species[sp]
		true_move = -np.sort(-initial.moves[foe, slot])
		ok = sets.level[rows] == initial.level[foe, slot]
		ok &= (sets.moves[rows] == true_move).all(1)
		ok &= sets.item[rows] == initial.item[foe, slot]
		ok &= sets.ability[rows] == initial.ability[foe, slot]
		ok &= (sets.ivs[rows] == initial.ivs[foe, slot]).all(1)
		ok &= (sets.evs[rows] == initial.evs[foe, slot]).all(1)
		ok &= sets.nature[rows] == initial.nature[foe, slot]
		true_teams.append(rows[ok])
	return any(
		all(
			np.intersect1d(truth, candidate).size
			for truth, candidate in zip(true_teams, rows, strict=True)
		)
		for _, _, _, rows, _ in belief._hypotheses(evidence)
	)


@jax.jit
def game_play(position, key, tables):
	move, chance = jax.random.split(key)
	raw = jnp.where(observed_mask(position), 0.0, -jnp.inf)
	actions = jax.random.categorical(move, raw).astype(jnp.int32)
	updated, outcome = advance_batch(
		position, actions, jax.random.split(chance, BATCH), None, tables
	)
	return updated, outcome, actions


def test_many_random_requests_reproduce_view_and_keep_true_team(env, pool, catalogue, tables):
	rng = np.random.default_rng(4)
	supported = np.asarray(env.team_indices)
	indices = rng.choice(supported, size=(2, BATCH))
	for i, name in enumerate(("zoroark", "ditto")):
		contains = (pool[0].species[supported] == tables.species.index[name]).any(1)
		if contains.any():
			indices[1, i * 2 : i * 2 + 2] = supported[contains][0]
			indices[0, i * 2 : i * 2 + 2] = supported[contains][0]
	teams = teams_from_pool(pool[0], *indices)
	keys = jax.random.split(KEY, BATCH)
	battle = jax.jit(jax.vmap(reset, in_axes=(0, 0, None)))(keys, teams, env.tables)
	position = jax.jit(jax.vmap(init, in_axes=(0, None)))(battle, env.tables)
	seed = 14
	method = Method1(
		TinyValue(len(tables.moves.ids)), env.tables, catalogue, SearchConfig(worlds=8)
	)
	beliefs = method.new_beliefs(seed, BATCH)
	sides = jnp.arange(BATCH) % 2
	host_sides = np.arange(BATCH) % 2
	rows = jnp.arange(BATCH)
	refused = jnp.zeros(BATCH, jnp.bool_)
	actions = jnp.zeros((BATCH, 2), jnp.int32)
	original = jax.device_get(teams)
	world_count = switches = pivots = illusion = transformed = 0
	for ply in range(75):
		view = views(position, sides, env.tables, refused)
		public = jax.device_get(view)
		actual = jax.device_get(position.state)
		histories, particles = [], []
		for i, belief in enumerate(beliefs):
			observation = row(public.obs, i)
			live = public.result[i] == ONGOING
			histories.append(belief.update(observation, public.refused[i], propose=live))
			part, count, _ = belief.sample(8) if live else belief.empty(8)
			particles.append(part)
			if not live:
				continue
			assert _true_compatible(
				belief, catalogue, row(original, i), row(actual, i), host_sides[i]
			), (i, ply, belief.signature, actual.species_base[i], actual.seen_order[i])
			assert count == 128
			switches += int(observation.request[SWITCH])
			pivots += int(
				observation.request[SWITCH]
				and not (observation.own.fainted * observation.own.is_active).any()
			)
			illusion += int(actual.volatiles[i, 1 - host_sides[i], ILLUSION_SLOT] > 0)
			transformed += int(observation.opponent.transformed)
		worlds, valid = _worlds(
			view,
			stack(particles),
			stack(histories),
			actions[rows, sides],
			jnp.zeros((BATCH, len(tables.moves.ids) + 9)),
			keys,
			method.data,
			env.tables,
		)
		valid = np.asarray(valid)
		live = public.result == ONGOING
		mask = np.asarray(observed_mask(worlds))[np.arange(BATCH), :, host_sides]
		expected = np.broadcast_to(public.obs.legal_actions[:, None], mask.shape)
		np.testing.assert_array_equal(mask[live], expected[live])
		# Samples incompatible with a discovered trap are discarded by the same
		# validation gate used for deployment. Test every retained observation.
		checked = live & ~np.asarray(actual.trap_known)[np.arange(BATCH), host_sides]
		assert valid[checked].all(), (ply, np.flatnonzero(~valid.all(1) & checked))
		world_count += int(valid[live].sum())
		position, outcome, actions = game_play(position, jax.random.fold_in(KEY, ply), env.tables)
		refused = outcome.rejected[rows, sides]
	assert world_count > 10_000
	assert switches > 100 and pivots > 0
	assert illusion > 0 and transformed > 0
	print(
		f"{world_count:,} valid worlds, {switches} switch requests, {pivots} pivots, "
		f"{illusion} Illusion and {transformed} Transform views"
	)


def test_decisions_ignore_hidden_sets_and_held_opponent_actions(env, catalogue, tables):
	state = row(jax.jit(env.new_battles)(KEY), 0)
	side = jnp.asarray([0])
	init_batch = jax.jit(jax.vmap(init, in_axes=(0, None)))
	# Change only the unseen enemy, its live hidden item, and its private queued
	# data. The battle can have the very same visible prefix.
	twin = state._replace(
		item=state.item.at[1, 0].set(catalogue.sets.item[0]),
		moves=state.moves.at[1, 1].set(catalogue.sets.moves[0]),
		turn_moves=state.turn_moves.at[1].set(catalogue.sets.moves[0, 0]),
		turn_actions=state.turn_actions.at[1].set(2),
	)
	method = Method1(
		TinyValue(len(tables.moves.ids)),
		env.tables,
		catalogue,
		SearchConfig(worlds=4, sample_strength=4.0),
	)

	def query(battle, held):
		position = init_batch(jax.tree.map(lambda x: x[None], battle), env.tables)
		position = position._replace(
			held=position.held.at[0, 1].set(held), has_held=position.has_held.at[0, 1].set(True)
		)
		view = views(position, side, env.tables, jnp.asarray([False]))
		result = method.request(
			{},
			view,
			method.new_beliefs(87, 1),
			jnp.zeros((1, 2)),
			KEY[None],
			jnp.zeros((1, 9)),
			jnp.zeros(1, jnp.int32),
		)
		return view, result

	first, a = query(state, 0)
	second, b = query(twin, 3)
	assert_tree(first, second)
	assert_tree(a, b)
	assert np.asarray(a[0].searched).all()


def test_hp_rounding_and_empty_belief_falls_back(env, catalogue, tables):
	from pokejax.engine.events import protocol_percent
	from pokejax.search.rebuild import _hp_from_percent

	maxima = jnp.arange(1, 405)[:, None]
	hp = jnp.minimum(jnp.arange(405)[None, :], maxima)
	percents = protocol_percent(hp, maxima)
	restored = _hp_from_percent(percents / 100, maxima)
	np.testing.assert_array_equal(protocol_percent(restored, maxima), percents)

	state = jax.jit(env.new_battles)(KEY)
	position = jax.jit(jax.vmap(init, in_axes=(0, None)))(row(state, slice(0, 1)), env.tables)
	view = views(position, jnp.zeros(1, jnp.int32), env.tables, jnp.zeros(1, bool))
	foe = view.obs.opponent._replace(level=view.obs.opponent.level.at[0, 0].set(2.55))
	view = view._replace(obs=view.obs._replace(opponent=foe))
	method = Method1(TinyValue(len(tables.moves.ids)), env.tables, catalogue)
	decision, metrics, _ = method.request(
		{},
		view,
		method.new_beliefs(87, 1),
		jnp.zeros((1, 2)),
		KEY[None],
		jnp.zeros((1, 9)),
		jnp.zeros(1, jnp.int32),
	)
	assert metrics.no_particle[0]
	assert not decision.searched[0]
	assert decision.action[0] == decision.raw[0]


def test_pivot_ko_leaves_fainted_opponent_waiting(env, catalogue, tables):
	battle = row(jax.jit(env.new_battles)(KEY), 0)
	battle = battle._replace(
		hp=battle.hp.at[1, battle.active[1]].set(0),
		request=battle.request.at[0].set(SWITCH).at[1].set(NONE),
		turn_stage=jnp.asarray(TURN_RESIDUAL, battle.turn_stage.dtype),
	)
	position = jax.jit(jax.vmap(init, in_axes=(0, None)))(
		jax.tree.map(lambda x: x[None], battle), env.tables
	)
	view = views(position, jnp.zeros(1, jnp.int32), env.tables, jnp.zeros(1, bool))
	belief = Belief(catalogue, np.random.default_rng(234))
	history = belief.update(row(jax.device_get(view.obs), 0))
	particles, count, _ = belief.sample(4)
	assert count > 0
	method = Method1(
		TinyValue(len(tables.moves.ids)), env.tables, catalogue, SearchConfig(worlds=4)
	)
	result, valid = _worlds(
		view,
		stack([particles]),
		stack([history]),
		jnp.zeros(1, jnp.int32),
		jnp.zeros((1, len(tables.moves.ids) + 9)),
		KEY[None],
		method.data,
		env.tables,
	)
	assert np.asarray(valid).all()
	np.testing.assert_array_equal(result.state.request[0], [[SWITCH, NONE]] * 4)
	np.testing.assert_array_equal(result.state.turn_stage[0], [TURN_RESIDUAL] * 4)


def test_copied_ability_reveal_survives_switch_without_rejecting_ditto(
	env, pool, catalogue, tables
):
	ditto = tables.species.index["ditto"]
	intimidate = tables.abilities.index["intimidate"]
	our_team, our_lead = np.argwhere(pool[0].ability == intimidate)[0]
	foe_team, foe_lead = np.argwhere(pool[0].species == ditto)[0]
	teams = teams_from_pool(pool[0], our_team, foe_team)
	order = np.broadcast_to(np.arange(TEAM_SLOTS), (2, TEAM_SLOTS)).copy()
	for side, lead in enumerate((our_lead, foe_lead)):
		order[side, [0, lead]] = order[side, [lead, 0]]
	teams = jax.tree.map(lambda x: x[np.arange(2)[:, None], order], teams)
	state = jax.jit(reset)(KEY, teams, env.tables)
	observation = jax.jit(observe)
	obs = jax.device_get(observation(state, 0, env.tables))
	assert obs.opponent.transformed
	assert obs.opponent.ability[0] == intimidate
	belief = Belief(catalogue, np.random.default_rng(10))
	belief.update(obs)
	assert belief.bank is not None

	# An announced ability stays on the roster after its engine ability is restored.
	next_state = jax.jit(switch_members)(
		state, jnp.asarray([False, True]), jnp.asarray([0, 1]), env.tables
	)
	obs = jax.device_get(observation(next_state, 0, env.tables))
	assert not obs.opponent.transformed and obs.opponent.species[0] == ditto
	assert obs.opponent.ability[0] == intimidate
	belief.update(obs)
	worlds, count, _ = belief.sample(32)
	assert count == 128
	assert (catalogue.sets.species[worlds.sets[:, 0]] == ditto).all()


def test_sleep_origin_from_public_engine_events(env, catalogue):
	state = row(jax.jit(env.new_battles)(KEY), 0)
	obs = jax.device_get(observe(state, 0, env.tables))
	belief = Belief(catalogue, np.random.default_rng(42))
	belief.update(obs, propose=False)
	status = np.array(obs.opponent.status)
	active = int(np.argmax(obs.opponent.is_active))
	status[active] = SLEEP
	moves = np.array(obs.events.move)
	moves[1] = catalogue.move["rest"]
	event_status = np.array(obs.events.status)
	event_status[1] = SLEEP
	changed = obs._replace(
		opponent=obs.opponent._replace(status=status),
		events=obs.events._replace(move=moves, status=event_status),
	)
	history = belief.update(changed, propose=False)
	assert not history.sleep_from_foe[active]
	assert not history.sleep_unknown[active]
	other = Belief(catalogue, np.random.default_rng(43))
	other.update(obs, propose=False)
	moves[0], moves[1] = catalogue.move["spore"], 0
	spore = changed._replace(events=changed.events._replace(move=moves))
	history = other.update(spore, propose=False)
	assert history.sleep_from_foe[active]
	assert not history.sleep_unknown[active]
	yawn = Belief(catalogue, np.random.default_rng(44))
	yawned = obs._replace(opponent=obs.opponent._replace(yawn=1))
	yawn.update(yawned, propose=False)
	moves[0] = 0
	asleep = changed._replace(events=changed.events._replace(move=moves))
	history = yawn.update(asleep, propose=False)
	assert history.sleep_from_foe[active]
	assert not history.sleep_unknown[active]
