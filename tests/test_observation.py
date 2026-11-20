"""Public information boundaries exercised on real-team random rollouts."""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine import state as S
from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import Observation, observe
from pokejax.engine.randomness import CRIT, DAMAGE_ROLL, HIT, ORDER, UNFORCED
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import MOVE, NONE, SWITCH
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import load_team_pool

KEY = jax.random.key(83)
RUN = jax.jit(step)
OBSERVE = jax.jit(observe)


@pytest.fixture(scope="module")
def battles(tmp_path_factory):
	tables = load_tables()
	fixture = json.loads(Path("tests/fixtures/reset.json").read_text())
	path = tmp_path_factory.mktemp("observation") / "teams.json"
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
	pool = load_team_pool(path, tables)
	t = build_engine_tables(tables)
	indices = np.random.default_rng(23).integers(0, len(pool.team_size), (8, 2))
	teams = teams_from_pool(pool, indices[:, 0], indices[:, 1])
	initial = jax.vmap(reset, in_axes=(0, 0, None))(jax.random.split(KEY, 8), teams, t)

	@jax.jit
	def rollout(states):
		def advance(state, key):
			mask = jax.vmap(legal_actions, in_axes=(0, None))(state, t)
			action_key, step_key = jax.random.split(key)
			actions = jax.random.categorical(action_key, jnp.where(mask, 0.0, -jnp.inf))
			out = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
				state, actions, jax.random.split(step_key, 8), UNFORCED, t
			)
			return out, out

		return jax.lax.scan(advance, states, jax.random.split(KEY, 24))[1]

	history = rollout(initial)
	states = jax.tree.map(lambda x: x.reshape((-1, *x.shape[2:])), history)
	return tables, t, jax.tree.map(lambda x: x[0], initial), states


def assert_equal(a, b):
	for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
		np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("side", [0, 1])
def test_no_hidden_information(battles, side):
	_, t, _, states = battles
	foe = 1 - side

	def change(s):
		unseen = s.seen_order[foe] == 0
		updates = {}
		# These fields are entirely private, even for seen members.
		for name in (
			"pp",
			"max_pp",
			"evs",
			"ivs",
			"ability",
			"ability_base",
			"item",
			"nature",
			"happiness",
			"gender",
		):
			x = getattr(s, name)
			updates[name] = x.at[foe].set(jnp.zeros_like(x[foe]))
		# Change every stored stat, including max HP, while preserving public HP.
		stats = s.stats.at[foe].set(s.stats[foe] * 2)
		hp = s.hp.at[foe].set(s.hp[foe] * 2)
		updates.update(stats=stats, hp=hp)
		for name in ("species", "level", "types", "status"):
			x = getattr(s, name)
			mask = unseen.reshape((6,) + (1,) * (x.ndim - 2))
			updates[name] = x.at[foe].set(jnp.where(mask, jnp.ones_like(x[foe]), x[foe]))
		moves = jnp.where(s.revealed_moves[foe] != 0, s.moves[foe], 1)
		updates["moves"] = s.moves.at[foe].set(moves)
		return s._replace(**updates)

	changed = jax.jit(jax.vmap(change))(states)
	get = jax.jit(jax.vmap(observe, in_axes=(0, None, None)))
	assert_equal(get(states, side, t), get(changed, side, t))
	assert np.any(np.asarray(states.seen_order) == 0)
	assert np.any(np.asarray(states.revealed_moves) != 0)


@pytest.mark.parametrize("side", [0, 1])
def test_private_slot_permutations(battles, side):
	_, t, _, states = battles
	foe = 1 - side
	permutation = jnp.array([4, 2, 0, 5, 1, 3])
	inverse = jnp.argsort(permutation)

	def permute(s):
		updates = {}
		for name in s._fields:
			x = getattr(s, name)
			if x.ndim >= 2 and x.shape[:2] == (2, 6):
				updates[name] = x.at[foe].set(x[foe, permutation])
		updates["active"] = s.active.at[foe].set(inverse[s.active[foe]].astype(s.active.dtype))
		s = s._replace(**updates)
		for name in ("moves", "pp", "max_pp", "revealed_moves"):
			x = getattr(s, name)
			s = s._replace(**{name: x.at[foe].set(x[foe, :, ::-1])})
		return s

	get = jax.jit(jax.vmap(observe, in_axes=(0, None, None)))
	assert_equal(get(states, side, t), get(jax.jit(jax.vmap(permute))(states), side, t))


def test_reveal_updates(battles):
	_, t, s, _ = battles
	np.testing.assert_array_equal(s.seen_order, [[1, 0, 0, 0, 0, 0]] * 2)
	assert not np.any(s.revealed_moves)
	# Switch to slot 5 first, then slot 2, then back to lead.
	for actions, rank, slot in (([8, 8], 2, 5), ([6, 6], 3, 2), ([4, 4], 1, 0)):
		s = RUN(s, jnp.array(actions), KEY, UNFORCED, t)
		np.testing.assert_array_equal(s.active, [slot, slot])
		np.testing.assert_array_equal(s.seen_order[:, slot], [rank, rank])
	o = OBSERVE(s, 0, t)
	np.testing.assert_array_equal(o.opponent.species[:3], s.species[1, jnp.array([0, 5, 2])])
	np.testing.assert_array_equal(o.opponent.is_active, [1, 0, 0, 0, 0, 0])
	assert not np.any(s.revealed_moves)
	# Inert status moves still reveal; a miss also reveals. No damage cancels moves.
	moves = s.moves.at[:, 0, 2].set((t.struggle_id - 1).astype(s.moves.dtype))
	t = t._replace(move_base_power=jnp.zeros_like(t.move_base_power))
	s = s._replace(moves=moves, pp=s.pp.at[:, 0, 2].set(10))
	forced = UNFORCED.at[HIT, :, 0].set(0)
	out = RUN(s, jnp.array([2, 2]), KEY, forced, t)
	assert np.all(out.revealed_moves[:, 0, 2] == 1)
	assert np.sum(out.revealed_moves) == 2
	np.testing.assert_array_equal(OBSERVE(out, 0, t).opponent.moves[0], [moves[1, 0, 2], 0, 0, 0])
	struggle = s._replace(pp=jnp.zeros_like(s.pp))
	out = RUN(struggle, jnp.array([0, 0]), KEY, UNFORCED, t)
	assert not np.any(out.revealed_moves)
	# A KO before the second action reveals only the move actually attempted.
	s = s._replace(hp=s.hp.at[:, 0].set(1), moves=s.moves.at[:, 0, 0].set(1))
	t = t._replace(
		move_base_power=jnp.full_like(t.move_base_power, 100),
		move_always_hits=jnp.ones_like(t.move_always_hits),
		type_immune=jnp.zeros_like(t.type_immune),
	)
	s = s._replace(stats=s.stats.at[:, 0, 5].set(100), pp=s.pp.at[:, 0, 0].set(10))
	forced = UNFORCED.at[ORDER, 0, 0].set(0).at[CRIT, :, 0].set(0).at[DAMAGE_ROLL, :, 0].set(0)
	out = RUN(s, jnp.array([0, 0]), KEY, forced, t)
	assert out.revealed_moves[0, 0, 0] == 1
	assert not np.any(out.revealed_moves[1])
	assert out.request[1] == SWITCH
	out = RUN(out, jnp.array([0, 5]), KEY, forced, t)
	assert out.seen_order[1, 2] == 3  # Already seen: retain its first-appearance rank.


def test_hp_rounding_and_visible_features(battles):
	_, t, s, _ = battles
	# pokemon.ts:2096-2102: ceil, except damaged 100 becomes 99.
	for hp, maximum, percent in (
		(0, 301, 0),
		(1, 301, 1),
		(100, 301, 34),
		(300, 301, 99),
		(301, 301, 100),
		(1, 1, 100),
		(50, 100, 50),
	):
		s = s._replace(hp=s.hp.at[1, 0].set(hp), stats=s.stats.at[1, 0, 0].set(maximum))
		o = OBSERVE(s, 0, t)
		np.testing.assert_allclose(o.opponent.hp[0], percent / 100)
		assert o.opponent.fainted[0] == (hp == 0)
		np.testing.assert_allclose(o.own.hp, np.asarray(s.hp[0]) / np.asarray(s.stats[0, :, 0]))
	for request in (NONE, MOVE, SWITCH):
		s = s._replace(request=s.request.at[0].set(request))
		np.testing.assert_array_equal(OBSERVE(s, 0, t).request, np.eye(3)[request])
	s = s._replace(winner=jnp.int8(0))
	np.testing.assert_array_equal(OBSERVE(s, 0, t).request, [1, 0, 0])


def test_public_status_uses_seen_order(battles):
	_, t, s, _ = battles
	status = s.status.at[0].set(jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.int8))
	status = status.at[1].set(jnp.array([1, 2, 3, 4, 5, 6], dtype=jnp.int8))
	seen = s.seen_order.at[1, 5].set(2).at[1, 2].set(3)
	s = s._replace(status=status, seen_order=seen)
	o = OBSERVE(s, 0, t)
	np.testing.assert_array_equal(o.own.status, [1, 2, 3, 4, 5, 6])
	np.testing.assert_array_equal(o.opponent.status, [1, 6, 3, 0, 0, 0])


def test_weather_and_hazards_are_public_and_perspective_ordered(battles):
	_, t, s, _ = battles
	s = s._replace(
		weather=jnp.int8(3),
		weather_turns=jnp.int8(2),
		hazards=jnp.asarray([[2, 1], [3, 0]], jnp.int8),
	)
	for side, expected in ((0, [[2, 1], [3, 0]]), (1, [[3, 0], [2, 1]])):
		o = OBSERVE(s, side, t)
		assert int(o.weather) == 3
		assert float(o.weather_turns) == 2
		np.testing.assert_array_equal(o.hazards, expected)


def test_hp_against_pinned_protocol(battles):
	root = Path(__file__).resolve().parents[1]
	if not (root / "reference/vendor/pokemon-showdown/dist").is_dir():
		pytest.skip("Run node reference/setup.cjs for the protocol oracle")
	cases = [(hp, maximum) for maximum in (1, 100, 301, 714) for hp in range(maximum + 1)]
	script = """
	const fs = require('node:fs');
	const { Battle } = require('./reference/vendor/pokemon-showdown/dist/sim/battle');
	const teams = JSON.parse(fs.readFileSync('tests/fixtures/reset.json')).teamSets;
	const cases = JSON.parse(fs.readFileSync(0, 'utf8'));
	const battle = new Battle({formatid: 'gen5randombattle', seed: [1, 2, 3, 4],
		p1: {name: 'A', team: teams[0]}, p2: {name: 'B', team: teams[1]}});
	const pokemon = battle.sides[1].active[0];
	const health = cases.map(([hp, maximum]) => {
		pokemon.hp = hp; pokemon.maxhp = maximum;
		return parseInt(pokemon.getHealth().shared, 10);
	});
	console.log(JSON.stringify(health));
	battle.destroy();
	"""
	result = subprocess.run(
		["node", "-e", script],
		cwd=root,
		input=json.dumps(cases),
		capture_output=True,
		text=True,
		check=True,
	)
	_, t, s, _ = battles
	hp, maximum = jnp.asarray(cases).T
	states = jax.vmap(
		lambda h, m: s._replace(
			hp=s.hp.at[1, 0].set(h.astype(s.hp.dtype)),
			stats=s.stats.at[1, 0, 0].set(m.astype(s.stats.dtype)),
		)
	)(hp, maximum)
	observations = jax.jit(jax.vmap(observe, in_axes=(0, None, None)))(states, 0, t)
	np.testing.assert_allclose(
		observations.opponent.hp[:, 0], np.array(json.loads(result.stdout)) / 100
	)


def test_shapes_dtypes_jit_vmap(battles):
	_, t, s, states = battles
	observations = jax.jit(
		jax.vmap(lambda x: jax.vmap(observe, in_axes=(None, 0, None))(x, jnp.arange(2), t))
	)(states)
	assert isinstance(observations, Observation)
	o = OBSERVE(s, 0, t)
	np.testing.assert_array_equal(o.own.species, s.species[0])
	np.testing.assert_array_equal(o.own.moves, s.moves[0])
	np.testing.assert_array_equal(o.own.types, s.types[0] + 1)
	np.testing.assert_allclose(o.own.level, np.asarray(s.level[0]) / 100)
	np.testing.assert_allclose(o.own.stats, np.asarray(s.stats[0]) / 1000)
	np.testing.assert_allclose(o.own.pp, np.asarray(s.pp[0]) / np.maximum(s.max_pp[0], 1))
	boosted = s._replace(boosts=s.boosts.at[0, 0].set(-6).at[1, 1].set(3))
	o_boosted = OBSERVE(boosted, 0, t)
	assert o_boosted.own.boosts[0] == -1
	assert o_boosted.opponent.boosts[1] == 0.5
	for team in (o.own, o.opponent):
		for name, x in zip(team._fields, team):
			integer_fields = {"species", "types", "moves", "status", "item", "ability"}
			boolean_fields = {"item_consumed", "item_knocked_off"} if team is o.own else set()
			assert x.dtype == (
				jnp.int16
				if name in integer_fields
				else jnp.bool_
				if name in boolean_fields
				else jnp.float32
			)
			assert x.shape == {
				"types": (6, 2),
				"moves": (6, 4),
				"pp": (6, 4),
				"stats": (6, 6),
				"boosts": (7,),
				"confused": (),
				"charging": (),
				"recharging": (),
				"locked": (),
				"trapped": (),
			}.get(name, (6,))
	assert o.turn.shape == () and o.turn.dtype == jnp.float32
	assert o.request.shape == (3,) and o.request.dtype == jnp.float32
	assert o.legal_actions.shape == (9,) and o.legal_actions.dtype == jnp.bool_
	assert o.weather.shape == () and o.weather.dtype == jnp.int16
	assert o.weather_turns.shape == () and o.weather_turns.dtype == jnp.float32
	assert o.hazards.shape == (2, 2) and o.hazards.dtype == jnp.float32
	np.testing.assert_array_equal(o.legal_actions, legal_actions(s, t)[0])
	for x in jax.tree.leaves(observations):
		assert x.shape[:2] == (192, 2)
		assert np.all(np.isfinite(x))
	for name, x in zip(o.opponent._fields, o.opponent):
		if name not in (
			"unseen",
			"boosts",
			"confused",
			"charging",
			"recharging",
			"locked",
			"trapped",
		):
			assert np.all(np.asarray(x)[1:] == 0)
	np.testing.assert_array_equal(o.opponent.unseen, [0, 1, 1, 1, 1, 1])


@pytest.mark.parametrize("side", [0, 1])
def test_illusion_disguise_hides_the_real_species(battles, side):
	"""A disguised holder is shown as its disguise: its real species never reaches the observer."""
	tables, t, _, states = battles
	foe = 1 - side
	slots = len(tables.species.ids)

	def disguise(s):
		active = s.active[foe]
		bench = (active + 1) % 6
		volatiles = s.volatiles.at[foe, S.ILLUSION_SLOT].set(bench + 1)
		shown = s.shown_species.at[foe, active].set(s.species[foe, bench])
		return s._replace(volatiles=volatiles, shown_species=shown)

	def hide_real_species(s):
		active = s.active[foe]
		species = s.species.at[foe, active].set((s.species[foe, active] + 7) % slots)
		return s._replace(species=species)

	shown = jax.jit(jax.vmap(disguise))(states)
	changed = jax.jit(jax.vmap(hide_real_species))(shown)
	get = jax.jit(jax.vmap(observe, in_axes=(0, None, None)))
	assert_equal(get(shown, side, t), get(changed, side, t))
