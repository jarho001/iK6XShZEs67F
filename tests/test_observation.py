"""Public information boundaries exercised on real-team random rollouts and forced outcomes."""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine import events as EV
from pokejax.engine import state as S
from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import Observation, observe
from pokejax.engine.randomness import (
	ACTION_ORDER,
	CONFUSION_HIT,
	CRIT,
	DAMAGE_ROLL,
	DRAG_IN,
	FULL_PARALYSIS,
	HIT,
	HIT_COUNT,
	ORDER,
	UNFORCED,
)
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import MOVE, NONE, SWITCH
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import load_team_pool

KEY = jax.random.key(83)
RUN = jax.jit(step)
OBSERVE = jax.jit(observe)
FORCED = UNFORCED.at[HIT, :, 0].set(1).at[CRIT, :, :].set(0).at[DAMAGE_ROLL, :, :].set(0)


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


def play(s, actions, t, forced=UNFORCED):
	return RUN(s, jnp.array(actions), KEY, forced, t)


def set_active_move(s, tables, side, slot, name):
	member = s.active[side]
	return s._replace(
		moves=s.moves.at[side, member, slot].set(tables.moves.index[name]),
		pp=s.pp.at[side, member, slot].set(10),
	)


def set_active_types(s, tables, side, names):
	member = s.active[side]
	types = [tables.types.index[name] for name in names] + [-1] * (2 - len(names))
	return s._replace(types=s.types.at[side, member].set(jnp.array(types, jnp.int8)))


def no_items(s):
	"""Held items add their own HP changes (Life Orb recoil, Leftovers); the HP tests clear them."""
	return s._replace(item=jnp.zeros_like(s.item))


def alive_bench(s, side):
	slots = np.arange(6)
	return int(next(x for x in slots if x != int(s.active[side]) and int(s.hp[side, x]) > 0))


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
		# A benched member shows the HP it left with: the true HP behind it (Regenerator) is private.
		seen = s.hp_seen.at[foe].set(s.hp_seen[foe] * 2)
		updates.update(stats=stats, hp=hp, hp_seen=seen)
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


PRIVATE_COUNTERS = (
	S.CONFUSION,
	S.ENCORE_REMAINING,
	S.TAUNT,
	S.DISABLE_REMAINING,
	S.YAWN,
	S.SLOW_START,
	S.SUBSTITUTE,
)


@pytest.mark.parametrize("side", [0, 1])
def test_private_counters_and_durations_do_not_leak(battles, side):
	"""Counters and HP that the protocol never states, on both sides, change no observation."""
	_, t, _, states = battles
	# Give every battle the restricting volatiles, so the counters are live for both sides.
	volatiles = states.volatiles
	for column, value in (
		(S.CONFUSION, 3),
		(S.ENCORE_REMAINING, 3),
		(S.ENCORE_MOVE, 5),
		(S.TAUNT, 3),
		(S.DISABLE_REMAINING, 4),
		(S.DISABLE_MOVE, 6),
		(S.YAWN, 2),
		(S.SLOW_START, 5),
		(S.SUBSTITUTE, 9),
	):
		volatiles = volatiles.at[:, :, column].set(value)
	states = states._replace(
		volatiles=volatiles,
		trick_room=jnp.full_like(states.trick_room, 5),
		weather=jnp.full_like(states.weather, S.RAIN),
		weather_turns=jnp.full_like(states.weather_turns, 8),
		wish_turns=jnp.full_like(states.wish_turns, 2),
		wish_hp=jnp.full_like(states.wish_hp, 60),
	)

	def change(s):
		volatiles = s.volatiles
		for column in PRIVATE_COUNTERS:
			# Presence is kept (a counter stays positive), the value is not.
			keep = volatiles[:, column] > 0
			volatiles = volatiles.at[:, column].set(
				jnp.where(keep, jnp.where(volatiles[:, column] > 2, 2, 5), 0)
			)
		volatiles = volatiles.at[:, S.SUBSTITUTE].set(
			jnp.where(s.volatiles[:, S.SUBSTITUTE] > 0, 1, 0)
		)
		return s._replace(
			volatiles=volatiles,
			trick_room=jnp.full_like(s.trick_room, 1),
			weather_turns=jnp.full_like(s.weather_turns, 1),
			wish_turns=jnp.full_like(s.wish_turns, 1),
			wish_hp=jnp.full_like(s.wish_hp, 7),
		)

	get = jax.jit(jax.vmap(observe, in_axes=(0, None, None)))
	changed = jax.jit(jax.vmap(change))(states)
	assert_equal(get(states, side, t), get(changed, side, t))


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


def test_weather_trick_room_and_wish_are_ages_and_flags(battles):
	_, t, s, _ = battles
	s = s._replace(
		weather=jnp.int8(S.RAIN),
		weather_turns=jnp.int8(8),
		weather_age=jnp.int8(3),
		hazards=jnp.asarray([[2, 1], [3, 0]], jnp.int8),
		trick_room=jnp.int8(4),
		trick_room_age=jnp.int8(2),
		wish_turns=jnp.asarray([2, 0], jnp.int8),
		wish_age=jnp.asarray([1, 0], jnp.int8),
		healing_wish=jnp.asarray([False, True]),
	)
	for side, expected in ((0, [[2, 1], [3, 0]]), (1, [[3, 0], [2, 1]])):
		o = OBSERVE(s, side, t)
		assert int(o.weather) == S.RAIN
		assert float(o.weather_age) == 3
		assert not hasattr(o, "weather_turns")
		np.testing.assert_array_equal(o.hazards, expected)
		assert float(o.trick_room) == 1 and float(o.trick_room_age) == 2
		if side == 0:
			np.testing.assert_array_equal(o.wish, [1, 0])
			np.testing.assert_array_equal(o.wish_age, [1, 0])
			np.testing.assert_array_equal(o.healing_wish, [0, 1])
		else:
			np.testing.assert_array_equal(o.wish, [0, 1])
			np.testing.assert_array_equal(o.wish_age, [0, 1])
			np.testing.assert_array_equal(o.healing_wish, [1, 0])
	# The remaining counts are private: changing them changes nothing public.
	changed = s._replace(
		weather_turns=jnp.int8(1),
		trick_room=jnp.int8(1),
		wish_turns=jnp.asarray([1, 0], jnp.int8),
	)
	assert_equal(OBSERVE(s, 0, t), OBSERVE(changed, 0, t))


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


def test_public_volatiles_follow_the_protocol(battles):
	"""Each public flag and count is set from the state that the matching protocol line sets."""
	tables, t, s, _ = battles
	s = set_active_move(s, tables, 0, 0, "tackle")
	volatiles = s.volatiles.at[1, S.SUBSTITUTE].set(3)
	volatiles = (
		volatiles.at[1, S.ENCORE_REMAINING]
		.set(2)
		.at[1, S.ENCORE_MOVE]
		.set(tables.moves.index["tackle"])
	)
	volatiles = volatiles.at[1, S.TAUNT].set(3).at[1, S.DISABLE_REMAINING].set(4)
	volatiles = volatiles.at[1, S.DISABLE_MOVE].set(tables.moves.index["tackle"])
	volatiles = volatiles.at[1, S.LEECH_SEED].set(1).at[1, S.YAWN].set(2).at[1, S.CURSE].set(1)
	volatiles = volatiles.at[1, S.DESTINY_BOND].set(1).at[1, S.TRANSFORMED].set(1)
	volatiles = volatiles.at[1, S.FLASH_FIRE].set(1).at[1, S.SLOW_START].set(5)
	volatiles = volatiles.at[1, S.AUTOTOMIZE].set(2).at[1, S.PROTECT].set(1)
	volatiles = volatiles.at[1, S.ROOST].set(1)
	s2 = s._replace(volatiles=volatiles)
	o = OBSERVE(s2, 0, t)
	opp = o.opponent
	assert float(opp.substitute) == 1
	assert float(opp.encore) == 1 and int(opp.encore_move) == tables.moves.index["tackle"]
	assert float(opp.taunt) == 1 and float(opp.disable) == 1
	assert int(opp.disabled_move) == tables.moves.index["tackle"]
	assert float(opp.leech_seed) == 1 and float(opp.yawn) == 1 and float(opp.curse) == 1
	assert float(opp.destiny_bond) == 1 and float(opp.transformed) == 1
	assert float(opp.flash_fire) == 1 and float(opp.slow_start) == 1
	assert float(opp.autotomize) == 2 and float(opp.protect) == 1 and float(opp.roost) == 1
	# Without the counter, the move ID is not shown; the flag follows the remaining count only.
	cleared = s2._replace(volatiles=volatiles.at[1, S.ENCORE_REMAINING].set(0))
	o = OBSERVE(cleared, 0, t)
	assert float(o.opponent.encore) == 0 and int(o.opponent.encore_move) == 0
	# The same fields for the own side, from its own volatiles.
	own = OBSERVE(s2._replace(volatiles=volatiles.at[0, S.TAUNT].set(1)), 0, t).own
	assert float(own.taunt) == 1 and float(own.substitute) == 0


def test_choice_lock_is_own_only(battles):
	tables, t, s, _ = battles
	s = set_active_move(s, tables, 0, 0, "tackle")
	member = s.active[0]
	s = s._replace(
		item=s.item.at[0, member].set(tables.items.index["choiceband"]),
		volatiles=s.volatiles.at[0, S.CHOICE_MOVE].set(tables.moves.index["tackle"]),
	)
	o = OBSERVE(s, 0, t)
	assert float(o.own.choice_locked) == 1
	assert int(o.own.choice_move) == tables.moves.index["tackle"]
	assert not hasattr(o.opponent, "choice_locked")
	# A lock without a Choice item is inert, and the opponent's lock is never shown.
	inert = s._replace(item=s.item.at[0, member].set(0))
	assert float(OBSERVE(inert, 0, t).own.choice_locked) == 0
	foe_locked = s._replace(volatiles=s.volatiles.at[1, S.LOCKED_MOVE].set(3))
	assert not hasattr(OBSERVE(foe_locked, 0, t).opponent, "choice_locked")


def test_opponent_types_follow_disguise_and_roost(battles):
	tables, t, s, _ = battles
	s = set_active_types(s, tables, 1, ["fire", "flying"])
	member = s.active[1]
	o = OBSERVE(s, 0, t)
	np.testing.assert_array_equal(
		o.opponent.types[int(np.argmax(np.asarray(o.opponent.is_active)))],
		[tables.types.index["fire"] + 1, tables.types.index["flying"] + 1],
	)
	# Roost's Flying removal is not announced: the observation shows the pre-roost types.
	roosting = s._replace(
		volatiles=s.volatiles.at[1, S.ROOST].set(1),
		types=s.types.at[1, member].set(jnp.array([tables.types.index["fire"], -1], jnp.int8)),
		roost_types=s.roost_types.at[1, member].set(
			jnp.array([tables.types.index["fire"], tables.types.index["flying"]], jnp.int8)
		),
	)
	o = OBSERVE(roosting, 0, t)
	active = int(np.argmax(np.asarray(o.opponent.is_active)))
	np.testing.assert_array_equal(
		o.opponent.types[active],
		[tables.types.index["fire"] + 1, tables.types.index["flying"] + 1],
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
	integer_fields = {"species", "types", "moves", "status", "item", "ability"}
	integer_fields |= {"encore_move", "disabled_move", "choice_move"}
	scalar_fields = {
		"confused",
		"charging",
		"recharging",
		"locked",
		"trapped",
		"substitute",
		"encore",
		"encore_move",
		"taunt",
		"disable",
		"disabled_move",
		"leech_seed",
		"yawn",
		"curse",
		"destiny_bond",
		"transformed",
		"flash_fire",
		"slow_start",
		"autotomize",
		"protect",
		"roost",
		"choice_locked",
		"choice_move",
	}
	shapes = {
		"types": (6, 2),
		"moves": (6, 4),
		"pp": (6, 4),
		"stats": (6, 6),
		"boosts": (7,),
	}
	for team in (o.own, o.opponent):
		for name, x in zip(team._fields, team):
			boolean_fields = {"item_consumed", "item_knocked_off"} if team is o.own else set()
			assert x.dtype == (
				jnp.int16
				if name in integer_fields
				else jnp.bool_
				if name in boolean_fields
				else jnp.float32
			), name
			if name in shapes:
				assert x.shape == shapes[name], name
			elif name in scalar_fields:
				assert x.shape == (), name
			else:
				assert x.shape == (6,), name
	for name, x in zip(o.events._fields, o.events):
		assert x.shape == (2,), name
		want = (
			jnp.int16
			if name in {"order", "move", "cant", "effectiveness", "entered", "status"}
			else jnp.float32
		)
		assert x.dtype == want, name
	assert o.turn.shape == () and o.turn.dtype == jnp.float32
	assert o.request.shape == (3,) and o.request.dtype == jnp.float32
	assert o.legal_actions.shape == (9,) and o.legal_actions.dtype == jnp.bool_
	assert o.weather.shape == () and o.weather.dtype == jnp.int16
	assert o.weather_age.shape == () and o.weather_age.dtype == jnp.float32
	assert o.hazards.shape == (2, 2) and o.hazards.dtype == jnp.float32
	for name in ("trick_room", "trick_room_age"):
		assert getattr(o, name).shape == () and getattr(o, name).dtype == jnp.float32
	for name in ("wish", "wish_age", "healing_wish"):
		assert getattr(o, name).shape == (2,) and getattr(o, name).dtype == jnp.float32
	np.testing.assert_array_equal(o.legal_actions, legal_actions(s, t)[0])
	for x in jax.tree.leaves(observations):
		assert x.shape[:2] == (192, 2)
		assert np.all(np.isfinite(x))
	for name, x in zip(o.opponent._fields, o.opponent):
		if name not in ("unseen", "boosts") and name not in scalar_fields:
			assert np.all(np.asarray(x)[1:] == 0), name
	np.testing.assert_array_equal(o.opponent.unseen, [0, 1, 1, 1, 1, 1])
	# Event codes stay inside their enumerations over every recorded rollout state.
	events = observations.events
	assert np.all((np.asarray(events.order) >= 0) & (np.asarray(events.order) <= 2))
	assert np.all((np.asarray(events.cant) >= 0) & (np.asarray(events.cant) <= 10))
	assert np.all((np.asarray(events.effectiveness) >= 0) & (np.asarray(events.effectiveness) <= 4))
	assert np.all((np.asarray(events.entered) >= 0) & (np.asarray(events.entered) <= 4))
	assert np.all((np.asarray(events.status) >= 0) & (np.asarray(events.status) <= 6))
	assert np.all(np.asarray(events.hits) <= 5)
	assert np.all(np.abs(np.asarray(events.hp_change)) <= 1)


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


@pytest.mark.parametrize("side", [0, 1])
def test_illusion_identity_is_private(battles, side):
	"""Which bench member hides behind a disguise is private; the disguise species is shown.

	Two bench members that look alike (same species and level) are interchangeable as the source.
	"""
	_tables, t, _, states = battles
	foe = 1 - side

	def disguise(s, offset):
		active = s.active[foe]
		twin, bench = (active + 1) % 6, (active + offset) % 6
		# A seen member is part of the roster: only an unseen look-alike stands in for the twin.
		bench = jnp.where(s.seen_order[foe, bench] > 0, twin, bench)
		species = s.species.at[foe, bench].set(s.species[foe, twin])
		level = s.level.at[foe, bench].set(s.level[foe, twin])
		volatiles = s.volatiles.at[foe, S.ILLUSION_SLOT].set(bench + 1)
		shown = s.shown_species.at[foe, active].set(species[foe, twin])
		return s._replace(volatiles=volatiles, shown_species=shown, species=species, level=level)

	first = jax.jit(jax.vmap(lambda s: disguise(s, 1)))(states)
	second = jax.jit(jax.vmap(lambda s: disguise(s, 2)))(states)
	get = jax.jit(jax.vmap(observe, in_axes=(0, None, None)))
	assert_equal(get(first, side, t), get(second, side, t))


# Event record: each test forces the outcomes of one step and checks the record it leaves.


def test_event_record_order_moves_and_outcome(battles):
	tables, t, s, _ = battles
	s = no_items(s)
	s = set_active_move(s, tables, 0, 0, "tackle")
	s = set_active_move(s, tables, 1, 0, "tackle")
	s = set_active_types(s, tables, 0, ["normal"])
	s = set_active_types(s, tables, 1, ["normal"])
	forced = FORCED.at[ORDER, 0, ACTION_ORDER].set(0)
	out = play(s, [0, 0], t, forced)
	event = np.asarray(out.event)
	# Speed decides the order unless it ties; either way the two moves take positions 1 and 2.
	assert sorted(event[:, EV.EV_ORDER]) == [1, 2]
	np.testing.assert_array_equal(out.event_move, [tables.moves.index["tackle"]] * 2)
	np.testing.assert_array_equal(event[:, EV.EV_CANT], [0, 0])
	np.testing.assert_array_equal(event[:, EV.EV_OUTCOME], [0, 0])
	np.testing.assert_array_equal(event[:, EV.EV_EFFECT], [EV.EFFECT_NEUTRAL] * 2)
	np.testing.assert_array_equal(event[:, EV.EV_ENTRY], [0, 0])
	np.testing.assert_array_equal(event[:, EV.EV_FAINTED], [0, 0])
	assert np.all(event[:, EV.EV_HP] < 0)
	o = OBSERVE(out, 0, t)
	np.testing.assert_array_equal(o.events.order, event[:, EV.EV_ORDER])
	np.testing.assert_array_equal(o.events.move, [tables.moves.index["tackle"]] * 2)
	# The next step clears the record: a switch-only step records only the switch.
	out2 = play(out, [4, 0], t, FORCED)
	assert int(out2.event[0, EV.EV_ENTRY]) == EV.ENTRY_CHOSEN
	assert int(out2.event[0, EV.EV_ORDER]) == 0
	assert int(out2.event_move[0]) == 0


def test_event_record_miss_crit_and_hits(battles):
	tables, t, s, _ = battles
	s = no_items(s)
	s = set_active_move(s, tables, 0, 0, "tackle")
	s = set_active_move(s, tables, 1, 0, "bulletseed")
	s = set_active_types(s, tables, 0, ["normal"])
	s = set_active_types(s, tables, 1, ["normal"])
	forced = (
		UNFORCED.at[ORDER, 0, ACTION_ORDER]
		.set(0)
		.at[HIT, 0, 0]
		.set(0)
		.at[HIT, 1, 0]
		.set(1)
		.at[CRIT, 1]
		.set(1)
		.at[DAMAGE_ROLL, :, :]
		.set(0)
		.at[HIT_COUNT, 1, 0]
		.set(3)
	)
	out = play(s, [0, 0], t, forced)
	event = np.asarray(out.event)
	assert event[0, EV.EV_OUTCOME] == EV.OUTCOME_MISS
	assert event[0, EV.EV_EFFECT] == EV.EFFECT_NONE
	# Side 0's active is still hit by the bullet seeds, so its change is negative.
	assert event[0, EV.EV_HP] < 0
	assert event[1, EV.EV_OUTCOME] & EV.OUTCOME_CRIT
	assert event[1, EV.EV_HITS] == 3
	assert event[1, EV.EV_OUTCOME] & EV.OUTCOME_MISS == 0
	assert event[1, EV.EV_EFFECT] == EV.EFFECT_NEUTRAL


def test_event_record_super_resisted_and_neutral(battles):
	tables, t, s, _ = battles
	s = set_active_move(s, tables, 0, 0, "ember")
	s = set_active_move(s, tables, 1, 0, "tackle")
	for defender, effect in (
		(["grass"], EV.EFFECT_SUPER),
		(["water"], EV.EFFECT_RESISTED),
		(["normal"], EV.EFFECT_NEUTRAL),
	):
		case = set_active_types(s, tables, 1, defender)
		out = play(case, [0, 0], t, FORCED)
		assert int(out.event[0, EV.EV_EFFECT]) == effect, defender


def test_event_record_immunity_and_failure(battles):
	tables, t, s, _ = battles
	s = no_items(s)
	# Toxic on a Poison type is refused with -immune: the attacker's move fails with immunity.
	s = set_active_move(s, tables, 0, 0, "toxic")
	s = set_active_move(s, tables, 1, 0, "tackle")
	s = set_active_types(s, tables, 1, ["poison"])
	out = play(s, [0, 0], t, FORCED)
	assert int(out.event[0, EV.EV_EFFECT]) == EV.EFFECT_IMMUNE
	assert int(out.event[0, EV.EV_OUTCOME]) & EV.OUTCOME_FAILED
	assert int(out.status[1, out.active[1]]) == 0 and int(out.event[1, EV.EV_STATUS]) == 0
	# A second Substitute while one stands fails. The foe's move is inert, so nothing breaks it.
	s = set_active_move(s, tables, 1, 0, "splash")
	s = set_active_move(s, tables, 0, 0, "substitute")
	s = s._replace(volatiles=s.volatiles.at[0, S.SUBSTITUTE].set(3))
	out = play(s, [0, 0], t, FORCED)
	assert int(out.event[0, EV.EV_OUTCOME]) == EV.OUTCOME_FAILED
	# Recover at full HP fails (-fail heal).
	s = set_active_move(s, tables, 0, 0, "recover")
	member = s.active[0]
	s = s._replace(hp=s.hp.at[0, member].set(s.stats[0, member, 0]))
	out = play(s, [0, 0], t, FORCED)
	assert int(out.event[0, EV.EV_OUTCOME]) == EV.OUTCOME_FAILED
	# A Wish while one is pending fails.
	s = set_active_move(s, tables, 0, 0, "wish")
	s = s._replace(wish_turns=s.wish_turns.at[0].set(2))
	out = play(s, [0, 0], t, FORCED)
	assert int(out.event[0, EV.EV_OUTCOME]) == EV.OUTCOME_FAILED


@pytest.mark.parametrize(
	"name, reason",
	[
		("paralysis", EV.CANT_PARALYSIS),
		("sleep", EV.CANT_SLEEP),
		("flinch", EV.CANT_FLINCH),
		("confusion", EV.CANT_CONFUSION),
		("recharge", EV.CANT_RECHARGE),
	],
)
def test_event_record_cant_reasons(battles, name, reason):
	tables, t, s, _ = battles
	s = set_active_move(s, tables, 0, 0, "tackle")
	s = set_active_move(s, tables, 1, 0, "tackle")
	forced = FORCED
	member = s.active[0]
	if name == "paralysis":
		s = s._replace(status=s.status.at[0, member].set(S.PARALYSIS))
		forced = forced.at[FULL_PARALYSIS, 0, 0].set(1)
	elif name == "sleep":
		s = s._replace(
			status=s.status.at[0, member].set(S.SLEEP),
			status_counter=s.status_counter.at[0, member].set(3),
		)
	elif name == "flinch":
		s = s._replace(volatiles=s.volatiles.at[0, S.FLINCH].set(1))
	elif name == "confusion":
		s = s._replace(volatiles=s.volatiles.at[0, S.CONFUSION].set(3))
		forced = forced.at[CONFUSION_HIT, 0, 0].set(1)
	elif name == "recharge":
		s = s._replace(volatiles=s.volatiles.at[0, S.MUST_RECHARGE].set(1))
	out = play(s, [0, 0], t, forced)
	event = np.asarray(out.event)
	assert event[0, EV.EV_CANT] == reason
	assert event[0, EV.EV_ORDER] != 0
	assert np.asarray(out.event_move)[0] == 0
	assert event[1, EV.EV_CANT] == 0


def test_event_record_entries(battles):
	tables, t, s, _ = battles
	# Chosen switch: the bench member enters on a move request.
	out = play(s, [4, 0], t, FORCED)
	assert int(out.event[0, EV.EV_ENTRY]) == EV.ENTRY_CHOSEN
	assert int(out.event[1, EV.EV_ENTRY]) == 0
	# Dragged in by Roar: the foe is pulled to a chosen bench member.
	s2 = set_active_move(s, tables, 0, 0, "roar")
	forced = FORCED.at[DRAG_IN, 1, 0].set(alive_bench(s2, 1))
	out = play(s2, [0, 0], t, forced)
	assert int(out.event[1, EV.EV_ENTRY]) == EV.ENTRY_DRAGGED
	# Replacement after a faint: the KO leaves a SWITCH request, then the bench enters.
	s3 = set_active_move(s, tables, 0, 0, "tackle")
	defender = s3.active[1]
	s3 = s3._replace(hp=s3.hp.at[1, defender].set(1))
	out = play(s3, [0, 0], t, FORCED.at[ORDER, 0, ACTION_ORDER].set(0))
	assert int(out.event[1, EV.EV_FAINTED]) == 1
	assert int(out.request[1]) == SWITCH
	out = play(out, [0, 4], t, FORCED)
	assert int(out.event[1, EV.EV_ENTRY]) == EV.ENTRY_REPLACEMENT
	assert int(out.event[0, EV.EV_ENTRY]) == 0


def test_event_record_pivot_entry(battles):
	tables, t, s, _ = battles
	s = set_active_move(s, tables, 0, 0, "uturn")
	s = set_active_move(s, tables, 1, 0, "tackle")
	out = play(s, [0, 0], t, FORCED.at[ORDER, 0, ACTION_ORDER].set(0))
	# U-turn leaves the user alive and waiting for its switch.
	assert int(out.request[0]) == SWITCH and int(out.hp[0, out.active[0]]) > 0
	out = play(out, [4, 0], t, FORCED)
	assert int(out.event[0, EV.EV_ENTRY]) == EV.ENTRY_PIVOT


def test_event_record_status_hp_and_faint(battles):
	tables, t, s, _ = battles
	s = no_items(s)
	s = set_active_move(s, tables, 0, 0, "thunderwave")
	s = set_active_move(s, tables, 1, 0, "tackle")
	s = set_active_types(s, tables, 1, ["normal"])
	out = play(s, [0, 0], t, FORCED)
	assert int(out.event[1, EV.EV_STATUS]) == S.PARALYSIS
	assert int(out.event[0, EV.EV_STATUS]) == 0
	assert int(out.event[1, EV.EV_HP]) == 0
	# A KO by a tackle: the foe's percentage falls to zero and the faint is recorded.
	s = set_active_move(s, tables, 0, 0, "tackle")
	s = set_active_move(s, tables, 1, 0, "tackle")
	s = set_active_types(s, tables, 1, ["normal"])
	defender = s.active[1]
	s = s._replace(hp=s.hp.at[1, defender].set(1))
	out = play(s, [0, 0], t, FORCED)
	assert int(out.event[1, EV.EV_FAINTED]) == 1
	assert int(out.event[1, EV.EV_HP]) == -1
	o = OBSERVE(out, 0, t)
	np.testing.assert_allclose(o.events.hp_change[1], -1 / 100)
	assert float(o.events.fainted[1]) == 1


# The player view (tests/test-player-view.cjs, pokejax/obs_diff.py) found each of these: what the
# protocol names or leaves out differs from what the first observation showed.


def test_benched_foe_shows_the_hp_it_left_with(battles):
	"""Regenerator heals on the bench without a protocol line: the other player keeps the HP it saw."""
	tables, t, s, _ = battles
	member = int(s.active[1])
	maximum = int(s.stats[1, member, 0])
	low = max(maximum * 3 // 10, 1)
	s = no_items(s)
	s = s._replace(
		hp=s.hp.at[1, member].set(low),
		ability=s.ability.at[1, member].set(tables.abilities.index["regenerator"]),
	)
	out = play(s, [0, 4], t, FORCED)
	assert int(out.hp[1, member]) > low  # healed on leaving, silently
	o = OBSERVE(out, 0, t)
	np.testing.assert_allclose(o.opponent.hp[0], ((100 * low + maximum - 1) // maximum) / 100)
	assert float(o.opponent.is_active[0]) == 0 and float(o.opponent.is_active[1]) == 1


def test_item_flags_follow_how_the_item_left(battles):
	_, t, s, _ = battles
	foe, mine = int(s.active[1]), int(s.active[0])
	for gone, consumed, taken in ((S.ITEM_HELD, 0, 0), (S.ITEM_USED, 1, 0), (S.ITEM_TAKEN, 0, 1)):
		named = s._replace(
			item_gone=s.item_gone.at[1, foe].set(gone).at[0, mine].set(gone),
			item_revealed=s.item_revealed.at[1, foe].set(True),
			item_public=s.item_public.at[1, foe].set(5),
		)
		o = OBSERVE(named, 0, t)
		assert float(o.opponent.item_consumed[0]) == consumed
		assert float(o.opponent.item_knocked_off[0]) == taken
		assert bool(o.own.item_consumed[mine]) == bool(consumed)
		assert bool(o.own.item_knocked_off[mine]) == bool(taken)
	# Until the protocol has named the opponent's item, nothing about it is shown.
	hidden = s._replace(item_gone=s.item_gone.at[1, foe].set(S.ITEM_TAKEN))
	o = OBSERVE(hidden, 0, t)
	assert float(o.opponent.item_knocked_off[0]) == 0 and float(o.opponent.item_revealed[0]) == 0


def test_knock_off_and_an_eaten_berry_are_named(battles):
	tables, t, s, _ = battles
	foe = int(s.active[1])
	s = no_items(s)
	s = set_active_move(s, tables, 0, 0, "knockoff")
	leftovers = tables.items.index["leftovers"]
	knocked = s._replace(item=s.item.at[1, foe].set(leftovers))
	out = play(knocked, [0, 0], t, FORCED)
	o = OBSERVE(out, 0, t)
	assert int(o.opponent.item[0]) == leftovers and float(o.opponent.item_knocked_off[0]) == 1
	assert float(o.opponent.item_consumed[0]) == 0
	assert bool(OBSERVE(out, 1, t).own.item_knocked_off[foe])
	# A Sitrus Berry that is eaten shows as consumed, never as knocked off.
	sitrus = tables.items.index["sitrusberry"]
	maximum = int(s.stats[1, foe, 0])
	eaten = set_active_move(s, tables, 0, 0, "tackle")
	eaten = eaten._replace(
		item=eaten.item.at[1, foe].set(sitrus), hp=eaten.hp.at[1, foe].set(maximum // 2 + 1)
	)
	eaten = set_active_move(eaten, tables, 1, 0, "splash")
	out = play(eaten, [0, 0], t, FORCED)
	o = OBSERVE(out, 0, t)
	assert float(o.opponent.item_consumed[0]) == 1 and float(o.opponent.item_knocked_off[0]) == 0
	assert int(o.opponent.item[0]) == sitrus


def test_transformed_member_shows_its_own_stats_and_ability(battles):
	"""Transform copies the foe's stats and ability, which only the foe's request would show."""
	tables, t, s, _ = battles
	member = int(s.active[0])
	own = s.stats[0, member]
	copied = jnp.array([9, 9, 9, 9, 9], dtype=own.dtype)
	base = int(s.ability_base[0, member])
	foreign = tables.abilities.index["sturdy"]
	changed = s._replace(
		volatiles=s.volatiles.at[0, S.TRANSFORMED].set(1),
		stats=s.stats.at[0, member, 1:].set(copied),
		saved_stats=s.saved_stats.at[0].set(own[1:]),
		ability=s.ability.at[0, member].set(foreign),
		ability_public=s.ability_public.at[0, member].set(0),
	)
	o = OBSERVE(changed, 0, t)
	np.testing.assert_allclose(o.own.stats[member], np.asarray(own) / 1000)
	assert int(o.own.ability[member]) == base
	# An ability that the protocol named for it (Imposter's, a copied Intimidate's) is shown.
	named = changed._replace(ability_public=changed.ability_public.at[0, member].set(foreign))
	assert int(OBSERVE(named, 0, t).own.ability[member]) == foreign


def test_trapping_abilities_are_not_in_the_observed_legal_actions(battles):
	"""The request only says maybeTrapped: whether the foe really has Shadow Tag is private."""
	tables, t, s, _ = battles
	for name in ("shadowtag", "arenatrap", "magnetpull"):
		trapper = s._replace(
			ability=s.ability.at[1, s.active[1]].set(tables.abilities.index[name]),
			types=s.types.at[0, s.active[0]].set(
				jnp.array([tables.types.index["steel"], -1], jnp.int8)
			),
		)
		engine_mask = np.asarray(legal_actions(trapper, t)[0])
		seen = np.asarray(OBSERVE(trapper, 0, t).legal_actions)
		assert not engine_mask[4:].any() and seen[4:].all(), name
	partial = s._replace(
		volatiles=s.volatiles.at[0, S.PARTIALLY_TRAPPED].set(3).at[0, S.PARTIAL_TRAP_SOURCE].set(1)
	)
	assert not np.asarray(OBSERVE(partial, 0, t).legal_actions)[4:].any()  # that one is announced


def test_a_discovered_trap_shows_in_the_observed_legal_actions_until_an_active_changes(battles):
	"""A refused switch tells the player it is trapped (trap_known): its mask loses the switches,
	the foe's flag is never read, a forced switch is never refused, and step clears the flag when
	either active changes."""
	tables, t, s, _ = battles
	trapper = s._replace(
		ability=s.ability.at[1, s.active[1]].set(tables.abilities.index["shadowtag"])
	)
	trapper = set_active_move(
		set_active_move(trapper, tables, 0, 0, "splash"), tables, 1, 0, "splash"
	)
	known = trapper._replace(trap_known=jnp.array([True, False]))
	seen = np.asarray(OBSERVE(known, 0, t).legal_actions)
	assert seen[0] and not seen[4:].any()
	np.testing.assert_array_equal(seen[:4], np.asarray(OBSERVE(trapper, 0, t).legal_actions)[:4])
	assert np.asarray(OBSERVE(trapper, 0, t).legal_actions)[4:].any()
	# The mask of the foe is untouched by the flag of the other side; so is the whole observation
	# of a side whose own flag is clear.
	only_foe = trapper._replace(trap_known=jnp.array([False, True]))
	assert_equal(OBSERVE(only_foe, 0, t), OBSERVE(trapper, 0, t))
	assert not np.asarray(OBSERVE(only_foe, 1, t).legal_actions)[4:].any()
	# A forced switch is a different request: the trap does not apply to it.
	switching = known._replace(request=known.request.at[0].set(SWITCH))
	assert np.asarray(OBSERVE(switching, 0, t).legal_actions)[4:].any()
	# Nobody leaves: the flag stays. The trapper leaves: step clears it.
	assert bool(play(known, [0, 0], t, FORCED).trap_known[0])
	out = play(known, [0, 4], t, FORCED)
	assert int(out.active[1]) != int(known.active[1])
	assert not np.asarray(out.trap_known).any()
	assert np.asarray(OBSERVE(out, 0, t).legal_actions)[4:].any()


def test_fainted_members_show_no_status(battles):
	_, t, s, _ = battles
	fainted = s._replace(
		hp=s.hp.at[0, 3].set(0).at[1, 3].set(0),
		hp_seen=s.hp_seen.at[1, 3].set(0),
		status=s.status.at[0, 3].set(S.FREEZE).at[1, 3].set(S.FREEZE),
		seen_order=s.seen_order.at[1, 3].set(2),
	)
	o = OBSERVE(fainted, 0, t)
	assert int(o.own.status[3]) == 0 and float(o.own.fainted[3]) == 1
	assert int(o.opponent.status[1]) == 0 and float(o.opponent.fainted[1]) == 1


def test_a_disguised_holder_shows_the_level_of_its_disguise(battles):
	_, t, s, _ = battles
	foe = 1
	active, bench = int(s.active[foe]), 3
	disguised = s._replace(
		level=s.level.at[foe, active].set(81).at[foe, bench].set(60),
		volatiles=s.volatiles.at[foe, S.ILLUSION_SLOT].set(bench + 1),
		shown_species=s.shown_species.at[foe, active].set(s.species[foe, bench]),
	)
	o = OBSERVE(disguised, 0, t)
	np.testing.assert_allclose(o.opponent.level[0], 0.60)
	revealed = disguised._replace(volatiles=disguised.volatiles.at[foe, S.ILLUSION_SLOT].set(0))
	np.testing.assert_allclose(OBSERVE(revealed, 0, t).opponent.level[0], 0.81)


def test_a_lock_shows_in_a_move_request_only_and_not_for_the_foe(battles):
	_, t, s, _ = battles
	locked = s._replace(
		volatiles=s.volatiles.at[0, S.LOCKED_MOVE].set(3).at[1, S.LOCKED_MOVE].set(3)
	)
	o = OBSERVE(locked, 0, t)
	assert float(o.own.locked) == 1
	assert not hasattr(o.opponent, "locked")
	waiting = locked._replace(request=locked.request.at[0].set(NONE))
	assert float(OBSERVE(waiting, 0, t).own.locked) == 0
	switching = locked._replace(request=locked.request.at[0].set(SWITCH))
	assert float(OBSERVE(switching, 0, t).own.locked) == 0


def test_a_locked_move_continuing_reveals_nothing_new(battles):
	tables, t, s, _ = battles
	foe, member = 1, int(s.active[1])
	outrage = tables.moves.index["outrage"]
	locked = s._replace(
		moves=s.moves.at[foe, member, 2].set(outrage),
		pp=s.pp.at[foe, member, 2].set(5),
		volatiles=s.volatiles.at[foe, S.LOCKED_MOVE]
		.set(outrage)
		.at[foe, S.LOCKED_REMAINING]
		.set(2)
		.at[foe, S.LOCKED_TRUE_DURATION]
		.set(3),
	)
	out = play(locked, [0, 0], t, FORCED)
	assert not np.any(np.asarray(out.revealed_moves[foe]))


def test_contact_items_are_named_only_when_their_damage_is(battles):
	"""Rough Skin faints the attacker first, so the Helmet prints nothing; Life Orb after the
	user fainted to its move's recoil does not print either."""
	tables, t, s, _ = battles
	s = set_active_move(s, tables, 0, 0, "tackle")
	active0, active1 = int(s.active[0]), int(s.active[1])
	holder = s._replace(
		item=s.item.at[1, active1].set(tables.items.index["rockyhelmet"]),
		ability=s.ability.at[1, active1].set(tables.abilities.index["roughskin"]),
		hp=s.hp.at[0, active0].set(1),
	)
	forced = FORCED.at[ORDER, 0, ACTION_ORDER].set(1)
	out = play(set_active_move(holder, tables, 1, 0, "splash"), [0, 0], t, forced)
	assert int(out.hp[0, active0]) == 0
	assert not bool(out.item_revealed[1, active1])
	# With the attacker alive after Rough Skin, the Helmet's damage line names it.
	healthy = holder._replace(hp=holder.hp.at[0, active0].set(holder.stats[0, active0, 0]))
	out = play(set_active_move(healthy, tables, 1, 0, "splash"), [0, 0], t, forced)
	assert int(out.hp[0, active0]) > 0 and bool(out.item_revealed[1, active1])
	# A Life Orb holder that faints to its move's recoil takes no Life Orb damage: not named.
	orb = set_active_move(s, tables, 0, 0, "doubleedge")
	orb = orb._replace(
		item=orb.item.at[0, active0].set(tables.items.index["lifeorb"]),
		hp=orb.hp.at[0, active0].set(1),
	)
	out = play(set_active_move(orb, tables, 1, 0, "splash"), [0, 0], t, forced)
	assert int(out.hp[0, active0]) == 0 and not bool(out.item_revealed[0, active0])


def test_a_reflected_leech_seed_is_refused_by_a_grass_type_user(battles):
	tables, t, s, _ = battles
	s = no_items(s)
	s = set_active_move(s, tables, 0, 0, "leechseed")
	s = set_active_types(s, tables, 0, ["grass"])
	s = s._replace(ability=s.ability.at[1, s.active[1]].set(tables.abilities.index["magicbounce"]))
	out = play(s, [0, 0], t, FORCED)
	holder = 1
	assert int(out.event[holder, EV.EV_EFFECT]) == EV.EFFECT_IMMUNE
	assert (
		int(out.ability_public[holder, out.active[holder]]) == tables.abilities.index["magicbounce"]
	)
