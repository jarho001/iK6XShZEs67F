"""Phase 1 outcomes from the pinned Showdown formulas and decision boundaries."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.order import action_order
from pokejax.engine.randomness import (
	ACTION_ORDER,
	CRIT,
	DAMAGE_ROLL,
	HIT,
	ORDER,
	SWITCH_IN_ORDER,
	UNFORCED,
)
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import MOVE, NONE, ONGOING, SWITCH
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import load_npz, load_team_pool

KEY = jax.random.key(17)
FORCED = UNFORCED.at[HIT, :, 0].set(1).at[CRIT, :, 0].set(0).at[DAMAGE_ROLL, :, 0].set(0)
RUN = jax.jit(step)


@pytest.fixture(scope="module")
def battle_pool(tmp_path_factory):
	path = Path("work/teams/gen5-teams.npz")
	if path.exists():
		return load_npz(path)
	# Keep the suite runnable in a clean checkout using the recorded pool subset.
	fixture = json.loads(Path("tests/fixtures/reset.json").read_text())
	path = tmp_path_factory.mktemp("step-pool") / "teams.json"
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
	return load_team_pool(path, load_tables())


@pytest.fixture(scope="module")
def tables_and_state(battle_pool):
	tables = load_tables()
	t = build_engine_tables(tables)
	state = reset(KEY, teams_from_pool(battle_pool, 0, 1), t)
	# Hand-built equal level/stats, large HP, one plain move, no STAB.
	state = state._replace(
		level=jnp.full_like(state.level, 100),
		stats=jnp.full_like(state.stats, 100).at[:, :, 0].set(1000),
		hp=jnp.full_like(state.hp, 1000),
		types=jnp.full_like(state.types, -1).at[:, :, 0].set(tables.types.index["water"]),
		moves=jnp.zeros_like(state.moves).at[:, :, 0].set(tables.moves.index["tackle"]),
		pp=jnp.zeros_like(state.pp).at[:, :, 0].set(10),
	)
	return tables, t, state


def play(state, t, actions=(0, 0), forced=FORCED):
	return RUN(state, jnp.asarray(actions, jnp.int32), KEY, forced, t)


def test_legal_requests(tables_and_state):
	_, t, s = tables_and_state
	mask = legal_actions(s, t)
	np.testing.assert_array_equal(mask[0], [True, False, False, False, *([True] * 5)])
	s = s._replace(pp=jnp.zeros_like(s.pp), active=jnp.array([2, 4], jnp.int8))
	s = s._replace(hp=s.hp.at[0, 1].set(0), request=jnp.array([SWITCH, NONE], jnp.int8))
	mask = legal_actions(s, t)
	np.testing.assert_array_equal(mask[0], [False] * 4 + [True, False, True, True, True])
	np.testing.assert_array_equal(mask[1], [True] + [False] * 8)
	s = s._replace(request=jnp.full_like(s.request, MOVE))
	np.testing.assert_array_equal(legal_actions(s, t)[0, :4], [True, False, False, False])
	s = s._replace(winner=jnp.int8(0))
	np.testing.assert_array_equal(legal_actions(s, t), [[True] + [False] * 8] * 2)


def test_exact_damage_roll_miss_crit_and_boosts(tables_and_state):
	_, t, s = tables_and_state
	# sim/battle-actions.ts:1770-1821: floor(floor(42*50*100/100)/50)+2 = 44.
	# :1824-1875: roll, STAB then effectiveness. Roll 7 gives floor(44*.93)=40.
	out = play(s, t, forced=FORCED.at[DAMAGE_ROLL, :, 0].set(7))
	np.testing.assert_array_equal(out.hp[:, 0], [960, 960])
	np.testing.assert_array_equal(out.pp[:, 0, 0], [9, 9])
	assert int(out.turn) == 2
	out = play(s, t, forced=FORCED.at[HIT, :, 0].set(0))
	np.testing.assert_array_equal(out.hp, s.hp)
	np.testing.assert_array_equal(out.pp[:, 0, 0], [9, 9])
	# :1737-1754 crit ignores negative offense and positive defense, doubles damage.
	s = s._replace(boosts=s.boosts.at[0, 0].set(-6).at[1, 1].set(6))
	out = play(s, t, forced=FORCED.at[CRIT, 0, 0].set(1))
	assert int(out.hp[1, 0]) == 912
	# Positive offense and negative defense still apply on crit (42*50*200/50/50+2)*2.
	s = s._replace(boosts=s.boosts.at[0, 0].set(2).at[1, 1].set(-2))
	out = play(s, t, forced=FORCED.at[CRIT, 0, 0].set(1))
	assert int(out.hp[1, 0]) == 660


@pytest.mark.parametrize(
	"move,user_types,target_types,expected",
	[
		("tackle", ["normal"], ["water"], 66),
		("tackle", ["water"], ["ghost"], 0),
		("ember", ["water"], ["grass", "steel"], 140),
		("ember", ["water"], ["fire", "water"], 8),
		("aerialace", ["water"], ["water"], 52),
		("protect", ["water"], ["water"], 0),
		("seismictoss", ["water"], ["water"], 0),
	],
)
def test_types_stab_always_hit_and_inert_moves(
	tables_and_state, move, user_types, target_types, expected
):
	tables, t, s = tables_and_state
	for side, types in enumerate((user_types, target_types)):
		ids = [tables.types.index[k] for k in types] + [-1] * (2 - len(types))
		s = s._replace(types=s.types.at[side, 0].set(jnp.array(ids, jnp.int8)))
	s = s._replace(moves=s.moves.at[0, 0, 0].set(tables.moves.index[move]))
	# battle-actions.ts:1800-1901: ember core 33+2=35; 4x=140, .25x=floor(35/4)=8.
	forced = FORCED.at[HIT, 0, 0].set(0) if move == "aerialace" else FORCED
	out = play(s, t, forced=forced)
	assert int(out.hp[1, 0]) == 1000 - expected
	assert int(out.pp[0, 0, 0]) == 9


@pytest.mark.parametrize("first", [0, 1])
def test_speed_ties_and_ko_cancels_move(tables_and_state, first):
	_, t, s = tables_and_state
	s = s._replace(hp=s.hp.at[:, 0].set(1))
	out = play(s, t, forced=FORCED.at[ORDER, 0, ACTION_ORDER].set(first))
	assert int(out.hp[first, 0]) == 1
	assert int(out.hp[1 - first, 0]) == 0
	assert int(out.pp[first, 0, 0]) == 9
	assert int(out.pp[1 - first, 0, 0]) == 10
	np.testing.assert_array_equal(out.request, [NONE, SWITCH] if first == 0 else [SWITCH, NONE])
	assert int(out.turn) == 1


def test_speed_boost_priority_and_switch_order(tables_and_state):
	tables, t, s = tables_and_state
	s = s._replace(hp=s.hp.at[:, 0].set(1), stats=s.stats.at[0, 0, 5].set(120))
	assert int(play(s, t).hp[0, 0]) == 1
	s = s._replace(boosts=s.boosts.at[1, 4].set(1))
	assert int(play(s, t).hp[1, 0]) == 1
	s = s._replace(moves=s.moves.at[0, 0, 0].set(tables.moves.index["quickattack"]))
	assert int(play(s, t).hp[0, 0]) == 1
	# battle-queue.ts:200-225 switches order 103, moves order 200, regardless of priority.
	out = play(s, t, actions=(0, 4))
	assert int(out.active[1]) == 1
	assert int(out.hp[1, 0]) == 1
	assert int(out.hp[1, 1]) == 965  # Quick Attack: floor(42*40/50)+2 = 35.
	moves = s.moves[jnp.arange(2), s.active, 0]
	for replacement, occurrence in ((False, ACTION_ORDER), (True, SWITCH_IN_ORDER)):
		for first in (0, 1):
			equal = s._replace(boosts=jnp.zeros_like(s.boosts), stats=s.stats.at[0, 0, 5].set(100))
			order = action_order(
				equal,
				jnp.array([4, 4]),
				moves,
				replacement,
				KEY,
				FORCED.at[ORDER, 0, occurrence].set(first),
				t,
			)
			assert int(order) == first


def test_replacement_turn_and_boost_reset(tables_and_state):
	_, t, s = tables_and_state
	s = s._replace(hp=s.hp.at[1, 0].set(1), boosts=s.boosts.at[1, 0].set(3))
	out = play(s, t, forced=FORCED.at[ORDER, 0, 0].set(0))
	np.testing.assert_array_equal(out.request, [NONE, SWITCH])
	assert int(out.turn) == 1
	assert np.all(out.boosts[1] == 0)  # pokemon.ts:1529 clearVolatile on faint.
	before = out.pp
	out = play(out, t, actions=(0, 5))
	np.testing.assert_array_equal(out.active, [0, 2])
	np.testing.assert_array_equal(out.request, [MOVE, MOVE])
	np.testing.assert_array_equal(out.pp, before)
	assert int(out.turn) == 2
	s = s._replace(boosts=s.boosts.at[0, 0].set(4))
	out = play(s, t, actions=(4, 4))
	assert np.all(out.boosts == 0)


@pytest.mark.parametrize("first", [0, 1])
def test_struggle_recoil_double_ko_and_terminal_noop(tables_and_state, first):
	tables, t, s = tables_and_state
	# moves.ts:18208-18226: typeless, 50 BP, always hits; battle-actions.ts:1437
	# rounds 101/4 to 25, 102/4 to 26 (not Gen 4's floor).
	for maximum, recoil in ((101, 25), (102, 26), (1, 1)):
		case = s._replace(
			pp=jnp.zeros_like(s.pp),
			stats=s.stats.at[:, 0, 0].set(maximum),
			types=s.types.at[:, 0, 0].set(tables.types.index["ghost"]),
		)
		out = play(case, t, forced=FORCED.at[HIT, :, 0].set(0))
		np.testing.assert_array_equal(out.hp[:, 0], [1000 - 44 - recoil] * 2)
		assert np.all(out.pp == 0)
	# battle.ts:2660-2664 last faint side wins, not a tie. Struggle's user is last.
	s = s._replace(hp=jnp.zeros_like(s.hp).at[:, 0].set(1), pp=jnp.zeros_like(s.pp))
	out = play(s, t, forced=FORCED.at[ORDER, 0, 0].set(first))
	assert int(out.winner) == first
	assert int(out.turn) == 1
	assert np.all(out.request == NONE)
	assert np.all(out.hp == 0)
	again = play(out, t)
	for a, b in zip(out, again):
		np.testing.assert_array_equal(a, b)


def test_last_pokemon_win_and_no_target_pp(tables_and_state):
	_, t, s = tables_and_state
	s = s._replace(hp=s.hp.at[1].set(0).at[1, 0].set(1))
	out = play(s, t, forced=FORCED.at[ORDER, 0, 0].set(0))
	assert int(out.winner) == 0
	assert np.all(out.request == NONE)
	assert int(out.turn) == 1
	# Faster Struggle user dies to recoil, leaving no target. The surviving
	# foe still spends PP (runMove:339 precedes useMove's no-target check:574).
	s = s._replace(hp=jnp.full_like(s.hp, 1000).at[0, 0].set(1), pp=s.pp.at[0].set(0))
	out = play(s, t, forced=FORCED.at[ORDER, 0, 0].set(0))
	assert int(out.hp[1, 0]) == 956
	assert int(out.pp[1, 0, 0]) == 9
	assert int(out.hp[0, 0]) == 0


def test_minimum_damage_after_resistance(tables_and_state):
	tables, t, s = tables_and_state
	s = s._replace(
		level=s.level.at[0, 0].set(1),
		stats=s.stats.at[0, 0, 3].set(1),
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["ember"]),
		types=s.types.at[1, 0].set(
			jnp.array([tables.types.index["fire"], tables.types.index["water"]], jnp.int8)
		),
	)
	# battle-actions.ts:1882-1901: core 0 + 2, roll 15 => 1, two halvings
	# => 0, then Gen 5's minimum-one check BEFORE the final identity modifier.
	out = play(s, t, forced=FORCED.at[DAMAGE_ROLL, 0, 0].set(15))
	assert int(out.hp[1, 0]) == 999


def test_last_pp_becomes_struggle_without_wrapping(tables_and_state):
	tables, t, s = tables_and_state
	s = s._replace(
		moves=s.moves.at[:, 0, 0].set(tables.moves.index["protect"]),
		pp=s.pp.at[:, 0, 0].set(1),
	)
	out = play(s, t)
	assert np.all(out.pp[:, 0] == 0)
	np.testing.assert_array_equal(out.hp, s.hp)
	out = play(out, t)
	assert np.all(out.pp[:, 0] == 0)
	np.testing.assert_array_equal(out.hp[:, 0], [706, 706])  # 44 damage + 250 recoil.


def test_jit_vmap_64_pool_battles_200_legal_steps(tables_and_state, battle_pool):
	_, t, _ = tables_and_state
	pool = battle_pool
	rng = np.random.default_rng(21)
	indices = rng.integers(0, len(pool.team_size), size=(64, 2))
	teams = teams_from_pool(pool, indices[:, 0], indices[:, 1])
	states = jax.vmap(reset, in_axes=(0, 0, None))(jax.random.split(KEY, 64), teams, t)

	@jax.jit
	def rollout(states):
		def body(carry, key):
			mask = jax.vmap(legal_actions, in_axes=(0, None))(carry, t)
			# Uniform among legal choices, including terminal no-ops.
			actions = jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))
			out = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
				carry, actions, jax.random.split(key, 64), UNFORCED, t
			)
			valid = jnp.all(out.hp >= 0) & jnp.all(out.pp <= states.max_pp)
			return out, valid

		return jax.lax.scan(body, states, jax.random.split(KEY, 200))

	out, valid = rollout(states)
	assert np.all(valid)
	assert np.all(out.hp >= 0)
	assert np.all(out.pp <= states.max_pp)
	assert np.all((out.winner >= ONGOING) & (out.winner <= 1))
