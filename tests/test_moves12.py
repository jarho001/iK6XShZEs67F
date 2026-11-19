"""Focused checks for task 12 moves: variable power, stat sources and conditional success."""

import jax
import jax.numpy as jnp
import pytest

from pokejax.engine import variable_moves as vm
from pokejax.engine.data import build_engine_tables
from pokejax.engine.randomness import ACTION_ORDER, CRIT, DAMAGE_ROLL, HIT, ORDER, UNFORCED
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import AUTOTOMIZE, MOVE, SPE, SWITCH
from pokejax.engine.step import step
from pokejax.mechanics.hidden_power import hidden_power
from pokejax.teams import HIDDEN_POWER_TYPES

KEY = jax.random.key(12)
RUN = jax.jit(step)
FORCED = (
	UNFORCED.at[ORDER, 0, ACTION_ORDER]
	.set(0)
	.at[HIT, :, :]
	.set(1)
	.at[CRIT, :, :]
	.set(0)
	.at[DAMAGE_ROLL, :, :]
	.set(0)
)


@pytest.fixture
def battle(tables, synthetic_pool):
	teams = teams_from_pool(synthetic_pool, 0, 1)
	engine = build_engine_tables(tables)
	teams = teams._replace(ability=jnp.zeros_like(teams.ability), item=jnp.zeros_like(teams.item))
	state = reset(KEY, teams, engine)
	state = state._replace(
		stats=jnp.full_like(state.stats, 100).at[:, :, 0].set(1000),
		hp=jnp.full_like(state.hp, 1000),
		level=jnp.full_like(state.level, 100),
		types=jnp.full_like(state.types, -1),
		request=jnp.array([MOVE, MOVE], dtype=state.request.dtype),
	)
	# The foe uses a status move unless a test sets its move: damage it deals would be recorded.
	return engine, with_moves(state, engine, 1, ["nastyplot"], tables)


def with_moves(state, engine, side, names, tables):
	ids = [tables.moves.index[name] for name in names]
	ids += [0] * (4 - len(ids))
	ids = jnp.asarray(ids, state.moves.dtype)
	pp = engine.move_max_pp[ids]
	return state._replace(
		moves=state.moves.at[side, 0].set(ids),
		pp=state.pp.at[side, 0].set(pp.astype(state.pp.dtype)),
		max_pp=state.max_pp.at[side, 0].set(pp.astype(state.max_pp.dtype)),
	)


def hp_of(state, side, slot=0):
	return int(state.hp[side, slot])


def test_hidden_power_type_and_power_match_showdown():
	# Expected values are Showdown's getHiddenPower(ivs) output at the pinned commit.
	expected = {
		(31, 31, 31, 31, 31, 31): ("dark", 70),
		(31, 30, 31, 30, 31, 31): ("grass", 70),
		(31, 0, 31, 31, 31, 31): ("dragon", 68),
		(30, 31, 30, 30, 30, 30): ("fighting", 70),
		(31, 2, 31, 30, 31, 30): ("fire", 70),
	}
	for ivs, (type_name, power) in expected.items():
		kind, got_power = hidden_power(jnp.asarray(ivs, jnp.int32))
		assert HIDDEN_POWER_TYPES[int(kind)] == type_name
		assert int(got_power) == power


def test_pursuit_hits_a_switching_target_before_it_leaves(battle, tables):
	engine, state = battle
	state = with_moves(state, engine, 1, ["pursuit"], tables)
	state = with_moves(state, engine, 0, ["tackle"], tables)
	before = hp_of(state, 0)
	out = RUN(state, jnp.array([4, 0]), KEY, FORCED, engine)
	assert int(out.active[0]) == 1  # the switch still happens after the hit
	assert hp_of(out, 0, 0) < before  # the old active took the Pursuit hit
	assert hp_of(out, 0, 1) == hp_of(state, 0, 1)


def test_pursuit_behind_uturn_hits_before_the_pivot_pause(battle, tables):
	engine, state = battle
	state = state._replace(stats=state.stats.at[1, 0, SPE].set(10))  # Pursuit user is slower
	state = with_moves(state, engine, 0, ["uturn"], tables)
	state = with_moves(state, engine, 1, ["pursuit"], tables)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.request[0]) == SWITCH
	assert hp_of(out, 0, 0) < 1000


def test_sucker_punch_fails_against_a_status_move(battle, tables):
	engine, state = battle
	state = with_moves(state, engine, 0, ["suckerpunch"], tables)
	state = with_moves(state, engine, 1, ["toxic"], tables)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) == 1000


def test_focus_punch_fails_after_a_damaging_hit_but_not_after_a_status_move(battle, tables):
	engine, state = battle
	state = with_moves(state, engine, 0, ["focuspunch"], tables)
	hit_first = with_moves(state, engine, 1, ["tackle"], tables)
	out = RUN(hit_first, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) == 1000  # Focus Punch failed: the foe hit first
	status_first = with_moves(state, engine, 1, ["nastyplot"], tables)
	out = RUN(status_first, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) < 1000


def test_counter_needs_physical_damage_and_mirror_coat_needs_special(battle, tables):
	engine, state = battle
	special_taken = state.turn_damage.at[0, 2].set(50)
	state = state._replace(turn_damage=special_taken, turn_hit=state.turn_hit.at[0].set(True))
	counter = with_moves(state, engine, 0, ["counter"], tables)
	out = RUN(counter, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) == 1000
	mirror = with_moves(state, engine, 0, ["mirrorcoat"], tables)
	out = RUN(mirror, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) == 1000 - 100


def test_metal_burst_returns_one_and_a_half_times_the_last_hit(battle, tables):
	engine, state = battle
	state = state._replace(
		turn_damage=state.turn_damage.at[0, 0].set(30), turn_hit=state.turn_hit.at[0].set(True)
	)
	state = with_moves(state, engine, 0, ["metalburst"], tables)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) == 1000 - 45
	quiet = with_moves(battle[1], engine, 0, ["metalburst"], tables)
	out = RUN(quiet, jnp.array([0, 0]), KEY, FORCED, engine)
	assert hp_of(out, 1, 0) == 1000


def _drop(before, after):
	return before - after


def test_retaliate_and_fusion_flare_double_after_their_triggers(battle, tables):
	engine, state = battle
	plain = with_moves(state, engine, 0, ["retaliate"], tables)
	doubled = plain._replace(fainted_last=plain.fainted_last.at[0].set(True))
	low = _drop(1000, hp_of(RUN(plain, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	high = _drop(1000, hp_of(RUN(doubled, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	assert high > 1.8 * low

	flare = with_moves(state, engine, 0, ["fusionflare"], tables)
	bolt_first = flare._replace(
		last_success=jnp.asarray(tables.moves.index["fusionbolt"], jnp.int16)
	)
	low = _drop(1000, hp_of(RUN(flare, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	high = _drop(1000, hp_of(RUN(bolt_first, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	assert high > 1.8 * low


def test_acrobatics_doubles_without_an_item_or_once_its_gem_is_used_up(battle, tables):
	engine, state = battle
	acro = with_moves(state, engine, 0, ["acrobatics"], tables)
	held = acro._replace(item=acro.item.at[0, 0].set(tables.items.index["leftovers"]))
	bare = acro._replace(item=acro.item.at[0, 0].set(0))
	low = _drop(1000, hp_of(RUN(held, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	high = _drop(1000, hp_of(RUN(bare, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	assert high > 1.8 * low
	# A Flying Gem is used up at TryPrimaryHit, before BasePower, so Acrobatics doubles.
	gem = acro._replace(item=acro.item.at[0, 0].set(tables.items.index["flyinggem"]))
	gem_drop = _drop(1000, hp_of(RUN(gem, jnp.array([0, 0]), KEY, FORCED, engine), 1, 0))
	assert gem_drop > 1.8 * low


def test_autotomize_lightens_the_weight_used_by_weight_moves(battle, tables):
	engine, state = battle
	state = with_moves(state, engine, 0, ["autotomize"], tables)
	base = int(engine.species_weight_hg[state.species[0, 0]])
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.volatiles[0, AUTOTOMIZE]) == (1 if base > 1 else 0)
	assert int(vm.active_weight(out, 0, engine)) == max(base - 1000, 1) if base > 1 else True
	assert int(vm.active_weight(out, 1, engine)) == max(
		int(engine.species_weight_hg[state.species[1, 0]]), 1
	)
