"""Gen 5 damage, recovery and self-faint effects from the pinned Showdown source."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_status

from pokejax.engine.move_effects import apply_healing
from pokejax.engine.randomness import (
	CRIT,
	DAMAGE_ROLL,
	HIT,
	ORDER,
	SECONDARY,
	UNFORCED,
)
from pokejax.engine.state import HEALTHY, NONE, PARALYSIS, POISON, SIDE0, SIDE1
from pokejax.engine.step import step

KEY = jax.random.key(42)
RUN = jax.jit(step)
FORCED = UNFORCED.at[HIT].set(1).at[CRIT].set(0).at[DAMAGE_ROLL].set(0).at[ORDER, 0, 0].set(0)


@pytest.fixture(scope="module")
def setup():
	return test_status.setup.__wrapped__()


def set_move(state, tables, side, move):
	return state._replace(moves=state.moves.at[side, 0, 0].set(tables.moves.index[move]))


def last_members_only(state, side0_hp, side1_hp):
	hp = jnp.zeros_like(state.hp).at[0, 0].set(side0_hp).at[1, 0].set(side1_hp)
	return state._replace(hp=hp)


def test_recoil_ko_queues_user_last_and_awards_tie_to_user(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "bravebird")
	state = last_members_only(state, 1, 1)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert np.all(np.asarray(out.hp) == 0)
	assert int(out.winner) == SIDE0


def test_explosion_faints_before_ghost_immunity_is_checked(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "explosion")
	state = last_members_only(state, 100, 100)
	state = state._replace(
		types=state.types.at[1, 0].set(jnp.array([tables.types.index["ghost"], -1], jnp.int8))
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 0
	assert int(out.hp[1, 0]) == 100
	assert int(out.winner) == SIDE1

	miss = state._replace(request=state.request.at[1].set(NONE))
	out = RUN(miss, jnp.array([0, 0]), KEY, FORCED.at[HIT, 0, 0].set(0), engine)
	assert int(out.hp[0, 0]) == 0
	assert int(out.hp[1, 0]) == 100

	double_ko = last_members_only(state, 1, 1)._replace(
		types=state.types.at[1, 0].set(jnp.array([tables.types.index["normal"], -1], jnp.int8))
	)
	out = RUN(double_ko, jnp.array([0, 0]), KEY, FORCED, engine)
	assert np.all(np.asarray(out.hp) == 0)
	assert int(out.winner) == SIDE1


def test_high_jump_kick_crashes_on_immunity_and_miss(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "highjumpkick")
	immune = state._replace(
		types=state.types.at[1, 0].set(jnp.array([tables.types.index["ghost"], -1], jnp.int8))
	)
	immune = immune._replace(request=immune.request.at[1].set(NONE))
	out = RUN(immune, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 80
	assert int(out.hp[1, 0]) == 160

	miss = state._replace(request=state.request.at[1].set(NONE))
	out = RUN(miss, jnp.array([0, 0]), KEY, FORCED.at[HIT, 0, 0].set(0), engine)
	assert int(out.hp[0, 0]) == 80
	assert int(out.hp[1, 0]) == 160


def test_drain_uses_actual_damage_and_does_not_overheal(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "gigadrain")
	state = state._replace(
		hp=state.hp.at[0, 0].set(100).at[1, 0].set(11),
		stats=state.stats.at[0, 0, 5].set(200),
	)
	state = state._replace(request=state.request.at[1].set(NONE))
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[1, 0]) == 0
	assert int(out.hp[0, 0]) == 106

	full = state._replace(hp=state.hp.at[0, 0].set(160))
	out = RUN(full, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 160


def test_heal_at_full_hp_fails_without_changing_hp(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "recover")
	state = state._replace(request=state.request.at[1].set(NONE))
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 160
	assert int(out.pp[0, 0, 0]) == 19


def test_heal_fraction_rounding_follows_the_showdown_call_site(setup):
	tables, engine, state = setup
	state = state._replace(
		stats=state.stats.at[0, 0, 0].set(321),
		hp=state.hp.at[0, 0].set(10),
		request=state.request.at[1].set(NONE),
	)
	recover = set_move(state, tables, 0, "recover")
	out = RUN(recover, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 171  # MoveTable heal uses Math.round(321 / 2).
	synthesis = set_move(state, tables, 0, "synthesis")
	out = RUN(synthesis, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 170  # Its callback goes through modify(321, 0.5).


def test_pain_split_sets_both_to_the_floor_average(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "painsplit")
	state = state._replace(
		stats=state.stats.at[0, 0, 0].set(100).at[1, 0, 0].set(120),
		hp=state.hp.at[0, 0].set(50).at[1, 0].set(101),
		request=state.request.at[1].set(NONE),
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 75
	assert int(out.hp[1, 0]) == 75


def test_multihit_secondaries_use_a_separate_address_for_each_hit(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "twineedle")
	state = state._replace(stats=state.stats.at[0, 0, 5].set(200))
	state = state._replace(request=state.request.at[1].set(NONE))
	forced = FORCED.at[SECONDARY, 0, 0].set(0).at[SECONDARY, 0, 2].set(1)
	out = RUN(state, jnp.array([0, 0]), KEY, forced, engine)
	assert int(out.status[1, 0]) == POISON


def test_multihit_stops_when_the_target_faints(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "twineedle")
	state = state._replace(
		hp=state.hp.at[1, 0].set(1),
		request=state.request.at[1].set(NONE),
	)
	forced = FORCED.at[SECONDARY, 0, 2].set(1)
	out = RUN(state, jnp.array([0, 0]), KEY, forced, engine)
	assert int(out.hp[1, 0]) == 0
	assert int(out.status[1, 0]) != POISON


@pytest.mark.parametrize(
	"target_type,initial_status,expected_status",
	[
		("normal", PARALYSIS, PARALYSIS),
		("poison", HEALTHY, HEALTHY),
		("steel", HEALTHY, HEALTHY),
	],
)
def test_later_twineedle_secondary_respects_status_and_type_immunity(
	setup, target_type, initial_status, expected_status
):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "twineedle")
	state = state._replace(
		status=state.status.at[1, 0].set(initial_status),
		types=state.types.at[1, 0].set(jnp.array([tables.types.index[target_type], -1], jnp.int8)),
		request=state.request.at[1].set(NONE),
	)
	forced = FORCED.at[SECONDARY, 0, 0].set(0).at[SECONDARY, 0, 2].set(1)
	out = RUN(state, jnp.array([0, 0]), KEY, forced, engine)
	assert int(out.status[1, 0]) == expected_status


def test_roost_removes_flying_until_the_opponents_move_then_restores_it(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "roost")
	state = set_move(state, tables, 1, "earthquake")
	state = state._replace(
		types=state.types.at[0, 0].set(
			jnp.array([tables.types.index["water"], tables.types.index["flying"]], jnp.int8)
		),
		hp=state.hp.at[0, 0].set(80),
		stats=state.stats.at[0, 0, 5].set(200),
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) < 160
	assert np.array_equal(
		out.types[0, 0], [tables.types.index["water"], tables.types.index["flying"]]
	)
	assert int(out.volatiles[0, 2]) == 0

	pure_flying = state._replace(
		types=state.types.at[0, 0].set(jnp.array([tables.types.index["flying"], -1], jnp.int8)),
		hp=state.hp.at[0, 0].set(80),
	)
	roost_state, healed = apply_healing(
		pure_flying, 0, tables.moves.index["roost"], True, True, engine
	)
	assert bool(healed)
	assert np.array_equal(roost_state.types[0, 0], [tables.types.index["normal"], -1])


def test_fixed_damage_ignores_defense_roll_and_resistance_but_checks_immunity(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "seismictoss")
	state = state._replace(
		request=state.request.at[1].set(NONE),
		stats=state.stats.at[1, 0, 2].set(999),
		level=state.level.at[0, 0].set(100),
		types=state.types.at[1, 0].set(jnp.array([tables.types.index["psychic"], -1], jnp.int8)),
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[DAMAGE_ROLL, 0, 0].set(15), engine)
	assert int(out.hp[1, 0]) == 60

	immune = state._replace(
		types=state.types.at[1, 0].set(jnp.array([tables.types.index["ghost"], -1], jnp.int8))
	)
	out = RUN(immune, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[1, 0]) == 160


def test_super_fang_halves_current_hp_rounded_down(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "superfang")
	state = state._replace(hp=state.hp.at[1, 0].set(101), request=state.request.at[1].set(NONE))
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[1, 0]) == 51


def test_numeric_fixed_damage_uses_its_constant_amount(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "dragonrage")
	state = state._replace(request=state.request.at[1].set(NONE))
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[1, 0]) == 120


def test_high_ratio_and_always_critical_moves(setup):
	tables, engine, state = setup
	state = set_move(state, tables, 0, "crosschop")
	state = state._replace(
		hp=state.hp.at[1, 0].set(1000),
		stats=state.stats.at[1, 0, 0].set(1000),
		request=state.request.at[1].set(NONE),
	)
	ordinary = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[CRIT, 0, 0].set(0), engine)
	critical = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[CRIT, 0, 0].set(1), engine)
	assert int(critical.hp[1, 0]) < int(ordinary.hp[1, 0])

	state = set_move(state, tables, 0, "frostbreath")
	no_crit_outcome = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[CRIT, 0, 0].set(0), engine)
	crit_outcome = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[CRIT, 0, 0].set(1), engine)
	assert int(no_crit_outcome.hp[1, 0]) == int(crit_outcome.hp[1, 0])
