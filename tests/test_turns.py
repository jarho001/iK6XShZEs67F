"""Gen 5 protection, charge, recharge, lock, Fake Out and trapping rules."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_status

from pokejax.engine.actions import legal_actions
from pokejax.engine.observation import observe
from pokejax.engine.randomness import (
	CONFUSION_DURATION,
	HIT,
	ORDER,
	SECONDARY,
	STALL_SUCCESS,
	UNFORCED,
)
from pokejax.engine.residual import residual
from pokejax.engine.state import (
	CONFUSION,
	FAKE_OUT_USED,
	FLINCH,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	MUST_RECHARGE,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	POISON,
	PROTECT,
	SLEEP,
	STALL,
	SWITCH,
	TWO_TURN_MOVE,
)
from pokejax.engine.step import step
from pokejax.engine.turn_moves import try_stalling_move

KEY = jax.random.key(73)
FORCED = UNFORCED.at[HIT, :, 0].set(1).at[ORDER, 0, 0].set(0)
RUN = jax.jit(step)


@pytest.fixture(scope="module")
def setup():
	return test_status.setup.__wrapped__()


def move(state, tables, side, name):
	return state._replace(moves=state.moves.at[side, 0, 0].set(tables.moves.index[name]))


def test_protect_repeat_failure_and_stall_reset(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "protect")
	first = RUN(state, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(first.hp[1, 0]) == int(state.hp[1, 0])
	assert int(first.volatiles[0, PROTECT]) == 0  # one-turn guard expired
	assert int(first.volatiles[0, STALL]) == 2  # Gen 5's first repeat denominator

	repeated = RUN(first, jnp.array([0, 0]), KEY, FORCED.at[STALL_SUCCESS, 0, 0].set(1), t)
	assert int(repeated.volatiles[0, STALL]) == 4  # The shared Gen 5 counter doubles.

	failed = RUN(repeated, jnp.array([0, 0]), KEY, FORCED.at[STALL_SUCCESS, 0, 0].set(0), t)
	assert int(failed.hp[0, 0]) < int(first.hp[0, 0])
	assert int(failed.volatiles[0, STALL]) == 0  # failure drops the shared counter

	near_cap = state._replace(volatiles=state.volatiles.at[0, STALL].set(128))
	capped = RUN(near_cap, jnp.array([0, 0]), KEY, FORCED.at[STALL_SUCCESS, 0, 0].set(1), t)
	assert int(capped.volatiles[0, STALL]) == 256
	capped_again = RUN(capped, jnp.array([0, 0]), KEY, FORCED.at[STALL_SUCCESS, 0, 0].set(1), t)
	assert int(capped_again.volatiles[0, STALL]) == 256

	nonstall = first._replace(moves=first.moves.at[0, 0, 0].set(tables.moves.index["tackle"]))
	cleared = RUN(nonstall, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(cleared.volatiles[0, STALL]) == 0


def test_protect_samples_gen5_chances_through_the_two_to_32_cap(setup):
	tables, t, state = setup
	protect = tables.moves.index["protect"]

	def sample_batch(current, values):
		values = jnp.asarray(values, dtype=jnp.uint32)
		keys = jnp.zeros((values.shape[0],) + UNFORCED.shape, dtype=jnp.uint32)
		keys = keys.at[:, STALL_SUCCESS, 0, 0].set(values)
		return jax.vmap(
			lambda key: try_stalling_move(current, 0, protect, True, True, key, UNFORCED, t)
		)(keys)

	# The first use always passes and initializes the next denominator to two.
	states, succeeded = sample_batch(state, [0, 1])
	assert np.asarray(succeeded).all()
	state = jax.tree.map(lambda values: values[0], states)
	assert int(state.volatiles[0, STALL]) == 2

	# Enumerating each raw residue verifies exactly one passing sample per denominator.
	for denominator in (2, 4, 8, 16, 32, 64, 128):
		assert int(state.volatiles[0, STALL]) == denominator
		states, succeeded = sample_batch(state, range(denominator))
		assert int(jnp.sum(succeeded)) == 1
		state = jax.tree.map(lambda values: values[0], states)
		assert int(state.volatiles[0, STALL]) == min(denominator * 2, 256)

	# At 256, Gen 5 switches to randomChance(1, 2**32): only a zero uint32 passes.
	states, succeeded = sample_batch(state, [0, 1, 255, 256, 2**32 - 1])
	assert np.array_equal(np.asarray(succeeded), [True, False, False, False, False])
	assert np.asarray(states.volatiles[:, 0, STALL]).tolist() == [256, 0, 0, 0, 0]


def test_fake_out_only_on_first_active_move_action(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "fakeout")
	first = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[SECONDARY, 0, 0].set(1), t)
	assert int(first.volatiles[0, FAKE_OUT_USED]) == 1
	assert int(first.hp[1, 0]) < int(state.hp[1, 0])

	second = RUN(first, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(second.hp[1, 0]) == int(first.hp[1, 0])
	assert int(second.hp[0, 0]) < int(first.hp[0, 0])

	switched = RUN(first, jnp.array([4, 0]), KEY, FORCED, t)
	assert int(switched.active[0]) == 1
	assert int(switched.volatiles[0, FAKE_OUT_USED]) == 0


def test_two_turn_charge_blocks_switch_and_hits_after_target_switches(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "fly")
	state = state._replace(
		moves=state.moves.at[1, 0, 0].set(tables.moves.index["tackle"]),
		stats=state.stats.at[0, 0, 5].set(200),
	)
	charging = RUN(state, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(charging.volatiles[0, TWO_TURN_MOVE]) == tables.moves.index["fly"]
	assert int(charging.hp[0, 0]) == int(state.hp[0, 0])
	assert not bool(legal_actions(charging, t)[0, 4:].any())

	# A switch selected during the forced second turn occurs before Fly lands.
	switched = RUN(charging, jnp.array([0, 4]), KEY, FORCED, t)
	assert int(switched.active[1]) == 1
	assert int(switched.hp[1, 1]) < int(charging.hp[1, 1])
	assert int(switched.volatiles[0, TWO_TURN_MOVE]) == 0


def test_two_turn_charge_survives_target_faint_replacement_and_clears_on_flinch(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "fly")
	state = state._replace(
		hp=state.hp.at[1, 0].set(1),
		status=state.status.at[1, 0].set(POISON),
		active_turns=state.active_turns.at[1, 0].set(1),
	)
	charging = RUN(state, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(charging.request[1]) == SWITCH
	assert int(charging.volatiles[0, TWO_TURN_MOVE]) > 0
	replaced = RUN(charging, jnp.array([0, 4]), KEY, FORCED, t)
	assert int(replaced.active[1]) == 1
	assert int(replaced.volatiles[0, TWO_TURN_MOVE]) > 0
	attacked = RUN(replaced, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(attacked.hp[1, 1]) < int(replaced.hp[1, 1])

	interrupted = replaced._replace(volatiles=replaced.volatiles.at[0, FLINCH].set(1))
	interrupted = RUN(interrupted, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(interrupted.volatiles[0, TWO_TURN_MOVE]) == 0
	assert bool(legal_actions(interrupted, t)[0, 1:4].any())


def test_recharge_miss_and_target_faint_rules(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "hydrocannon")
	miss = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[HIT, 0, 0].set(0), t)
	assert int(miss.volatiles[0, MUST_RECHARGE]) == 0

	# Gen 5 Showdown applies the self volatile even when the hit KOs its target.
	ko = state._replace(hp=state.hp.at[1, 0].set(1))
	out = RUN(ko, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.hp[1, 0]) == 0
	assert int(out.volatiles[0, MUST_RECHARGE]) == 1
	assert bool(legal_actions(out, t)[0, 0])
	assert not bool(legal_actions(out, t)[0, 1:4].any())
	assert not bool(legal_actions(out, t)[0, 4:].any())


def test_locked_move_ends_on_target_ko_and_confuses(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "outrage")
	state = state._replace(
		volatiles=state.volatiles.at[0, LOCKED_MOVE]
		.set(tables.moves.index["outrage"])
		.at[0, LOCKED_REMAINING]
		.set(1)
		.at[0, LOCKED_TRUE_DURATION]
		.set(1),
		hp=state.hp.at[1, 0].set(1),
	)
	assert not bool(legal_actions(state, t)[0, 1:4].any())
	assert not bool(legal_actions(state, t)[0, 4:].any())
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED.at[CONFUSION_DURATION, 0, 0].set(0), t)
	assert int(out.hp[1, 0]) == 0
	assert int(out.volatiles[0, LOCKED_MOVE]) == 0
	assert int(out.volatiles[0, CONFUSION]) == 2
	assert int(out.request[1]) == SWITCH


def test_sleep_removes_locked_move_before_true_duration_decrement(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "outrage")
	state = state._replace(
		volatiles=state.volatiles.at[0, LOCKED_MOVE]
		.set(tables.moves.index["outrage"])
		.at[0, LOCKED_REMAINING]
		.set(2)
		.at[0, LOCKED_TRUE_DURATION]
		.set(2),
		status=state.status.at[0, 0].set(SLEEP),
		status_counter=state.status_counter.at[0, 0].set(2),
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.volatiles[0, LOCKED_MOVE]) == 0
	assert int(out.volatiles[0, CONFUSION]) == 0


def test_partial_trap_switch_mask_source_departure_and_rapid_spin(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "rapidspin")
	state = state._replace(
		volatiles=state.volatiles.at[0, PARTIALLY_TRAPPED].set(5).at[0, PARTIAL_TRAP_SOURCE].set(1),
		active_turns=state.active_turns.at[1, 0].set(1),
	)
	assert not bool(legal_actions(state, t)[0, 4:].any())

	freed = RUN(state, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(freed.volatiles[0, PARTIALLY_TRAPPED]) == 0
	assert bool(legal_actions(freed, t)[0, 4:].any())

	trapped = state._replace(moves=state.moves.at[0, 0, 0].set(tables.moves.index["tackle"]))
	departed = RUN(trapped, jnp.array([0, 4]), KEY, FORCED, t)
	assert int(departed.active[1]) == 1
	assert int(departed.volatiles[0, PARTIALLY_TRAPPED]) == 0
	assert bool(legal_actions(departed, t)[0, 4:].any())

	# Showdown decrements volatile duration before its residual callback, so a
	# timer at one ends without one more 1/16 max HP tick.
	ending = state._replace(
		volatiles=state.volatiles.at[0, PARTIALLY_TRAPPED].set(1).at[0, PARTIAL_TRAP_SOURCE].set(1)
	)
	end_state = residual(ending, jnp.asarray(True), KEY, FORCED, t)
	assert int(end_state.hp[0, 0]) == int(ending.hp[0, 0])
	assert int(end_state.volatiles[0, PARTIALLY_TRAPPED]) == 0


def test_flinched_locked_move_expires_without_confusion_when_true_duration_remains(setup):
	tables, t, state = setup
	state = move(state, tables, 0, "outrage")
	state = state._replace(
		volatiles=state.volatiles.at[0, LOCKED_MOVE]
		.set(tables.moves.index["outrage"])
		.at[0, LOCKED_REMAINING]
		.set(1)
		.at[0, LOCKED_TRUE_DURATION]
		.set(2)
		.at[0, FLINCH]
		.set(1)
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.volatiles[0, LOCKED_MOVE]) == 0
	assert int(out.volatiles[0, CONFUSION]) == 0


def test_turn_state_observation_hides_durations(setup):
	_, t, state = setup
	state = state._replace(
		volatiles=state.volatiles.at[0, TWO_TURN_MOVE]
		.set(1)
		.at[0, LOCKED_MOVE]
		.set(2)
		.at[0, LOCKED_REMAINING]
		.set(2)
		.at[0, PARTIALLY_TRAPPED]
		.set(5)
		.at[0, PARTIAL_TRAP_SOURCE]
		.set(1)
		.at[0, MUST_RECHARGE]
		.set(1),
		active_turns=state.active_turns.at[1, 0].set(1),
	)
	longer = state._replace(
		volatiles=state.volatiles.at[0, LOCKED_REMAINING].set(3).at[0, PARTIALLY_TRAPPED].set(6)
	)
	first = observe(state, 0, t)
	second = observe(longer, 0, t)
	for left, right in zip(jax.tree.leaves(first), jax.tree.leaves(second), strict=True):
		np.testing.assert_array_equal(left, right)
	assert bool(first.own.charging)
	assert bool(first.own.locked)
	assert bool(first.own.trapped)
	assert bool(first.own.recharging)
