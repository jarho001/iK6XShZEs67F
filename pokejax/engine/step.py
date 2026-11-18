"""Pure decision-point transition for Gen 5 singles.

Actions must be legal. Random outcomes retain engine.randomness addresses;
masked arithmetic can evaluate unused outcomes without consuming randomness.
"""

import jax.numpy as jnp

from .actions import SIDES, bench_slots
from .boosts import apply_boosts
from .damage_effects import SELF_FAINT_ALWAYS, apply_damage_effects
from .field import (
	apply_move_field_effects,
	apply_switch_hazards_ordered,
)
from .order import action_order
from .randomness import random_bits
from .residual import residual
from .secondaries import apply_secondaries
from .state import (
	FAKE_OUT_USED,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	MOVE,
	MUST_RECHARGE,
	NONE,
	ONGOING,
	QUICK_GUARD,
	ROOST,
	SLEEP,
	SUN,
	SWITCH,
	TEAM_SLOTS,
	TOXIC,
	TWO_TURN_MOVE,
	WIDE_GUARD,
)
from .status import after_hit, apply_move, before_move
from .turn_moves import (
	apply_charge_boosts,
	apply_move_volatiles,
	clear_broken_guards,
	set_active_volatile,
	target_is_blocked,
	try_stalling_move,
)
from .volatiles import advance_active_turns, confuse, end_turn


def _switch_both(state, actions, enabled):
	"""Switch both sides with one set of fixed-team writes."""
	choices = jnp.clip(actions - 4, 0, 4)
	targets = bench_slots(state.active)[SIDES, choices]
	outgoing = state.active
	active = jnp.where(enabled, targets, outgoing).astype(state.active.dtype)
	boosts = jnp.where(enabled[:, None], 0, state.boosts)
	volatile_columns = jnp.arange(state.volatiles.shape[1])
	side_condition = (volatile_columns == QUICK_GUARD) | (volatile_columns == WIDE_GUARD)
	clear_volatile = enabled[:, None] & ~side_condition[None, :]
	volatiles = jnp.where(clear_volatile, 0, state.volatiles)

	team_slots = jnp.arange(TEAM_SLOTS)
	outgoing_mask = team_slots[None, :] == outgoing[:, None]
	restore_roost = enabled & (state.volatiles[:, ROOST] > 0)
	types = jnp.where(
		(restore_roost[:, None] & outgoing_mask)[:, :, None], state.roost_types, state.types
	)

	target_order = state.seen_order[SIDES, targets]
	next_order = jnp.max(state.seen_order, axis=1) + 1
	seen_order = jnp.where(
		(enabled & (target_order == 0))[:, None] & (team_slots[None, :] == targets[:, None]),
		next_order[:, None],
		state.seen_order,
	)

	status = state.status[SIDES, targets]
	counter = state.status_counter[SIDES, targets]
	counter = jnp.where(status == TOXIC, 0, counter)
	counter = jnp.where(status == SLEEP, state.sleep_start[SIDES, targets], counter)
	target_mask = enabled[:, None] & (team_slots[None, :] == targets[:, None])
	active_turns = jnp.where(target_mask, 0, state.active_turns)
	status_counter = jnp.where(target_mask, counter[:, None], state.status_counter)
	return state._replace(
		active=active,
		active_turns=active_turns,
		boosts=boosts,
		volatiles=volatiles,
		types=types,
		seen_order=seen_order,
		status_counter=status_counter,
	)


def _move(
	state,
	side,
	action,
	move,
	enabled,
	charged_at_start,
	first,
	requests,
	actions,
	key,
	forced,
	tables,
):
	user, foe = state.active[side], 1 - side
	enabled &= (state.hp[side, user] > 0) & jnp.any(state.hp[foe] > 0) & (state.winner == ONGOING)
	charge_canceled = charged_at_start & (state.volatiles[side, TWO_TURN_MOVE] == 0)
	enabled &= ~charge_canceled
	started = enabled
	charging = state.volatiles[side, TWO_TURN_MOVE] > 0
	locked = state.volatiles[side, LOCKED_MOVE] > 0
	recharge = state.volatiles[side, MUST_RECHARGE] > 0
	continuation = charging | locked
	fakeout_used = state.volatiles[side, FAKE_OUT_USED] > 0
	state = set_active_volatile(
		state,
		side,
		FAKE_OUT_USED,
		jnp.where(started, 1, state.volatiles[side, FAKE_OUT_USED]),
	)
	state, enabled = before_move(state, side, move, enabled, key, forced, tables, recharge=recharge)
	move_enabled = enabled
	struggle = move == tables.struggle_id
	slot = jnp.minimum(action, 3)
	pp = state.pp[side, user, slot].astype(jnp.int32)
	consume_pp = enabled & ~recharge & ~continuation & ~struggle
	new_pp = jnp.maximum(pp - consume_pp, 0).astype(state.pp.dtype)
	revealed = state.revealed_moves[side, user, slot] | (enabled & ~recharge & ~struggle)
	team_slots = jnp.arange(TEAM_SLOTS)
	move_slots = jnp.arange(state.pp.shape[-1])
	selected_move = (
		(SIDES == side)[:, None, None]
		& (team_slots[None, :, None] == user)
		& (move_slots[None, None, :] == slot)
	)
	state = state._replace(
		pp=jnp.where(selected_move, new_pp, state.pp),
		revealed_moves=jnp.where(selected_move, revealed, state.revealed_moves),
	)
	will_act = (side == first) & (requests[foe] == MOVE) & (actions[foe] < 4)
	state, stalled = try_stalling_move(state, side, move, enabled, will_act, key, forced, tables)
	stall_kind = tables.move_stall_kind[move]
	move_allowed = enabled & ((stall_kind == 0) | stalled)
	fakeout_fails = tables.move_fakeout[move] & fakeout_used
	move_allowed &= ~fakeout_fails
	solar_skip = tables.move_charge_sun_skip[move] & (state.weather == SUN)
	charge_start = move_allowed & tables.move_charge[move] & ~charging & ~solar_skip
	charge_attack = ~tables.move_charge[move] | charging | solar_skip
	attack_allowed = move_allowed & ~charge_start & charge_attack & ~recharge
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(charge_start, move, state.volatiles[side, TWO_TURN_MOVE]),
	)
	state = apply_charge_boosts(state, side, move, charge_start, tables)
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(
			charging & (move_enabled | (started & ~move_enabled)),
			-jnp.abs(state.volatiles[side, TWO_TURN_MOVE]),
			state.volatiles[side, TWO_TURN_MOVE],
		),
	)
	breaks_guard = attack_allowed & tables.move_breaks_protect[move]
	state = clear_broken_guards(state, side, breaks_guard)
	blocked = target_is_blocked(state, side, move, tables)
	state, success, hit, hit_landed, field_hit = apply_damage_effects(
		state, side, move, attack_allowed, blocked, key, forced, tables
	)
	state = apply_move(state, side, move, attack_allowed, hit, key, forced, tables)
	state = apply_boosts(state, side, move, attack_allowed, hit, success, tables)
	state = confuse(state, foe, attack_allowed & hit & tables.move_confusion[move], key, forced)
	state = apply_secondaries(state, side, move, hit_landed[0], hit_landed[1:], key, forced, tables)
	state = after_hit(state, side, move, attack_allowed & success, tables)
	state = apply_move_field_effects(
		state, side, move, attack_allowed, field_hit, hit, key, forced, tables
	)
	state = apply_move_volatiles(state, side, move, attack_allowed & field_hit, key, forced, tables)
	lock_continuation = locked & move_enabled
	lock_remaining = state.volatiles[side, LOCKED_REMAINING]
	lock_ends = lock_continuation & (lock_remaining == 1)
	state = set_active_volatile(
		state,
		side,
		LOCKED_MOVE,
		jnp.where(lock_ends, 0, state.volatiles[side, LOCKED_MOVE]),
	)
	state = set_active_volatile(
		state,
		side,
		LOCKED_REMAINING,
		jnp.where(lock_ends, 0, state.volatiles[side, LOCKED_REMAINING]),
	)
	confusion = lock_ends & (state.volatiles[side, LOCKED_TRUE_DURATION] <= 1)
	state = set_active_volatile(
		state,
		side,
		LOCKED_TRUE_DURATION,
		jnp.where(lock_ends, 0, state.volatiles[side, LOCKED_TRUE_DURATION]),
	)
	state = confuse(state, side, confusion, key, forced)
	hp = state.hp.astype(jnp.int32)
	boosts = jnp.where((hp[SIDES, state.active] > 0)[:, None], state.boosts, 0)
	return state._replace(boosts=boosts)


def step(state, actions, key, forced, engine_tables):
	"""Run both legal choices to the next move, replacement, or end request.

	Faint replacements precede Battle.nextTurn, so their request retains the
	old turn number. Finishing the replacement step advances it; a win does not.
	"""
	t = engine_tables
	key = random_bits(key)
	ongoing = state.winner == ONGOING
	normal = jnp.any(state.request == MOVE) & ongoing
	replacement = ~normal & jnp.any(state.request == SWITCH) & ongoing
	moves = state.moves[SIDES, state.active, jnp.minimum(actions, 3)]
	charging = state.volatiles[:, TWO_TURN_MOVE]
	locked = state.volatiles[:, LOCKED_MOVE]
	recharge = state.volatiles[:, MUST_RECHARGE] > 0
	forced_move = (charging > 0) | (locked > 0) | recharge
	moves = jnp.where(charging > 0, charging, moves)
	moves = jnp.where(locked > 0, locked, moves)
	moves = jnp.where(recharge, 0, moves)
	usable = (state.moves[SIDES, state.active] != 0) & (state.pp[SIDES, state.active] > 0)
	moves = jnp.where(jnp.any(usable, axis=1) | forced_move, moves, t.struggle_id)
	first = action_order(state, actions, moves, replacement, key, forced, t)
	requests = state.request
	will_switch = (actions >= 4) & (requests != NONE) & ongoing
	last_faint = jnp.asarray(0, state.winner.dtype)
	alive = jnp.any(state.hp > 0, axis=1)
	# Switches have higher queue priority than every move in this pool. Showdown
	# resolves the grouped SwitchIn handlers after the switch actions and before
	# the first move, so compute their shared speed order once here.
	state = _switch_both(state, actions, will_switch)
	alive_after = jnp.any(state.hp > 0, axis=1)
	newly_dead = alive & ~alive_after
	last_faint = jnp.where(
		jnp.any(newly_dead),
		jnp.where(jnp.sum(newly_dead) == 1, jnp.argmax(newly_dead), 1 - first),
		last_faint,
	)
	alive = alive_after
	state = apply_switch_hazards_ordered(state, will_switch, jnp.any(will_switch), key, forced, t)
	alive_after = jnp.any(state.hp > 0, axis=1)
	newly_dead = alive & ~alive_after
	last_faint = jnp.where(
		jnp.any(newly_dead),
		jnp.where(jnp.sum(newly_dead) == 1, jnp.argmax(newly_dead), 1 - first),
		last_faint,
	)
	alive = alive_after
	for side in (first, 1 - first):
		action = actions[side]
		move = normal & (requests[side] == MOVE) & (action < 4)
		state = _move(
			state,
			side,
			action,
			moves[side],
			move,
			charging[side] > 0,
			first,
			requests,
			actions,
			key,
			forced,
			t,
		)
		alive_after = jnp.any(state.hp > 0, axis=1)
		newly_dead = alive & ~alive_after
		faint_order = jnp.where(t.move_self_faint[moves[side]] == SELF_FAINT_ALWAYS, 1 - side, side)
		faint_order = jnp.where(
			jnp.any(newly_dead) & (jnp.sum(newly_dead) == 1),
			jnp.argmax(newly_dead.astype(jnp.int32)),
			faint_order,
		)
		last_faint = jnp.where(jnp.any(newly_dead), faint_order, last_faint)
		alive = alive_after
	winner = jnp.where(~alive[0], 1, 0)
	winner = jnp.where(~jnp.any(alive), last_faint, winner)
	winner = jnp.where(jnp.all(alive), ONGOING, winner).astype(state.winner.dtype)
	winner = jnp.where(state.winner == ONGOING, winner, state.winner)
	state = state._replace(winner=winner)
	state = residual(state, normal & (state.winner == ONGOING), key, forced, t)
	ended = state.winner != ONGOING
	fainted = state.hp[SIDES, state.active] == 0
	needs_switch = fainted & jnp.any(state.hp > 0, axis=1)
	clear_fainted = needs_switch & ~ended
	roost = (state.volatiles[SIDES, ROOST] > 0) & fainted
	volatile_columns = jnp.arange(state.volatiles.shape[1])
	side_condition = (volatile_columns == QUICK_GUARD) | (volatile_columns == WIDE_GUARD)
	team_slots = jnp.arange(TEAM_SLOTS)
	active_slots = team_slots[None, :] == state.active[:, None]
	types = jnp.where((active_slots & roost[:, None])[:, :, None], state.roost_types, state.types)
	status = jnp.where(active_slots & clear_fainted[:, None], 0, state.status)
	status_counter = jnp.where(active_slots & clear_fainted[:, None], 0, state.status_counter)
	state = state._replace(
		types=types,
		volatiles=jnp.where(fainted[:, None] & ~side_condition[None, :], 0, state.volatiles),
		status=status,
		status_counter=status_counter,
	)
	waiting = jnp.any(needs_switch)
	request = jnp.where(waiting, jnp.where(needs_switch, SWITCH, NONE), MOVE)
	request = jnp.where(ended, NONE, request).astype(state.request.dtype)
	advance = (normal | replacement) & ~waiting & ~ended
	state = end_turn(state, normal & ~ended, key, forced)
	state = advance_active_turns(state, advance)
	return state._replace(request=request, turn=state.turn + advance.astype(jnp.int32))
