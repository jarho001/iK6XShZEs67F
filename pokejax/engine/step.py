"""Pure decision-point transition for Gen 5 singles.

Actions must be legal. Random outcomes retain engine.randomness addresses;
masked arithmetic can evaluate unused outcomes without consuming randomness.
"""

import jax.numpy as jnp
from jax import lax

from .abilities import (
	OWN_TEMPO,
	SHEER_FORCE,
	active_abilities,
	effective_weather,
	has_hook,
	move_abilities,
)
from .actions import SIDES, move_availability
from .boosts import apply_boosts
from .damage_effects import SELF_FAINT_ALWAYS, apply_damage_effects
from .field import apply_move_field_effects
from .items import (
	apply_after_move_items,
	clear_broken_choice_locks,
	gather_active_items,
	gather_active_update_hooks,
	update_active_items,
)
from .order import action_order
from .randomness import random_bits
from .residual import residual
from .secondaries import apply_secondaries
from .state import (
	CHOICE_MOVE,
	CONFUSION,
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
	SUN,
	SWITCH,
	TEAM_SLOTS,
	TURN_IDLE,
	TURN_MOVES,
	TURN_REPLACEMENT,
	TURN_RESIDUAL,
	TWO_TURN_MOVE,
	WIDE_GUARD,
)
from .status import after_hit, apply_move, before_move
from .switching import drag_in, run_switch_in, switch_actions, switch_handler_first
from .turn_moves import (
	apply_charge_boosts,
	apply_move_volatiles,
	clear_broken_guards,
	set_active_volatile,
	target_move_guards,
	try_stalling_move,
)
from .volatiles import advance_active_turns, confuse, end_turn


def _move(
	state,
	side,
	action,
	move,
	enabled,
	charged_at_start,
	first,
	opponent_pending,
	key,
	forced,
	tables,
	ability_rows,
	alive_before,
):
	user, foe = state.active[side], 1 - side
	item_rows = gather_active_items(state, tables)
	rows = move_abilities(ability_rows, side)
	weather = effective_weather(state, rows)
	sheer = has_hook(rows[side], SHEER_FORCE) & tables.move_sheer_force[move]
	enabled &= (state.hp[side, user] > 0) & alive_before[foe] & (state.winner == ONGOING)
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
	choice_start = move_enabled & item_rows.choice[side]
	choice_start &= state.volatiles[side, CHOICE_MOVE] == 0
	choice_start &= ~continuation & ~recharge
	choice_start &= move != 0
	state = set_active_volatile(
		state,
		side,
		CHOICE_MOVE,
		jnp.where(choice_start, move, state.volatiles[side, CHOICE_MOVE]),
	)
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
	will_act = (side == first) & opponent_pending
	state, stalled = try_stalling_move(state, side, move, enabled, will_act, key, forced, tables)
	stall_kind = tables.move_stall_kind[move]
	move_allowed = enabled & ((stall_kind == 0) | stalled)
	fakeout_fails = tables.move_fakeout[move] & fakeout_used
	move_allowed &= ~fakeout_fails
	solar_skip = tables.move_charge_sun_skip[move] & (weather == SUN)
	charge_start = move_allowed & tables.move_charge[move] & ~charging & ~solar_skip
	charge_attack = ~tables.move_charge[move] | charging | solar_skip
	attack_allowed = move_allowed & ~charge_start & charge_attack & ~recharge
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(charge_start, move, state.volatiles[side, TWO_TURN_MOVE]),
	)
	state = apply_charge_boosts(state, side, move, charge_start, tables, rows)
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
	guarded, invulnerable = target_move_guards(state, side, move, tables, rows)
	state, success, hit, hit_landed, field_hit = apply_damage_effects(
		state,
		side,
		move,
		attack_allowed,
		guarded,
		invulnerable,
		key,
		forced,
		tables,
		item_rows,
		rows,
		weather,
		sheer,
		~will_act,
	)
	state = apply_move(state, side, move, attack_allowed, hit, key, forced, tables, rows, weather)
	state = apply_boosts(state, side, move, attack_allowed & ~sheer, hit, success, tables, rows)
	state = confuse(
		state,
		foe,
		attack_allowed & hit & tables.move_confusion[move] & ~has_hook(rows[foe], OWN_TEMPO),
		key,
		forced,
	)
	state = apply_secondaries(
		state,
		side,
		move,
		hit_landed[0],
		hit_landed[1:],
		key,
		forced,
		tables,
		rows,
		weather,
		sheer,
		item_rows,
	)
	state = after_hit(state, side, move, attack_allowed & success, tables, weather)
	state = apply_move_field_effects(
		state, side, move, attack_allowed, field_hit, hit, key, forced, tables, item_rows
	)
	state = apply_move_volatiles(
		state, side, move, attack_allowed & field_hit, key, forced, tables, ability_rows
	)
	user_alive_before_after_move_item = state.hp[side, user] > 0
	state = apply_after_move_items(state, side, move, move_enabled, hit, item_rows, tables, sheer)
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
	confusion &= ~has_hook(rows[side], OWN_TEMPO)
	fatigue_only = confusion & (state.volatiles[side, CONFUSION] == 0)
	state = confuse(state, side, confusion, key, forced)
	state = clear_broken_choice_locks(state, tables)
	user_alive = state.hp[side, user] > 0
	target_alive = state.hp[foe, state.active[foe]] > 0
	bench_alive = (state.hp[side] > 0) & (jnp.arange(TEAM_SLOTS) != state.active[side])
	pivot = enabled & tables.move_self_switch[move] & hit & user_alive
	pivot &= jnp.any(bench_alive)
	drag = (
		enabled
		& tables.move_force_switch[move]
		& hit
		& user_alive_before_after_move_item
		& target_alive
	)
	state, dragged, ability_rows = drag_in(state, foe, drag, key, forced, tables, side)
	# A final hit can run Update before the queue declares the winner.
	# Fatigue confusion starts later, so only an ongoing game can cure it here.
	alive_after = jnp.any(state.hp > 0, axis=1)
	terminal = ~jnp.all(alive_after)
	update = started & (state.winner == ONGOING) & (~terminal | jnp.any(hit_landed))
	ignore_confusion = (SIDES == side) & terminal & fatigue_only
	state = update_active_items(
		state,
		update,
		gather_active_update_hooks(state, tables),
		ability_rows,
		ignore_confusion,
	)
	hp = state.hp.astype(jnp.int32)
	boosts = jnp.where((hp[SIDES, state.active] > 0)[:, None], state.boosts, 0)
	return state._replace(boosts=boosts), pivot, dragged, ability_rows, alive_after


def _resolve_action_winner(state, alive_before, alive, side, move, tables):
	"""Resolve a move's faint result before the next queued Showdown action."""
	newly_dead = alive_before & ~alive
	last_faint = jnp.where(tables.move_self_faint[move] == SELF_FAINT_ALWAYS, 1 - side, side)
	last_faint = jnp.where(
		jnp.any(newly_dead) & (jnp.sum(newly_dead) == 1),
		jnp.argmax(newly_dead.astype(jnp.int32)),
		last_faint,
	)
	winner = jnp.where(~alive[0], 1, 0)
	winner = jnp.where(~jnp.any(alive), last_faint, winner)
	winner = jnp.where(jnp.all(alive), ONGOING, winner).astype(state.winner.dtype)
	return state._replace(winner=jnp.where(state.winner == ONGOING, winner, state.winner))


def _clear_turn_plan(state, clear):
	return state._replace(
		turn_stage=jnp.where(clear, TURN_IDLE, state.turn_stage).astype(state.turn_stage.dtype),
		turn_actions=jnp.where(clear, 0, state.turn_actions),
		turn_moves=jnp.where(clear, 0, state.turn_moves),
		turn_pending=jnp.where(clear, False, state.turn_pending),
		turn_first=jnp.where(clear, 0, state.turn_first).astype(state.turn_first.dtype),
	)


def step(state, actions, key, forced, engine_tables):
	"""Run a fixed move queue and pause at each public switch decision."""
	t = engine_tables
	key = random_bits(key)
	ongoing = state.winner == ONGOING
	requests = state.request
	start_turn = (state.turn_stage == TURN_IDLE) & jnp.any(requests == MOVE) & ongoing
	legacy_replacement = (state.turn_stage == TURN_IDLE) & jnp.any(requests == SWITCH) & ongoing

	# Fix the chosen actions, resolved moves, and queue order exactly once. Resume
	# calls use these saved values and never recompute priority or speed order.
	move_slots = jnp.minimum(actions, 3)
	active_moves, usable, choice_active, choice_usable = move_availability(state, t)
	moves = active_moves[SIDES, move_slots]
	charging = state.volatiles[:, TWO_TURN_MOVE]
	locked = state.volatiles[:, LOCKED_MOVE]
	recharge = state.volatiles[:, MUST_RECHARGE] > 0
	forced_move = (charging > 0) | (locked > 0) | recharge
	moves = jnp.where(charging > 0, charging, moves)
	moves = jnp.where(locked > 0, locked, moves)
	moves = jnp.where(recharge, 0, moves)
	choice_has_pp = jnp.any(choice_usable, axis=1)
	item_rows = gather_active_items(state, t)
	choice_struggle = choice_active & ~choice_has_pp & ~forced_move
	moves = jnp.where(jnp.any(usable, axis=1) | forced_move, moves, t.struggle_id)
	moves = jnp.where(choice_struggle, t.struggle_id, moves)
	ability_rows = active_abilities(state, t)
	state, first = action_order(
		state, actions, moves, jnp.any(requests == SWITCH), key, forced, item_rows, t, ability_rows
	)
	pending = (requests == MOVE) & (actions < 4) & ongoing
	state = state._replace(
		turn_stage=jnp.where(start_turn, TURN_MOVES, state.turn_stage).astype(
			state.turn_stage.dtype
		),
		turn_actions=jnp.where(start_turn, move_slots, state.turn_actions).astype(
			state.turn_actions.dtype
		),
		turn_moves=jnp.where(start_turn, moves, state.turn_moves).astype(state.turn_moves.dtype),
		turn_pending=jnp.where(start_turn, pending, state.turn_pending),
		turn_first=jnp.where(start_turn, first, state.turn_first).astype(state.turn_first.dtype),
	)

	# Both voluntary and requested switches share the same dense switch-out and
	# switch-in paths. Switches happen before the saved moves, as in Showdown.
	switch_mask = (actions >= 4) & (requests != NONE) & ongoing
	previous = (state, item_rows, ability_rows, start_turn)
	state = switch_actions(state, actions, switch_mask)
	item_rows = gather_active_items(state, t)
	ability_rows = active_abilities(state, t)
	handler_first = switch_handler_first(state, key, forced, item_rows, ability_rows)
	handler_first = jnp.where(start_turn, first, handler_first)
	state, ability_rows = run_switch_in(
		state, switch_mask, handler_first, t, previous=previous, rows=ability_rows
	)
	state = update_active_items(state, jnp.any(switch_mask), item_rows, ability_rows)

	# Run the two fixed move slots through the same transition. A pivot masks
	# any remaining slot until the player has chosen a replacement.
	run_moves = state.turn_stage == TURN_MOVES
	first = state.turn_first.astype(jnp.int32)

	def execute(index, carry):
		state, abilities, pivot_sides = carry
		side = first ^ index
		enabled = run_moves & state.turn_pending[side] & ~jnp.any(pivot_sides)
		enabled &= state.winner == ONGOING
		alive_before = jnp.any(state.hp > 0, axis=1)
		move = state.turn_moves[side]
		state, pivot, dragged, abilities, alive = _move(
			state,
			side,
			state.turn_actions[side],
			move,
			enabled,
			state.volatiles[side, TWO_TURN_MOVE] > 0,
			first,
			state.turn_pending[1 - side],
			key,
			forced,
			t,
			abilities,
			alive_before,
		)
		pivot_sides |= (SIDES == side) & pivot & (state.winner == ONGOING)
		pending = state.turn_pending & ~((SIDES == side) & enabled) & ~dragged
		state = state._replace(turn_pending=pending)
		state = _resolve_action_winner(state, alive_before, alive, side, move, t)
		return state, abilities, pivot_sides

	# GPU unrolling lets XLA fuse the two slots into the surrounding rollout.
	# The compact loop keeps the CPU PPO compile tractable. The backend choice
	# is resolved at lowering, independent of any batched battle values.
	def run_queue(carry, *, unroll=False):
		return lax.fori_loop(0, 2, execute, carry, unroll=unroll)

	state, ability_rows, pivot_sides = lax.platform_dependent(
		(state, ability_rows, jnp.zeros((2,), jnp.bool_)),
		cuda=lambda carry: run_queue(carry, unroll=True),
		default=run_queue,
	)
	pivot_pause = jnp.any(pivot_sides) & (state.winner == ONGOING)
	state = state._replace(
		turn_stage=jnp.where(
			run_moves & ~jnp.any(state.turn_pending), TURN_RESIDUAL, state.turn_stage
		).astype(state.turn_stage.dtype),
		request=jnp.where(
			pivot_pause,
			jnp.where(pivot_sides, SWITCH, NONE),
			state.request,
		).astype(state.request.dtype),
	)

	# Residual is a separate saved stage. End-of-turn volatiles expire only after
	# any replacement request is completed, just before Battle.endTurn advances.
	run_residual = (state.turn_stage == TURN_RESIDUAL) & ~pivot_pause & (state.winner == ONGOING)
	state = residual(state, run_residual, key, forced, t, ability_rows)
	state = end_turn(state, run_residual & (state.winner == ONGOING), key, forced)
	state = state._replace(
		turn_stage=jnp.where(
			run_residual & (state.winner == ONGOING), TURN_REPLACEMENT, state.turn_stage
		).astype(state.turn_stage.dtype)
	)

	# A fainted active is requested only once the rest of the queued turn and its
	# residual have run. Hazard KOs on a replacement repeat this same request.
	fainted = state.hp[SIDES, state.active] == 0
	needs_switch = fainted & jnp.any(state.hp > 0, axis=1)
	waiting = jnp.any(needs_switch)
	ended = state.winner != ONGOING
	active_slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	replacement_stage = state.turn_stage == TURN_REPLACEMENT
	clear_fainted = needs_switch & ~ended & replacement_stage
	ability = jnp.where(active_slots & fainted[:, None], state.ability_base, state.ability)
	status = jnp.where(active_slots & clear_fainted[:, None], 0, state.status)
	status_counter = jnp.where(active_slots & clear_fainted[:, None], 0, state.status_counter)
	roost = fainted & (state.volatiles[:, ROOST] > 0)
	types = jnp.where((active_slots & roost[:, None])[:, :, None], state.roost_types, state.types)
	columns = jnp.arange(state.volatiles.shape[1])
	side_condition = (columns == QUICK_GUARD) | (columns == WIDE_GUARD)
	volatiles = jnp.where(fainted[:, None] & ~side_condition[None, :], 0, state.volatiles)
	state = state._replace(
		types=types,
		ability=ability,
		volatiles=volatiles,
		status=status,
		status_counter=status_counter,
	)
	finish_replacement = replacement_stage & ~waiting & ~ended
	finish_legacy = legacy_replacement & ~waiting & ~ended
	finish_turn = finish_replacement | finish_legacy
	state = advance_active_turns(state, finish_turn)
	request = jnp.where(waiting, jnp.where(needs_switch, SWITCH, NONE), MOVE)
	request = jnp.where(pivot_pause, state.request, request)
	request = jnp.where(ended, NONE, request).astype(state.request.dtype)
	state = state._replace(
		request=request,
		turn=state.turn + finish_turn.astype(jnp.int32),
	)
	clear_plan = finish_turn | ended
	return _clear_turn_plan(state, clear_plan)
