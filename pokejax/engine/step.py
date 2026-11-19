"""Pure decision-point transition for Gen 5 singles.

Actions must be legal. Random outcomes retain engine.randomness addresses;
masked arithmetic can evaluate unused outcomes without consuming randomness.
"""

import jax.numpy as jnp
from jax import lax

from .abilities import (
	CONTACT,
	CONTACT_COLOR_CHANGE,
	OWN_TEMPO,
	PP_COST,
	PP_COST_PRESSURE,
	SHEER_FORCE,
	active_abilities,
	effective_weather,
	has_hook,
	move_abilities,
	reveal_abilities,
)
from .actions import SIDES, move_availability
from .boosts import apply_boosts
from .damage_effects import SELF_FAINT_ALWAYS, apply_damage_effects, apply_poison_touch
from .field import apply_move_field_effects, effective_move_type
from .items import (
	apply_after_move_items,
	clear_broken_choice_locks,
	gather_active_items,
	gather_active_update_hooks,
	update_active_items,
	update_item_state,
)
from .move_data import read_move
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
from .variable_moves import (
	SUCCESS_FOCUS_PUNCH,
	apply_autotomize,
	steal_berry,
	success_allowed,
)
from .volatiles import advance_active_turns, confuse, end_turn


def _move(
	state,
	side,
	action,
	row,
	enabled,
	first,
	opponent_pending,
	key,
	forced,
	tables,
	ability_rows,
	alive_before,
	target_switching,
):
	user, foe = state.active[side], 1 - side
	item_rows = gather_active_items(state, tables)
	rows = move_abilities(ability_rows, side)
	weather = effective_weather(state, rows)
	sheer = has_hook(rows[side], SHEER_FORCE) & row.move_sheer_force
	enabled &= (state.hp[side, user] > 0) & alive_before[foe] & (state.winner == ONGOING)
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
	state, enabled = before_move(
		state, side, row, enabled, key, forced, tables, ability_rows, recharge=recharge
	)
	# Focus Punch's beforeMoveCallback fails before any PP is spent.
	enabled &= ~((row.move_success_rule == SUCCESS_FOCUS_PUNCH) & state.focus_lost[side])
	move_enabled = enabled
	choice_start = move_enabled & item_rows.choice[side]
	choice_start &= state.volatiles[side, CHOICE_MOVE] == 0
	choice_start &= ~continuation & ~recharge
	choice_start &= row.id != 0
	state = set_active_volatile(
		state,
		side,
		CHOICE_MOVE,
		jnp.where(choice_start, row.id, state.volatiles[side, CHOICE_MOVE]),
	)
	struggle = row.id == tables.struggle_id
	slot = jnp.minimum(action, 3)
	pp = state.pp[side, user, slot].astype(jnp.int32)
	consume_pp = enabled & ~recharge & ~continuation & ~struggle
	pressure = rows[foe, PP_COST] == PP_COST_PRESSURE
	pressure_target_alive = state.hp[foe, state.active[foe]] > 0
	user_types = state.types[side, user]
	curse_self = (row.move_volatile_status == 5) & ~jnp.any(user_types == tables.ghost_type)
	pressure_target = row.move_pressure_target & ~curse_self
	pp_cost = 1 + (pressure & pressure_target_alive & pressure_target).astype(jnp.int32)
	new_pp = jnp.maximum(pp - consume_pp.astype(jnp.int32) * pp_cost, 0).astype(state.pp.dtype)
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
	state, stalled = try_stalling_move(state, side, row, enabled, will_act, key, forced, tables)
	stall_kind = row.move_stall_kind
	move_allowed = enabled & ((stall_kind == 0) | stalled)
	fakeout_fails = row.move_fakeout & fakeout_used
	move_allowed &= ~fakeout_fails
	solar_skip = row.move_charge_sun_skip & (weather == SUN)
	charge_start = move_allowed & row.move_charge & ~charging & ~solar_skip
	charge_attack = ~row.move_charge | charging | solar_skip
	attack_allowed = move_allowed & ~charge_start & charge_attack & ~recharge
	attack_allowed &= success_allowed(state, side, row, tables, rows)
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(charge_start, row.id, state.volatiles[side, TWO_TURN_MOVE]),
	)
	state = apply_charge_boosts(state, side, row, charge_start, tables, rows)
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(
			charging & started,
			-jnp.abs(state.volatiles[side, TWO_TURN_MOVE]),
			state.volatiles[side, TWO_TURN_MOVE],
		),
	)
	breaks_guard = attack_allowed & row.move_breaks_protect
	state = clear_broken_guards(state, side, breaks_guard)
	guarded, invulnerable = target_move_guards(state, side, row, tables, rows)
	state, result = apply_damage_effects(
		state,
		side,
		row,
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
		switching_foe=target_switching,
	)
	hit = result.hit
	state, status_reveals = apply_move(
		state, side, row, attack_allowed, hit, key, forced, tables, rows, weather
	)
	state = reveal_abilities(state, result.ability_reveals | status_reveals)
	state = apply_boosts(
		state, side, row, attack_allowed & ~sheer, hit, result.success, tables, rows
	)
	state = confuse(
		state,
		foe,
		attack_allowed & hit & row.move_confusion & ~has_hook(rows[foe], OWN_TEMPO),
		key,
		forced,
	)
	state, lum_between_hits = apply_secondaries(
		state,
		side,
		row,
		result.landed[0],
		result.landed[1:],
		key,
		forced,
		tables,
		rows,
		weather,
		sheer,
		item_rows,
	)
	state = apply_poison_touch(state, side, row, result.landed, key, forced, tables, rows)
	state = steal_berry(
		state, side, attack_allowed & row.move_steal_berry & result.hit, tables, rows
	)
	color_change = rows[foe, CONTACT] == CONTACT_COLOR_CHANGE
	color_type = effective_move_type(state, side, row, tables, weather).astype(state.types.dtype)
	color_target = state.active[foe]
	color_types = state.types[foe, color_target]
	color_trigger = attack_allowed & result.success & color_change & (row.move_category != 2)
	color_trigger &= (state.hp[foe, color_target] > 0) & (color_type >= 0)
	color_trigger &= ~jnp.any(color_types == color_type)
	color_member = (SIDES == foe)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == color_target)
	changed_types = jnp.stack((color_type, jnp.asarray(-1, state.types.dtype)))
	types = jnp.where(
		color_member[:, :, None] & color_trigger,
		changed_types,
		state.types,
	)
	roosting = state.volatiles[foe, ROOST] > 0
	roost_target = color_member & color_trigger & roosting
	roost_types = jnp.where(roost_target[:, :, None], changed_types, state.roost_types)
	state = state._replace(types=types, roost_types=roost_types)
	state = reveal_abilities(state, (SIDES == foe) & color_trigger)
	state = after_hit(state, side, row, attack_allowed & result.success, tables, weather)
	state = apply_move_field_effects(
		state, side, row, attack_allowed, result.field_hit, hit, key, forced, tables, item_rows
	)
	state = apply_move_volatiles(
		state, side, row, attack_allowed & result.field_hit, key, forced, tables, ability_rows
	)
	state = apply_autotomize(state, side, attack_allowed & row.move_autotomize, tables)
	user_alive_before_after_move_item = state.hp[side, user] > 0
	post_ability = tables.ability_hooks[state.ability[side, user].astype(jnp.int32)]
	post_ability_rows = jnp.where((SIDES == side)[:, None], post_ability[None, :], ability_rows)
	post_sheer = has_hook(post_ability_rows[side], SHEER_FORCE) & row.move_sheer_force
	state, herb, life_orb = apply_after_move_items(
		state, side, row, move_enabled, hit, item_rows, tables, post_sheer, post_ability_rows
	)
	# Every event still refers to the same two item holders until a drag.
	state = update_item_state(
		state,
		result.item_consumptions | lum_between_hits | herb,
		result.item_reveals | life_orb,
		tables,
	)
	did_something = attack_allowed & (hit | jnp.any(result.landed))
	state = state._replace(
		last_success=jnp.where(
			started & did_something, row.id, jnp.where(started, 0, state.last_success)
		).astype(state.last_success.dtype)
	)
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
	pivot = enabled & row.move_self_switch & hit & user_alive
	pivot &= jnp.any(bench_alive)
	drag = enabled & row.move_force_switch & hit & user_alive_before_after_move_item & target_alive
	state, dragged, ability_rows = drag_in(
		state, foe, drag, key, forced, tables, side, ability_rows
	)
	# A final hit can run Update before the queue declares the winner.
	# Fatigue confusion starts later, so only an ongoing game can cure it here.
	alive_after = jnp.any(state.hp > 0, axis=1)
	terminal = ~jnp.all(alive_after)
	update = started & (state.winner == ONGOING) & (~terminal | jnp.any(result.landed))
	ignore_confusion = (SIDES == side) & terminal & fatigue_only
	state = update_active_items(
		state,
		update,
		gather_active_update_hooks(state, tables),
		ability_rows,
		ignore_confusion,
		tables,
	)
	hp = state.hp.astype(jnp.int32)
	boosts = jnp.where((hp[SIDES, state.active] > 0)[:, None], state.boosts, 0)
	return state._replace(boosts=boosts), pivot, dragged, ability_rows, alive_after


def _resolve_action_winner(state, alive_before, alive, side, row):
	"""Resolve a move's faint result before the next queued Showdown action."""
	del alive_before
	winner = jnp.where(~alive[0], 1, 0)
	double_ko_winner = jnp.where(row.move_self_faint == SELF_FAINT_ALWAYS, 1 - side, side)
	winner = jnp.where(~jnp.any(alive), double_ko_winner, winner)
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
	"""Run a fixed move queue and pause at each public switch decision.

	Queue position 0 runs before voluntary switches and position 1 after them.
	A Pursuit user therefore hits a switching target before it leaves, and a
	Pursuit user queued behind a pivot hits it before the pivot pauses.
	"""
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
	state, first_order = action_order(
		state, actions, moves, jnp.any(requests == SWITCH), key, forced, item_rows, t, ability_rows
	)
	pending = (requests == MOVE) & (actions < 4) & ongoing
	switch_mask = (actions >= 4) & (requests != NONE) & ongoing
	# A Pursuit user whose target chose a switch goes first, before the switch-out.
	pursuer = pending & (moves == t.pursuit_id) & switch_mask[::-1] & (requests[::-1] == MOVE)
	first = jnp.where(jnp.any(pursuer), jnp.argmax(pursuer), first_order).astype(jnp.int32)
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

	run_moves = state.turn_stage == TURN_MOVES
	queue_first = state.turn_first.astype(jnp.int32)

	def run_slot(index, state, abilities, pivot_sides, alive_before, switching):
		"""One queue position: the move of queue_first ^ index, if still pending."""
		side = queue_first ^ index
		foe = 1 - side
		pursuit_behind_pivot = (state.turn_moves[side] == t.pursuit_id) & pivot_sides[foe]
		enabled = run_moves & state.turn_pending[side]
		enabled &= ~jnp.any(pivot_sides) | pursuit_behind_pivot
		enabled &= state.winner == ONGOING
		row = read_move(t, state.turn_moves[side])
		leaving = switching | pivot_sides
		state, pivot, dragged, abilities, alive = _move(
			state,
			side,
			state.turn_actions[side],
			row,
			enabled,
			queue_first,
			state.turn_pending[foe],
			key,
			forced,
			t,
			abilities,
			alive_before,
			leaving[foe],
		)
		pivot_sides = pivot_sides | ((SIDES == side) & pivot & (state.winner == ONGOING))
		pending = state.turn_pending & ~((SIDES == side) & enabled) & ~dragged
		state = state._replace(turn_pending=pending)
		state = _resolve_action_winner(state, alive_before, alive, side, row)
		return state, abilities, pivot_sides, alive

	def cancel_pursued_switches(state, switching):
		"""A switch whose target fell to a Pursuit hit never happens; the faint request covers it."""
		return switching & ((requests != MOVE) | (state.hp[SIDES, state.active] > 0))

	def switch_block(state, abilities, switching):
		"""Voluntary and requested switches: one dense switch-out and switch-in path."""
		item_rows = gather_active_items(state, t)
		previous = (state, item_rows, abilities, start_turn)
		state = switch_actions(state, actions, switching, t, abilities)
		item_rows = gather_active_items(state, t)
		abilities = active_abilities(state, t)
		handler_first = switch_handler_first(state, key, forced, item_rows, abilities)
		handler_first = jnp.where(start_turn, first_order, handler_first)
		state, abilities = run_switch_in(
			state, switching, handler_first, t, previous=previous, rows=abilities
		)
		state = update_active_items(state, jnp.any(switching), item_rows, abilities, tables=t)
		return state, abilities

	def unrolled_queue(carry):
		"""GPU: position 0, the switch block once, then position 1, straight-line."""
		state, abilities, pivot_sides, alive, switching = carry
		state, abilities, pivot_sides, alive = run_slot(
			0, state, abilities, pivot_sides, alive, switching
		)
		switching = cancel_pursued_switches(state, switching)
		state, abilities = switch_block(state, abilities, switching)
		state, abilities, pivot_sides, _ = run_slot(
			1, state, abilities, pivot_sides, alive, switching
		)
		return state, abilities, pivot_sides

	def rolled_queue(carry):
		"""CPU: a rolled loop. The switch block is masked to run after position 0 only."""

		def position(index, carry):
			state, abilities, pivot_sides, alive, switching = carry
			state, abilities, pivot_sides, alive = run_slot(
				index, state, abilities, pivot_sides, alive, switching
			)
			first = index == 0
			switching = jnp.where(first, cancel_pursued_switches(state, switching), switching)
			# The loop index is unbatched, so this cond skips the switch block at position 1.
			state, abilities = lax.cond(
				first,
				lambda operand: switch_block(*operand),
				lambda operand: operand[:2],
				(state, abilities, switching),
			)
			return state, abilities, pivot_sides, alive, switching

		state, abilities, pivot_sides, _, _ = lax.fori_loop(0, 2, position, carry)
		return state, abilities, pivot_sides

	# Both paths run the same operations; only the loop form differs by platform.
	no_pivot = jnp.zeros((2,), jnp.bool_)
	carry = (state, ability_rows, no_pivot, jnp.any(state.hp > 0, axis=1), switch_mask)
	state, ability_rows, pivot_sides = lax.platform_dependent(
		carry, cuda=unrolled_queue, default=rolled_queue
	)
	# A Pursuit hit can KO the pivoting Pokemon, which then waits for the faint request.
	pivot_sides = pivot_sides & (state.hp[SIDES, state.active] > 0)
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
	# Turn-scoped bookkeeping resets with Battle.endTurn; faints are seen per turn end.
	alive_now = state.hp > 0
	fainted = jnp.any(state.alive_mark & ~alive_now, axis=1)
	state = state._replace(
		fainted_last=jnp.where(finish_turn, fainted, state.fainted_last),
		alive_mark=jnp.where(finish_turn, alive_now, state.alive_mark),
		turn_damage=jnp.where(finish_turn, 0, state.turn_damage).astype(state.turn_damage.dtype),
		turn_hit=jnp.where(finish_turn, False, state.turn_hit),
		focus_lost=jnp.where(finish_turn, False, state.focus_lost),
		last_success=jnp.where(finish_turn, 0, state.last_success).astype(state.last_success.dtype),
	)
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
