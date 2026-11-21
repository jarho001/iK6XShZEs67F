"""Pure decision-point transition for Gen 5 singles.

Actions must be legal. Random outcomes retain engine.randomness addresses;
masked arithmetic can evaluate unused outcomes without consuming randomness.
"""

import jax
import jax.numpy as jnp
from jax import lax

from .abilities import (
	CONTACT,
	CONTACT_COLOR_CHANGE,
	MAGIC_BOUNCE,
	NOTRACE,
	OWN_TEMPO,
	PP_COST,
	PP_COST_PRESSURE,
	SHEER_FORCE,
	TRACE,
	UNNERVE,
	absorb_hit,
	absorption,
	active_abilities,
	effective_weather,
	has_hook,
	move_abilities,
	reveal_abilities,
)
from .actions import SIDES, move_availability
from .boosts import BELLY_DRUM, apply_boosts
from .damage import accuracy_hit
from .damage_effects import SELF_FAINT_ALWAYS, apply_damage_effects, apply_poison_touch
from .delayed import (
	SELF_DESTINY_BOND,
	SLOT_HEALING_WISH,
	SLOT_WISH,
	advance_wish,
	apply_destiny_bond,
	apply_healing_wish,
	apply_trick_room,
	apply_wish,
	faint_active,
)
from .events import (
	CANT_FOCUS_PUNCH,
	CANT_NONE,
	EFFECT_IMMUNE,
	EFFECT_NONE,
	OUTCOME_CRIT,
	OUTCOME_FAILED,
	OUTCOME_MISS,
	clear_events,
	entry_kinds,
	finish_events,
	first_cut,
	record_step_bounds,
	write_attempt,
	write_entry,
)
from .field import apply_move_field_effects, effective_move_type
from .forms import (
	relic_song_forme,
	restore_form,
	switch_foe_view,
	sync_weather_formes,
)
from .illusion import break_illusion
from .item_moves import apply_item_swap, apply_knock_off
from .items import (
	BERRY_SITRUS,
	apply_after_move_items,
	clear_broken_choice_locks,
	consume_active_items,
	gather_active_items,
	gather_active_update_hooks,
	update_active_items,
	update_item_state,
)
from .move_data import move_column, read_move
from .order import action_order
from .randomness import BOUNCE_HIT, random_bits
from .residual import residual
from .restrictions import (
	VOLATILE_LEECH_SEED,
	apply_restrictions,
	encore_override,
)
from .secondaries import apply_secondaries
from .sleep_talk import choose_called_slot
from .state import (
	CHOICE_MOVE,
	CONFUSION,
	DESTINY_BOND,
	FAKE_OUT_USED,
	HP,
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
	SPIKES,
	SUN,
	SWITCH,
	TEAM_SLOTS,
	TOXIC_SPIKES,
	TRACE_SEEK,
	TRANSFORMED,
	TURN_IDLE,
	TURN_MOVES,
	TURN_REPLACEMENT,
	TURN_RESIDUAL,
	TWO_TURN_MOVE,
	WIDE_GUARD,
)
from .status import after_hit, apply_move, before_move
from .substitute import SELF_SUBSTITUTE, apply_substitute, substitute_allowed
from .switching import (
	copy_on_entry,
	drag_in,
	reset_status_counters,
	run_switch_in,
	switch_actions,
	switch_handler_first,
	switch_in_hazards,
)
from .trace import trace_copy
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


def select_row(pred, chosen, other):
	"""Branch-free choice between two packed move rows."""
	return type(other)(*(jnp.where(pred, a, b) for a, b in zip(chosen, other)))


def slot_mask(state, side, user, slot):
	"""One dense [side, member, move slot] mask."""
	team = jnp.arange(TEAM_SLOTS)
	moves = jnp.arange(state.pp.shape[-1])
	return (
		(SIDES == side)[:, None, None]
		& (team[None, :, None] == user)
		& (moves[None, None, :] == slot)
	)


def pp_cost_for(row, pressure, target_alive, user_types, tables):
	"""PP spent by one use: Pressure adds one more on foe-targeted moves (DeductPP)."""
	curse_self = (row.move_volatile_status == 5) & ~jnp.any(user_types == tables.ghost_type)
	pressure_target = row.move_pressure_target & ~curse_self
	return 1 + (pressure & target_alive & pressure_target).astype(jnp.int32)


def write_last_move(state, side, user, move_id, enabled):
	"""Showdown Pokemon.moveUsed: the member's last move, cleared when it switches out."""
	member = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	mask = member & enabled
	return state._replace(
		last_move=jnp.where(mask, move_id.astype(state.last_move.dtype), state.last_move)
	)


def _move(
	state,
	side,
	action,
	queued,
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
	# Encore's OverrideAction swaps a chosen non-Struggle move before BeforeMove.
	# The effective move ID is resolved before its wide row is loaded (one gather per use).
	encore_on, encored, encore_slot = encore_override(state, side, queued, tables)
	encore_on &= ~continuation & ~recharge
	row = read_move(tables, jnp.where(encore_on, encored, queued))
	action = jnp.where(encore_on, encore_slot, action)
	state, enabled, cant = before_move(
		state, side, row, enabled, key, forced, tables, ability_rows, recharge=recharge
	)
	# Focus Punch's beforeMoveCallback fails before any PP is spent.
	before = enabled
	enabled &= ~((row.move_success_rule == SUCCESS_FOCUS_PUNCH) & state.focus_lost[side])
	cant = first_cut(cant, before, enabled, CANT_FOCUS_PUNCH)
	move_enabled = enabled
	choice_start = move_enabled & item_rows.choice[side]
	choice_start &= state.volatiles[side, CHOICE_MOVE] == 0
	# A Choice item locks the move even when it continues a locked move (Showdown ModifyMove).
	choice_start &= ~recharge
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
	# Sleep Talk calls one callable move (Showdown useMove, which skips runMove): the call
	# pays no PP, runs no BeforeMove checks, and leaves lastMove on Sleep Talk. Pressure
	# still charges Sleep Talk's slot one extra PP when the called move targets the foe.
	sleep_call = row.move_sleep_talk & enabled & (state.status[side, user] == SLEEP)
	can_call, call_slot = choose_called_slot(state, side, tables, key, forced)
	sleep_call &= can_call
	# Sleep Talk fails while its user is awake or has no move to call (onTry, onHit).
	talk_fail = row.move_sleep_talk & enabled & ~sleep_call
	called_id = state.moves[side, user, call_slot].astype(jnp.int32)
	pp_cost = pp_cost_for(row, pressure, pressure_target_alive, user_types, tables)
	# Sleep Talk's called move and Nature Power's static call are disjoint (the move table
	# has no move with both flags), so one resolved ID and one wide row serve both calls.
	natured = row.move_call > 0
	calling = sleep_call | natured
	call_row = read_move(tables, jnp.where(sleep_call, called_id, row.move_call.astype(jnp.int32)))
	# A called move's Pressure surcharge is charged to the caller's slot (useMove sourceEffect).
	call_extra = pp_cost_for(call_row, pressure, pressure_target_alive, user_types, tables) - 1
	pp_cost += calling.astype(jnp.int32) * call_extra
	new_pp = jnp.maximum(pp - consume_pp.astype(jnp.int32) * pp_cost, 0).astype(state.pp.dtype)
	# A locked or charging move was revealed when it began; the action's slot is not its slot.
	revealed = state.revealed_moves[side, user, slot] | (
		enabled & ~recharge & ~struggle & ~continuation
	)
	selected_move = slot_mask(state, side, user, slot)
	state = state._replace(
		pp=jnp.where(selected_move, new_pp, state.pp),
		revealed_moves=jnp.where(selected_move, revealed, state.revealed_moves),
	)
	# Showdown records the last move once PP is paid (or a continuation or Struggle runs).
	last_id = row.id
	last_moved = enabled & ~recharge & ((pp > 0) | continuation | struggle)
	# Showdown's moveUsed runs before the hit, so an Encore reflected back at this user reads it.
	state = write_last_move(state, side, user, last_id, last_moved)
	# Nature Power is a static call: the user uses Earthquake, and Nature Power keeps its PP.
	row = select_row(calling, call_row, row)
	# Sheer Force's ModifyMove reads the move that is really used: after an Encore override or a
	# call by Sleep Talk or Nature Power.
	sheer = has_hook(rows[side], SHEER_FORCE) & row.move_sheer_force
	# A move aimed at a fainted foe is retargeted to nothing: |move| with [notarget], then -fail.
	# Curse used by a non-Ghost targets its user (Showdown onModifyMove), so it has no foe target.
	curse_self = (row.move_volatile_status == 5) & ~jnp.any(
		state.types[side, state.active[side]] == tables.ghost_type
	)
	target_gone = (
		started & row.move_target_foe & ~curse_self & (state.hp[foe, state.active[foe]] == 0)
	)
	struggle = row.id == tables.struggle_id
	will_act = (side == first) & opponent_pending
	state, stalled = try_stalling_move(state, side, row, enabled, will_act, key, forced, tables)
	# Protect fails when no action follows it in the queue, or its StallMove check fails (onPrepareHit).
	stall_failed = enabled & (row.move_stall_kind > 0) & ~stalled
	stall_kind = row.move_stall_kind
	move_allowed = enabled & ((stall_kind == 0) | stalled)
	fakeout_fails = row.move_fakeout & fakeout_used
	move_allowed &= ~fakeout_fails
	solar_skip = row.move_charge_sun_skip & (weather == SUN)
	charge_start = move_allowed & row.move_charge & ~charging & ~solar_skip
	charge_attack = ~row.move_charge | charging | solar_skip
	attack_allowed = move_allowed & ~charge_start & charge_attack & ~recharge
	succeeds = success_allowed(state, side, row, tables, rows)
	attack_allowed &= succeeds
	success_fail = move_allowed & ~charge_start & charge_attack & ~recharge & ~succeeds
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
	sub_fail = (
		attack_allowed
		& (row.move_self_volatile == SELF_SUBSTITUTE)
		& ~substitute_allowed(state, side)
	)
	state = apply_substitute(
		state, side, attack_allowed & (row.move_self_volatile == SELF_SUBSTITUTE)
	)
	state = apply_destiny_bond(
		state, side, attack_allowed & (row.move_self_volatile == SELF_DESTINY_BOND)
	)
	state = apply_trick_room(state, attack_allowed & row.move_trick_room)
	wish_fail = (
		attack_allowed & (row.move_slot_condition == SLOT_WISH) & (state.wish_turns[side] != 0)
	)
	state = apply_wish(state, side, attack_allowed & (row.move_slot_condition == SLOT_WISH))
	healing_move = attack_allowed & (row.move_slot_condition == SLOT_HEALING_WISH)
	state, healing_placed, no_replacement = apply_healing_wish(state, side, healing_move)
	healing_fail = healing_move & ~healing_placed
	attack_allowed &= ~no_replacement
	breaks_guard = attack_allowed & row.move_breaks_protect
	state = clear_broken_guards(state, side, breaks_guard)
	guarded, invulnerable = target_move_guards(state, side, row, tables, rows)
	# TryHit: Magic Bounce reflects a reflectable move before its accuracy roll; the original
	# move then does nothing, and the reflected move runs further down with the holder acting.
	reflect = attack_allowed & row.move_reflectable & has_hook(rows[foe], MAGIC_BOUNCE)
	# A fainted target never reaches TryHit: the move already failed with [notarget].
	reflect &= ~guarded & ~invulnerable & (state.hp[foe, state.active[foe]] > 0)
	attack_allowed = attack_allowed & ~reflect
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
	# DamagingHit: a damaging hit on an Illusion holder ends the disguise.
	state = break_illusion(state, (SIDES == foe) & result.success, tables)
	hit = result.hit
	connected = hit | result.sub_hit
	# A reflected move (Magic Bounce) is used by the holder against the original user: its own
	# accuracy check decides it, and the effects below act for the holder (acting) through the
	# same effect calls, so each effect kind is compiled once.
	bounce_hit = accuracy_hit(state, foe, row, key, forced, tables, rows, weather, kind=BOUNCE_HIT)
	# TryHit of the reflected move: the original user's absorbing ability takes it before any
	# accuracy check (Volt Absorb against a reflected Thunder Wave).
	holder_types = state.types[side, user]
	reflect_immune = row.move_checks_immunity & jnp.any(
		(holder_types >= 0) & tables.type_immune[row.move_type, jnp.maximum(holder_types, 0)]
	)
	reflect_absorbed = absorption(
		state, foe, rows[side], row.move_type, 0, reflect_immune, reflect, row, tables
	)
	state, healed_hp = absorb_hit(
		state, side, rows[side], reflect_absorbed, state.hp[side, user].astype(jnp.int32)
	)
	user_slot = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	state = state._replace(hp=jnp.where(user_slot, healed_hp.astype(state.hp.dtype), state.hp))
	state = reveal_abilities(state, (SIDES == side) & reflect_absorbed)
	# The reflected |move| line names Magic Bounce, whatever the move does next.
	state = reveal_abilities(state, (SIDES == foe) & reflect)
	bounced = reflect & bounce_hit & ~reflect_absorbed & (state.hp[side, user] > 0)
	acting = jnp.where(reflect, foe, side)
	effect_on = attack_allowed | reflect
	effect_hit = jnp.where(reflect, bounced, hit)
	state, status_reveals, status_immune, status_failed = apply_move(
		state,
		acting,
		row,
		effect_on,
		effect_hit,
		key,
		forced,
		tables,
		rows,
		weather,
		reach=jnp.where(reflect, reflect & bounce_hit & ~reflect_absorbed, result.reach),
	)
	state = reveal_abilities(state, result.ability_reveals | status_reveals)
	# Roar and Whirlwind fail when the foe has no living bench member to drag in (canSwitch).
	no_bench = ~jnp.any((state.hp[foe] > 0) & (jnp.arange(TEAM_SLOTS) != state.active[foe]))
	drag_fail = attack_allowed & row.move_force_switch & hit & no_bench & (row.move_category == 2)
	state, swapped = apply_item_swap(
		state, side, attack_allowed & row.move_item_swap & hit, tables, rows
	)
	trick_fail = attack_allowed & row.move_item_swap & hit & ~swapped
	item_rows = gather_active_items(state, tables)
	# Belly Drum fails at or below half HP, at +6 Attack, and for a 1-HP user (onHit).
	drum_fail = (
		attack_allowed
		& (row.move_boost_effect == BELLY_DRUM)
		& (
			(
				2 * state.hp[side, user].astype(jnp.int32)
				<= state.stats[side, user, 0].astype(jnp.int32)
			)
			| (state.boosts[side, 0] >= 6)
			| (state.stats[side, user, 0] <= 1)
		)
	)
	state = apply_boosts(
		state,
		acting,
		row,
		(attack_allowed & ~sheer) | reflect,
		effect_hit,
		jnp.where(reflect, bounced, result.success | result.sub_hit),
		tables,
		rows,
	)
	state = confuse(
		state,
		foe,
		attack_allowed & hit & row.move_confusion & ~has_hook(rows[foe], OWN_TEMPO),
		key,
		forced,
	)
	state, lum_between_hits, sync_failed, sync_immune = apply_secondaries(
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
		result.sub_landed[0],
		result.sub_landed[1:],
	)
	state = apply_poison_touch(state, side, row, result.landed, key, forced, tables, rows)
	knock = attack_allowed & row.move_knock_off & jnp.any(result.landed)
	knock &= state.hp[side, user] > 0
	# A Focus Sash that saved the target was already used up by its Damage event (takeItem
	# finds nothing), so the sash counts as consumed, not knocked off.
	knock &= ~result.item_consumptions[foe]
	state = apply_knock_off(state, side, knock, tables, rows)
	item_rows = gather_active_items(state, tables)
	state = steal_berry(
		state, side, attack_allowed & row.move_steal_berry & result.hit, tables, rows
	)
	color_change = rows[foe, CONTACT] == CONTACT_COLOR_CHANGE
	color_type = effective_move_type(state, side, row, tables, weather).astype(state.types.dtype)
	color_target = state.active[foe]
	color_types = state.types[foe, color_target]
	# Sheer Force removes the secondary hit that Color Change needs (AfterMoveSecondary).
	color_trigger = attack_allowed & result.success & color_change & (row.move_category != 2)
	color_trigger &= ~sheer & (row.id != tables.struggle_id)
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
	# A Spikes or Toxic Spikes move with the layers already at their cap, or a weather move that
	# repeats the standing weather, fails (Field.setWeather -fail, the hazard's maximum).
	field_hit = jnp.where(reflect, bounced, result.field_hit)
	field_fail = (effect_on & field_hit) & (
		((row.move_hazard == 1) & (state.hazards[1 - acting, SPIKES] >= 3))
		| ((row.move_hazard == 2) & (state.hazards[1 - acting, TOXIC_SPIKES] >= 2))
		| ((row.move_weather > 0) & (row.move_weather == state.weather))
	)
	# Rapid Spin runs as onAfterHit, or onAfterSubDamage on a substitute hit, only while its user
	# lives: Rocky Helmet or Rough Skin can faint the user first.
	spin_hit = (effect_hit | result.sub_hit) & (state.hp[side, user] > 0)
	state = apply_move_field_effects(
		state,
		acting,
		row,
		effect_on,
		jnp.where(reflect, bounced, result.field_hit),
		spin_hit,
		key,
		forced,
		tables,
		item_rows,
	)
	state = apply_move_volatiles(
		state,
		side,
		row,
		attack_allowed & result.field_hit,
		key,
		forced,
		tables,
		ability_rows,
		connected=attack_allowed & connected,
	)
	state = apply_autotomize(state, side, attack_allowed & row.move_autotomize, tables)
	state, restrict_failed, restrict_immune = apply_restrictions(
		state, acting, row, effect_hit, opponent_pending & ~reflect, tables, rows, weather
	)
	# Relic Song (onAfterMoveSecondarySelf): a Meloetta changes forme after a connecting hit.
	relic = attack_allowed & row.move_relic_forme & connected
	state = relic_song_forme(state, (SIDES == side) & relic, tables)
	# Destiny Bond: a KO by this move's damage takes the attacker down too.
	db_trigger = attack_allowed & result.success & (state.volatiles[foe, DESTINY_BOND] > 0)
	db_trigger &= (state.hp[foe, state.active[foe]] == 0) & (state.hp[side, user] > 0)
	state = faint_active(state, side, db_trigger)
	state = faint_active(state, side, healing_placed)
	post_ability = tables.ability_hooks[state.ability[side, user].astype(jnp.int32)]
	post_ability_rows = jnp.where((SIDES == side)[:, None], post_ability[None, :], ability_rows)
	post_sheer = has_hook(post_ability_rows[side], SHEER_FORCE) & row.move_sheer_force
	state, herb, life_orb = apply_after_move_items(
		state, side, row, move_enabled, connected, item_rows, tables, post_sheer, post_ability_rows
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
	user_alive = state.hp[side, user] > 0
	target_alive = state.hp[foe, state.active[foe]] > 0
	bench_alive = (state.hp[side] > 0) & (jnp.arange(TEAM_SLOTS) != state.active[side])
	pivot = enabled & row.move_self_switch & connected & user_alive
	pivot &= jnp.any(bench_alive)
	# forceSwitch is checked in the same hit, before the contact damage that may faint the user, and
	# the drag itself happens after the move, so a user that fainted to Rough Skin still drags.
	drag = enabled & row.move_force_switch & hit & target_alive
	# Trace's Update copies a copyable foe; its Start effects run with this move's switch-in.
	state, traced = trace_copy(state, tables)
	# A reflected Roar or Whirlwind drags the original user; Transform copies for the user.
	bounce_drag = bounced & row.move_force_switch & (state.hp[side, user] > 0)
	transform_on = attack_allowed & row.move_transform & hit
	# Update runs after the hit and before the phazed target leaves, so its berry still goes off
	# (Sitrus Berry after Dragon Tail). The incoming Pokemon gets its own Update below.
	dragged_side = jnp.where(reflect, side, foe)
	dragged_slot = state.active[dragged_side]
	dragged_hp = state.hp[dragged_side, dragged_slot].astype(jnp.int32)
	dragged_max = state.stats[dragged_side, dragged_slot, HP].astype(jnp.int32)
	unnerved = has_hook(rows[1 - dragged_side], UNNERVE) & (
		state.hp[1 - dragged_side, state.active[1 - dragged_side]] > 0
	)
	eats = started & (drag | bounce_drag) & (state.winner == ONGOING) & ~unnerved
	eats &= item_rows.berry_kind[dragged_side] == BERRY_SITRUS
	eats &= (dragged_hp > 0) & (2 * dragged_hp <= dragged_max) & (dragged_hp < dragged_max)
	dragged_member = (SIDES == dragged_side)[:, None] & (
		jnp.arange(TEAM_SLOTS)[None, :] == dragged_slot
	)
	healed = jnp.minimum(dragged_max, dragged_hp + jnp.maximum(dragged_max // 4, 1))
	state = state._replace(
		hp=jnp.where(dragged_member & eats, healed.astype(state.hp.dtype), state.hp)
	)
	state = consume_active_items(state, (SIDES == dragged_side) & eats)
	state, dragged, ability_rows = drag_in(
		state,
		jnp.where(reflect, side, foe),
		drag | bounce_drag,
		key,
		forced,
		tables,
		jnp.where(reflect, foe, side),
		ability_rows,
		copy_mask=(SIDES == side) & transform_on,
		extra_starts=traced,
	)
	# The Choice lock ends once its move or item is gone, also when Transform replaced the moves.
	state = clear_broken_choice_locks(state, tables)
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
	state = update_forms(state, tables)
	hp = state.hp.astype(jnp.int32)
	boosts = jnp.where((hp[SIDES, state.active] > 0)[:, None], state.boosts, 0)
	# Event record (events.py): this side's move attempt. A reflected move is the holder's attempt.
	user = (SIDES == side) & started
	user_immune = (status_immune | restrict_immune) & ~reflect
	acting_fail = status_failed | restrict_failed | field_fail | status_immune
	failed = (
		sync_failed
		| result.guarded
		| result.immune
		| target_gone
		| trick_fail
		| result.heal_full
		| result.sub_refused
		| drum_fail
		| drag_fail
		| stall_failed
		| fakeout_fails
		| success_fail
		| talk_fail
		| sub_fail
		| wish_fail
		| healing_fail
		| (acting_fail & ~reflect)
	)
	outcome = (
		jnp.where(result.missed, OUTCOME_MISS, 0)
		| jnp.where(failed, OUTCOME_FAILED, 0)
		| jnp.where(result.critical, OUTCOME_CRIT, 0)
	)
	effect = jnp.where(user_immune | sync_immune, EFFECT_IMMUNE, result.effect)
	state = write_attempt(
		state,
		user,
		move=jnp.where(move_enabled, row.id, 0),
		cant=jnp.where(move_enabled, CANT_NONE, cant),
		outcome=jnp.where(move_enabled, outcome, 0),
		effect=jnp.where(move_enabled, effect, EFFECT_NONE),
		hits=jnp.where(move_enabled, result.hits, 0),
		order=True,
	)
	holder = (SIDES == foe) & reflect
	holder_failed = acting_fail & reflect
	# A reflected Leech Seed is refused by a Grass-type original user with -immune.
	reflected_seed = reflect & (row.move_volatile_status == VOLATILE_LEECH_SEED)
	reflected_seed &= jnp.any(state.types[side, state.active[side]] == tables.grass_type)
	holder_immune = ((status_immune | restrict_immune) & reflect) | reflected_seed
	holder_outcome = jnp.where(reflect & ~bounce_hit, OUTCOME_MISS, 0) | jnp.where(
		holder_failed, OUTCOME_FAILED, 0
	)
	state = write_attempt(
		state,
		holder,
		move=row.id,
		cant=CANT_NONE,
		outcome=holder_outcome,
		effect=jnp.where(holder_immune, EFFECT_IMMUNE, EFFECT_NONE),
		hits=0,
	)
	return (
		state._replace(boosts=boosts),
		pivot,
		dragged,
		ability_rows,
		alive_after,
		result.instafaint_kill,
	)


def update_hooks(state, tables):
	"""Showdown's Update event for formes and seeking abilities.

	A seeking Trace copies its foe and runs the copied ability's Start effects. Forecast and
	Flower Gift follow the effective weather of the active Pokemon.
	"""
	state, copied = trace_copy(state, tables)
	state, _ = run_switch_in(
		state, copied, 0, tables, hazards=False, rows=active_abilities(state, tables)
	)
	return sync_weather_formes(state, tables, active_abilities(state, tables))


def update_forms(state, tables):
	"""Forme changes that follow the weather, without any Start effects (the residual's update)."""
	return sync_weather_formes(state, tables, active_abilities(state, tables))


def _resolve_action_winner(state, alive_before, alive, side, self_faint, last_first=False):
	"""Resolve a move's faint result before the next queued Showdown action.

	With both sides out, checkWin names the side of the last faint processed. The queue
	holds the attacker's self-faint first, then the target (ALWAYS); an instafaint KO of
	the attacker (Aftermath, Rough Skin, Iron Barbs, Rocky Helmet; last_first) is moved to
	the front, so the target wins.
	"""
	del alive_before
	winner = jnp.where(~alive[0], 1, 0)
	double_ko_winner = jnp.where(last_first | (self_faint == SELF_FAINT_ALWAYS), 1 - side, side)
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
	state = clear_events(state, ongoing)
	bounds = record_step_bounds(state)
	entries = entry_kinds(state, requests, bounds)
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
	# Pursuit's onBeforeSwitchOut removes the leaving Pokemon's Destiny Bond before it is hit.
	leaving_pursued = switch_mask & pursuer[::-1]
	bond_column = jnp.arange(state.volatiles.shape[1]) == DESTINY_BOND
	state = state._replace(
		volatiles=jnp.where(
			leaving_pursued[:, None] & bond_column[None, :], 0, state.volatiles
		).astype(state.volatiles.dtype)
	)
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
		queued = state.turn_moves[side].astype(jnp.int32)
		leaving = switching | pivot_sides
		state, pivot, dragged, abilities, alive, last_first = _move(
			state,
			side,
			state.turn_actions[side],
			queued,
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
		state = _resolve_action_winner(
			state, alive_before, alive, side, move_column(t, queued, "move_self_faint"), last_first
		)
		return state, abilities, pivot_sides, alive

	def cancel_pursued_switches(state, switching):
		"""A switch whose target fell to a Pursuit hit never happens; the faint request covers it."""
		return switching & ((requests != MOVE) | (state.hp[SIDES, state.active] > 0))

	def switch_block(state, abilities, switching):
		"""Voluntary and requested switches: one dense switch-out and switch-in path."""
		state = write_entry(state, switching, entries)
		item_rows = gather_active_items(state, t)
		previous = (state, item_rows, abilities, start_turn)
		state = switch_actions(state, actions, switching, t, abilities)
		# runSwitch caches each entrant's speed when it is queued. Its SwitchIn hazards run
		# before any Imposter or Trace copy, and the copy before the Start effects.
		entrant_rows = active_abilities(state, t)
		handler_first = switch_handler_first(
			state, key, forced, gather_active_items(state, t), entrant_rows
		)
		handler_first = jnp.where(start_turn, first_order, handler_first)
		before = state.hp[SIDES, state.active] > 0
		state = switch_in_hazards(state, switching, handler_first, t, entrant_rows, previous)
		# Each entrant's status SwitchIn handler runs before its own hazards, in speed order; when
		# the faster side's hazards end the battle, the slower side's handler never runs.
		cut_short = (SIDES != handler_first) & (state.winner != ONGOING)
		cut_short &= state.hp[handler_first, state.active[handler_first]] == 0
		state = reset_status_counters(state, switching & ~cut_short)
		# Imposter (onSwitchIn) and Trace (onStart) copy the foe as it stands when this side's
		# switch-in runs; a foe entering after it is still the old active.
		view = switch_foe_view(
			previous[0], state, switching, first_order, jnp.any(requests == SWITCH)
		)
		state, _ = copy_on_entry(
			state, switching, jnp.zeros((2,), jnp.bool_), t, view, SIDES == handler_first
		)
		# A seeking Trace whose holder leaves after its foe entered copies the new foe in that
		# Update, before its own switch-out: only the public ability keeps the copy.
		leaves_late = switching & switching[::-1] & (first_order == 1 - SIDES)
		foe_now = state.active[::-1]
		foe_ability = state.ability[SIDES[::-1], foe_now]
		foe_rows = t.ability_hooks[foe_ability.astype(jnp.int32)]
		copies = (
			leaves_late
			& (previous[0].volatiles[:, TRACE_SEEK] > 0)
			& has_hook(active_abilities(previous[0], t), TRACE)
			& (foe_ability != 0)
			& (foe_ability != t.noability_id)
			& ~has_hook(foe_rows, NOTRACE)
			& (state.hp[SIDES[::-1], foe_now] > 0)
		)
		slots = jnp.arange(TEAM_SLOTS)[None, :] == previous[0].active[:, None]
		copy_slots = slots & copies[:, None]
		public = jnp.where(
			copy_slots,
			foe_ability[:, None].astype(state.ability_public.dtype),
			state.ability_public,
		)
		state = state._replace(ability_public=public)
		# That copy's Start effects (Intimidate, weather setters) run in the same Update, against
		# the entrant, before its own switch-in. The holder has left, so run from its old slot.
		copy_state = state._replace(
			active=jnp.where(copies, previous[0].active, state.active),
			ability=jnp.where(
				copy_slots, foe_ability[:, None].astype(state.ability.dtype), state.ability
			),
		)
		copy_state, _ = run_switch_in(copy_state, copies, 0, t, hazards=False)
		state = copy_state._replace(active=state.active, ability=state.ability)
		item_rows = gather_active_items(state, t)
		abilities = active_abilities(state, t)
		# An Imposter whose foe starts first copies it after that Start (its Download boost).
		late_copy = switching & (state.volatiles[:, TRANSFORMED] > 0)
		late_copy &= switching[::-1] & (SIDES != handler_first)
		state, abilities = run_switch_in(
			state,
			switching,
			handler_first,
			t,
			previous=previous,
			rows=abilities,
			hazards=False,
			before=before,
			late_copy=late_copy,
		)
		# The Update after the faster entrant's switch-in also reaches the Pokemon that has still to
		# leave: a berry it could not eat under that foe's Unnerve goes off before it leaves.
		old = previous[0]
		mid = state._replace(
			active=jnp.where(leaves_late, old.active, state.active),
			volatiles=jnp.where(leaves_late[:, None], old.volatiles, state.volatiles),
			boosts=jnp.where(leaves_late[:, None], old.boosts, state.boosts),
		)
		leaver_rows = jax.tree.map(
			lambda old_rows, new_rows: jnp.where(
				leaves_late.reshape((2,) + (1,) * (old_rows.ndim - 1)), old_rows, new_rows
			),
			previous[1],
			item_rows,
		)
		mid = update_active_items(
			mid, leaves_late, leaver_rows, jnp.where(leaves_late[:, None], previous[2], abilities)
		)
		state = state._replace(
			hp=mid.hp,
			status=mid.status,
			status_counter=mid.status_counter,
			item=mid.item,
			item_consumed=mid.item_consumed,
			last_item=mid.last_item,
			item_revealed=mid.item_revealed,
			item_public=mid.item_public,
			item_gone=mid.item_gone,
			ability_public=mid.ability_public,
		)
		state = update_active_items(state, jnp.any(switching), item_rows, abilities, tables=t)
		state = update_hooks(state, t)
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
	state = update_forms(state, t)
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
	state = restore_form(state, fainted, t)
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
	state = advance_wish(state, finish_turn)
	request = jnp.where(waiting, jnp.where(needs_switch, SWITCH, NONE), MOVE)
	request = jnp.where(pivot_pause, state.request, request)
	request = jnp.where(ended, NONE, request).astype(state.request.dtype)
	state = state._replace(
		request=request,
		turn=state.turn + finish_turn.astype(jnp.int32),
	)
	clear_plan = finish_turn | ended
	return _clear_turn_plan(finish_events(state, bounds, ongoing), clear_plan)
