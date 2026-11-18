"""Dense fixed-slot switch writes and shared switch-in hazard handling."""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .abilities import (
	START,
	START_ANNOUNCE,
	START_ANTICIPATION,
	START_DOWNLOAD,
	START_FRISK,
	START_INTIMIDATE,
	WEATHER,
	active_abilities,
	boost_delta,
	effective_weather,
	move_abilities,
	reveal_abilities,
	suppresses_weather,
)
from .actions import SIDES, bench_slots
from .field import apply_active_switch_hazards, set_weather
from .items import reveal_active_items, update_item_state
from .order import active_speed
from .randomness import DRAG_IN, SWITCH_HANDLER_ORDER, UNFORCED, first_side, integer
from .state import (
	DEF,
	NUM_VOLATILES,
	ONGOING,
	QUICK_GUARD,
	ROOST,
	SLEEP,
	SPD,
	TEAM_SLOTS,
	TOXIC,
	WIDE_GUARD,
)


def switch_members(state, switch_mask, targets):
	"""Replace selected active members, clearing active-only state densely."""
	outgoing = state.active
	active = jnp.where(switch_mask, targets, outgoing).astype(state.active.dtype)
	boosts = jnp.where(switch_mask[:, None], 0, state.boosts)
	columns = jnp.arange(NUM_VOLATILES)
	side_condition = (columns == QUICK_GUARD) | (columns == WIDE_GUARD)
	clear_volatile = switch_mask[:, None] & ~side_condition[None, :]
	volatiles = jnp.where(clear_volatile, 0, state.volatiles)

	slots = jnp.arange(TEAM_SLOTS)
	outgoing_mask = slots[None, :] == outgoing[:, None]
	ability = jnp.where(outgoing_mask & switch_mask[:, None], state.ability_base, state.ability)
	clear_fainted = switch_mask & (outgoing != targets) & (state.hp[SIDES, outgoing] == 0)
	status_team = jnp.where(outgoing_mask & clear_fainted[:, None], 0, state.status)
	status_counter_team = jnp.where(outgoing_mask & clear_fainted[:, None], 0, state.status_counter)
	restore_roost = switch_mask & (state.volatiles[:, ROOST] > 0)
	types = jnp.where(
		(restore_roost[:, None] & outgoing_mask)[:, :, None], state.roost_types, state.types
	)
	target_seen = state.seen_order[SIDES, targets]
	next_seen = jnp.max(state.seen_order, axis=1) + 1
	seen_order = jnp.where(
		(switch_mask & (target_seen == 0))[:, None] & (slots[None, :] == targets[:, None]),
		next_seen[:, None],
		state.seen_order,
	)

	target_status = state.status[SIDES, targets]
	counter = state.status_counter[SIDES, targets]
	counter = jnp.where(target_status == TOXIC, 0, counter)
	counter = jnp.where(target_status == SLEEP, state.sleep_start[SIDES, targets], counter)
	target_mask = switch_mask[:, None] & (slots[None, :] == targets[:, None])
	active_turns = jnp.where(target_mask, 0, state.active_turns)
	status_counter_team = jnp.where(target_mask, counter[:, None], status_counter_team)
	return state._replace(
		active=active,
		ability=ability,
		active_turns=active_turns,
		boosts=boosts,
		volatiles=volatiles,
		types=types,
		status=status_team,
		status_counter=status_counter_team,
		seen_order=seen_order,
	)


def switch_actions(state, actions, switch_mask):
	"""Translate action IDs 4..8 to fixed team slots and switch both sides."""
	choices = jnp.clip(actions - 4, 0, 4)
	targets = bench_slots(state.active)[SIDES, choices]
	return switch_members(state, switch_mask, targets)


def drag_in(state, side, enabled, key, forced, tables, source=None):
	"""Uniformly select a usable bench member and run its normal hazards."""
	slots = jnp.arange(TEAM_SLOTS)
	eligible = (state.hp[side] > 0) & (slots != state.active[side])
	count = jnp.sum(eligible.astype(jnp.int32))
	rank = integer(key, UNFORCED, DRAG_IN, side, 0, jnp.maximum(count, 1))
	cumulative = jnp.cumsum(eligible.astype(jnp.int32))
	selected = eligible & (cumulative == rank + 1)
	sampled = jnp.sum(jnp.where(selected, slots, 0))
	forced_slot = forced[DRAG_IN, side, 0].astype(jnp.int32)
	target = jnp.where(forced_slot >= 0, forced_slot, sampled)
	forced_valid = (forced_slot < 0) | eligible[jnp.clip(forced_slot, 0, TEAM_SLOTS - 1)]
	active = enabled & (count > 0) & forced_valid
	switch_mask = (SIDES == side) & active
	targets = jnp.where(SIDES == side, target, state.active)
	state = switch_members(state, switch_mask, targets)
	state, rows = run_switch_in(state, switch_mask, side, tables, source)
	return (state, switch_mask) if source is None else (state, switch_mask, rows)


def switch_handler_first(state, key, forced, items, rows):
	"""Incoming speed is cached when runSwitch is queued."""
	speed = active_speed(state, items, rows)
	tie = first_side(key, forced, SWITCH_HANDLER_ORDER)
	return jnp.where(speed[0] == speed[1], tie, jnp.where(speed[0] > speed[1], 0, 1))


def run_switch_in(state, switchers, first, tables, mold_source=None, previous=None, rows=None):
	"""Resolve hazards and Start effects in the order of their holders."""
	if rows is None:
		rows = active_abilities(state, tables)
	before = state.hp[SIDES, state.active] > 0
	ongoing = state.winner == ONGOING
	faster = SIDES == first
	early = jnp.zeros((2,), jnp.bool_)
	weather = effective_weather(state, rows)
	if previous is not None:
		old, old_items, old_abilities, sequential = previous
		early = faster & sequential & jnp.all(switchers)
		old_target = early[::-1]
		old_alive = old.hp[SIDES, old.active] > 0
		first_rows = jnp.where(old_target[:, None], old_abilities, rows)
		first_alive = jnp.where(old_target, old_alive, before)
		first_weather = jnp.where(suppresses_weather(first_rows, first_alive), 0, state.weather)
		weather = jnp.where(early, first_weather, weather)

	hazard_rows = rows if mold_source is None else move_abilities(rows, mold_source)
	state = apply_active_switch_hazards(
		state,
		state.active,
		switchers,
		first,
		tables,
		hazard_rows,
		weather,
		rows,
	)
	alive = state.hp[SIDES, state.active] > 0
	starts = switchers & alive & ongoing & (faster | (state.winner == ONGOING))
	start = rows[:, START]
	if previous is not None:
		old_start = starts & early & old_alive[::-1]
		old_intimidate = (old_start & (start == START_INTIMIDATE))[::-1]
		old_frisk = (old_start & (start == START_FRISK))[::-1] & (old_items.ids != 0)
		delta = -((jnp.arange(7) == 0) & old_intimidate[:, None]).astype(jnp.int8)
		delta = boost_delta(delta, old_abilities, True)
		boosts = old.boosts + delta
		blocked = old_intimidate & (delta[:, 0] == 0) & (old.boosts[:, 0] > -6)
		# Only the outgoing target's item and public ability can survive its switch.
		temporary = state._replace(active=old.active, ability=old.ability)
		herb = old_intimidate & old_items.boost_restore & jnp.any(boosts < 0, axis=1)
		temporary = update_item_state(temporary, herb, old_frisk)
		temporary = reveal_abilities(temporary, blocked)
		state = state._replace(
			item=temporary.item,
			item_consumed=temporary.item_consumed,
			item_revealed=temporary.item_revealed,
			item_public=temporary.item_public,
			ability_public=temporary.ability_public,
		)
	else:
		old_target = early
		old_alive = alive
		old = state
	# The faster grouped handler can see the slower entrant before its hazards.
	grouped = jnp.all(switchers) & ~jnp.any(early)
	foe_alive = jnp.where(grouped & faster, before[::-1], alive[::-1])
	foe_alive = jnp.where(early, old_alive[::-1], foe_alive)
	intimidate = starts & foe_alive & (start == START_INTIMIDATE)
	download = starts & foe_alive & (start == START_DOWNLOAD)
	frisk = starts & foe_alive & (start == START_FRISK)
	frisk_target_has_item = (
		jnp.where(old_target, old.item[SIDES, old.active], state.item[SIDES, state.active])[::-1]
		!= 0
	)
	frisk &= frisk_target_has_item
	state = reveal_active_items(state, (frisk & ~early)[::-1])

	stats = jnp.where(
		old_target[:, None], old.stats[SIDES, old.active], state.stats[SIDES, state.active]
	)
	target_boosts = jnp.where(old_target[:, None], old.boosts, state.boosts)
	defense = scale_stat_stage(
		stats[:, DEF].astype(jnp.int32), target_boosts[:, DEF - 1].astype(jnp.int32)
	)
	spdef = scale_stat_stage(
		stats[:, SPD].astype(jnp.int32), target_boosts[:, SPD - 1].astype(jnp.int32)
	)
	download_stat = jnp.where(defense[::-1] >= spdef[::-1], 2, 0)
	stages = jnp.arange(7)
	up = download[:, None] & (stages == download_stat[:, None])
	intimidate_active = (intimidate & ~early)[::-1]
	down = intimidate_active[:, None] & (stages == 0)
	delta = boost_delta(up.astype(jnp.int8) - down.astype(jnp.int8), rows, True)
	boosts = jnp.clip(state.boosts + delta, -6, 6)
	blocked_intimidate = (
		intimidate_active & (delta[:, 0] == 0) & (state.boosts[:, 0] > -6) & ~up[:, 0]
	)
	state = state._replace(boosts=boosts)
	# Anticipation uses the move dex's base type and effectiveness, before abilities.
	foe_moves = jnp.where(
		old_target[:, None], old.moves[SIDES, old.active], state.moves[SIDES, state.active]
	)[::-1]
	move_types = tables.move_type[foe_moves]
	types = state.types[SIDES, state.active]
	strong = tables.anticipation_type[move_types, types[:, None, 0] + 1, types[:, None, 1] + 1]
	strong &= tables.move_category[foe_moves] != 2
	anticipation = starts & foe_alive & (start == START_ANTICIPATION) & jnp.any(strong, axis=1)

	weather = rows[:, WEATHER]
	last = 1 - first
	state, first_changed = set_weather(state, weather[first], starts[first], 0, ability=True)
	state, last_changed = set_weather(state, weather[last], starts[last], 0, ability=True)
	revealed = (
		(starts & (start == START_ANNOUNCE))
		| intimidate
		| blocked_intimidate
		| download
		| frisk
		| anticipation
	)
	revealed |= jnp.where(SIDES == first, first_changed, last_changed)
	return reveal_abilities(state, revealed), rows
