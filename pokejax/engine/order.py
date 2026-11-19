"""Two-action queue order (sim/battle-queue.ts and Battle.comparePriority)."""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .abilities import (
	ORDER,
	ORDER_PRANKSTER,
	SPEED,
	SPEED_QUICK_FEET,
	SPEED_SLOW_START,
	SPEED_UNBURDEN,
	active_abilities,
	berry_allowed,
	effective_weather,
)
from .actions import SIDES
from .items import (
	BERRY_CUSTAP,
	apply_active_stat_modifiers,
	consume_active_items,
)
from .modifiers import chain
from .randomness import ACTION_ORDER, SWITCH_IN_ORDER, first_side
from .state import (
	BOOST_SPE,
	HEALTHY,
	MOVE,
	MUST_RECHARGE,
	ONGOING,
	PARALYSIS,
	SLOW_START,
	SPE,
	TURN_IDLE,
	UNBURDEN,
)


def action_order(state, actions, moves, replacement, key, forced, rows, tables, abilities=None):
	"""Switches precede moves; then priority, boosted speed, random ties.

	Both voluntary and replacement switches use the outgoing Pokemon's speed
	(Battle.getActionSpeed). Gen 5 freezes this order for the whole queue.
	"""
	switch = actions >= 4
	priority = jnp.where(switch, 0, tables.move_priority[moves]).astype(jnp.int32)
	if abilities is None:
		abilities = active_abilities(state, tables)
	status_move = tables.move_category[moves] == 2
	prankster = (abilities[:, ORDER] == ORDER_PRANKSTER) & status_move & ~switch
	priority += prankster.astype(jnp.int32)
	speed = action_speed(state, rows, abilities)
	hp = state.hp[SIDES, state.active].astype(jnp.int32)
	maximum = state.stats[SIDES, state.active, 0].astype(jnp.int32)
	custap = (state.request == MOVE) & (actions < 4) & (state.turn_stage == TURN_IDLE)
	custap &= (state.winner == ONGOING) & (priority <= 0)
	custap &= state.volatiles[:, MUST_RECHARGE] == 0
	custap &= berry_allowed(state, abilities)
	custap &= (rows.berry_kind == BERRY_CUSTAP) & (4 * hp <= maximum) & (maximum > 0)
	state = consume_active_items(state, custap, tables)
	priority = priority * 10 + custap.astype(jnp.int32)
	context = jnp.where(replacement, SWITCH_IN_ORDER, ACTION_ORDER)
	tie = first_side(key, forced, context)
	same_kind = switch[0] == switch[1]
	same_priority = priority[0] == priority[1]
	first = jnp.where(speed[0] == speed[1], tie, jnp.where(speed[0] > speed[1], 0, 1))
	first = jnp.where(same_priority, first, jnp.where(priority[0] > priority[1], 0, 1))
	first = jnp.where(same_kind, first, jnp.where(switch[0], 0, 1))
	return state, first


def action_speed(state, items, abilities):
	"""Pokemon.getActionSpeed: Trick Room inverts the capped speed as 10000 - speed."""
	raw = _capped_speed(state, items, abilities)
	trick_room = state.trick_room > 0
	return jnp.where(trick_room, (10000 - raw) & 8191, raw & 8191)


def active_speed(state, items, abilities):
	"""GetActionSpeed: stages, chained speed hooks, paralysis, then width caps."""
	return _capped_speed(state, items, abilities) & 8191


def _capped_speed(state, items, abilities):
	"""Speed after stages, hooks and paralysis, capped at 10000 (Pokemon.getStat)."""
	status = state.status[SIDES, state.active]
	speed = scale_stat_stage(
		state.stats[SIDES, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	weather = effective_weather(state, abilities)
	hook = abilities[:, SPEED]
	quick_feet = hook == SPEED_QUICK_FEET
	modifier = jnp.where(
		(hook != 0) & (hook == weather),
		8192,
		jnp.where(quick_feet & (status != HEALTHY), 6144, 4096),
	)
	unburden = (hook == SPEED_UNBURDEN) & (state.volatiles[SIDES, UNBURDEN] > 0)
	unburden &= state.item[SIDES, state.active] == 0
	slow_start = (hook == SPEED_SLOW_START) & (state.volatiles[SIDES, SLOW_START] > 0)
	modifier = chain(modifier, jnp.where(unburden, 8192, jnp.where(slow_start, 2048, 4096)))
	speed = apply_active_stat_modifiers(speed, SPE, items, modifier)
	speed = jnp.where((status == PARALYSIS) & ~quick_feet, speed // 4, speed)
	return jnp.minimum(speed, 10000)
