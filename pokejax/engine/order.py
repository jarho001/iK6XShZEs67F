"""Two-action queue order (sim/battle-queue.ts and Battle.comparePriority)."""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .randomness import ACTION_ORDER, SWITCH_IN_ORDER, first_side
from .state import BOOST_SPE, PARALYSIS, SPE


def action_order(state, actions, moves, replacement, key, forced, tables):
	"""Switches precede moves; then priority, boosted speed, random ties.

	Both voluntary and replacement switches use the outgoing Pokemon's speed
	(Battle.getActionSpeed). Gen 5 freezes this order for the whole queue.
	"""
	switch = actions >= 4
	priority = jnp.where(switch, 0, tables.move_priority[moves]).astype(jnp.int32)
	speed = scale_stat_stage(
		state.stats[SIDES, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = jnp.where(state.status[SIDES, state.active] == PARALYSIS, speed // 4, speed)
	# Pokemon.getStat caps speed; getActionSpeed then truncates to 13 bits.
	speed = jnp.minimum(speed, 10000) & 8191
	context = jnp.where(replacement, SWITCH_IN_ORDER, ACTION_ORDER)
	tie = first_side(key, forced, context)
	same_kind = switch[0] == switch[1]
	same_priority = priority[0] == priority[1]
	first = jnp.where(speed[0] == speed[1], tie, jnp.where(speed[0] > speed[1], 0, 1))
	first = jnp.where(same_priority, first, jnp.where(priority[0] > priority[1], 0, 1))
	return jnp.where(same_kind, first, jnp.where(switch[0], 0, 1))
