"""End-of-turn handlers, in Showdown residual order.

Rows are (order, suborder, handler). Add effects here as their groups arrive.
"""

from itertools import groupby

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .randomness import RESIDUAL_ORDER, first_side
from .state import BOOST_SPE, BURN, ONGOING, PARALYSIS, POISON, SPE, TOXIC


def _poison(state, enabled):
	slot = state.active
	status = state.status[SIDES, slot]
	damage = jnp.where(
		enabled & (status == POISON), jnp.maximum(state.stats[SIDES, slot, 0] // 8, 1), 0
	)
	return state, damage


def _toxic(state, enabled):
	slot = state.active
	status = state.status[SIDES, slot]
	stage = state.status_counter[SIDES, slot]
	toxic = enabled & (status == TOXIC) & (state.hp[SIDES, slot] > 0)
	stage = jnp.where(toxic, jnp.minimum(stage + 1, 15), stage)
	damage = jnp.where(toxic, jnp.maximum(state.stats[SIDES, slot, 0] // 16, 1) * stage, 0)
	counter = state.status_counter.at[SIDES, slot].set(stage)
	return state._replace(status_counter=counter), damage


def _burn(state, enabled):
	slot = state.active
	status = state.status[SIDES, slot]
	damage = jnp.where(
		enabled & (status == BURN), jnp.maximum(state.stats[SIDES, slot, 0] // 8, 1), 0
	)
	return state, damage


RESIDUAL_HANDLERS = ((9, 0, _poison), (9, 0, _toxic), (10, 0, _burn))


def residual(state, enabled, key, forced):
	"""Apply residual rows, resolving faints after each individual handler."""
	speed = scale_stat_stage(
		state.stats[SIDES, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = jnp.where(state.status[SIDES, state.active] == PARALYSIS, speed // 4, speed)
	first = jnp.where(
		speed[0] == speed[1],
		first_side(key, forced, RESIDUAL_ORDER),
		jnp.where(speed[0] > speed[1], 0, 1),
	)
	for _, group in groupby(
		sorted(RESIDUAL_HANDLERS, key=lambda row: row[:2]), key=lambda row: row[:2]
	):
		handlers = tuple(row[2] for row in group)
		for side in (first, 1 - first):
			for handler in handlers:
				active = enabled & (state.winner == ONGOING)
				state, damage = handler(state, active & (SIDES == side))
				hp = state.hp.astype(jnp.int32)
				hp = hp.at[SIDES, state.active].set(
					jnp.maximum(hp[SIDES, state.active] - damage, 0)
				)
				alive = jnp.any(hp > 0, axis=1)
				winner = jnp.where(jnp.all(alive), ONGOING, jnp.where(alive[0], 0, 1))
				state = state._replace(
					hp=hp.astype(state.hp.dtype),
					winner=jnp.where(active, winner, state.winner).astype(state.winner.dtype),
				)
	return state._replace(
		boosts=jnp.where((state.hp[SIDES, state.active] > 0)[:, None], state.boosts, 0),
	)
