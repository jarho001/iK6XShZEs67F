"""Gen 5 end-of-turn handlers in Showdown's residual order."""

from typing import NamedTuple

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .field import weather_residual
from .randomness import RESIDUAL_ORDER, first_side
from .state import (
	BOOST_SPE,
	BURN,
	ONGOING,
	PARALYSIS,
	POISON,
	ROOST,
	SPE,
	TOXIC,
)


class ResidualRow(NamedTuple):
	order: int
	suborder: int
	name: str
	scope: str
	handler: object


def _poison_toxic(state, enabled):
	slot = state.active
	status = state.status[SIDES, slot]
	hp = state.hp[SIDES, slot].astype(jnp.int32)
	maximum = state.stats[SIDES, slot, 0].astype(jnp.int32)
	toxic = enabled & (status == TOXIC) & (hp > 0)
	stage = state.status_counter[SIDES, slot].astype(jnp.int32)
	stage = jnp.where(toxic, jnp.minimum(stage + 1, 15), stage)
	poison = enabled & (status == POISON) & (hp > 0)
	damage = jnp.where(
		poison,
		jnp.maximum(maximum // 8, 1),
		jnp.where(toxic, jnp.maximum(maximum // 16, 1) * stage, 0),
	)
	counter = state.status_counter.at[SIDES, slot].set(stage.astype(state.status_counter.dtype))
	return state._replace(status_counter=counter), damage


def _burn(state, enabled):
	slot = state.active
	status = state.status[SIDES, slot]
	maximum = state.stats[SIDES, slot, 0].astype(jnp.int32)
	damage = jnp.where(enabled & (status == BURN), jnp.maximum(maximum // 8, 1), 0)
	return state, damage


def _end_roost(state, enabled, key, forced, tables):
	del key, forced, tables
	active = state.active
	roost = enabled & (state.volatiles[:, ROOST] > 0)
	types = state.types.at[SIDES, active].set(
		jnp.where(roost[:, None], state.roost_types[SIDES, active], state.types[SIDES, active])
	)
	volatiles = state.volatiles.at[:, ROOST].set(jnp.where(roost, 0, state.volatiles[:, ROOST]))
	return state._replace(types=types, volatiles=volatiles)


# Rows are (Showdown order, suborder, name, scope, handler), sorted as the
# pinned Gen 5 event dispatcher sorts them. Empty handlers are reserved for
# pool mechanics added by later tasks; their rows establish the complete order.
# Status order and suborder come from data/conditions.ts. Field, move, item and
# ability rows come from data/conditions.ts, data/moves.ts, data/items.ts and
# data/abilities.ts plus Gen 5's inherited condition overrides.
RESIDUAL_TABLE = (
	ResidualRow(1, 5, "weather", "field", weather_residual),
	ResidualRow(3, 3, "Future Sight", "slot", None),
	ResidualRow(4, 3, "Wish", "slot", None),
	ResidualRow(5, 1, "Pledge field effects", "field", None),
	ResidualRow(5, 2, "Grassy Terrain", "field", None),
	ResidualRow(5, 3, "Hydration, Shed Skin and other abilities", "pokemon", None),
	ResidualRow(5, 4, "Leftovers and Black Sludge", "pokemon", None),
	ResidualRow(6, 2, "Aqua Ring", "pokemon", None),
	ResidualRow(7, 2, "Ingrain", "pokemon", None),
	ResidualRow(8, 2, "Leech Seed", "pokemon", None),
	ResidualRow(9, 2, "Poison and Toxic", "pokemon", _poison_toxic),
	ResidualRow(10, 2, "Burn", "pokemon", _burn),
	ResidualRow(11, 2, "Nightmare", "pokemon", None),
	ResidualRow(12, 2, "Curse", "pokemon", None),
	ResidualRow(13, 2, "Partial trapping", "pokemon", None),
	ResidualRow(15, 2, "Taunt", "pokemon", None),
	ResidualRow(16, 2, "Encore", "pokemon", None),
	ResidualRow(17, 2, "Disable", "pokemon", None),
	ResidualRow(18, 2, "Magnet Rise", "pokemon", None),
	ResidualRow(19, 2, "Telekinesis", "pokemon", None),
	ResidualRow(20, 2, "Heal Block", "pokemon", None),
	ResidualRow(21, 2, "Embargo", "pokemon", None),
	ResidualRow(22, 2, "Throat Chop", "pokemon", None),
	ResidualRow(23, 2, "Yawn", "pokemon", None),
	ResidualRow(24, 2, "Perish Song", "pokemon", None),
	ResidualRow(25, 2, "Roost", "global", _end_roost),
	ResidualRow(26, 1, "Side condition timers", "side", None),
	ResidualRow(27, 1, "Field condition timers", "field", None),
	ResidualRow(28, 2, "Residual abilities", "pokemon", None),
	ResidualRow(28, 3, "Residual item effects", "pokemon", None),
	ResidualRow(29, 0, "Ability and item end-of-turn hooks", "pokemon", None),
)


def _resolve_handler_faint(state, side, damage, enabled):
	slot = state.active[side]
	old = state.hp[side, slot].astype(jnp.int32)
	hp_value = jnp.maximum(old - damage.astype(jnp.int32), 0)
	hp = state.hp.at[side, slot].set(hp_value.astype(state.hp.dtype))
	alive = jnp.any(hp > 0, axis=1)
	update = enabled & (damage > 0) & ~jnp.all(alive) & (state.winner == ONGOING)
	winner = jnp.where(~alive[0], 1, 0)
	boosts = state.boosts.at[side].set(jnp.where(hp_value > 0, state.boosts[side], 0))
	return state._replace(
		hp=hp,
		boosts=boosts,
		winner=jnp.where(update, winner, state.winner).astype(state.winner.dtype),
	)


def residual(state, enabled, key, forced, tables):
	"""Run each registered handler and resolve its faint queue in Showdown order."""
	speed = scale_stat_stage(
		state.stats[SIDES, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = jnp.where(state.status[SIDES, state.active] == PARALYSIS, speed // 4, speed)
	tie_first = first_side(key, forced, RESIDUAL_ORDER)
	first = jnp.where(speed[0] == speed[1], tie_first, jnp.where(speed[0] > speed[1], 0, 1))

	for row in RESIDUAL_TABLE:
		active = enabled & (state.winner == ONGOING)
		if row.handler is None:
			continue
		if row.scope in ("field", "global"):
			state = row.handler(state, active, key, forced, tables)
		elif row.scope == "pokemon":
			for side in (first, 1 - first):
				handler_enabled = enabled & (state.winner == ONGOING) & (SIDES == side)
				state, damage = row.handler(state, handler_enabled)
				state = _resolve_handler_faint(state, side, damage[side], handler_enabled[side])
	return state._replace(
		boosts=jnp.where((state.hp[SIDES, state.active] > 0)[:, None], state.boosts, 0),
	)
