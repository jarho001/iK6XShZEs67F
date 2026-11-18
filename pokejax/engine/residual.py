"""Gen 5 end-of-turn handlers in Showdown's residual order."""

from typing import NamedTuple

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .field import weather_residual
from .items import (
	apply_active_stat_modifiers,
	gather_active_items,
	residual_item_hook,
	update_active_items,
)
from .randomness import ORDER, RESIDUAL_ORDER, integer
from .state import (
	BOOST_SPE,
	BURN,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	NUM_VOLATILES,
	ONGOING,
	PARALYSIS,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	POISON,
	ROOST,
	SLEEP,
	SPE,
	TEAM_SLOTS,
	TOXIC,
)
from .volatiles import confuse_sides


class ResidualRow(NamedTuple):
	order: int
	suborder: int
	name: str
	scope: str
	handler: object


def _residual_tie_first(key, forced, context):
	"""Read current row addresses and retain older schema-v2 context-2 records."""
	sampled = integer(key, forced, ORDER, 0, context, 2)
	current = forced[ORDER, 0, context].astype(jnp.int32)
	legacy = forced[ORDER, 0, RESIDUAL_ORDER].astype(jnp.int32)
	return jnp.where(current >= 0, current, jnp.where(legacy >= 0, legacy, sampled))


def _poison_toxic(state, enabled, key, forced, tables):
	del key, forced, tables
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
	active_slots = jnp.arange(state.status_counter.shape[1])[None, :] == slot[:, None]
	counter = jnp.where(
		active_slots & toxic[:, None],
		stage[:, None].astype(state.status_counter.dtype),
		state.status_counter,
	)
	return state._replace(status_counter=counter), damage


def _burn(state, enabled, key, forced, tables):
	del key, forced, tables
	slot = state.active
	status = state.status[SIDES, slot]
	maximum = state.stats[SIDES, slot, 0].astype(jnp.int32)
	damage = jnp.where(enabled & (status == BURN), jnp.maximum(maximum // 8, 1), 0)
	return state, damage


def _partial_trap(state, enabled, key, forced, tables):
	"""Gen 5 Bind-family residual, source validity and hidden duration."""
	del key, forced, tables
	sides = SIDES
	foes = 1 - sides
	active = state.active
	timer = state.volatiles[:, PARTIALLY_TRAPPED].astype(jnp.int32)
	source_slot = state.volatiles[:, PARTIAL_TRAP_SOURCE].astype(jnp.int32) - 1
	source_active = source_slot == state.active[foes]
	source_alive = state.hp[foes, jnp.clip(source_slot, 0, 5)] > 0
	trapping = timer > 0
	active_target = state.hp[sides, active] > 0
	source_started = state.active_turns[foes, jnp.clip(source_slot, 0, 5)] > 0
	valid = enabled & (timer > 1) & source_active & source_alive & source_started & active_target
	maximum = state.stats[sides, active, 0].astype(jnp.int32)
	damage = jnp.where(valid, jnp.maximum(maximum // 16, 1), 0)
	expires = enabled & trapping & (timer <= 1)
	clears = (
		(enabled & trapping & ~source_active)
		| (enabled & trapping & ~source_alive)
		| (enabled & trapping & ~source_started)
		| expires
	)
	volatiles = state.volatiles
	volatiles = volatiles.at[:, PARTIALLY_TRAPPED].set(
		jnp.where(valid, jnp.maximum(timer - 1, 0), jnp.where(clears, 0, timer)).astype(
			volatiles.dtype
		)
	)
	volatiles = volatiles.at[:, PARTIAL_TRAP_SOURCE].set(
		jnp.where(clears, 0, volatiles[:, PARTIAL_TRAP_SOURCE])
	)
	return state._replace(volatiles=volatiles), damage


def _locked_move(state, enabled, key, forced, tables):
	"""Advance Outrage, Thrash and Petal Dance in their residual row."""
	del tables
	sides = SIDES
	active = state.active
	remaining = state.volatiles[:, LOCKED_REMAINING].astype(jnp.int32)
	true_duration = state.volatiles[:, LOCKED_TRUE_DURATION].astype(jnp.int32)
	locked = (state.volatiles[:, LOCKED_MOVE] > 0) & (remaining > 0) & enabled
	sleeping = state.status[sides, active] == SLEEP
	residual_runs = locked & (remaining > 1)
	true_next = jnp.maximum(true_duration - residual_runs.astype(jnp.int32), 0)
	sleep_ends = residual_runs & sleeping
	lock_ends = (locked & (remaining == 1)) | sleep_ends
	volatiles = state.volatiles
	volatiles = volatiles.at[:, LOCKED_REMAINING].set(
		jnp.where(
			residual_runs & ~sleep_ends, remaining - 1, volatiles[:, LOCKED_REMAINING]
		).astype(volatiles.dtype)
	)
	volatiles = volatiles.at[:, LOCKED_TRUE_DURATION].set(
		jnp.where(residual_runs, true_next, volatiles[:, LOCKED_TRUE_DURATION]).astype(
			volatiles.dtype
		)
	)
	volatiles = volatiles.at[:, LOCKED_MOVE].set(jnp.where(lock_ends, 0, volatiles[:, LOCKED_MOVE]))
	volatiles = volatiles.at[:, LOCKED_REMAINING].set(
		jnp.where(lock_ends, 0, volatiles[:, LOCKED_REMAINING])
	)
	volatiles = volatiles.at[:, LOCKED_TRUE_DURATION].set(
		jnp.where(lock_ends, 0, volatiles[:, LOCKED_TRUE_DURATION])
	)
	state = state._replace(volatiles=volatiles)
	confusion_due = jnp.where(sleep_ends, true_duration <= 1, true_next <= 1)
	state = confuse_sides(state, lock_ends & confusion_due, key, forced)
	return state


def _end_roost(state, enabled, key, forced, tables):
	del key, forced, tables
	active = state.active
	roost = enabled & (state.volatiles[:, ROOST] > 0)
	active_slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	types = jnp.where((active_slots & roost[:, None])[:, :, None], state.roost_types, state.types)
	columns = jnp.arange(NUM_VOLATILES)
	volatiles = jnp.where(
		(columns[None, :] == ROOST) & roost[:, None],
		0,
		state.volatiles,
	)
	return state._replace(types=types, volatiles=volatiles)


def _item_residual(state, enabled, key, forced, tables, item_rows, order):
	return residual_item_hook(state, enabled, key, forced, tables, item_rows, order)


# Rows are (Showdown order, suborder, name, scope, handler), sorted as the
# pinned Gen 5 event dispatcher sorts them. Empty handlers are reserved for
# pool mechanics added by later tasks; their rows establish the complete order.
# Status order and suborder come from data/conditions.ts. Field, move, item and
# ability rows come from data/conditions.ts, data/moves.ts, data/items.ts and
# data/abilities.ts plus Gen 5's inherited condition overrides.
RESIDUAL_TABLE = (
	ResidualRow(0, 2, "Locked moves", "global", _locked_move),
	ResidualRow(1, 5, "weather", "field", weather_residual),
	ResidualRow(3, 3, "Future Sight", "slot", None),
	ResidualRow(4, 3, "Wish", "slot", None),
	ResidualRow(5, 1, "Pledge field effects", "field", None),
	ResidualRow(5, 2, "Grassy Terrain", "field", None),
	ResidualRow(5, 3, "Hydration, Shed Skin and other abilities", "pokemon", None),
	ResidualRow(5, 4, "Leftovers and Black Sludge", "pokemon", _item_residual),
	ResidualRow(6, 2, "Aqua Ring", "pokemon", None),
	ResidualRow(7, 2, "Ingrain", "pokemon", None),
	ResidualRow(8, 2, "Leech Seed", "pokemon", None),
	ResidualRow(9, 2, "Poison and Toxic", "pokemon", _poison_toxic),
	ResidualRow(10, 2, "Burn", "pokemon", _burn),
	ResidualRow(11, 2, "Nightmare", "pokemon", None),
	ResidualRow(12, 2, "Curse", "pokemon", None),
	ResidualRow(13, 2, "Partial trapping", "pokemon", _partial_trap),
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
	ResidualRow(28, 3, "Toxic Orb and Flame Orb", "pokemon", _item_residual),
	ResidualRow(29, 0, "Ability and item end-of-turn hooks", "pokemon", _item_residual),
)


def _resolve_handler_faint(state, damage, enabled):
	active_slots = jnp.arange(state.hp.shape[1])[None, :] == state.active[:, None]
	old = state.hp[SIDES, state.active].astype(jnp.int32)
	hp_value = jnp.maximum(old - damage.astype(jnp.int32), 0)
	write = active_slots & enabled[:, None]
	hp = jnp.where(write, hp_value[:, None].astype(state.hp.dtype), state.hp)
	alive = jnp.any(hp > 0, axis=1)
	update = enabled & (damage > 0) & ~alive & (state.winner == ONGOING)
	update_battle = jnp.any(update)
	loser = jnp.argmax(update.astype(jnp.int32))
	winner = 1 - loser
	return state._replace(
		hp=hp,
		winner=jnp.where(update_battle, winner, state.winner).astype(state.winner.dtype),
	)


def residual(state, enabled, key, forced, tables):
	"""Run the full residual event, then resolve its queued faint messages."""
	residual_enabled = enabled & (state.winner == ONGOING)
	item_rows = gather_active_items(state, tables)
	speed = scale_stat_stage(
		state.stats[SIDES, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = apply_active_stat_modifiers(speed, SPE, item_rows)
	speed = jnp.where(state.status[SIDES, state.active] == PARALYSIS, speed // 4, speed)
	for row in RESIDUAL_TABLE:
		active = residual_enabled & (state.winner == ONGOING)
		if row.handler is None:
			continue
		if row.scope in ("field", "global"):
			state = row.handler(state, active, key, forced, tables)
		elif row.scope == "pokemon":
			context = {0: 2, 5: 2, 9: 6, 10: 7, 13: 8, 28: 2, 29: 2}[row.order]
			tie_first = _residual_tie_first(key, forced, context)
			first = jnp.where(speed[0] == speed[1], tie_first, jnp.where(speed[0] > speed[1], 0, 1))
			for side in (first, 1 - first):
				handler_enabled = residual_enabled & (state.winner == ONGOING) & (SIDES == side)
				if row.handler is _item_residual:
					state, damage = row.handler(
						state, handler_enabled, key, forced, tables, item_rows, row.order
					)
				else:
					state, damage = row.handler(state, handler_enabled, key, forced, tables)
				state = _resolve_handler_faint(state, damage, handler_enabled)
	state = update_active_items(state, residual_enabled, item_rows)
	return state._replace(
		boosts=jnp.where((state.hp[SIDES, state.active] > 0)[:, None], state.boosts, 0),
	)
