"""Gen 5 end-of-turn handlers in Showdown's residual order."""

from typing import NamedTuple

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .field import weather_residual
from .items import (
	apply_active_stat_modifiers,
	apply_residual_status_items,
	apply_residual_white_herb,
	gather_active_items,
	residual_hp_item,
	reveal_active_items,
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


def _apply_damage(state, damage, enabled, first, bench_alive, healing=0):
	"""Commit a Pokemon residual row, stopping after a battle-ending KO."""
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	before = state.hp[SIDES, state.active].astype(jnp.int32)
	after = jnp.maximum(before - damage, 0) + healing
	loses = (damage > 0) & (after == 0) & ~bench_alive
	first_mask = SIDES == first
	enabled = enabled & (first_mask | ~jnp.any(loses & first_mask & enabled))
	loses &= enabled
	winner = jnp.where(loses[0], 1, 0).astype(state.winner.dtype)
	return state._replace(
		hp=jnp.where(slots & enabled[:, None], after[:, None].astype(state.hp.dtype), state.hp),
		winner=jnp.where(jnp.any(loses), winner, state.winner),
	), enabled


def _status_damage(state, enabled, first, bench_alive, key, forced, speed_tie):
	"""Resolve poison/toxic (order 9) before burn (order 10)."""
	slot = state.active
	status = state.status[SIDES, slot]
	hp = state.hp[SIDES, slot]
	maximum = state.stats[SIDES, slot, 0].astype(jnp.int32)
	poisoned = (status == POISON) | (status == TOXIC)
	toxic = enabled & (status == TOXIC) & (hp > 0)
	poison = (status == POISON) & (hp > 0)
	stage = jnp.minimum(state.status_counter[SIDES, slot].astype(jnp.int32) + 1, 15)
	eighth = jnp.maximum(maximum // 8, 1)
	damage = jnp.where(
		toxic,
		jnp.maximum(maximum // 16, 1) * stage,
		jnp.where(enabled & (poison | (status == BURN)), eighth, 0),
	)
	tie = jnp.where(
		poisoned[0], _residual_tie_first(key, forced, 6), _residual_tie_first(key, forced, 7)
	)
	first = jnp.where(speed_tie, tie, first)
	first = jnp.where(poisoned[0] == poisoned[1], first, jnp.where(poisoned[0], 0, 1))
	state, ran = _apply_damage(state, damage, enabled, first, bench_alive)
	active_slots = jnp.arange(TEAM_SLOTS)[None, :] == slot[:, None]
	counter = jnp.where(
		active_slots & (toxic & ran)[:, None],
		stage[:, None].astype(state.status_counter.dtype),
		state.status_counter,
	)
	return state._replace(status_counter=counter)


def _partial_trap(state, enabled, first, bench_alive, tables, item_rows):
	"""Gen 5 Bind-family residual, source validity and hidden duration."""
	del tables, item_rows
	foes = 1 - SIDES
	active = state.active
	timer = state.volatiles[:, PARTIALLY_TRAPPED].astype(jnp.int32)
	source_slot = state.volatiles[:, PARTIAL_TRAP_SOURCE].astype(jnp.int32) - 1
	source_active = source_slot == active[foes]
	source_started = state.active_turns[foes, active[foes]] > 0
	hp = state.hp[SIDES, active]
	source_alive = hp[foes] > 0
	valid = enabled & (timer > 1) & source_active & source_alive & source_started & (hp > 0)
	maximum = state.stats[SIDES, active, 0].astype(jnp.int32)
	damage = jnp.where(valid, jnp.maximum(maximum // 16, 1), 0)

	# The first callback can faint the source of the second callback even if
	# its team still has a usable replacement.
	source_just_fainted = (SIDES != first) & (damage[foes] > 0) & (damage[foes] >= hp[foes])
	source_alive &= ~source_just_fainted
	valid &= ~source_just_fainted
	state, ran = _apply_damage(state, jnp.where(valid, damage, 0), enabled, first, bench_alive)
	clears = ran & (timer > 0) & (~source_active | ~source_alive | ~source_started | (timer <= 1))
	timer = jnp.where(valid & ran, timer - 1, jnp.where(clears, 0, timer))
	columns = jnp.arange(NUM_VOLATILES)
	volatiles = jnp.where(
		columns == PARTIALLY_TRAPPED, timer[:, None].astype(state.volatiles.dtype), state.volatiles
	)
	volatiles = jnp.where(clears[:, None] & (columns == PARTIAL_TRAP_SOURCE), 0, volatiles)
	return state._replace(volatiles=volatiles)


def _item_hp(state, enabled, first, bench_alive, tables, item_rows):
	healing, damage = residual_hp_item(state, enabled, tables, item_rows)
	state, ran = _apply_damage(state, damage, enabled, first, bench_alive, healing)
	return reveal_active_items(state, ran & ((healing > 0) | (damage > 0)))


def _white_herb(state, enabled, key, forced, tables, item_rows):
	del key, forced, tables
	return apply_residual_white_herb(state, enabled, item_rows)


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
	ResidualRow(5, 4, "Leftovers and Black Sludge", "pokemon", _item_hp),
	ResidualRow(6, 2, "Aqua Ring", "pokemon", None),
	ResidualRow(7, 2, "Ingrain", "pokemon", None),
	ResidualRow(8, 2, "Leech Seed", "pokemon", None),
	ResidualRow(9, 2, "Poison, Toxic and Burn (order 10)", "status", _status_damage),
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
	ResidualRow(28, 3, "Toxic Orb and Flame Orb", "item", apply_residual_status_items),
	ResidualRow(29, 0, "Ability and item end-of-turn hooks", "item", _white_herb),
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
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	bench_alive = jnp.any((state.hp > 0) & ~slots, axis=1)
	speed_tie = speed[0] == speed[1]
	faster = jnp.where(speed[0] > speed[1], 0, 1)
	for row in RESIDUAL_TABLE:
		active = residual_enabled & (state.winner == ONGOING)
		if row.handler is None:
			continue
		if row.scope in ("field", "global"):
			state = row.handler(state, active, key, forced, tables)
		elif row.scope == "item":
			state = row.handler(state, active, key, forced, tables, item_rows)
		elif row.scope == "status":
			state = row.handler(state, active, faster, bench_alive, key, forced, speed_tie)
		elif row.scope == "pokemon":
			context = {5: 2, 13: 8}[row.order]
			tie_first = _residual_tie_first(key, forced, context)
			first = jnp.where(speed_tie, tie_first, faster)
			state = row.handler(state, active, first, bench_alive, tables, item_rows)
	state = update_active_items(state, residual_enabled, item_rows)
	return state._replace(
		boosts=jnp.where((state.hp[SIDES, state.active] > 0)[:, None], state.boosts, 0),
	)
