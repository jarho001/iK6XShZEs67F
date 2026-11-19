"""Gen 5 end-of-turn handlers in Showdown's residual order."""

from typing import NamedTuple

import jax.numpy as jnp

from .abilities import (
	CONTACT,
	CONTACT_LIQUID_OOZE,
	MAGIC_GUARD,
	OWN_TEMPO,
	RESIDUAL,
	RESIDUAL_BAD_DREAMS,
	RESIDUAL_HARVEST,
	RESIDUAL_HYDRATION,
	RESIDUAL_POISON_HEAL,
	RESIDUAL_SHED_SKIN,
	RESIDUAL_SLOW_START,
	RESIDUAL_SPEED_BOOST,
	active_abilities,
	apply_boost_delta,
	effective_weather,
	has_hook,
	reveal_abilities,
)
from .actions import SIDES
from .delayed import trick_room_residual, wish_residual
from .field import weather_residual
from .items import (
	HARVEST_BERRY,
	apply_residual_status_items,
	apply_residual_white_herb,
	gather_active_items,
	residual_hp_item,
	reveal_active_items,
	update_active_items,
)
from .order import active_speed
from .randomness import (
	ABILITY_RESIDUAL_ORDER,
	HARVEST,
	ORDER,
	RESIDUAL_CHANCE,
	RESIDUAL_ORDER,
	chance,
	integer,
)
from .restrictions import encore_residual, taunt_residual
from .state import (
	BOOST_SPE,
	BURN,
	CURSE,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	HP,
	LEECH_SEED,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	NUM_VOLATILES,
	ONGOING,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	POISON,
	RAIN,
	ROOST,
	SLEEP,
	SLOW_START,
	SUN,
	TEAM_SLOTS,
	TOXIC,
	YAWN,
)
from .status import inflict_status
from .volatiles import confuse_sides


class ResidualRow(NamedTuple):
	order: int
	suborder: int
	name: str
	scope: str
	handler: object


def _residual_tie_first(key, forced, context, kind=ORDER):
	"""Read current row addresses and retain older schema-v2 context-2 records."""
	sampled = integer(key, forced, kind, 0, context, 2)
	current = forced[kind, 0, context].astype(jnp.int32)
	legacy = forced[ORDER, 0, RESIDUAL_ORDER].astype(jnp.int32)
	if kind != ORDER:
		return jnp.where(current >= 0, current, sampled)
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


def _status_damage(state, enabled, first, bench_alive, key, forced, speed_tie, rows):
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
	poison_heal = (rows[:, RESIDUAL] == RESIDUAL_POISON_HEAL) & poisoned & (hp > 0)
	healing = jnp.where(
		poison_heal & (hp < maximum), jnp.minimum(jnp.maximum(maximum // 8, 1), maximum - hp), 0
	)
	damage = jnp.where(poison_heal | has_hook(rows, MAGIC_GUARD), 0, damage)
	tie = jnp.where(
		poisoned[0], _residual_tie_first(key, forced, 6), _residual_tie_first(key, forced, 7)
	)
	first = jnp.where(speed_tie, tie, first)
	first = jnp.where(poisoned[0] == poisoned[1], first, jnp.where(poisoned[0], 0, 1))
	state, ran = _apply_damage(state, damage, enabled, first, bench_alive, healing)
	active_slots = jnp.arange(TEAM_SLOTS)[None, :] == slot[:, None]
	counter = jnp.where(
		active_slots & (toxic & ran)[:, None],
		stage[:, None].astype(state.status_counter.dtype),
		state.status_counter,
	)
	state = state._replace(status_counter=counter)
	return reveal_abilities(state, enabled & poison_heal & (healing > 0))


def _partial_trap(state, enabled, first, bench_alive, tables, item_rows, rows):
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
	damage = jnp.where(valid & ~has_hook(rows, MAGIC_GUARD), jnp.maximum(maximum // 16, 1), 0)

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


def _item_hp(state, enabled, first, bench_alive, tables, item_rows, rows):
	healing, damage = residual_hp_item(state, enabled, tables, item_rows, rows)
	state, ran = _apply_damage(state, damage, enabled, first, bench_alive, healing)
	return reveal_active_items(state, ran & ((healing > 0) | (damage > 0)))


def _ability_order5(state, enabled, first, bench_alive, tables, item_rows, rows, key, forced):
	del first, bench_alive, item_rows
	active = state.active
	hp = state.hp[SIDES, active].astype(jnp.int32)
	status = state.status[SIDES, active]
	hook = rows[:, RESIDUAL]
	live = enabled & (hp > 0)
	weather = effective_weather(state, rows)
	hydration = live & (hook == RESIDUAL_HYDRATION) & (status != 0) & (weather == RAIN)
	shed_skin = live & (hook == RESIDUAL_SHED_SKIN) & (status != 0)
	shed_roll = chance(key, forced, RESIDUAL_CHANCE, SIDES, 0, 33, 100)
	cure = hydration | (shed_skin & shed_roll)
	slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	status_write = slots & cure[:, None]
	state = state._replace(
		status=jnp.where(status_write, 0, state.status),
		status_counter=jnp.where(status_write, 0, state.status_counter),
		sleep_from_foe=jnp.where(status_write, 0, state.sleep_from_foe),
	)
	return reveal_abilities(state, cure)


def _ability_order28(state, enabled, first, bench_alive, tables, item_rows, rows, key, forced):
	del item_rows
	active = state.active
	hp = state.hp[SIDES, active].astype(jnp.int32)
	hook = rows[:, RESIDUAL]
	live = enabled & (hp > 0)
	speed_boost = live & (hook == RESIDUAL_SPEED_BOOST)
	speed_boost &= state.active_turns[SIDES, active] > 0
	delta = (speed_boost[:, None] & (jnp.arange(state.boosts.shape[-1]) == BOOST_SPE)).astype(
		state.boosts.dtype
	)
	boosts = apply_boost_delta(state.boosts, delta, rows, False)
	boosted = speed_boost & (state.boosts[:, BOOST_SPE] < 6)
	slow_start = live & (hook == RESIDUAL_SLOW_START)
	slow_start &= state.active_turns[SIDES, active] > 0
	slow_counter = jnp.where(
		slow_start & (state.volatiles[SIDES, SLOW_START] > 0),
		state.volatiles[SIDES, SLOW_START] - 1,
		state.volatiles[SIDES, SLOW_START],
	)
	columns = jnp.arange(NUM_VOLATILES)
	volatiles = jnp.where((columns[None, :] == SLOW_START), slow_counter[:, None], state.volatiles)
	state = state._replace(boosts=boosts, volatiles=volatiles)
	foe = 1 - SIDES
	foe_hp = state.hp[foe, active[foe]].astype(jnp.int32)
	foe_max = state.stats[foe, active[foe], 0].astype(jnp.int32)
	foe_sleeping = state.status[foe, active[foe]] == SLEEP
	dreams = live & (hook == RESIDUAL_BAD_DREAMS) & foe_sleeping & (foe_hp > 0)
	dream_damage = jnp.where(dreams, jnp.maximum(foe_max // 8, 1), 0)
	dream_damage = jnp.where(has_hook(rows[foe], MAGIC_GUARD), 0, dream_damage)
	damage = dream_damage[::-1]
	state, still_enabled = _apply_damage(state, damage, enabled, first, bench_alive)
	state = _harvest(state, still_enabled, key, forced, tables, rows)
	return reveal_abilities(state, boosted | dreams)


def _harvest(state, enabled, key, forced, tables, rows):
	active = state.active
	hp = state.hp[SIDES, active]
	last_item = state.item_public[SIDES, active].astype(jnp.int32)
	item_rows = tables.item_hooks.rows[last_item]
	weather = effective_weather(state, rows)
	sun = weather == SUN
	roll = chance(key, forced, HARVEST, SIDES, 0, 1, 2)
	try_restore = sun | roll
	restore = enabled & (hp > 0) & try_restore & (state.item[SIDES, active] == 0)
	restore &= rows[:, RESIDUAL] == RESIDUAL_HARVEST
	restore &= state.item_consumed[SIDES, active] & ~state.item_knocked_off[SIDES, active]
	restore &= item_rows[:, HARVEST_BERRY] != 0
	slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	write = slots & restore[:, None]
	state = state._replace(
		item=jnp.where(write, last_item[:, None].astype(state.item.dtype), state.item),
		item_consumed=jnp.where(write, False, state.item_consumed),
		item_revealed=state.item_revealed | write,
	)
	return reveal_abilities(state, restore)


def _end_turn_items(state, enabled, key, forced, tables, item_rows, rows):
	del key, forced, rows
	return apply_residual_white_herb(state, enabled, item_rows, tables)


def _locked_move(state, enabled, key, forced, tables, rows):
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
	state = confuse_sides(
		state, lock_ends & confusion_due & ~has_hook(rows, OWN_TEMPO), key, forced
	)
	return state


def _end_roost(state, enabled, key, forced, tables, rows):
	del key, forced, tables, rows
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


def _disable_residual(state, enabled, first, bench_alive, tables, item_rows, rows):
	del first, bench_alive, tables, item_rows, rows
	remaining = state.volatiles[:, DISABLE_REMAINING].astype(jnp.int32)
	tick = enabled & (remaining > 0)
	next_remaining = jnp.where(tick, jnp.maximum(remaining - 1, 0), remaining)
	clear = tick & (next_remaining == 0)
	columns = jnp.arange(NUM_VOLATILES)
	volatiles = jnp.where(
		columns[None, :] == DISABLE_REMAINING,
		next_remaining[:, None].astype(state.volatiles.dtype),
		state.volatiles,
	)
	volatiles = jnp.where(clear[:, None] & (columns[None, :] == DISABLE_MOVE), 0, volatiles)
	return state._replace(volatiles=volatiles)


def _curse_residual(state, enabled, first, bench_alive, tables, item_rows, rows):
	del tables, item_rows
	active = state.active
	hp = state.hp[SIDES, active].astype(jnp.int32)
	maximum = state.stats[SIDES, active, 0].astype(jnp.int32)
	curse = enabled & (hp > 0) & (state.volatiles[:, CURSE] > 0)
	damage = jnp.where(curse, jnp.maximum(maximum // 4, 1), 0)
	damage = jnp.where(has_hook(rows, MAGIC_GUARD), 0, damage)
	state, _ = _apply_damage(state, damage, enabled, first, bench_alive)
	return state


def _leech_pass(state, side, enabled, bench_alive, tables, rows):
	"""One Leech Seed callback: the seeded side loses max HP / 8 to the seeder's slot.

	The heal goes to the Pokemon in the seeder's slot. Liquid Ooze on the seeded
	Pokemon turns the heal into damage for the seeder.
	"""
	seeder = 1 - side
	seeded_slot = state.active[side]
	seeder_slot = state.active[seeder]
	seeded_hp = state.hp[side, seeded_slot].astype(jnp.int32)
	seeder_hp = state.hp[seeder, seeder_slot].astype(jnp.int32)
	seeder_max = state.stats[seeder, seeder_slot, HP].astype(jnp.int32)
	seeded_max = state.stats[side, seeded_slot, HP].astype(jnp.int32)
	live = enabled & (state.volatiles[side, LEECH_SEED] > 0) & (seeded_hp > 0) & (seeder_hp > 0)
	magic = has_hook(rows[side], MAGIC_GUARD)
	amount = jnp.where(live & ~magic, jnp.maximum(seeded_max // 8, 1), 0)
	dealt = jnp.minimum(amount, seeded_hp)
	ooze = rows[side, CONTACT] == CONTACT_LIQUID_OOZE
	seeder_side = SIDES == seeder
	damage = jnp.where(SIDES == side, dealt, 0) + jnp.where(seeder_side & ooze, dealt, 0)
	room = jnp.maximum(seeder_max - seeder_hp, 0)
	healing = jnp.where(seeder_side & ~ooze, jnp.minimum(dealt, room), 0)
	# The seeded side's faint does not stop the seeder's heal in the same callback, so
	# the seeder is the first side for the gate.
	state, _ = _apply_damage(state, damage, live, seeder, bench_alive, healing)
	state = reveal_abilities(state, (SIDES == side) & ooze & live)
	return state, live


def _leech_seed(state, enabled, first, bench_alive, key, forced, tables, rows):
	"""Leech Seed (order 8), faster holder first."""
	del key, forced
	state, _ = _leech_pass(state, first, enabled, bench_alive, tables, rows)
	state, _ = _leech_pass(state, 1 - first, enabled, bench_alive, tables, rows)
	return state


def _yawn(state, enabled, first, bench_alive, key, forced, tables, rows):
	"""Yawn (order 23): the count ends, then the target falls asleep from the foe."""
	del bench_alive
	remaining = state.volatiles[:, YAWN].astype(jnp.int32)
	tick = enabled & (remaining > 0)
	left = jnp.where(tick, remaining - 1, remaining)
	ends = tick & (left == 0)
	columns = jnp.arange(NUM_VOLATILES)
	volatiles = jnp.where(
		columns[None, :] == YAWN, left[:, None].astype(state.volatiles.dtype), state.volatiles
	)
	state = state._replace(volatiles=volatiles)
	for side in (first, 1 - first):
		state = inflict_status(state, side, SLEEP, ends[side], key, forced, tables, rows)
	return state


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
	ResidualRow(4, 3, "Wish", "global", wish_residual),
	ResidualRow(5, 1, "Pledge field effects", "field", None),
	ResidualRow(5, 2, "Grassy Terrain", "field", None),
	ResidualRow(5, 3, "Hydration and Shed Skin", "pokemon", _ability_order5),
	ResidualRow(5, 4, "Leftovers and Black Sludge", "pokemon", _item_hp),
	ResidualRow(6, 2, "Aqua Ring", "pokemon", None),
	ResidualRow(7, 2, "Ingrain", "pokemon", None),
	ResidualRow(8, 2, "Leech Seed", "ordered", _leech_seed),
	ResidualRow(9, 2, "Poison, Toxic and Burn (order 10)", "status", _status_damage),
	ResidualRow(11, 2, "Nightmare", "pokemon", None),
	ResidualRow(12, 2, "Curse", "pokemon", _curse_residual),
	ResidualRow(13, 2, "Partial trapping", "pokemon", _partial_trap),
	ResidualRow(15, 2, "Taunt", "global", taunt_residual),
	ResidualRow(16, 2, "Encore", "global", encore_residual),
	ResidualRow(17, 2, "Disable", "pokemon", _disable_residual),
	ResidualRow(18, 2, "Magnet Rise", "pokemon", None),
	ResidualRow(19, 2, "Telekinesis", "pokemon", None),
	ResidualRow(20, 2, "Heal Block", "pokemon", None),
	ResidualRow(21, 2, "Embargo", "pokemon", None),
	ResidualRow(22, 2, "Throat Chop", "pokemon", None),
	ResidualRow(23, 2, "Yawn", "ordered", _yawn),
	ResidualRow(24, 2, "Perish Song", "pokemon", None),
	ResidualRow(25, 2, "Roost", "global", _end_roost),
	ResidualRow(26, 1, "Side condition timers", "side", None),
	ResidualRow(27, 1, "Trick Room timers", "global", trick_room_residual),
	ResidualRow(27, 2, "Field condition timers", "field", None),
	ResidualRow(
		28, 2, "Speed Boost, Bad Dreams, Slow Start and Truant", "pokemon", _ability_order28
	),
	ResidualRow(28, 3, "Toxic Orb and Flame Orb", "item", apply_residual_status_items),
	ResidualRow(29, 0, "End-of-turn item hooks", "item", _end_turn_items),
)


def residual(state, enabled, key, forced, tables, rows=None):
	"""Run the full residual event, then resolve its queued faint messages."""
	residual_enabled = enabled & (state.winner == ONGOING)
	item_rows = gather_active_items(state, tables)
	if rows is None:
		rows = active_abilities(state, tables)
	speed = active_speed(state, item_rows, rows)
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	bench_alive = jnp.any((state.hp > 0) & ~slots, axis=1)
	speed_tie = speed[0] == speed[1]
	faster = jnp.where(speed[0] > speed[1], 0, 1)
	for row in RESIDUAL_TABLE:
		active = residual_enabled & (state.winner == ONGOING)
		if row.handler is None:
			continue
		if row.scope == "field":
			state = row.handler(state, active, key, forced, tables, rows, speed)
		elif row.scope == "global":
			state = row.handler(state, active, key, forced, tables, rows)
		elif row.handler is apply_residual_status_items:
			state = row.handler(
				state, active, key, forced, tables, item_rows, rows, effective_weather(state, rows)
			)
		elif row.handler is _end_turn_items:
			state = row.handler(state, active, key, forced, tables, item_rows, rows)
		elif row.scope == "item":
			state = row.handler(state, active, key, forced, tables, item_rows)
		elif row.scope == "status":
			state = row.handler(state, active, faster, bench_alive, key, forced, speed_tie, rows)
		elif row.scope == "ordered":
			tie_first = _residual_tie_first(key, forced, 2)
			first = jnp.where(speed_tie, tie_first, faster)
			state = row.handler(state, active, first, bench_alive, key, forced, tables, rows)
		elif row.scope == "pokemon":
			context = 8 if row.order == 13 else 0 if row.order == 28 else 2
			kind = ABILITY_RESIDUAL_ORDER if row.order == 28 else ORDER
			tie_first = _residual_tie_first(key, forced, context, kind)
			first = jnp.where(speed_tie, tie_first, faster)
			if row.handler is _ability_order5 or row.handler is _ability_order28:
				state = row.handler(
					state, active, first, bench_alive, tables, item_rows, rows, key, forced
				)
			elif row.handler is _partial_trap or row.handler is _disable_residual:
				state = row.handler(state, active, first, bench_alive, tables, item_rows, rows)
			else:
				state = row.handler(state, active, first, bench_alive, tables, item_rows, rows)
	state = update_active_items(
		state, residual_enabled, gather_active_items(state, tables), rows, tables=tables
	)
	return state._replace(
		boosts=jnp.where((state.hp[SIDES, state.active] > 0)[:, None], state.boosts, 0),
	)
