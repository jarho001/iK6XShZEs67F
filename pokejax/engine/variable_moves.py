"""Variable-power, stat-source and conditional move rules selected by table columns.

Each rule kind is a branch-free select over the active member's data, so a
move only adds a row. Showdown sources: basePowerCallback, onBasePower,
onModifyType, onTry and damageCallback in data/moves.ts and the gen5 mod chain,
and sim/battle-actions.ts getDamage. Weights are hectograms (Pokemon.weighthg).
"""

import jax.numpy as jnp

from pokejax.mechanics.hidden_power import hidden_power

from .abilities import DROP, ITEM_GUARD, ITEM_GUARD_STICKY_HOLD, REVERSE_STAGES
from .actions import SIDES
from .items import BERRY_CHESTO, BERRY_KIND, BERRY_LUM, BERRY_SITRUS, update_item_state
from .order import active_speed
from .state import (
	AUTOTOMIZE,
	BOOST_SPE,
	CONFUSION,
	HEALTHY,
	HP,
	MUST_RECHARGE,
	SLEEP,
	TEAM_SLOTS,
)
from .turn_moves import set_active_volatile

# Base power rule (move_bp_rule): what the callback reads before doubling.
BP_TABLE, BP_HIDDEN_POWER, BP_RETURN, BP_TARGET_WEIGHT, BP_WEIGHT_RATIO = range(5)
BP_GYRO_BALL, BP_USER_HP, BP_STORED_POWER = range(5, 8)
# Doubling condition (move_bp_double); move_bp_partner names a required last move.
(
	DOUBLE_NEVER,
	DOUBLE_USER_STATUS,
	DOUBLE_USER_NO_ITEM,
	DOUBLE_FAINTED_LAST_TURN,
	DOUBLE_PARTNER,
	DOUBLE_TARGET_MOVED,
	DOUBLE_TARGET_SWITCHING,
) = range(7)
# Type rule (move_type_rule) and stat rule (move_stat_rule).
TYPE_TABLE, TYPE_HIDDEN_POWER, TYPE_PLATE = range(3)
STAT_TABLE, STAT_DEFENSE_DEF, STAT_TARGET_ATTACK = range(3)
# Success gate (move_success_rule), Showdown's onTry and beforeMove checks.
(
	SUCCESS_ALWAYS,
	SUCCESS_SUCKER_PUNCH,
	SUCCESS_FOCUS_PUNCH,
	SUCCESS_COUNTER,
	SUCCESS_MIRROR_COAT,
	SUCCESS_METAL_BURST,
	SUCCESS_AUTOTOMIZE,
) = range(7)
# Damage kind for moves whose damage is the damage they took (damageCallback).
RETURNED_DAMAGE = 4
CATEGORY_PHYSICAL, CATEGORY_SPECIAL, CATEGORY_STATUS = range(3)
# Columns of BattleState.turn_damage, indexed by the defending side.
DAMAGE_TAKEN_ANY, DAMAGE_TAKEN_PHYSICAL, DAMAGE_TAKEN_SPECIAL = range(3)
AUTOTOMIZE_STEP = 1000


def hidden_power_for(state, side, tables):
	"""Hidden Power type index and power from the active member's stored IVs."""
	ivs = state.ivs[side, state.active[side]].astype(jnp.int32)
	kind, power = hidden_power(ivs)
	return tables.hidden_power_type[kind], power


def active_weight(state, side, tables):
	"""Pokemon.getWeight for the active member: species weight less Autotomize."""
	species = state.species[side, state.active[side]].astype(jnp.int32)
	weight = tables.species_weight_hg[species].astype(jnp.int32)
	stages = state.volatiles[side, AUTOTOMIZE].astype(jnp.int32)
	lighter = jnp.maximum(weight - stages * AUTOTOMIZE_STEP, 1)
	return jnp.maximum(jnp.where(stages > 0, lighter, weight), 1)


def _grass_knot_power(weight):
	return jnp.where(
		weight >= 2000,
		120,
		jnp.where(
			weight >= 1000,
			100,
			jnp.where(
				weight >= 500, 80, jnp.where(weight >= 250, 60, jnp.where(weight >= 100, 40, 20))
			),
		),
	)


def _heavy_slam_power(user_weight, target_weight):
	return jnp.where(
		user_weight >= target_weight * 5,
		120,
		jnp.where(
			user_weight >= target_weight * 4,
			100,
			jnp.where(
				user_weight >= target_weight * 3,
				80,
				jnp.where(user_weight >= target_weight * 2, 60, 40),
			),
		),
	)


def base_power(state, side, row, tables, switching_foe, rows, item_rows, gem_consumed):
	"""BasePower before items and weather: the callback value, then doubling.

	Every doubling is exact (x2), so applying it here matches Showdown's
	basePowerCallback and chainModify(2) in onBasePower alike. switching_foe is
	true when the target is leaving: a voluntary switch or a pivot.
	gem_consumed is true when a gem of this move's type is used up before BasePower.
	"""
	foe = 1 - side
	user = state.active[side]
	target = state.active[foe]
	base = row.move_base_power.astype(jnp.int32)
	_, hidden_power_value = hidden_power_for(state, side, tables)
	kind = row.move_bp_rule.astype(jnp.int32)
	target_weight = active_weight(state, foe, tables)
	user_weight = active_weight(state, side, tables)
	# Pokemon.getStat('spe') runs ModifySpe, so items, abilities and paralysis count.
	speeds = active_speed(state, item_rows, rows)
	user_speed = speeds[side]
	target_speed = speeds[foe]
	user_hp = state.hp[side, user].astype(jnp.int32)
	user_max = state.stats[side, user, HP].astype(jnp.int32)
	stored = base + 20 * jnp.sum(jnp.maximum(state.boosts[side].astype(jnp.int32), 0))
	returned = jnp.maximum(state.happiness[side, user].astype(jnp.int32) * 10 // 25, 1)
	power = jnp.where(kind == BP_HIDDEN_POWER, hidden_power_value, base)
	power = jnp.where(kind == BP_RETURN, returned, power)
	power = jnp.where(kind == BP_TARGET_WEIGHT, _grass_knot_power(target_weight), power)
	power = jnp.where(kind == BP_WEIGHT_RATIO, _heavy_slam_power(user_weight, target_weight), power)
	gyro = jnp.minimum(25 * target_speed // jnp.maximum(user_speed, 1) + 1, 150)
	power = jnp.where(kind == BP_GYRO_BALL, gyro, power)
	power = jnp.where(kind == BP_USER_HP, base * user_hp // jnp.maximum(user_max, 1), power)
	power = jnp.where(kind == BP_STORED_POWER, stored, power)
	power = jnp.maximum(power, 1)

	double = row.move_bp_double.astype(jnp.int32)
	status = state.status[side, user].astype(jnp.int32)
	# Showdown's newlySwitched is set by clearVolatile, which a lead never runs, so a lead is
	# not newly switched; a replacement is until its turn ends (active_turns counts turn ends).
	target_newly = state.active_turns[foe, target] == 0
	doubled = (
		((double == DOUBLE_USER_STATUS) & (status != HEALTHY) & (status != SLEEP))
		| ((double == DOUBLE_USER_NO_ITEM) & ((state.item[side, user] == 0) | gem_consumed))
		| ((double == DOUBLE_FAINTED_LAST_TURN) & state.fainted_last[side])
		| ((double == DOUBLE_PARTNER) & (state.last_success == row.move_bp_partner))
		| ((double == DOUBLE_TARGET_MOVED) & ~state.turn_pending[foe] & ~target_newly)
		| ((double == DOUBLE_TARGET_SWITCHING) & switching_foe)
	)
	return jnp.where(doubled, 2 * power, power)


def move_type_id(state, side, row, tables, weather):
	"""onModifyType: Weather Ball, then Hidden Power's IV type or a plate's Judgment type."""
	table = row.move_type.astype(jnp.int32)
	hidden_type, _ = hidden_power_for(state, side, tables)
	plate = tables.plate_type[state.item[side, state.active[side]].astype(jnp.int32)]
	kind = row.move_type_rule.astype(jnp.int32)
	typed = jnp.where(kind == TYPE_HIDDEN_POWER, hidden_type, table)
	typed = jnp.where((kind == TYPE_PLATE) & (plate >= 0), plate, typed)
	weather_type = row.move_weather_type[weather].astype(jnp.int32)
	return jnp.where(weather_type >= 0, weather_type, typed)


def success_allowed(state, side, row, tables, rows):
	"""Showdown onTry and beforeMove gates: False means the move fails with no effect."""
	foe = 1 - side
	kind = row.move_success_rule.astype(jnp.int32)
	foe_move = state.turn_moves[foe].astype(jnp.int32)
	foe_category = tables.move_category[foe_move].astype(jnp.int32)
	# Sucker Punch needs a damaging move still queued for the target.
	sucker = (
		state.turn_pending[foe]
		& (foe_category != CATEGORY_STATUS)
		& (state.volatiles[foe, MUST_RECHARGE] == 0)
	)
	focus = ~state.focus_lost[side]
	counter = state.turn_damage[side, DAMAGE_TAKEN_PHYSICAL] > 0
	mirror = state.turn_damage[side, DAMAGE_TAKEN_SPECIAL] > 0
	metal = state.turn_hit[side]
	speed_limit = jnp.where(rows[side, DROP] == REVERSE_STAGES, -6, 6)
	autotomize = state.boosts[side, BOOST_SPE] != speed_limit
	return jnp.where(
		kind == SUCCESS_SUCKER_PUNCH,
		sucker,
		jnp.where(
			kind == SUCCESS_FOCUS_PUNCH,
			focus,
			jnp.where(
				kind == SUCCESS_COUNTER,
				counter,
				jnp.where(
					kind == SUCCESS_MIRROR_COAT,
					mirror,
					jnp.where(
						kind == SUCCESS_METAL_BURST,
						metal,
						jnp.where(kind == SUCCESS_AUTOTOMIZE, autotomize, True),
					),
				),
			),
		),
	)


def returned_damage(state, side, row):
	"""damageCallback values: twice the last physical or special hit, or 1.5x the last hit."""
	damage = state.turn_damage[side].astype(jnp.int32)
	counter = 2 * damage[DAMAGE_TAKEN_PHYSICAL]
	mirror = 2 * damage[DAMAGE_TAKEN_SPECIAL]
	metal = (3 * damage[DAMAGE_TAKEN_ANY]) // 2
	kind = row.move_success_rule.astype(jnp.int32)
	value = jnp.where(
		kind == SUCCESS_COUNTER, counter, jnp.where(kind == SUCCESS_MIRROR_COAT, mirror, metal)
	)
	return jnp.maximum(value, 1)


def record_damage_taken(state, side, row, hit_landed, dealt):
	"""Defender bookkeeping for this turn: last hits, Focus Punch and Metal Burst.

	hit_landed and dealt are per hit on the fixed five-hit axis. The defender is
	the foe of the attacking side.
	"""
	foe = 1 - side
	indices = jnp.arange(hit_landed.shape[0], dtype=jnp.int32)
	landed = jnp.any(hit_landed)
	last_landed = jnp.max(jnp.where(hit_landed, indices, -1))
	positive = hit_landed & (dealt > 0)
	last_positive = jnp.max(jnp.where(positive, indices, -1))
	any_damage = dealt[jnp.maximum(last_landed, 0)].astype(jnp.int16)
	positive_damage = dealt[jnp.maximum(last_positive, 0)].astype(jnp.int16)
	physical = (row.move_category == CATEGORY_PHYSICAL) & (last_positive >= 0)
	special = (row.move_category == CATEGORY_SPECIAL) & (last_positive >= 0)
	slots = jnp.arange(state.turn_damage.shape[1])
	defender = (SIDES == foe)[:, None]
	damage = jnp.where(
		defender & landed & (slots[None, :] == DAMAGE_TAKEN_ANY),
		any_damage,
		state.turn_damage,
	)
	damage = jnp.where(
		defender & physical & (slots[None, :] == DAMAGE_TAKEN_PHYSICAL),
		positive_damage,
		damage,
	)
	damage = jnp.where(
		defender & special & (slots[None, :] == DAMAGE_TAKEN_SPECIAL), positive_damage, damage
	)
	damaging = row.move_category != CATEGORY_STATUS
	return state._replace(
		turn_damage=damage.astype(state.turn_damage.dtype),
		turn_hit=state.turn_hit | ((SIDES == foe) & landed),
		focus_lost=state.focus_lost | ((SIDES == foe) & landed & damaging),
	)


def apply_autotomize(state, side, enabled, tables):
	"""addVolatile('autotomize'): the first use starts one stage above 1 hg, and each
	later use adds a stage while the base weight stays above 1 hg after the stages."""
	species = state.species[side, state.active[side]].astype(jnp.int32)
	weight = tables.species_weight_hg[species].astype(jnp.int32)
	current = state.volatiles[side, AUTOTOMIZE].astype(jnp.int32)
	started = (weight > 1).astype(jnp.int32)
	stepped = jnp.where(weight - current * AUTOTOMIZE_STEP > 1, current + 1, current)
	new = jnp.where(current == 0, started, stepped)
	return set_active_volatile(state, side, AUTOTOMIZE, jnp.where(enabled, new, current))


def steal_berry(state, side, enabled, tables, rows):
	"""Bug Bite and Pluck (onHit): the target's berry is taken and eaten by the user
	at once. Only the eat effect of each berry applies; the user keeps its own item.
	"""
	foe = 1 - side
	user = state.active[side]
	target = state.active[foe]
	berry = tables.item_hooks.rows[state.item[foe, target].astype(jnp.int32), BERRY_KIND]
	berry = berry.astype(jnp.int32)
	guarded = rows[foe, ITEM_GUARD] == ITEM_GUARD_STICKY_HOLD
	steal = enabled & (berry > 0) & ~guarded & (state.hp[side, user] > 0)
	taken = (SIDES == foe) & steal
	# takeItem clears the item without setting lastItem, so it is not "consumed".
	prior_consumed, prior_last = state.item_consumed, state.last_item
	state = update_item_state(state, taken, taken, tables)
	state = state._replace(
		item_consumed=jnp.where(taken[:, None], prior_consumed, state.item_consumed),
		last_item=jnp.where(taken[:, None], prior_last, state.last_item),
	)
	member = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	status = state.status[side, user]
	lum = steal & (berry == BERRY_LUM)
	chesto = steal & (berry == BERRY_CHESTO) & (status == SLEEP)
	sitrus = steal & (berry == BERRY_SITRUS)
	clear_status = lum | chesto
	hp = state.hp[side, user].astype(jnp.int32)
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	healed = jnp.minimum(maximum, hp + jnp.maximum(maximum // 4, 1))
	columns = jnp.arange(state.volatiles.shape[1])
	confusion = (SIDES == side)[:, None] & (columns[None, :] == CONFUSION) & lum
	return state._replace(
		hp=jnp.where(member & sitrus, healed.astype(state.hp.dtype), state.hp),
		status=jnp.where(member & clear_status, HEALTHY, state.status),
		status_counter=jnp.where(member & clear_status, 0, state.status_counter),
		volatiles=jnp.where(confusion, 0, state.volatiles),
	)
