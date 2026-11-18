"""Gen 5 damage calculation with shared move math and per-hit crit/rolls."""

import jax.numpy as jnp

from pokejax.mechanics.damage_roll import apply_damage_roll
from pokejax.mechanics.neutral_damage import neutral_damage_before_roll
from pokejax.mechanics.stat_stage import scale_stat_stage

from .field import WEATHER_ACCURACY_ALWAYS, effective_move_type, weather_modify
from .randomness import CRIT, DAMAGE_ROLL, HIT, MAGNITUDE, chance, integer
from .state import ACCURACY, BURN, EVASION, SMACKDOWN, TWO_TURN_MOVE

CRIT_DENOMINATORS = jnp.asarray([1, 16, 8, 4, 3, 2], jnp.int32)
LEVEL_DAMAGE, HALF_CURRENT_DAMAGE, CONSTANT_DAMAGE = 1, 2, 3


def accuracy_hit(state, side, move, key, forced, tables):
	"""Sample the single move accuracy check before a multi-hit sequence."""
	boost = jnp.clip(
		state.boosts[side, ACCURACY].astype(jnp.int32)
		- state.boosts[1 - side, EVASION].astype(jnp.int32),
		-6,
		6,
	)
	accuracy = tables.move_accuracy[move].astype(jnp.int32)
	weather_accuracy = tables.move_weather_accuracy[move, state.weather].astype(jnp.int32)
	accuracy = jnp.where(weather_accuracy >= 0, weather_accuracy, accuracy)
	accuracy = jnp.where(
		boost >= 0, accuracy * (3 + boost) // 3, accuracy * 3 // jnp.maximum(3 - boost, 1)
	)
	return (
		tables.move_always_hits[move]
		| (weather_accuracy == WEATHER_ACCURACY_ALWAYS)
		| chance(key, forced, HIT, side, 0, jnp.clip(accuracy, 0, 100), 100)
	)


def move_damage_bases(state, side, move, key, forced, tables):
	"""Compute the non-crit and crit pre-roll damage bases once per move."""
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	struggle = move == t.struggle_id
	type_ = effective_move_type(state, move, t)
	target_types = state.types[foe, target]
	present = target_types >= 0
	safe_types = jnp.maximum(target_types, 0)
	grounded_flying = (type_ == t.ground_type) & (state.volatiles[foe, SMACKDOWN] > 0)
	counts_as_type = present & ~((target_types == t.flying_type) & grounded_flying)
	immune = ~struggle & jnp.any(counts_as_type & t.type_immune[type_, safe_types])
	exponent = jnp.where(
		struggle,
		0,
		jnp.sum(jnp.where(counts_as_type, t.type_exponent[type_, safe_types], 0)),
	).astype(jnp.int32)
	physical = t.move_category[move] == 0
	attack_index = jnp.where(physical, 1, 3)
	defense_index = jnp.where(physical, 2, 4)
	atk_boost = state.boosts[side, attack_index - 1].astype(jnp.int32)
	def_boost = state.boosts[foe, defense_index - 1].astype(jnp.int32)
	crit_atk_boost = jnp.where(atk_boost < 0, 0, atk_boost)
	crit_def_boost = jnp.where(def_boost > 0, 0, def_boost)
	attack = state.stats[side, user, attack_index].astype(jnp.int32)
	defense = state.stats[foe, target, defense_index].astype(jnp.int32)
	attack_stages = jnp.stack((atk_boost, crit_atk_boost))
	defense_stages = jnp.stack((def_boost, crit_def_boost))
	attack_effective = scale_stat_stage(attack, attack_stages)
	defense_effective = scale_stat_stage(defense, defense_stages)
	weather_spdef = jnp.any(present & t.weather_spdef_boost[state.weather, safe_types])
	defense_modifier = jnp.where((~physical) & weather_spdef, 6144, 4096)
	defense_effective = weather_modify(defense_effective, defense_modifier)
	level = state.level[side, user].astype(jnp.int32)
	power = t.move_base_power[move].astype(jnp.int32)
	magnitude = integer(key, forced, MAGNITUDE, side, 0, 100)
	magnitude_power = jnp.where(
		magnitude < 5,
		10,
		jnp.where(
			magnitude < 15,
			30,
			jnp.where(
				magnitude < 35,
				50,
				jnp.where(
					magnitude < 65,
					70,
					jnp.where(magnitude < 85, 90, jnp.where(magnitude < 95, 110, 150)),
				),
			),
		),
	)
	power = jnp.where(t.move_magnitude[move], magnitude_power, power)
	power *= t.move_weather_power[move, state.weather].astype(jnp.int32)
	power = weather_modify(power, t.move_weather_power_scale[move, state.weather])
	base_scenarios = neutral_damage_before_roll(
		level, jnp.maximum(power, 1), attack_effective, defense_effective
	)
	weather_modifier = t.weather_damage_by_type[state.weather, type_]
	base_normal = weather_modify(base_scenarios[0], weather_modifier)
	base_crit = weather_modify(base_scenarios[1], weather_modifier) * 2
	damage_kind = t.move_damage_kind[move]
	fixed = jnp.where(
		damage_kind == LEVEL_DAMAGE,
		level,
		jnp.where(
			damage_kind == HALF_CURRENT_DAMAGE,
			jnp.maximum(1, state.hp[foe, target].astype(jnp.int32) // 2),
			t.move_fixed_damage[move].astype(jnp.int32),
		),
	)
	has_power = (power > 0) | (damage_kind > 0)
	can_damage = ~immune & has_power & (t.move_category[move] != 2)
	can_effect = ~immune | ~t.move_checks_immunity[move]
	return (
		base_normal,
		base_crit,
		fixed,
		damage_kind,
		can_damage,
		can_effect,
		exponent,
		struggle,
		physical,
	)


def damage_for_hits(state, side, move, hit_indices, bases, key, forced, tables):
	"""Apply each hit's crit and roll to the shared damage bases."""
	(
		base_normal,
		base_crit,
		fixed,
		damage_kind,
		can_damage,
		_,
		exponent,
		struggle,
		physical,
	) = bases
	ratio = jnp.clip(tables.move_crit_ratio[move].astype(jnp.int32), 0, 5)
	denominator = CRIT_DENOMINATORS[ratio]
	user, target = state.active[side], state.active[1 - side]
	type_ = effective_move_type(state, move, tables)
	stab = jnp.any(state.types[side, user] == type_)
	burned = state.status[side, user] == BURN

	crit_roll = chance(key, forced, CRIT, side, hit_indices, 1, denominator)
	critical = tables.move_crit_always[move] | ((ratio > 0) & crit_roll)
	roll = integer(key, forced, DAMAGE_ROLL, side, hit_indices, 16)
	pre_roll = jnp.where(critical, base_crit, base_normal)
	damage = apply_damage_roll(pre_roll, roll)
	damage = jnp.where(stab & ~struggle, damage + damage // 2, damage)
	burn_reduction = (burned & physical).astype(jnp.int32)
	shift = exponent - burn_reduction
	boosted = jnp.left_shift(damage, jnp.maximum(shift, 0))
	resisted = damage >> jnp.maximum(-shift, 0)
	damage = jnp.where(shift > 0, boosted, resisted)
	target_charge = jnp.maximum(state.volatiles[1 - side, TWO_TURN_MOVE], 0)
	target_semi = tables.move_charge_semi[target_charge].astype(jnp.int32)
	semi_power = tables.move_semi_power[move, target_semi]
	damage *= jnp.where(semi_power, 2, 1).astype(jnp.int32)
	damage = jnp.where(damage == 0, 1, damage) & 65535
	damage = jnp.where(damage_kind > 0, fixed, damage)
	can_damage &= state.hp[1 - side, target] > 0
	return damage, can_damage
