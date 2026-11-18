"""Gen 5 damage calculation with shared move math and per-hit crit/rolls."""

import jax.numpy as jnp

from pokejax.mechanics.damage_roll import apply_damage_roll
from pokejax.mechanics.neutral_damage import neutral_damage_before_roll
from pokejax.mechanics.stat_stage import scale_stat_stage

from .abilities import (
	ACCURACY as ACCURACY_HOOK,
)
from .abilities import (
	ACCURACY_WONDER,
	ADAPTABILITY,
	CRIT_BLOCK,
	CRIT_BONUS,
	DEFENSE,
	DEFENSE_MARVEL_SCALE,
	IGNORE_BOOSTS,
	LEVITATE,
	NO_GUARD,
	SCRAPPY,
	accuracy_modifier,
	active_abilities,
	attack_modifiers,
	base_power_modifier,
	berry_allowed,
	effective_weather,
	final_damage_modifier,
	has_hook,
	skip_burn,
)
from .field import WEATHER_ACCURACY_ALWAYS, effective_move_type
from .items import (
	BERRY_LUM,
	DAMAGE_EXPERT_BELT,
	DAMAGE_LIFE_ORB,
	apply_stat_modifier,
	item_base_power_modifier,
)
from .modifiers import chain, modify
from .randomness import CRIT, DAMAGE_ROLL, HIT, MAGNITUDE, SECONDARY, chance, integer
from .secondaries import secondary_chances
from .state import (
	ACCURACY,
	BURN,
	CONFUSION,
	EVASION,
	FLASH_FIRE,
	HEALTHY,
	HP,
	POISON,
	SMACKDOWN,
	TWO_TURN_MOVE,
)

CRIT_DENOMINATORS = jnp.asarray([1, 16, 8, 4, 3, 2], jnp.int32)
LEVEL_DAMAGE, HALF_CURRENT_DAMAGE, CONSTANT_DAMAGE = 1, 2, 3


def accuracy_hit(state, side, move, key, forced, tables, rows=None, weather=None):
	"""Sample the single move accuracy check before a multi-hit sequence."""
	if rows is None:
		rows = active_abilities(state, tables)
	if weather is None:
		weather = effective_weather(state, rows)
	boost = jnp.clip(
		jnp.where(has_hook(rows[1 - side], IGNORE_BOOSTS), 0, state.boosts[side, ACCURACY]).astype(
			jnp.int32
		)
		- jnp.where(has_hook(rows[side], IGNORE_BOOSTS), 0, state.boosts[1 - side, EVASION]).astype(
			jnp.int32
		),
		-6,
		6,
	)
	accuracy = tables.move_accuracy[move].astype(jnp.int32)
	weather_accuracy = tables.move_weather_accuracy[move, weather].astype(jnp.int32)
	accuracy = jnp.where(weather_accuracy >= 0, weather_accuracy, accuracy)
	category = tables.move_category[move]
	targeted = tables.move_target_foe[move]
	wonder = targeted & (rows[1 - side, ACCURACY_HOOK] == ACCURACY_WONDER) & (category == 2)
	accuracy = modify(
		jnp.where(wonder, 50, accuracy),
		accuracy_modifier(
			rows,
			side,
			category == 0,
			targeted & (state.volatiles[1 - side, CONFUSION] > 0),
			weather,
		),
	)
	accuracy = jnp.where(
		boost >= 0, accuracy * (3 + boost) // 3, accuracy * 3 // jnp.maximum(3 - boost, 1)
	)
	return (
		tables.move_always_hits[move]
		| has_hook(rows[side], NO_GUARD)
		| (targeted & has_hook(rows[1 - side], NO_GUARD))
		| (weather_accuracy == WEATHER_ACCURACY_ALWAYS)
		| chance(key, forced, HIT, side, 0, jnp.clip(accuracy, 0, 100), 100)
	)


def type_matchup(state, side, move, type_, tables, rows):
	"""Apply type-chart immunities and groundedness before TryHit ability hooks."""
	foe = 1 - side
	target_types = state.types[foe, state.active[foe]]
	present = target_types >= 0
	safe_types = jnp.maximum(target_types, 0)
	ground = type_ == tables.ground_type
	grounded = state.volatiles[foe, SMACKDOWN] > 0
	flying = (target_types == tables.flying_type) & ground & grounded
	scrappy = has_hook(rows[side], SCRAPPY) & (
		(type_ == tables.normal_type) | (type_ == tables.fighting_type)
	)
	immune_type = jnp.any((present & ~flying) & tables.type_immune[type_, safe_types])
	levitate = ground & ~grounded & has_hook(rows[foe], LEVITATE)
	struggle = move == tables.struggle_id
	immune = ~struggle & ((immune_type & ~scrappy) | levitate)
	exponent = jnp.sum(jnp.where(present & ~flying, tables.type_exponent[type_, safe_types], 0))
	return immune, jnp.where(struggle, 0, exponent).astype(jnp.int32), levitate & ~immune_type


def move_damage_bases(
	state,
	side,
	move,
	key,
	forced,
	tables,
	item_rows,
	gem_trigger,
	rows=None,
	weather=None,
	type_=None,
	immune=None,
	exponent=None,
	sheer=False,
	analytic=False,
):
	"""Compute the non-crit and crit pre-roll damage bases once per move."""
	if rows is None:
		rows = active_abilities(state, tables)
	if weather is None:
		weather = effective_weather(state, rows)
	if type_ is None:
		type_ = effective_move_type(state, move, tables, weather)
	if immune is None:
		immune, exponent, _ = type_matchup(state, side, move, type_, tables, rows)
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	struggle = move == t.struggle_id
	item_type = jnp.where(struggle, -1, type_)
	target_types = state.types[foe, target]
	present = target_types >= 0
	safe_types = jnp.maximum(target_types, 0)
	physical = t.move_category[move] == 0
	attack_index = jnp.where(physical, 1, 3)
	defense_index = jnp.where(physical, 2, 4)
	atk_boost = state.boosts[side, attack_index - 1].astype(jnp.int32)
	def_boost = state.boosts[foe, defense_index - 1].astype(jnp.int32)
	atk_boost = jnp.where(has_hook(rows[foe], IGNORE_BOOSTS), 0, atk_boost)
	def_boost = jnp.where(has_hook(rows[side], IGNORE_BOOSTS), 0, def_boost)
	crit_atk_boost = jnp.where(atk_boost < 0, 0, atk_boost)
	crit_def_boost = jnp.where(def_boost > 0, 0, def_boost)
	attack = state.stats[side, user, attack_index].astype(jnp.int32)
	defense = state.stats[foe, target, defense_index].astype(jnp.int32)
	attack_stages = jnp.stack((atk_boost, crit_atk_boost))
	defense_stages = jnp.stack((def_boost, crit_def_boost))
	attack_effective = scale_stat_stage(attack, attack_stages)
	defense_effective = scale_stat_stage(defense, defense_stages)
	fire = type_ == t.fire_type
	user_status = state.status[side, user]
	flash_boost = fire & (state.volatiles[side, FLASH_FIRE] > 0)
	attack_effective, attack_modifier = attack_modifiers(
		rows,
		side,
		attack_effective,
		physical,
		user_status,
		fire | (type_ == t.ice_type),
		flash_boost,
	)
	attack_effective = apply_stat_modifier(
		attack_effective, side, attack_index, item_rows, attack_modifier
	)
	weather_spdef = jnp.any(present & t.weather_spdef_boost[weather, safe_types])
	defense_effective = modify(defense_effective, jnp.where(~physical & weather_spdef, 6144, 4096))
	marvel = physical & (rows[foe, DEFENSE] == DEFENSE_MARVEL_SCALE)
	statused = state.status[foe, target] != HEALTHY
	# In the supported move table Twineedle is the multi-hit move that can
	# inflict a status during the sequence. Its first poison can change Def
	# before hit two, unless the target's Lum Berry activates between hits.
	poison = (t.move_hit_max[move] > 1) & (t.move_secondary_status[move, 0] == POISON)
	poison &= ~jnp.asarray(sheer) & ~jnp.any(present & t.status_immune[POISON, safe_types])
	poison &= chance(
		key, forced, SECONDARY, side, 0, secondary_chances(t, move, rows[side])[0], 100
	)
	poison &= ~((item_rows.berry_kind[foe] == BERRY_LUM) & berry_allowed(state, rows)[foe])
	marvel_later = marvel & ~statused & poison
	defense_eff = apply_stat_modifier(
		defense_effective, foe, defense_index, item_rows, jnp.where(marvel & statused, 6144, 4096)
	)
	defense_later = apply_stat_modifier(defense_effective, foe, defense_index, item_rows, 6144)
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
	power *= t.move_weather_power[move, weather].astype(jnp.int32)

	item_modifier = item_base_power_modifier(
		side, item_type, gem_trigger, t.move_damage_kind[move], item_rows
	)
	power_modifier = base_power_modifier(
		rows,
		side,
		move,
		power,
		chain(t.move_weather_power_scale[move, weather], item_modifier),
		physical,
		user_status,
		fire,
		sheer,
		analytic,
		t,
	)
	power = modify(power, power_modifier)
	base_scenarios = neutral_damage_before_roll(
		level, jnp.maximum(power, 1), attack_effective, defense_eff
	)
	weather_modifier = t.weather_damage_by_type[weather, type_]
	later_scenarios = neutral_damage_before_roll(
		level, jnp.maximum(power, 1), attack_effective, defense_later
	)
	late_bases = modify(later_scenarios, weather_modifier)
	base_normal = modify(base_scenarios[0], weather_modifier)
	base_crit = modify(base_scenarios[1], weather_modifier) * 2
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
		marvel_later,
		late_bases,
	)


def damage_for_hits(
	state, side, move, hit_indices, bases, key, forced, tables, item_rows, rows, type_
):
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
		marvel_later,
		late_bases,
	) = bases
	stick_bonus = jnp.where(item_rows.stat_species[side], item_rows.crit_bonus[side], 0)
	ratio = jnp.clip(
		tables.move_crit_ratio[move].astype(jnp.int32)
		+ stick_bonus
		+ has_hook(rows[side], CRIT_BONUS),
		0,
		5,
	)
	denominator = CRIT_DENOMINATORS[ratio]
	user, target = state.active[side], state.active[1 - side]
	stab = jnp.any(state.types[side, user] == type_)
	burned = (state.status[side, user] == BURN) & ~skip_burn(rows[side])

	crit_roll = chance(key, forced, CRIT, side, hit_indices, 1, denominator)
	critical = tables.move_crit_always[move] | ((ratio > 0) & crit_roll)
	critical &= ~has_hook(rows[1 - side], CRIT_BLOCK)
	roll = integer(key, forced, DAMAGE_ROLL, side, hit_indices, 16)
	pre_roll = jnp.where(critical, base_crit, base_normal)
	late_roll = jnp.where(critical, 2 * late_bases[1], late_bases[0])
	pre_roll = jnp.where((hit_indices > 0) & marvel_later, late_roll, pre_roll)
	damage = apply_damage_roll(pre_roll, roll)
	damage = jnp.where(
		stab & ~struggle,
		damage + jnp.where(has_hook(rows[side], ADAPTABILITY), damage, damage // 2),
		damage,
	)
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
	modifier = jnp.where(
		item_rows.damage_kind[side] == DAMAGE_LIFE_ORB,
		5324,
		jnp.where(
			(item_rows.damage_kind[side] == DAMAGE_EXPERT_BELT) & (exponent > 0),
			4915,
			4096,
		),
	).astype(jnp.int32)
	full_hp = (hit_indices == 0) & (state.hp[1 - side, target] >= state.stats[1 - side, target, HP])
	damage = (
		modify(damage, final_damage_modifier(rows, side, critical, exponent, full_hp, modifier))
		& 65535
	)
	damage = jnp.where(damage_kind > 0, fixed, damage)
	can_damage &= state.hp[1 - side, target] > 0
	return damage, can_damage
