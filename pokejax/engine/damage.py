"""Plain Gen 5 move arithmetic; sim/battle-actions.ts:1658-1901.

Move callbacks, flags, effects, abilities and items are intentionally outside
Phase 1. Struggle alone overrides its table type to ??? (data/moves.ts).
"""

import jax.numpy as jnp

from pokejax.mechanics.damage_composition import damage_from_preroll
from pokejax.mechanics.neutral_damage import neutral_damage_before_roll
from pokejax.mechanics.stat_stage import scale_stat_stage

from .randomness import CRIT, DAMAGE_ROLL, HIT, chance, integer
from .state import ACCURACY, EVASION

CRIT_DENOMINATORS = jnp.asarray([1, 16, 8, 4, 3, 2], jnp.int32)


def move_damage(state, side, move, key, forced, tables):
	"""Return damage and successful hit; callers mask actions that cannot run."""
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	struggle = move == t.struggle_id
	type_ = t.move_type[move]
	target_types = state.types[foe, target]
	present = target_types >= 0
	safe_types = jnp.maximum(target_types, 0)
	immune = ~struggle & jnp.any(present & t.type_immune[type_, safe_types])
	exponent = jnp.where(
		struggle, 0, jnp.sum(jnp.where(present, t.type_exponent[type_, safe_types], 0))
	).astype(jnp.int32)
	boost = jnp.clip(
		state.boosts[side, ACCURACY].astype(jnp.int32)
		- state.boosts[foe, EVASION].astype(jnp.int32),
		-6,
		6,
	)
	accuracy = t.move_accuracy[move].astype(jnp.int32)
	accuracy = jnp.where(
		boost >= 0, accuracy * (3 + boost) // 3, accuracy * 3 // jnp.maximum(3 - boost, 1)
	)
	hit = t.move_always_hits[move] | chance(
		key, forced, HIT, side, 0, jnp.clip(accuracy, 0, 100), 100
	)
	ratio = jnp.clip(t.move_crit_ratio[move].astype(jnp.int32), 0, 5)
	crit = (ratio > 0) & chance(key, forced, CRIT, side, 0, 1, CRIT_DENOMINATORS[ratio])
	physical = t.move_category[move] == 0
	attack_index = jnp.where(physical, 1, 3)
	defense_index = jnp.where(physical, 2, 4)
	atk_boost = state.boosts[side, attack_index - 1].astype(jnp.int32)
	def_boost = state.boosts[foe, defense_index - 1].astype(jnp.int32)
	atk_boost = jnp.where(crit & (atk_boost < 0), 0, atk_boost)
	def_boost = jnp.where(crit & (def_boost > 0), 0, def_boost)
	attack = scale_stat_stage(state.stats[side, user, attack_index].astype(jnp.int32), atk_boost)
	defense = scale_stat_stage(state.stats[foe, target, defense_index].astype(jnp.int32), def_boost)
	power = t.move_base_power[move].astype(jnp.int32)
	pre_roll = neutral_damage_before_roll(
		state.level[side, user].astype(jnp.int32), jnp.maximum(power, 1), attack, defense
	)
	pre_roll *= jnp.where(crit, 2, 1)
	roll = integer(key, forced, DAMAGE_ROLL, side, 0, 16)
	damage = damage_from_preroll(
		pre_roll,
		roll,
		jnp.any(state.types[side, user] == type_),
		False,
		struggle,
		exponent,
		False,
		physical,
		False,
		4096,
	)
	success = hit & ~immune & (power > 0) & (t.move_category[move] != 2)
	success &= state.hp[foe, target] > 0
	return jnp.where(success, damage, 0), success
