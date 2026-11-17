"""Healing, Pain Split and temporary type effects from moves."""

import jax.numpy as jnp

from .actions import SIDES
from .state import HP, NO_TYPE, ROOST


def _heal_fraction(value, numerator, denominator, callback):
	"""Heal fields use Math.round; weather-heal callbacks use modify's half-down rule."""
	denominator = jnp.maximum(denominator.astype(jnp.int32), 1)
	doubled = 2 * value * numerator.astype(jnp.int32)
	math_round = (doubled + denominator) // (2 * denominator)
	modify_round = (doubled + denominator - 1) // (2 * denominator)
	return jnp.where(callback, modify_round, math_round)


def apply_healing_active(state, side, move, enabled, hit, active_hp, tables):
	"""Apply table-driven recovery to active HP and update Roost's type fields."""
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	hp = active_hp[side]
	numerator, denominator = t.move_heal[move]
	can_heal = enabled & hit & (numerator > 0) & (hp > 0) & (hp < maximum)
	amount = jnp.maximum(
		1,
		_heal_fraction(maximum, numerator, denominator, t.move_heal_weather[move] > 0),
	)
	healed_hp = jnp.minimum(maximum, hp + amount)
	healed = can_heal & (healed_hp > hp)

	target_hp = active_hp[foe]
	average = jnp.maximum((hp + target_hp) // 2, 1)
	pain_split = enabled & hit & t.move_pain_split[move] & (hp > 0) & (target_hp > 0)
	target_max = state.stats[foe, target, HP].astype(jnp.int32)
	user_hp = jnp.where(can_heal, healed_hp, hp)
	user_hp = jnp.where(pain_split, jnp.minimum(maximum, average), user_hp)
	target_new_hp = jnp.minimum(target_max, average)
	active_hp = jnp.where(SIDES == side, user_hp, active_hp)
	active_hp = jnp.where(SIDES == foe, jnp.where(pain_split, target_new_hp, target_hp), active_hp)

	roost = healed & t.move_roost[move]
	old_types = state.types[side, user]
	new_types = jnp.where(old_types == t.flying_type, NO_TYPE, old_types)
	new_types = jnp.where(
		jnp.all(new_types < 0), jnp.asarray([t.normal_type, NO_TYPE], old_types.dtype), new_types
	)
	types = state.types.at[side, user].set(jnp.where(roost, new_types, old_types))
	roost_types = state.roost_types.at[side, user].set(
		jnp.where(roost, old_types, state.roost_types[side, user])
	)
	volatiles = state.volatiles.at[side, ROOST].set(
		jnp.where(roost, 1, state.volatiles[side, ROOST])
	)
	return (
		state._replace(types=types, roost_types=roost_types, volatiles=volatiles),
		active_hp,
		healed,
	)


def apply_healing(state, side, move, enabled, hit, tables):
	"""Apply recovery and update the standalone state for focused rule checks."""
	active_hp = state.hp[SIDES, state.active].astype(jnp.int32)
	state, active_hp, healed = apply_healing_active(
		state, side, move, enabled, hit, active_hp, tables
	)
	hp = state.hp.at[SIDES, state.active].set(active_hp.astype(state.hp.dtype))
	return state._replace(hp=hp), healed
