"""Healing, Pain Split and temporary type effects from moves."""

import jax.numpy as jnp
from jax import lax

from .actions import SIDES
from .state import HP, NO_TYPE, NUM_VOLATILES, ROOST, TEAM_SLOTS


def _heal_fraction(value, numerator, denominator, callback):
	"""Heal fields use Math.round; weather-heal callbacks use modify's half-down rule."""
	denominator = jnp.maximum(denominator.astype(jnp.int32), 1)
	bias = denominator // 2 - callback.astype(jnp.int32)
	product = value * numerator.astype(jnp.int32)
	# Unused healing rows can have a zero numerator. Both rounding paths
	# still produce a zero heal once the caller applies its minimum amount.
	return lax.div(jnp.maximum(product + bias, 0), denominator)


def apply_healing_active(state, side, move, enabled, hit, active_hp, tables, weather=None):
	"""Apply table-driven recovery to active HP and update Roost's type fields."""
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	hp = active_hp[side]
	numerator, denominator = t.move_weather_heal[
		move, state.weather if weather is None else weather
	]
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
	team_slots = jnp.arange(TEAM_SLOTS)
	user_mask = (SIDES == side)[:, None] & (team_slots[None, :] == user)
	types = jnp.where(user_mask[:, :, None] & roost, new_types[None, None, :], state.types)
	roost_types = jnp.where(
		user_mask[:, :, None] & roost, old_types[None, None, :], state.roost_types
	)
	columns = jnp.arange(NUM_VOLATILES)
	roost_mask = (SIDES == side)[:, None] & (columns[None, :] == ROOST) & roost
	volatiles = jnp.where(roost_mask, 1, state.volatiles)
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
	active_slots = jnp.arange(state.hp.shape[1])[None, :] == state.active[:, None]
	hp = jnp.where(active_slots, active_hp[:, None].astype(state.hp.dtype), state.hp)
	return state._replace(hp=hp), healed
