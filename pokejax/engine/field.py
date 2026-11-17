"""Weather, entry hazards and their move-table effects for Gen 5 singles."""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .randomness import SWITCH_HANDLER_ORDER, WEATHER_CHANGE_ORDER, WEATHER_ORDER, first_side
from .state import (
	BOOST_SPE,
	HAIL,
	HEALTHY,
	ONGOING,
	PARALYSIS,
	POISON,
	SAND,
	SPE,
	SPIKES,
	TOXIC,
	TOXIC_SPIKES,
)

WEATHER_ACCURACY_ALWAYS = -2
WEATHER_ACCURACY_BASE = -1
WEATHER_DURATION = 5


def effective_move_type(state, move, tables):
	"""Return the move-table type after its weather type override."""
	weather_type = tables.move_weather_type[move, state.weather].astype(jnp.int32)
	return jnp.where(weather_type >= 0, weather_type, tables.move_type[move].astype(jnp.int32))


def weather_modify(value, modifier):
	"""Showdown's Gen 5 Battle.modify rounding for a fixed-point modifier."""
	value = value.astype(jnp.int32)
	modifier = modifier.astype(jnp.int32)
	return (value * modifier + 2047) // 4096


def apply_move_field_effects(state, side, move, enabled, field_hit, hit, key, forced, tables):
	"""Apply weather, hazards, and Rapid Spin through per-move table columns."""
	apply = enabled & field_hit
	weather = tables.move_weather[move].astype(state.weather.dtype)
	weather_change = apply & (weather > 0) & (weather != state.weather)
	# Field.setWeather emits WeatherChange even when the weather field was clear.
	_ = first_side(key, forced, WEATHER_CHANGE_ORDER)
	weather_state = jnp.where(weather_change, weather, state.weather)
	weather_turns = jnp.where(weather_change, WEATHER_DURATION, state.weather_turns)
	state = state._replace(weather=weather_state, weather_turns=weather_turns)

	foe = 1 - side
	hazard = tables.move_hazard[move]
	spikes = state.hazards[foe, SPIKES].astype(jnp.int32)
	toxic_spikes = state.hazards[foe, TOXIC_SPIKES].astype(jnp.int32)
	add_spikes = apply & (hazard == 1) & (spikes < 3)
	add_toxic_spikes = apply & (hazard == 2) & (toxic_spikes < 2)
	hazard_layers = state.hazards.astype(jnp.int32)
	adds = jnp.stack((add_spikes, add_toxic_spikes)).astype(jnp.int32)
	hazards = hazard_layers + (SIDES == foe)[:, None] * adds[None, :]
	clear = enabled & hit & tables.move_rapid_spin[move]
	hazards = jnp.where((SIDES == side)[:, None] & clear, 0, hazards)

	order = state.hazard_order.astype(jnp.int32)
	foe_order = order[foe]
	new_order = jnp.max(foe_order) + 1
	first_layers = jnp.stack((add_spikes & (spikes == 0), add_toxic_spikes & (toxic_spikes == 0)))
	foe_order = jnp.where(first_layers, new_order, foe_order)
	hazard_order = jnp.where((SIDES == foe)[:, None], foe_order[None, :], order)
	hazard_order = jnp.where((SIDES == side)[:, None] & clear, 0, hazard_order)
	return state._replace(
		hazards=hazards.astype(state.hazards.dtype),
		hazard_order=hazard_order.astype(state.hazard_order.dtype),
	)


def _resolve_entry_faint(state, side, enabled):
	alive = jnp.any(state.hp > 0, axis=1)
	ended = enabled & ~alive[side] & (state.winner == ONGOING)
	winner = jnp.where(alive[1 - side], 1 - side, side)
	return state._replace(winner=jnp.where(ended, winner, state.winner).astype(state.winner.dtype))


def _apply_spikes_handler(state, side, slot, enabled, tables):
	types = state.types[side, slot]
	grounded = ~jnp.any(types == tables.flying_type)
	hp = state.hp[side, slot].astype(jnp.int32)
	maximum = state.stats[side, slot, 0].astype(jnp.int32)
	spikes = state.hazards[side, SPIKES].astype(jnp.int32)
	damage_numerator = jnp.where(spikes == 1, 3, jnp.where(spikes == 2, 4, 6))
	spike_damage = jnp.maximum(maximum * damage_numerator // 24, 1)
	take_spikes = enabled & grounded & (spikes > 0) & (hp > 0)
	hp = jnp.maximum(hp - jnp.where(take_spikes, spike_damage, 0), 0)
	state = state._replace(hp=state.hp.at[side, slot].set(hp.astype(state.hp.dtype)))
	return _resolve_entry_faint(state, side, take_spikes & (hp == 0))


def _apply_toxic_spikes_handler(state, side, slot, enabled, tables):
	types = state.types[side, slot]
	grounded = ~jnp.any(types == tables.flying_type)
	hp = state.hp[side, slot].astype(jnp.int32)
	toxic_layers = state.hazards[side, TOXIC_SPIKES].astype(jnp.int32)
	poison_type = jnp.any(types == tables.poison_type)
	absorb = enabled & grounded & poison_type & (toxic_layers > 0) & (hp > 0)
	toxic_spikes = state.hazards.at[side, TOXIC_SPIKES].set(
		jnp.where(absorb, 0, toxic_layers).astype(state.hazards.dtype)
	)
	status = state.status[side, slot]
	inflicted = jnp.where(toxic_layers >= 2, TOXIC, POISON)
	present = types >= 0
	immune = jnp.any(present & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	poison = enabled & grounded & (toxic_layers > 0) & ~poison_type & (hp > 0)
	poison &= (status == HEALTHY) & ~immune
	status = jnp.where(poison, inflicted, status)
	counter = state.status_counter[side, slot]
	counter = jnp.where(poison, 0, counter)
	hazard_order = state.hazard_order.at[side, TOXIC_SPIKES].set(
		jnp.where(absorb, 0, state.hazard_order[side, TOXIC_SPIKES])
	)
	state = state._replace(
		hazards=toxic_spikes,
		hazard_order=hazard_order,
		status=state.status.at[side, slot].set(status),
		status_counter=state.status_counter.at[side, slot].set(counter),
	)
	return _resolve_entry_faint(state, side, False)


def apply_switch_hazards(state, side, slot, enabled, tables):
	"""Apply a side's ordered Spikes and Toxic Spikes handlers on switch-in."""
	spikes_order = state.hazard_order[side, SPIKES]
	toxic_order = state.hazard_order[side, TOXIC_SPIKES]
	first_kind = jnp.where(
		(state.hazards[side, SPIKES] > 0)
		& ((state.hazards[side, TOXIC_SPIKES] == 0) | (spikes_order < toxic_order)),
		SPIKES,
		TOXIC_SPIKES,
	)
	for index in (0, 1):
		kind = jnp.where(index == 0, first_kind, 1 - first_kind)
		alive = state.hp[side, slot] > 0
		active = enabled & alive & (state.winner == ONGOING)
		state = _apply_spikes_handler(state, side, slot, active & (kind == SPIKES), tables)
		active = active & (state.hp[side, slot] > 0) & (state.winner == ONGOING)
		state = _apply_toxic_spikes_handler(
			state, side, slot, active & (kind == TOXIC_SPIKES), tables
		)
	return state


def apply_switch_hazards_ordered(state, switchers, enabled, key, forced, tables):
	"""Run grouped SwitchIn handlers in Showdown Speed and tie order."""
	sides = SIDES
	speed = scale_stat_stage(
		state.stats[sides, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = jnp.where(state.status[sides, state.active] == PARALYSIS, speed // 4, speed)
	tie_first = first_side(key, forced, SWITCH_HANDLER_ORDER)
	first = jnp.where(speed[0] == speed[1], tie_first, jnp.where(speed[0] > speed[1], 0, 1))
	for side in (first, 1 - first):
		active = enabled & switchers[side] & (state.winner == ONGOING)
		state = apply_switch_hazards(state, side, state.active[side], active, tables)
	return state


def weather_residual(state, enabled, key, forced, tables):
	"""Apply the weather field callback to both actives before resolving faints."""
	sides = SIDES
	weather = state.weather.astype(jnp.int32)
	turns = state.weather_turns.astype(jnp.int32)
	callback = enabled & (weather > 0) & ((turns == 0) | (turns > 1))
	decrement = enabled & (weather > 0) & (turns > 0)
	expired = decrement & (turns == 1)
	new_turns = jnp.where(decrement, jnp.maximum(turns - 1, 0), turns)
	new_weather = jnp.where(expired, 0, weather)
	# clearWeather emits WeatherChange when a finite weather expires.
	_ = first_side(key, forced, WEATHER_CHANGE_ORDER)

	speed = scale_stat_stage(
		state.stats[sides, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = jnp.where(state.status[sides, state.active] == PARALYSIS, speed // 4, speed)
	tie_order = first_side(key, forced, WEATHER_ORDER)
	first = jnp.where(speed[0] == speed[1], tie_order, jnp.where(speed[0] > speed[1], 0, 1))

	active_slots = state.active
	active_types = state.types[sides, active_slots]
	present = active_types >= 0
	safe_types = jnp.maximum(active_types, 0)
	immune = jnp.any(present & tables.weather_residual_immune[weather, safe_types], axis=1)
	can_damage = callback & ((weather == SAND) | (weather == HAIL))
	can_damage &= ~immune & (state.hp[sides, active_slots] > 0)
	maximum = state.stats[sides, active_slots, 0].astype(jnp.int32)
	damage = jnp.where(can_damage, jnp.maximum(maximum // 16, 1), 0)
	old_hp = state.hp[sides, active_slots].astype(jnp.int32)
	new_hp = jnp.maximum(old_hp - damage, 0)
	hp = state.hp.at[sides, active_slots].set(new_hp.astype(state.hp.dtype))

	newly_dead = (old_hp > 0) & (new_hp == 0) & can_damage
	alive = jnp.any(hp > 0, axis=1)
	double_end = ~alive[0] & ~alive[1]
	last_faint = jnp.where(
		newly_dead[0] & newly_dead[1],
		1 - first,
		jnp.where(newly_dead[0], 0, 1),
	)
	winner = jnp.where(double_end, last_faint, jnp.where(~alive[0], 1, 0))
	ended = callback & ~jnp.all(alive) & (state.winner == ONGOING)
	return state._replace(
		hp=hp,
		weather=new_weather.astype(state.weather.dtype),
		weather_turns=new_turns.astype(state.weather_turns.dtype),
		winner=jnp.where(ended, winner, state.winner).astype(state.winner.dtype),
	)
