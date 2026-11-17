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
	TEAM_SLOTS,
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


def _resolve_entry_faint(state, enabled_sides):
	alive = jnp.any(state.hp > 0, axis=1)
	ended = enabled_sides & ~alive & (state.winner == ONGOING)
	winners = jnp.where(alive[1 - SIDES], 1 - SIDES, SIDES)
	winner = jnp.where(ended[0], winners[0], winners[1])
	return state._replace(
		winner=jnp.where(jnp.any(ended), winner, state.winner).astype(state.winner.dtype)
	)


def _active_member_mask(slots):
	return jnp.arange(TEAM_SLOTS)[None, :] == slots[:, None]


def _apply_spikes_handler(state, slots, enabled, tables):
	types = state.types[SIDES, slots]
	grounded = ~jnp.any(types == tables.flying_type, axis=1)
	hp_before = state.hp[SIDES, slots].astype(jnp.int32)
	maximum = state.stats[SIDES, slots, 0].astype(jnp.int32)
	spikes = state.hazards[:, SPIKES].astype(jnp.int32)
	damage_numerator = jnp.where(spikes == 1, 3, jnp.where(spikes == 2, 4, 6))
	spike_damage = jnp.maximum(maximum * damage_numerator // 24, 1)
	take_spikes = enabled & grounded & (spikes > 0) & (hp_before > 0)
	hp_after = jnp.maximum(hp_before - jnp.where(take_spikes, spike_damage, 0), 0)
	member_mask = _active_member_mask(slots) & take_spikes[:, None]
	hp = jnp.where(member_mask, hp_after[:, None].astype(state.hp.dtype), state.hp)
	state = state._replace(hp=hp)
	return _resolve_entry_faint(state, take_spikes & (hp_after == 0))


def _apply_toxic_spikes_handler(state, slots, enabled, tables):
	types = state.types[SIDES, slots]
	grounded = ~jnp.any(types == tables.flying_type, axis=1)
	hp = state.hp[SIDES, slots]
	toxic_layers = state.hazards[:, TOXIC_SPIKES].astype(jnp.int32)
	poison_type = jnp.any(types == tables.poison_type, axis=1)
	absorb = enabled & grounded & poison_type & (toxic_layers > 0) & (hp > 0)
	hazards = state.hazards.at[:, TOXIC_SPIKES].set(
		jnp.where(absorb, 0, toxic_layers).astype(state.hazards.dtype)
	)
	status = state.status[SIDES, slots]
	inflicted = jnp.where(toxic_layers >= 2, TOXIC, POISON)
	present = types >= 0
	immune = jnp.any(
		present & tables.status_immune[inflicted[:, None], jnp.maximum(types, 0)], axis=1
	)
	poison = enabled & grounded & (toxic_layers > 0) & ~poison_type & (hp > 0)
	poison &= (status == HEALTHY) & ~immune
	member_mask = _active_member_mask(slots) & poison[:, None]
	status_array = jnp.where(member_mask, inflicted[:, None], state.status)
	counter_array = jnp.where(member_mask, 0, state.status_counter)
	hazard_order = state.hazard_order.at[:, TOXIC_SPIKES].set(
		jnp.where(absorb, 0, state.hazard_order[:, TOXIC_SPIKES])
	)
	return state._replace(
		hazards=hazards,
		hazard_order=hazard_order,
		status=status_array,
		status_counter=counter_array,
	)


def _apply_switch_hazards_sides(state, slots, enabled_sides, tables):
	spikes_order = state.hazard_order[:, SPIKES]
	toxic_order = state.hazard_order[:, TOXIC_SPIKES]
	first_kind = jnp.where(
		(state.hazards[:, SPIKES] > 0)
		& ((state.hazards[:, TOXIC_SPIKES] == 0) | (spikes_order < toxic_order)),
		SPIKES,
		TOXIC_SPIKES,
	)
	active = enabled_sides & (state.hp[SIDES, slots] > 0) & (state.winner == ONGOING)
	state = _apply_spikes_handler(state, slots, active & (first_kind == SPIKES), tables)
	state = _apply_toxic_spikes_handler(state, slots, active & (first_kind == TOXIC_SPIKES), tables)
	active &= (state.hp[SIDES, slots] > 0) & (state.winner == ONGOING)
	state = _apply_spikes_handler(state, slots, active & (first_kind != SPIKES), tables)
	return _apply_toxic_spikes_handler(state, slots, active & (first_kind != TOXIC_SPIKES), tables)


def apply_switch_hazards(state, side, slot, enabled, tables):
	"""Apply one incoming Pokemon's Spikes and Toxic Spikes callbacks."""
	slots = jnp.where(SIDES == side, slot, state.active)
	enabled_sides = (SIDES == side) & enabled
	return _apply_switch_hazards_sides(state, slots, enabled_sides, tables)


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
	first_sides = SIDES == first
	first_enabled = enabled & switchers & first_sides
	second_enabled = enabled & switchers & ~first_sides
	state = _apply_switch_hazards_sides(state, state.active, first_enabled, tables)
	state = _apply_switch_hazards_sides(state, state.active, second_enabled, tables)
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
