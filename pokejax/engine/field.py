"""Weather, entry hazards and their move-table effects for Gen 5 singles."""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .actions import SIDES
from .items import apply_active_stat_modifiers
from .randomness import SWITCH_HANDLER_ORDER, WEATHER_CHANGE_ORDER, WEATHER_ORDER, first_side
from .state import (
	BOOST_SPE,
	HAIL,
	HEALTHY,
	ONGOING,
	PARALYSIS,
	POISON,
	SAND,
	SMACKDOWN,
	SPE,
	SPIKES,
	TEAM_SLOTS,
	TOXIC,
	TOXIC_SPIKES,
)
from .turn_moves import clear_partial_trap

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


def apply_move_field_effects(
	state, side, move, enabled, field_hit, hit, key, forced, tables, item_rows
):
	"""Apply weather, hazards, and Rapid Spin through per-move table columns."""
	apply = enabled & field_hit
	weather = tables.move_weather[move].astype(state.weather.dtype)
	weather_change = apply & (weather > 0) & (weather != state.weather)
	# Field.setWeather emits WeatherChange even when the weather field was clear.
	_ = first_side(key, forced, WEATHER_CHANGE_ORDER)
	weather_state = jnp.where(weather_change, weather, state.weather)
	weather_rock = item_rows.weather_kind[side] == weather
	duration = jnp.where(
		weather_rock, item_rows.weather_duration[side].astype(jnp.int32), WEATHER_DURATION
	)
	weather_turns = jnp.where(weather_change, duration, state.weather_turns).astype(
		state.weather_turns.dtype
	)
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
	state = state._replace(
		hazards=hazards.astype(state.hazards.dtype),
		hazard_order=hazard_order.astype(state.hazard_order.dtype),
	)
	return clear_partial_trap(state, side, clear)


def _active_member_mask(slots):
	return jnp.arange(TEAM_SLOTS)[None, :] == slots[:, None]


def _apply_switch_hazards_sides(state, slots, enabled, first, tables):
	"""Evaluate each hazard once per side, then commit in switch-in order."""
	types = state.types[SIDES, slots]
	grounded = ~jnp.any(types == tables.flying_type, axis=1) | (state.volatiles[:, SMACKDOWN] > 0)
	before = state.hp[SIDES, slots].astype(jnp.int32)
	active = enabled & grounded & (before > 0) & (state.winner == ONGOING)
	spikes = state.hazards[:, SPIKES]
	toxic_layers = state.hazards[:, TOXIC_SPIKES]
	toxic_first = (toxic_layers > 0) & (
		state.hazard_order[:, TOXIC_SPIKES] <= state.hazard_order[:, SPIKES]
	)
	maximum = state.stats[SIDES, slots, 0].astype(jnp.int32)
	numerator = jnp.where(spikes == 1, 3, jnp.where(spikes == 2, 4, 6))
	damage = jnp.maximum(maximum * numerator // 24, 1)
	after = jnp.maximum(before - jnp.where(active & (spikes > 0), damage, 0), 0)
	faint = active & (after == 0)
	member_mask = _active_member_mask(slots)
	bench_alive = jnp.any((state.hp > 0) & ~member_mask, axis=1)
	last_member_faints = faint & ~bench_alive

	# A battle-ending Spikes KO stops the other switch-in. Toxic Spikes can
	# still run on the fainting side if its callback precedes Spikes.
	first_side = SIDES == first
	active &= first_side | ~jnp.any(first_side & last_member_faints)
	after = jnp.where(active, after, before)
	toxic_active = active & (toxic_layers > 0) & (toxic_first | (after > 0))
	poison_type = jnp.any(types == tables.poison_type, axis=1)
	absorb = toxic_active & poison_type
	inflicted = jnp.where(toxic_layers >= 2, TOXIC, POISON)
	immune = jnp.any(
		(types >= 0) & tables.status_immune[inflicted[:, None], jnp.maximum(types, 0)], axis=1
	)
	poison = toxic_active & ~poison_type & ~immune & (state.status[SIDES, slots] == HEALTHY)
	poison_slots = member_mask & poison[:, None]
	clear = absorb[:, None] & (jnp.arange(state.hazards.shape[1]) == TOXIC_SPIKES)

	ended = active & last_member_faints
	side_alive = bench_alive | (after > 0)
	loser = jnp.where(ended[0], 0, 1)
	winner = jnp.where(jnp.any(ended & side_alive[::-1]), 1 - loser, loser)
	return state._replace(
		hp=jnp.where(member_mask, after[:, None].astype(state.hp.dtype), state.hp),
		hazards=jnp.where(clear, 0, state.hazards),
		hazard_order=jnp.where(clear, 0, state.hazard_order),
		status=jnp.where(poison_slots, inflicted[:, None].astype(state.status.dtype), state.status),
		status_counter=jnp.where(poison_slots, 0, state.status_counter),
		winner=jnp.where(jnp.any(ended), winner, state.winner).astype(state.winner.dtype),
	)


def apply_switch_hazards(state, side, slot, enabled, tables):
	"""Apply one incoming Pokemon's Spikes and Toxic Spikes callbacks."""
	slots = jnp.where(SIDES == side, slot, state.active)
	enabled_sides = (SIDES == side) & enabled
	return _apply_switch_hazards_sides(state, slots, enabled_sides, side, tables)


def apply_switch_hazards_ordered(state, switchers, enabled, key, forced, item_rows, tables):
	"""Run grouped SwitchIn handlers in Showdown Speed and tie order."""
	sides = SIDES
	speed = scale_stat_stage(
		state.stats[sides, state.active, SPE].astype(jnp.int32),
		state.boosts[:, BOOST_SPE].astype(jnp.int32),
	)
	speed = apply_active_stat_modifiers(speed, SPE, item_rows)
	speed = jnp.where(state.status[sides, state.active] == PARALYSIS, speed // 4, speed)
	tie_first = first_side(key, forced, SWITCH_HANDLER_ORDER)
	first = jnp.where(speed[0] == speed[1], tie_first, jnp.where(speed[0] > speed[1], 0, 1))
	return _apply_switch_hazards_sides(state, state.active, enabled & switchers, first, tables)


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
