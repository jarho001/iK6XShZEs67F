"""Weather, entry hazards and their move-table effects for Gen 5 singles."""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .abilities import (
	WEATHER,
	WEATHER_IMMUNITY,
	active_abilities,
	effective_weather,
	grounded_ability,
	status_immunity,
	suppresses_weather,
)
from .actions import SIDES
from .randomness import WEATHER_CHANGE_ORDER, WEATHER_ORDER, first_side
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
	TWO_TURN_MOVE,
)
from .turn_moves import clear_partial_trap

WEATHER_ACCURACY_ALWAYS = -2
WEATHER_ACCURACY_BASE = -1
WEATHER_DURATION = 5


def effective_move_type(state, move, tables, weather=None):
	"""Return the move-table type after its weather type override."""
	weather_type = tables.move_weather_type[
		move, state.weather if weather is None else weather
	].astype(jnp.int32)
	return jnp.where(weather_type >= 0, weather_type, tables.move_type[move].astype(jnp.int32))


def set_weather(state, weather, enabled, duration, ability=False):
	"""An ability can replace existing finite weather of the same kind in Gen 5."""
	change = enabled & (weather > 0)
	change &= (weather != state.weather) | (ability & (state.weather_turns != 0))
	return state._replace(
		weather=jnp.where(change, weather, state.weather).astype(state.weather.dtype),
		weather_turns=jnp.where(change, duration, state.weather_turns).astype(
			state.weather_turns.dtype
		),
	), change


def apply_move_field_effects(
	state, side, move, enabled, field_hit, hit, key, forced, tables, item_rows
):
	"""Apply weather, hazards, and Rapid Spin through per-move table columns."""
	apply = enabled & field_hit
	weather = tables.move_weather[move].astype(state.weather.dtype)
	# Field.setWeather emits WeatherChange even when the weather field was clear.
	_ = first_side(key, forced, WEATHER_CHANGE_ORDER)
	weather_rock = item_rows.weather_kind[side] == weather
	duration = jnp.where(
		weather_rock, item_rows.weather_duration[side].astype(jnp.int32), WEATHER_DURATION
	)
	state, _ = set_weather(state, weather, apply, duration)

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


def apply_active_switch_hazards(state, slots, enabled, first, tables, rows, weather, entrants=None):
	"""Evaluate each hazard once per side, then commit in switch-in order."""
	types = state.types[SIDES, slots]
	grounded = ~jnp.any(types == tables.flying_type, axis=1) | (state.volatiles[:, SMACKDOWN] > 0)
	grounded &= grounded_ability(state, rows)
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
	if entrants is not None:
		# SwitchIn orders handlers by cached Speed before suborder. A faster
		# entrant sets its weather after its own hazards and before the slower
		# entrant's Toxic Spikes status check.
		first_weather = entrants[first, WEATHER]
		first_sets = enabled[first] & (after[first] > 0) & (first_weather > 0)
		later_weather = jnp.where(first_sets, first_weather, state.weather)
		alive_later = jnp.where(first_side, after > 0, before > 0)
		later_weather = jnp.where(suppresses_weather(entrants, alive_later), 0, later_weather)
		weather = jnp.where(first_side | ~jnp.all(enabled), weather, later_weather)
	immune |= status_immunity(rows, inflicted, weather)
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
	"""Apply one incoming member's hazards without changing its team slot."""
	slots = jnp.where(SIDES == side, slot, state.active)
	rows = tables.ability_hooks[state.ability[SIDES, slots]]
	return apply_active_switch_hazards(
		state, slots, (SIDES == side) & enabled, side, tables, rows, effective_weather(state, rows)
	)


def weather_residual(state, enabled, key, forced, tables, rows=None, speed=None):
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

	if rows is None:
		rows = active_abilities(state, tables)
	if speed is None:
		speed = scale_stat_stage(
			state.stats[sides, state.active, SPE].astype(jnp.int32),
			state.boosts[:, BOOST_SPE].astype(jnp.int32),
		)
		speed = jnp.where(state.status[sides, state.active] == PARALYSIS, speed // 4, speed)
	weather = effective_weather(state, rows)
	tie_order = first_side(key, forced, WEATHER_ORDER)
	first = jnp.where(speed[0] == speed[1], tie_order, jnp.where(speed[0] > speed[1], 0, 1))

	active_slots = state.active
	active_types = state.types[sides, active_slots]
	present = active_types >= 0
	safe_types = jnp.maximum(active_types, 0)
	immune = jnp.any(present & tables.weather_residual_immune[weather, safe_types], axis=1)
	immune |= (rows[:, WEATHER_IMMUNITY] & (1 << weather)) != 0
	# Dig and Dive's condition onImmunity callbacks block sand and hail.
	semi = tables.move_charge_semi[jnp.maximum(state.volatiles[:, TWO_TURN_MOVE], 0)]
	immune |= (semi == 2) | (semi == 3)
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
