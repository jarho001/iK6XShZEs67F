"""Declarative secondary rows, applied once after the hit sequence."""

import jax.numpy as jnp

from pokejax.tables import SECONDARY_SLOTS

from .actions import SIDES
from .randomness import (
	CONFUSION_DURATION,
	MAX_OCCURRENCES,
	SECONDARY,
	SECONDARY_STATUS,
	SLEEP_DURATION,
	chance,
	integer,
)
from .state import CONFUSION, FLINCH, HEALTHY, SLEEP, TEAM_SLOTS


def apply_secondaries(state, side, move, first_hit, later_hits, key, forced, tables):
	"""Roll both secondary rows together, then write active effects by side."""
	rows = jnp.arange(SECONDARY_SLOTS)
	later_forced = forced[SECONDARY, side, 2:MAX_OCCURRENCES:SECONDARY_SLOTS]
	later_bits = key[SECONDARY, side, 2:MAX_OCCURRENCES:SECONDARY_SLOTS]
	later_passed = jnp.where(
		later_forced >= 0,
		later_forced != 0,
		(later_bits % 100) < tables.move_secondary_chance[move, 0],
	)
	any_later = jnp.any(later_passed & later_hits)
	chance_ = tables.move_secondary_chance[move]
	required_species = tables.move_secondary_user_species[move]
	user_species = state.species[side, state.active[side]]
	eligible = (required_species == 0) | (user_species == required_species)
	first_pass = first_hit & chance(key, forced, SECONDARY, side, rows, chance_, 100)
	apply = eligible & (first_pass | ((rows == 0) & any_later))
	targets = jnp.where(tables.move_secondary_self[move], side, 1 - side)
	slots = state.active[targets]
	target_hp = state.hp[targets, slots]

	choices = tables.move_secondary_status_choices[move]
	choice = integer(key, forced, SECONDARY_STATUS, side, rows, 3)
	inflicted = jnp.where(
		jnp.any(choices != 0, axis=1), choices[rows, choice], tables.move_secondary_status[move]
	)
	types = state.types[targets, slots]
	present = types >= 0
	immune = jnp.any(
		present & tables.status_immune[inflicted[:, None], jnp.maximum(types, 0)], axis=1
	)
	immune |= tables.weather_status_immune[state.weather, inflicted]
	sleep_clause = jnp.any(
		(state.status[targets] == SLEEP)
		& (state.sleep_from_foe[targets] != 0)
		& (state.hp[targets] > 0),
		axis=1,
	)
	current_status = state.status[targets, slots]
	can_inflict = apply & (target_hp > 0) & (inflicted != HEALTHY) & (current_status == HEALTHY)
	can_inflict &= ~immune & ~((inflicted == SLEEP) & sleep_clause)
	same_target = targets[0] == targets[1]
	row1_status = jnp.where(same_target & can_inflict[0], inflicted[0], current_status[1])
	can_inflict = can_inflict.at[1].set(can_inflict[1] & (row1_status == HEALTHY))
	durations = (integer(key, forced, SLEEP_DURATION, targets, 0, 3) + 2).astype(jnp.int8)
	active = state.active
	active_status = state.status[SIDES, active]
	active_counter = state.status_counter[SIDES, active]
	active_sleep_start = state.sleep_start[SIDES, active]
	active_sleep_source = state.sleep_from_foe[SIDES, active]
	status_by_side = active_status
	counter_by_side = active_counter
	start_by_side = active_sleep_start
	source_by_side = active_sleep_source
	for row in range(SECONDARY_SLOTS):
		by_side = targets[row] == SIDES
		set_status = can_inflict[row] & by_side
		status_by_side = jnp.where(set_status, inflicted[row], status_by_side)
		counter_value = jnp.where(inflicted[row] == SLEEP, durations[row], 0)
		counter_by_side = jnp.where(set_status, counter_value, counter_by_side)
		start_by_side = jnp.where(
			set_status & (inflicted[row] == SLEEP), durations[row], start_by_side
		)
		source_by_side = jnp.where(set_status, inflicted[row] == SLEEP, source_by_side)
	member_mask = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	status_array = jnp.where(member_mask, status_by_side[:, None], state.status)
	counter_array = jnp.where(member_mask, counter_by_side[:, None], state.status_counter)
	start_array = jnp.where(member_mask, start_by_side[:, None], state.sleep_start)
	source_array = jnp.where(member_mask, source_by_side[:, None], state.sleep_from_foe)

	boosts = state.boosts
	volatiles = state.volatiles
	volatile_kind = tables.move_secondary_volatile[move]
	for row in range(SECONDARY_SLOTS):
		by_side = targets[row] == SIDES
		boost_enabled = apply[row] & (target_hp[row] > 0) & by_side
		boost_delta = jnp.where(
			by_side[:, None], tables.move_secondary_boosts[move, row], 0
		).astype(jnp.int32)
		boosts = jnp.clip(
			boosts.astype(jnp.int32) + jnp.where(boost_enabled[:, None], boost_delta, 0), -6, 6
		).astype(state.boosts.dtype)
		confusion = apply[row] & (volatile_kind[row] == 1) & by_side
		current_confusion = volatiles[:, CONFUSION]
		confusion_duration = (integer(key, forced, CONFUSION_DURATION, SIDES, 0, 4) + 2).astype(
			volatiles.dtype
		)
		new_confusion = (current_confusion == 0) & (state.hp[SIDES, active] > 0) & confusion
		volatiles = volatiles.at[:, CONFUSION].set(
			jnp.where(new_confusion, confusion_duration, current_confusion)
		)
		flinch = apply[row] & (volatile_kind[row] == 2) & by_side
		flinch &= state.hp[SIDES, active] > 0
		volatiles = volatiles.at[:, FLINCH].set(
			jnp.where(flinch, jnp.asarray(1, volatiles.dtype), volatiles[:, FLINCH])
		)

	return state._replace(
		status=status_array,
		status_counter=counter_array,
		sleep_start=start_array,
		sleep_from_foe=source_array,
		boosts=boosts,
		volatiles=volatiles,
	)
