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
	by_side = targets[:, None] == SIDES
	active = state.active
	active_hp = state.hp[SIDES, active]
	active_status = state.status[SIDES, active]
	target_hp = active_hp[targets]

	choices = tables.move_secondary_status_choices[move]
	choice = integer(key, forced, SECONDARY_STATUS, side, rows, 3)
	inflicted = jnp.where(
		jnp.any(choices != 0, axis=1), choices[rows, choice], tables.move_secondary_status[move]
	)
	types = state.types[SIDES, active][targets]
	present = types >= 0
	immune = jnp.any(
		present & tables.status_immune[inflicted[:, None], jnp.maximum(types, 0)], axis=1
	)
	immune |= tables.weather_status_immune[state.weather, inflicted]
	sleep_clause = jnp.any(
		(state.status == SLEEP) & (state.sleep_from_foe != 0) & (state.hp > 0),
		axis=1,
	)
	can_inflict = apply & (target_hp > 0) & (inflicted != HEALTHY)
	can_inflict &= (active_status[targets] == HEALTHY) & ~immune
	can_inflict &= ~((inflicted == SLEEP) & sleep_clause[targets])
	row0, row1 = can_inflict
	row1 &= ~(row0 & (targets[0] == targets[1]))
	status_effects = jnp.stack((row0, row1))[:, None] & by_side
	inflicted_by_side = jnp.where(status_effects[0], inflicted[0], inflicted[1])
	set_status = jnp.any(status_effects, axis=0)
	set_sleep = set_status & (inflicted_by_side == SLEEP)
	duration = (integer(key, forced, SLEEP_DURATION, SIDES, 0, 3) + 2).astype(jnp.int8)
	member_mask = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	status_mask = member_mask & set_status[:, None]
	status_array = jnp.where(status_mask, inflicted_by_side[:, None], state.status)
	counter_array = jnp.where(
		status_mask, jnp.where(set_sleep, duration, 0)[:, None], state.status_counter
	)
	start_array = jnp.where(member_mask & set_sleep[:, None], duration[:, None], state.sleep_start)
	source_array = jnp.where(status_mask, set_sleep[:, None], state.sleep_from_foe)

	boosts = state.boosts
	hit_effects = (apply & (target_hp > 0))[:, None] & by_side
	for row in range(SECONDARY_SLOTS):
		delta = jnp.where(hit_effects[row, :, None], tables.move_secondary_boosts[move, row], 0)
		boosts = jnp.clip(boosts + delta, -6, 6)

	volatile_kind = tables.move_secondary_volatile[move]
	confusion = jnp.any(hit_effects & (volatile_kind == 1)[:, None], axis=0)
	confusion &= state.volatiles[:, CONFUSION] == 0
	confusion_duration = (integer(key, forced, CONFUSION_DURATION, SIDES, 0, 4) + 2).astype(
		state.volatiles.dtype
	)
	flinch = jnp.any(hit_effects & (volatile_kind == 2)[:, None], axis=0)
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(
		confusion[:, None] & (columns == CONFUSION), confusion_duration[:, None], state.volatiles
	)
	volatiles = jnp.where(flinch[:, None] & (columns == FLINCH), 1, volatiles)

	return state._replace(
		status=status_array,
		status_counter=counter_array,
		sleep_start=start_array,
		sleep_from_foe=source_array,
		boosts=boosts,
		volatiles=volatiles,
	)
