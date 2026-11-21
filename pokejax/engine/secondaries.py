"""Declarative secondary rows, applied once after the hit sequence."""

import jax.numpy as jnp

from pokejax.tables import SECONDARY_SLOTS

from .abilities import (
	BOOST_EVENT,
	BOOST_EVENT_DEFIANT,
	INNER_FOCUS,
	OWN_TEMPO,
	SERENE_GRACE,
	SHIELD_DUST,
	STATUS_EVENT,
	STATUS_EVENT_SYNCHRONIZE,
	apply_boost_delta,
	berry_allowed,
	boost_delta,
	has_hook,
	reveal_abilities,
	status_immunity,
)
from .actions import SIDES
from .events import EV_STATUS, write_event
from .items import BERRY_LUM
from .move_data import read_move
from .randomness import (
	CONFUSION_DURATION,
	MAX_OCCURRENCES,
	SECONDARY,
	SECONDARY_STATUS,
	SLEEP_DURATION,
	address_bits,
	chance,
	integer,
)
from .state import BURN, CONFUSION, FLINCH, HEALTHY, PARALYSIS, POISON, SLEEP, TEAM_SLOTS, TOXIC
from .status import synchronize_status


def secondary_chances(tables, move, source):
	row = read_move(tables, move)
	return row.move_secondary_chance * jnp.where(has_hook(source, SERENE_GRACE), 2, 1)


def apply_secondaries(
	state,
	side,
	move,
	first_hit,
	later_hits,
	key,
	forced,
	tables,
	abilities,
	weather,
	sheer,
	items,
	first_sub=False,
	later_sub=False,
):
	"""Roll both secondary rows together, then write active effects by side.

	first_sub and later_sub mark hits that landed on a substitute. Showdown still rolls the
	secondaries for them (the target is null), so the user's own effects (a self boost such as
	Flame Charge) apply while the target's effects do not.
	"""
	row = read_move(tables, move)
	rows = jnp.arange(SECONDARY_SLOTS)
	later_forced = (
		jnp.full((len(range(2, MAX_OCCURRENCES, SECONDARY_SLOTS)),), -1, jnp.int8)
		if forced is None
		else forced[SECONDARY, side, 2:MAX_OCCURRENCES:SECONDARY_SLOTS]
	)
	later_bits = address_bits(key, SECONDARY, side, jnp.arange(2, MAX_OCCURRENCES, SECONDARY_SLOTS))
	chance_ = secondary_chances(tables, row, abilities[side])
	later_passed = jnp.where(
		later_forced >= 0,
		later_forced != 0,
		(later_bits % 100) < chance_[0],
	)
	any_later = jnp.any(later_passed & later_hits)
	any_later_self = jnp.any(later_passed & (later_hits | later_sub))
	required_species = row.move_secondary_user_species
	user_species = state.species[side, state.active[side]]
	eligible = (required_species == 0) | (user_species == required_species)
	rolled = chance(key, forced, SECONDARY, side, rows, chance_, 100)
	first_pass = first_hit & rolled
	self_effect = row.move_secondary_self
	apply_target = first_pass | ((rows == 0) & any_later)
	apply_self = ((first_hit | first_sub) & rolled) | ((rows == 0) & any_later_self)
	apply = eligible & jnp.where(self_effect, apply_self, apply_target)
	apply &= ~sheer & (self_effect | ~has_hook(abilities[1 - side], SHIELD_DUST))
	targets = jnp.where(self_effect, side, 1 - side)
	by_side = targets[:, None] == SIDES
	active = state.active
	active_hp = state.hp[SIDES, active]
	active_status = state.status[SIDES, active]
	target_hp = active_hp[targets]

	choices = row.move_secondary_status_choices
	choice = integer(key, forced, SECONDARY_STATUS, side, rows, 3)
	inflicted = jnp.where(
		jnp.any(choices != 0, axis=1), choices[rows, choice], row.move_secondary_status
	)
	types = state.types[SIDES, active][targets]
	present = types >= 0
	type_immune = jnp.any(
		present & tables.status_immune[inflicted[:, None], jnp.maximum(types, 0)], axis=1
	)
	weather_immune = tables.weather_status_immune[weather, inflicted]
	ability_immune = status_immunity(abilities[targets], inflicted, weather)
	immune = type_immune | weather_immune | ability_immune
	sleep_clause = jnp.any(
		(state.status == SLEEP) & (state.sleep_from_foe != 0) & (state.hp > 0),
		axis=1,
	)
	can_inflict = apply & (target_hp > 0) & (inflicted != HEALTHY)
	can_inflict &= (active_status[targets] == HEALTHY) & ~immune
	can_inflict &= ~((inflicted == SLEEP) & sleep_clause[targets])
	# Poison from the first hit can use Lum before the next hit rolls its
	# own secondary. A landed later hit also proves the target survived hit one.
	foe = 1 - side
	lum_between_hits = first_pass[0] & apply[0] & jnp.any(later_hits)
	lum_between_hits &= (targets[0] == foe) & (inflicted[0] == POISON)
	lum_between_hits &= (active_status[foe] == HEALTHY) & ~immune[0]
	lum_between_hits &= (items.berry_kind[foe] == BERRY_LUM) & berry_allowed(state, abilities)[foe]
	can_inflict &= ~((rows == 0) & lum_between_hits & ~any_later)
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
	defiant_revealed = jnp.zeros((2,), jnp.bool_)
	hit_effects = (apply & (target_hp > 0))[:, None] & by_side
	for index in range(SECONDARY_SLOTS):
		delta = jnp.where(hit_effects[index, :, None], row.move_secondary_boosts[index], 0)
		actual_delta = boost_delta(delta, abilities, SIDES != side)
		before_attack = boosts[:, 0]
		boosts = apply_boost_delta(boosts, delta, abilities, SIDES != side)
		defiant = abilities[:, BOOST_EVENT] == BOOST_EVENT_DEFIANT
		defiant &= (SIDES != side) & jnp.any(actual_delta < 0, axis=-1)
		defiant_revealed |= defiant & (boosts[:, 0] > before_attack)

	volatile_kind = row.move_secondary_volatile
	confusion = jnp.any(hit_effects & (volatile_kind == 1)[:, None], axis=0)
	confusion &= ~has_hook(abilities, OWN_TEMPO)
	confusion &= state.volatiles[:, CONFUSION] == 0
	confusion_duration = (integer(key, forced, CONFUSION_DURATION, SIDES, 0, 4) + 2).astype(
		state.volatiles.dtype
	)
	flinch = jnp.any(hit_effects & (volatile_kind == 2)[:, None], axis=0) & ~has_hook(
		abilities, INNER_FOCUS
	)
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(
		confusion[:, None] & (columns == CONFUSION), confusion_duration[:, None], state.volatiles
	)
	volatiles = jnp.where(flinch[:, None] & (columns == FLINCH), 1, volatiles)

	state = state._replace(
		status=status_array,
		status_counter=counter_array,
		sleep_start=start_array,
		sleep_from_foe=source_array,
		boosts=boosts,
		volatiles=volatiles,
	)
	state = write_event(state, set_status, EV_STATUS, inflicted_by_side)
	# A Synchronize holder copies a secondary status back to the user; the user's immunity can
	# refuse the copy, which Showdown reports as -immune on the user (events.py FAILED, IMMUNE).
	user_slot = state.active[side]
	user_healthy = state.status[side, user_slot] == HEALTHY
	user_types = state.types[side, user_slot]
	sync_refused = jnp.zeros((), jnp.bool_)
	sync_busy = jnp.zeros((), jnp.bool_)
	for index in range(SECONDARY_SLOTS):
		sync_trigger = can_inflict[index] & (targets[index] != side)
		inflicted_index = inflicted[index]
		copy_immune = jnp.any(
			(user_types >= 0) & tables.status_immune[inflicted_index, jnp.maximum(user_types, 0)]
		)
		copy_immune |= tables.weather_status_immune[weather, inflicted_index] | status_immunity(
			abilities[side], inflicted_index, weather
		)
		statusable = (
			(inflicted_index == BURN) | (inflicted_index == POISON) | (inflicted_index == TOXIC)
		)
		statusable |= inflicted_index == PARALYSIS
		holder = abilities[targets[index], STATUS_EVENT] == STATUS_EVENT_SYNCHRONIZE
		sync_refused |= sync_trigger & statusable & holder & user_healthy & copy_immune
		sync_busy |= sync_trigger & statusable & holder & ~user_healthy
		state = synchronize_status(
			state,
			targets[index],
			side,
			inflicted[index],
			sync_trigger,
			key,
			forced,
			tables,
			abilities,
		)
	state = reveal_abilities(state, defiant_revealed)
	return state, (SIDES == foe) & lum_between_hits, sync_refused | sync_busy, sync_refused
