"""Major status effects, move prevention and status move primitives."""

import jax.numpy as jnp

from .randomness import FULL_PARALYSIS, SLEEP_DURATION, THAW, chance, integer
from .state import FREEZE, HEALTHY, PARALYSIS, SLEEP, TOXIC

REST, PARTY_CURE, TRANSFER, REFRESH = range(1, 5)


def switch_in(state, side, slot, enabled):
	status = state.status[side, slot]
	counter = state.status_counter[side, slot]
	counter = jnp.where(status == TOXIC, 0, counter)
	counter = jnp.where(status == SLEEP, state.sleep_start[side, slot], counter)
	return state._replace(
		status_counter=state.status_counter.at[side, slot].set(
			jnp.where(enabled, counter, state.status_counter[side, slot])
		)
	)


def before_move(state, side, move, enabled, key, forced, tables):
	slot = state.active[side]
	status = state.status[side, slot]
	counter = state.status_counter[side, slot]
	sleeping = enabled & (status == SLEEP)
	counter = jnp.where(sleeping, counter - 1, counter)
	wake = sleeping & (counter <= 0)
	thaw = enabled & (status == FREEZE) & chance(key, forced, THAW, side, 0, 1, 5)
	can_defrost = tables.move_defrost[move]
	frozen = enabled & (status == FREEZE) & ~thaw & ~can_defrost
	paralyzed = enabled & (status == PARALYSIS) & chance(key, forced, FULL_PARALYSIS, side, 0, 1, 4)
	clear = wake | thaw
	state = state._replace(
		status=state.status.at[side, slot].set(jnp.where(clear, HEALTHY, status)),
		status_counter=state.status_counter.at[side, slot].set(jnp.where(clear, 0, counter)),
	)
	return state, enabled & ~(sleeping & ~wake) & ~frozen & ~paralyzed


def apply_move(state, side, move, enabled, hit, damage_success, damage, key, forced, tables):
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	effect = tables.move_effect[move]
	user_status = state.status[side, user]
	target_status = state.status[foe, target]
	# Defrost moves cure the user when actually executed; a damaging Fire hit thaws the target.
	self_thaw = enabled & (user_status == FREEZE) & tables.move_defrost[move]
	target_thaw = (
		enabled
		& damage_success
		& (state.hp[foe, target] > damage)
		& (target_status == FREEZE)
		& ((tables.move_type[move] == tables.fire_type) | tables.move_thaws_target[move])
	)
	state = state._replace(
		status=state.status.at[side, user]
		.set(jnp.where(self_thaw, HEALTHY, user_status))
		.at[foe, target]
		.set(jnp.where(target_thaw, HEALTHY, target_status))
	)
	user_status = state.status[side, user]
	target_status = state.status[foe, target]
	transfer = effect == TRANSFER
	inflicted = jnp.where(transfer, user_status, tables.move_status[move])
	types = state.types[foe, target]
	immune = jnp.any((types >= 0) & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	clause = jnp.any(
		(state.status[foe] == SLEEP) & (state.sleep_from_foe[foe] != 0) & (state.hp[foe] > 0)
	)
	set_foe = enabled & hit & ~tables.move_status_self[move]
	set_foe &= (state.hp[foe, target] > 0) & (inflicted != HEALTHY)
	set_foe &= (target_status == HEALTHY) & ~immune & ~((inflicted == SLEEP) & clause)
	duration = (integer(key, forced, SLEEP_DURATION, foe, 0, 3) + 2).astype(jnp.int8)
	status = state.status.at[foe, target].set(jnp.where(set_foe, inflicted, target_status))
	counter = state.status_counter.at[foe, target].set(
		jnp.where(
			set_foe, jnp.where(inflicted == SLEEP, duration, 0), state.status_counter[foe, target]
		)
	)
	start = state.sleep_start.at[foe, target].set(
		jnp.where(set_foe & (inflicted == SLEEP), duration, state.sleep_start[foe, target])
	)
	source = state.sleep_from_foe.at[foe, target].set(
		jnp.where(set_foe, inflicted == SLEEP, state.sleep_from_foe[foe, target])
	)
	clear_transfer = set_foe & transfer
	refresh = enabled & (effect == REFRESH) & (user_status != SLEEP) & (user_status != FREEZE)
	party = enabled & (effect == PARTY_CURE)
	clear_team = party & (state.hp[side] > 0)
	status = status.at[side].set(jnp.where(clear_team, HEALTHY, status[side]))
	status = status.at[side, user].set(
		jnp.where(clear_transfer | refresh, HEALTHY, status[side, user])
	)
	counter = counter.at[side].set(jnp.where(clear_team, 0, counter[side]))
	counter = counter.at[side, user].set(
		jnp.where(clear_transfer | refresh, 0, counter[side, user])
	)
	rest = (
		enabled
		& (effect == REST)
		& (user_status != SLEEP)
		& (state.hp[side, user] < state.stats[side, user, 0])
	)
	status = status.at[side, user].set(jnp.where(rest, SLEEP, status[side, user]))
	counter = counter.at[side, user].set(jnp.where(rest, 3, counter[side, user]))
	start = start.at[side, user].set(jnp.where(rest, 3, start[side, user]))
	source = source.at[side, user].set(jnp.where(rest, False, source[side, user]))
	hp = state.hp.at[side, user].set(
		jnp.where(rest, state.stats[side, user, 0], state.hp[side, user])
	)
	return state._replace(
		status=status, status_counter=counter, sleep_start=start, sleep_from_foe=source, hp=hp
	)
