"""Major status effects, move prevention and status move primitives."""

import jax.numpy as jnp

from .randomness import FULL_PARALYSIS, SLEEP_DURATION, THAW, chance, integer
from .state import FREEZE, HEALTHY, PARALYSIS, SLEEP, TOXIC
from .volatiles import before_move as volatile_before_move

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
	can_defrost = tables.move_defrost[move]
	thaw = enabled & (status == FREEZE) & ~can_defrost & chance(key, forced, THAW, side, 0, 1, 5)
	frozen = enabled & (status == FREEZE) & ~thaw & ~can_defrost
	clear = wake | thaw
	state = state._replace(
		status=state.status.at[side, slot].set(jnp.where(clear, HEALTHY, status)),
		status_counter=state.status_counter.at[side, slot].set(jnp.where(clear, 0, counter)),
	)
	enabled &= ~(sleeping & ~wake) & ~frozen
	state, enabled = volatile_before_move(state, side, enabled, key, forced)
	paralyzed = enabled & (status == PARALYSIS) & chance(key, forced, FULL_PARALYSIS, side, 0, 1, 4)
	return state, enabled & ~paralyzed


def inflict_status(state, side, inflicted, enabled, key, forced, tables):
	"""Pokemon.trySetStatus: HP, current status, type immunity, then Sleep Clause."""
	slot = state.active[side]
	types = state.types[side, slot]
	immune = jnp.any((types >= 0) & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	clause = jnp.any(
		(state.status[side] == SLEEP) & (state.sleep_from_foe[side] != 0) & (state.hp[side] > 0)
	)
	apply = enabled & (state.hp[side, slot] > 0) & (inflicted != HEALTHY)
	apply &= (state.status[side, slot] == HEALTHY) & ~immune & ~((inflicted == SLEEP) & clause)
	duration = (integer(key, forced, SLEEP_DURATION, side, 0, 3) + 2).astype(jnp.int8)
	return state._replace(
		status=state.status.at[side, slot].set(
			jnp.where(apply, inflicted, state.status[side, slot])
		),
		status_counter=state.status_counter.at[side, slot].set(
			jnp.where(
				apply, jnp.where(inflicted == SLEEP, duration, 0), state.status_counter[side, slot]
			)
		),
		sleep_start=state.sleep_start.at[side, slot].set(
			jnp.where(apply & (inflicted == SLEEP), duration, state.sleep_start[side, slot])
		),
		sleep_from_foe=state.sleep_from_foe.at[side, slot].set(
			jnp.where(apply, inflicted == SLEEP, state.sleep_from_foe[side, slot])
		),
	)


def apply_move(state, side, move, enabled, hit, key, forced, tables):
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	effect = tables.move_effect[move]
	user_status = state.status[side, user]
	target_status = state.status[foe, target]
	# ModifyMove cures a defrost move's user; DamagingHit thaws its target later,
	# AFTER secondaries (BattleActions.spreadMoveHit, Gen 5+ event order).
	self_thaw = enabled & (user_status == FREEZE) & tables.move_defrost[move]
	state = state._replace(
		status=state.status.at[side, user].set(jnp.where(self_thaw, HEALTHY, user_status))
	)
	user_status = state.status[side, user]
	target_status = state.status[foe, target]
	transfer = effect == TRANSFER
	inflicted = jnp.where(transfer, user_status, tables.move_status[move])
	state = inflict_status(
		state, foe, inflicted, enabled & hit & ~tables.move_status_self[move], key, forced, tables
	)
	clear_transfer = transfer & (target_status == HEALTHY) & (state.status[foe, target] != HEALTHY)
	status, counter = state.status, state.status_counter
	start, source = state.sleep_start, state.sleep_from_foe
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


def after_hit(state, side, move, enabled, tables):
	foe = 1 - side
	slot = state.active[foe]
	status = state.status[foe, slot]
	thaw = enabled & (state.hp[foe, slot] > 0) & (status == FREEZE)
	thaw &= (tables.move_type[move] == tables.fire_type) | tables.move_thaws_target[move]
	return state._replace(status=state.status.at[foe, slot].set(jnp.where(thaw, HEALTHY, status)))
