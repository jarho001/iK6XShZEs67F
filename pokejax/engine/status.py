"""Major status effects, move prevention and status move primitives."""

import jax.numpy as jnp

from .actions import SIDES
from .field import effective_move_type
from .randomness import FULL_PARALYSIS, SLEEP_DURATION, THAW, chance, integer
from .state import (
	FREEZE,
	HEALTHY,
	MUST_RECHARGE,
	PARALYSIS,
	SLEEP,
	TEAM_SLOTS,
	TOXIC,
	TWO_TURN_MOVE,
)
from .turn_moves import set_active_volatile
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


def before_move(state, side, move, enabled, key, forced, tables, recharge=False):
	started = enabled
	was_charging = state.volatiles[side, TWO_TURN_MOVE] > 0
	recharge_now = enabled & recharge
	state = set_active_volatile(
		state,
		side,
		MUST_RECHARGE,
		jnp.where(recharge_now, 0, state.volatiles[side, MUST_RECHARGE]),
	)
	enabled &= ~recharge_now
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
	member_mask = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == slot)
	state = state._replace(
		status=jnp.where(member_mask & clear, HEALTHY, state.status),
		status_counter=jnp.where(member_mask, jnp.where(clear, 0, counter), state.status_counter),
	)
	enabled &= ~(sleeping & ~wake) & ~frozen
	state, enabled = volatile_before_move(state, side, enabled, key, forced)
	paralyzed = enabled & (status == PARALYSIS) & chance(key, forced, FULL_PARALYSIS, side, 0, 1, 4)
	can_act = enabled & ~paralyzed
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(was_charging & started & ~can_act, 0, state.volatiles[side, TWO_TURN_MOVE]),
	)
	return state, can_act


def inflict_status(state, side, inflicted, enabled, key, forced, tables):
	"""Pokemon.trySetStatus: HP, current status, type immunity, then Sleep Clause."""
	slot = state.active[side]
	types = state.types[side, slot]
	immune = jnp.any((types >= 0) & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	immune |= tables.weather_status_immune[state.weather, inflicted]
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
	"""Apply move status effects with dense team masks and one field write."""
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	effect = tables.move_effect[move]
	user_status = state.status[side, user]
	target_status = state.status[foe, target]
	team_slots = jnp.arange(TEAM_SLOTS)
	user_mask = (SIDES == side)[:, None] & (team_slots == user)[None, :]
	target_mask = (SIDES == foe)[:, None] & (team_slots == target)[None, :]
	# ModifyMove cures a defrost move's user; DamagingHit thaws its target later,
	# AFTER secondaries (BattleActions.spreadMoveHit, Gen 5+ event order).
	self_thaw = enabled & (user_status == FREEZE) & tables.move_defrost[move]
	status = jnp.where(user_mask & self_thaw, HEALTHY, state.status)
	user_status = jnp.where(self_thaw, HEALTHY, user_status)
	transfer = effect == TRANSFER
	inflicted = jnp.where(transfer, user_status, tables.move_status[move])
	types = state.types[foe, target]
	immune = jnp.any((types >= 0) & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	immune |= tables.weather_status_immune[state.weather, inflicted]
	clause = jnp.any(
		(state.status[foe] == SLEEP) & (state.sleep_from_foe[foe] != 0) & (state.hp[foe] > 0)
	)
	inflict = enabled & hit & ~tables.move_status_self[move]
	inflict &= (state.hp[foe, target] > 0) & (inflicted != HEALTHY)
	inflict &= (target_status == HEALTHY) & ~immune & ~((inflicted == SLEEP) & clause)
	duration = (integer(key, forced, SLEEP_DURATION, foe, 0, 3) + 2).astype(jnp.int8)
	target_status_after = jnp.where(inflict, inflicted, target_status)
	status = jnp.where(target_mask & inflict, inflicted, status)
	counter = jnp.where(
		target_mask & inflict,
		jnp.where(inflicted == SLEEP, duration, 0),
		state.status_counter,
	)
	start = jnp.where(target_mask & inflict & (inflicted == SLEEP), duration, state.sleep_start)
	source = jnp.where(target_mask & inflict, inflicted == SLEEP, state.sleep_from_foe)
	clear_transfer = transfer & (target_status == HEALTHY) & (target_status_after != HEALTHY)
	refresh = enabled & (effect == REFRESH) & (user_status != SLEEP) & (user_status != FREEZE)
	party = enabled & (effect == PARTY_CURE)
	clear_team = party & (state.hp[side] > 0)
	clear_team_mask = (SIDES == side)[:, None] & clear_team[None, :]
	status = jnp.where(clear_team_mask, HEALTHY, status)
	counter = jnp.where(clear_team_mask, 0, counter)
	clear_user = clear_transfer | refresh
	status = jnp.where(user_mask & clear_user, HEALTHY, status)
	counter = jnp.where(user_mask & clear_user, 0, counter)
	rest = (
		enabled
		& (effect == REST)
		& (user_status != SLEEP)
		& (state.hp[side, user] < state.stats[side, user, 0])
	)
	status = jnp.where(user_mask & rest, SLEEP, status)
	counter = jnp.where(user_mask & rest, 3, counter)
	start = jnp.where(user_mask & rest, 3, start)
	source = jnp.where(user_mask & rest, False, source)
	hp = jnp.where(user_mask & rest, state.stats[..., 0], state.hp)
	return state._replace(
		status=status,
		status_counter=counter,
		sleep_start=start,
		sleep_from_foe=source,
		hp=hp,
	)


def after_hit(state, side, move, enabled, tables):
	foe = 1 - side
	slot = state.active[foe]
	status = state.status[foe, slot]
	thaw = enabled & (state.hp[foe, slot] > 0) & (status == FREEZE)
	thaw &= (
		effective_move_type(state, move, tables) == tables.fire_type
	) | tables.move_thaws_target[move]
	target_mask = (SIDES == foe)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == slot)
	return state._replace(status=jnp.where(target_mask & thaw, HEALTHY, state.status))
