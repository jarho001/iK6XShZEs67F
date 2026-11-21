"""Slot and field effects that resolve later: Wish, Healing Wish, Trick Room, Destiny Bond.

Wish and Healing Wish are slot conditions: they stay with the position, so
the Pokemon in that slot at resolution is the one healed. Trick Room is a pseudo
weather with five turns that Trick Room again ends. Destiny Bond is a user volatile
that lasts until the user's next move attempt.
"""

import jax.numpy as jnp

from .actions import SIDES
from .state import (
	DESTINY_BOND,
	HEALTHY,
	HP,
	NUM_VOLATILES,
	TEAM_SLOTS,
)

# Move-table codes: self volatile 4 is Destiny Bond; slot conditions are 1 (Wish) and 2.
SELF_DESTINY_BOND = 4
SLOT_WISH, SLOT_HEALING_WISH = 1, 2
WISH_TURNS = 2
TRICK_ROOM_TURNS = 5
MAX_AGE = 127


def apply_wish(state, side, enabled):
	"""Wish starts only when the slot has none; the heal is half the user's maximum HP."""
	user = state.active[side]
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	add = enabled & (state.wish_turns[side] == 0)
	sides = SIDES == side
	turns = jnp.where(sides & add, WISH_TURNS, state.wish_turns)
	amount = jnp.where(sides & add, jnp.maximum(maximum // 2, 1), state.wish_hp)
	age = jnp.where(sides & add, 0, state.wish_age)
	return state._replace(
		wish_turns=turns.astype(state.wish_turns.dtype),
		wish_hp=amount.astype(state.wish_hp.dtype),
		wish_age=age.astype(state.wish_age.dtype),
	)


def advance_wish(state, turn_advanced):
	"""The counter drops from two to one when the turn advances (Showdown counts turns).

	The public age (turns since the Wish was used) counts each turn end while a Wish is pending.
	"""
	turns = state.wish_turns
	drop = turn_advanced[None] & (turns == 2)
	pending = turn_advanced[None] & (turns > 0)
	age = state.wish_age.astype(jnp.int32)
	age = jnp.where(pending, jnp.minimum(age + 1, MAX_AGE), age)
	return state._replace(
		wish_turns=jnp.where(drop, 1, turns).astype(turns.dtype),
		wish_age=age.astype(state.wish_age.dtype),
	)


def wish_residual(state, enabled, key, forced, tables, rows):
	"""Wish (order 4): a Wish with one turn left resolves at this residual.

	Showdown's onResidual skips the turn it was used on, then onEnd heals the slot's
	Pokemon if it has not fainted. The counter drops in advance_wish, so mid-turn
	states (a pending replacement) still show two turns.
	"""
	del key, forced, tables, rows
	turns = state.wish_turns.astype(jnp.int32)
	resolve = enabled & (turns == 1)
	left = jnp.where(resolve, 0, turns)
	active = state.active
	hp = state.hp[SIDES, active].astype(jnp.int32)
	maximum = state.stats[SIDES, active, HP].astype(jnp.int32)
	heal = resolve & (hp > 0)
	healed = jnp.minimum(maximum, hp + state.wish_hp.astype(jnp.int32))
	hp = jnp.where(heal, healed, hp)
	slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	return state._replace(
		hp=jnp.where(slots, hp[:, None].astype(state.hp.dtype), state.hp),
		wish_turns=left.astype(state.wish_turns.dtype),
		wish_hp=jnp.where(resolve, 0, state.wish_hp).astype(state.wish_hp.dtype),
		wish_age=jnp.where(resolve, 0, state.wish_age).astype(state.wish_age.dtype),
	)


def trick_room_residual(state, enabled, key, forced, tables, rows):
	"""Trick Room (field order 27): the five turns count down and then end.

	The public age counts the turn ends since it started; it is zero once Trick Room ends.
	"""
	del key, forced, tables, rows
	turns = state.trick_room.astype(jnp.int32)
	tick = enabled & (turns > 0)
	left = jnp.where(tick, turns - 1, turns)
	age = state.trick_room_age.astype(jnp.int32)
	age = jnp.where(tick & (left > 0), jnp.minimum(age + 1, MAX_AGE), age)
	age = jnp.where(left == 0, 0, age)
	return state._replace(
		trick_room=left.astype(state.trick_room.dtype),
		trick_room_age=age.astype(state.trick_room_age.dtype),
	)


def apply_trick_room(state, enabled):
	"""Using Trick Room while it stands ends it; otherwise it starts for five turns.

	Starting and ending both restart the public age at zero (-fieldstart / -fieldend).
	"""
	new = jnp.where(state.trick_room > 0, 0, TRICK_ROOM_TURNS)
	return state._replace(
		trick_room=jnp.where(enabled, new, state.trick_room).astype(state.trick_room.dtype),
		trick_room_age=jnp.where(enabled, 0, state.trick_room_age).astype(
			state.trick_room_age.dtype
		),
	)


def apply_destiny_bond(state, side, enabled):
	"""Destiny Bond on the user; using it again refreshes it."""
	columns = jnp.arange(NUM_VOLATILES)
	mask = (SIDES == side)[:, None] & enabled & (columns[None, :] == DESTINY_BOND)
	return state._replace(
		volatiles=jnp.where(mask, 1, state.volatiles).astype(state.volatiles.dtype)
	)


def faint_active(state, side, enabled):
	"""Faint the side's active Pokemon (Destiny Bond's attacker, or a Healing Wish user)."""
	user = state.active[side]
	member = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	return state._replace(hp=jnp.where(member & enabled, 0, state.hp).astype(state.hp.dtype))


def apply_healing_wish(state, side, enabled):
	"""Healing Wish needs a living replacement and an empty slot; the user then faints.

	Returns (state, placed). The user faints only when the slot condition is placed.
	"""
	user = state.active[side]
	slots = jnp.arange(TEAM_SLOTS)
	bench = (state.hp[side] > 0) & (slots != user)
	placed = enabled & jnp.any(bench) & ~state.healing_wish[side]
	healing = jnp.where((SIDES == side) & placed, True, state.healing_wish)
	return state._replace(healing_wish=healing), placed


def healing_wish_switch_in(state, switch_mask, targets):
	"""A living entrant takes the pending Healing Wish: full HP, no status, slot cleared."""
	consume = switch_mask & state.healing_wish & (state.hp[SIDES, targets] > 0)
	member = consume[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == targets[:, None])
	maximum = state.stats[SIDES, targets, HP]
	hp = jnp.where(member, maximum[:, None].astype(state.hp.dtype), state.hp)
	status = jnp.where(member, HEALTHY, state.status)
	counter = jnp.where(member, 0, state.status_counter)
	return state._replace(
		hp=hp,
		status=status.astype(state.status.dtype),
		status_counter=counter.astype(state.status_counter.dtype),
		healing_wish=state.healing_wish & ~consume,
	)
