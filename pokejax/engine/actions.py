"""Nine actions per side, with stable fixed team-slot identities.

0..3 use move slots 0..3. If no move has PP, only move action 0 is legal
and means Struggle. 4..8 switch to the k-th non-active slot (k = action - 4),
listing non-active fixed team slots in ascending order. NONE uses action 0
as a no-op. Finished battles behave as NONE for both sides.
"""

import jax.numpy as jnp

from .items import CHOICE
from .state import (
	CHOICE_MOVE,
	LOCKED_MOVE,
	MOVE,
	MUST_RECHARGE,
	NONE,
	ONGOING,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	SWITCH,
	TWO_TURN_MOVE,
)

NUM_ACTIONS = 9
SIDES = jnp.arange(2)


def bench_slots(active):
	"""Map the five switch choices to fixed slots, skipping the active slot."""
	choices = jnp.arange(5)
	return choices + (choices >= active[..., None])


def legal_actions(state, engine_tables):
	"""Match Pokemon.getMoveRequestData and Side.chooseSwitch without effects."""
	request = jnp.where(state.winner == ONGOING, state.request, NONE)
	moves = state.moves[SIDES, state.active]
	pp = state.pp[SIDES, state.active]
	usable = (moves != 0) & (pp > 0)
	usable = usable.at[:, 0].set(usable[:, 0] | ~jnp.any(usable, axis=1))
	choice_move = state.volatiles[:, CHOICE_MOVE]
	item = state.item[SIDES, state.active].astype(jnp.int32)
	choice_item = engine_tables.item_hooks.rows[item, CHOICE] != 0
	choice_active = choice_item & (choice_move > 0)
	choice_active &= jnp.any(moves == choice_move[:, None], axis=1)
	choice_usable = (moves == choice_move[:, None]) & (pp > 0)
	choice_has_pp = jnp.any(choice_usable, axis=1)
	choice_mask = jnp.where(choice_has_pp[:, None], choice_usable, jnp.arange(4)[None, :] == 0)
	forced = (
		(state.volatiles[:, TWO_TURN_MOVE] > 0)
		| (state.volatiles[:, LOCKED_MOVE] > 0)
		| (state.volatiles[:, MUST_RECHARGE] > 0)
	)
	only_first = jnp.arange(4)[None, :] == 0
	move_mask = (request[:, None] == MOVE) & jnp.where(forced[:, None], only_first, usable)
	move_mask = jnp.where(
		(request[:, None] == MOVE) & choice_active[:, None] & ~forced[:, None],
		choice_mask,
		move_mask,
	)
	move_mask = move_mask.at[:, 0].set(move_mask[:, 0] | (request == NONE))
	bench_alive = state.hp[SIDES[:, None], bench_slots(state.active)] > 0
	can_switch = (request == SWITCH) | ((request == MOVE) & ~forced)
	source_slot = state.volatiles[:, PARTIAL_TRAP_SOURCE].astype(jnp.int32) - 1
	foe = 1 - SIDES
	source_active = source_slot == state.active[foe]
	source_alive = state.hp[foe, jnp.clip(source_slot, 0, 5)] > 0
	trapped = (state.volatiles[:, PARTIALLY_TRAPPED] > 0) & source_active & source_alive
	can_switch &= ~trapped
	can_switch |= request == SWITCH
	switch_mask = can_switch[:, None] & bench_alive
	return jnp.concatenate((move_mask, switch_mask), axis=1)
