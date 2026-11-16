"""Nine actions per side, with stable fixed team-slot identities.

0..3 use move slots 0..3. If no move has PP, only move action 0 is legal
and means Struggle. 4..8 switch to the k-th non-active slot (k = action - 4),
listing non-active fixed team slots in ascending order. NONE uses action 0
as a no-op. Finished battles behave as NONE for both sides.
"""

import jax.numpy as jnp

from .state import MOVE, NONE, ONGOING, SWITCH

NUM_ACTIONS = 9
SIDES = jnp.arange(2)


def bench_slots(active):
	"""Map the five switch choices to fixed slots, skipping the active slot."""
	choices = jnp.arange(5)
	return choices + (choices >= active[..., None])


def legal_actions(state, engine_tables):
	"""Match Pokemon.getMoveRequestData and Side.chooseSwitch without effects."""
	del engine_tables
	request = jnp.where(state.winner == ONGOING, state.request, NONE)
	moves = state.moves[SIDES, state.active]
	pp = state.pp[SIDES, state.active]
	usable = (moves != 0) & (pp > 0)
	usable = usable.at[:, 0].set(usable[:, 0] | ~jnp.any(usable, axis=1))
	move_mask = (request[:, None] == MOVE) & usable
	move_mask = move_mask.at[:, 0].set(move_mask[:, 0] | (request == NONE))
	bench_alive = state.hp[SIDES[:, None], bench_slots(state.active)] > 0
	switch_mask = ((request == MOVE) | (request == SWITCH))[:, None] & bench_alive
	return jnp.concatenate((move_mask, switch_mask), axis=1)
