"""Sleep Talk: a sleeping user calls one of its other moves at random.

Showdown picks uniformly among the moveset's filled slots that lack the nosleeptalk
flag (this includes moves with no PP and excludes two-turn charge moves). The call
spends the called move's PP, not Sleep Talk's second use; a called move with no PP
fails with no effect.
"""

import jax.numpy as jnp

from .randomness import SLEEP_TALK, integer


def callable_slots(state, side, tables):
	"""Moveset slots that Sleep Talk may call, in slot order."""
	user = state.active[side]
	moves = state.moves[side, user].astype(jnp.int32)
	blocked = (tables.move_no_sleep_talk[moves] != 0) | (tables.move_charge[moves] != 0)
	return (moves != 0) & ~blocked


def choose_called_slot(state, side, tables, key, forced):
	"""Return (any, slot): whether a move can be called, and the sampled moveset slot."""
	eligible = callable_slots(state, side, tables)
	count = jnp.sum(eligible.astype(jnp.int32))
	pick = integer(key, forced, SLEEP_TALK, side, 0, jnp.maximum(count, 1))
	cumulative = jnp.cumsum(eligible.astype(jnp.int32))
	chosen = eligible & (cumulative == pick + 1)
	return count > 0, jnp.argmax(chosen).astype(jnp.int32)
