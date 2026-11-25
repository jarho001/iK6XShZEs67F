"""The environment's request handling for a single battle, without episode resets."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pokejax.engine.actions import NUM_ACTIONS, action_masks, block_switches
from pokejax.engine.observation import observe
from pokejax.engine.state import MOVE, NONE, NUM_SIDES, ONGOING, TIE, BattleState
from pokejax.engine.step import step
from pokejax.env import TURN_LIMIT, side_reward


class SearchState(NamedTuple):
	state: BattleState
	held: jax.Array
	has_held: jax.Array
	base: jax.Array
	blocked: jax.Array


class Outcome(NamedTuple):
	reward: jax.Array
	rejected: jax.Array
	illegal: jax.Array


def init(state, tables):
	"""Wrap one battle at its initial request, with no choices held yet."""
	base, blocked = action_masks(state, tables)
	return SearchState(
		state, jnp.zeros(NUM_SIDES, jnp.int32), jnp.zeros(NUM_SIDES, jnp.bool_), base, blocked
	)


def observed_mask(position):
	state = position.state
	moving = (state.winner == ONGOING)[..., None] & (state.request == MOVE)
	return block_switches(position.base, state.trap_known & moving)


def observe_both(position, tables):
	return jax.vmap(observe, in_axes=(None, 0, None))(position.state, jnp.arange(NUM_SIDES), tables)


def advance(position, actions, key, forced, tables):
	"""Advance one battle using an engine key. Finished positions are absorbing."""
	state = position.state
	ongoing = state.winner == ONGOING
	actions = jnp.where(position.has_held, position.held, jnp.asarray(actions, jnp.int32))
	legal = block_switches(position.base, position.blocked)
	in_range = (actions >= 0) & (actions < NUM_ACTIONS)
	index = jnp.clip(actions, 0, NUM_ACTIONS - 1)[:, None]
	chosen_legal = in_range & jnp.take_along_axis(legal, index, axis=-1)[:, 0]
	chosen_base = in_range & jnp.take_along_axis(position.base, index, axis=-1)[:, 0]
	rejected = ongoing & ~chosen_legal & chosen_base & position.blocked & ~state.trap_known
	illegal = ongoing & ~chosen_legal & ~rejected
	actions = jnp.where(illegal, jnp.argmax(legal, axis=-1).astype(jnp.int32), actions)
	actions = jnp.where(ongoing & (state.request != NONE), actions, 0)
	refused = rejected.any()

	stepped = step(state, actions, key, forced, tables)
	kept = jax.tree.map(lambda old, new: jnp.where(refused | ~ongoing, old, new), state, stepped)
	known = jnp.where(refused, state.trap_known | rejected, kept.trap_known)
	over = (kept.turn > TURN_LIMIT) & (kept.winner == ONGOING)
	kept = kept._replace(
		trap_known=known, winner=jnp.where(over, TIE, kept.winner).astype(state.winner.dtype)
	)
	base, blocked = action_masks(kept, tables)
	has_held = refused & ~rejected & (kept.winner == ONGOING)
	result = SearchState(kept, jnp.where(has_held, actions, 0), has_held, base, blocked)
	result = jax.tree.map(lambda old, new: jnp.where(ongoing, new, old), position, result)
	reward = side_reward(kept.winner)
	return result, Outcome(jnp.stack((reward, -reward)), rejected, illegal)


def tile(tree, lanes):
	"""Copy the leaves of one position or both of its players' carries to N lanes."""
	return jax.tree.map(lambda x: jnp.broadcast_to(x, (lanes, *x.shape)), tree)


def advance_batch(positions, actions, keys, forced, tables):
	"""Advance independent lanes with native randomness or one forced table per lane."""
	return jax.vmap(advance, in_axes=(0, 0, 0, None if forced is None else 0, None))(
		positions, actions, keys, forced, tables
	)
