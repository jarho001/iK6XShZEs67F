"""Nine actions per side, with stable fixed team-slot identities.

0..3 use move slots 0..3. If no move has PP, only move action 0 is legal
and means Struggle. 4..8 switch to the k-th non-active slot (k = action - 4),
listing non-active fixed team slots in ascending order. NONE uses action 0
as a no-op. Finished battles behave as NONE for both sides.
"""

import jax.numpy as jnp

from .abilities import (
	LEVITATE,
	TRAP,
	TRAP_ARENA,
	TRAP_MAGNET,
	TRAP_SHADOW,
	active_abilities,
	has_hook,
)
from .items import CHOICE
from .restrictions import restrict_legal_moves
from .state import (
	CHOICE_MOVE,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	LOCKED_MOVE,
	MOVE,
	MUST_RECHARGE,
	NONE,
	ONGOING,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	SMACKDOWN,
	SWITCH,
	TWO_TURN_MOVE,
)

NUM_ACTIONS = 9
MOVE_ACTIONS = 4
SIDES = jnp.arange(2)


def bench_slots(active):
	"""Map the five switch choices to fixed slots, skipping the active slot."""
	choices = jnp.arange(5)
	return choices + (choices >= active[..., None])


def move_availability(state, engine_tables):
	"""Resolve move-slot usability and Choice lock for selection and execution."""
	moves = state.moves[SIDES, state.active]
	pp = state.pp[SIDES, state.active]
	usable = (moves != 0) & (pp > 0)
	usable &= (state.volatiles[:, DISABLE_REMAINING] == 0)[:, None] | (
		moves != state.volatiles[:, DISABLE_MOVE, None]
	)
	usable = restrict_legal_moves(state, moves, usable, engine_tables)
	choice_move = state.volatiles[:, CHOICE_MOVE]
	item = state.item[SIDES, state.active].astype(jnp.int32)
	choice_item = engine_tables.item_hooks.rows[item, CHOICE] != 0
	matches_choice = moves == choice_move[:, None]
	choice_active = choice_item & (choice_move > 0) & jnp.any(matches_choice, axis=1)
	choice_usable = matches_choice & usable
	return moves, usable, choice_active, choice_usable


def hidden_traps(state, engine_tables):
	"""[2] flags: the foe's active member traps this side's active now, whatever the request says.

	Arena Trap traps a grounded member, Shadow Tag any member but a Shadow Tag user, and Magnet Pull
	a Steel type. It reads the current abilities, so a trapper that stops trapping (Mummy gives it
	another ability) stops at once.
	"""
	foe = 1 - SIDES
	abilities = active_abilities(state, engine_tables)
	foe_ability = abilities[foe, TRAP]
	types = state.types[SIDES, state.active]
	smackdown = state.volatiles[:, SMACKDOWN] > 0
	grounded = ~jnp.any(types == engine_tables.flying_type, axis=-1) | smackdown
	grounded &= ~has_hook(abilities, LEVITATE) | smackdown
	steel = jnp.any(types == engine_tables.steel_type, axis=-1)
	foe_alive = state.hp[foe, state.active[foe]] > 0
	shadow_user = abilities[:, TRAP] == TRAP_SHADOW
	return foe_alive & (
		((foe_ability == TRAP_ARENA) & grounded)
		| ((foe_ability == TRAP_SHADOW) & ~shadow_user)
		| ((foe_ability == TRAP_MAGNET) & steel)
	)


def action_masks(state, engine_tables):
	"""Match Pokemon.getMoveRequestData and Side.chooseSwitch without effects.

	Returns the [2, 9] mask that the request describes (everything except hidden traps), and
	[2] flags for the sides whose voluntary switches a hidden trap forbids: Shadow Tag, Arena
	Trap and Magnet Pull trap without any protocol line, because the request only says
	maybeTrapped (whenever the foe's species can have such an ability). A player cannot see
	them, so legal_actions applies them and observed_actions does not, until the side has
	been refused a switch (BattleState.trap_known).
	"""
	request = jnp.where(state.winner == ONGOING, state.request, NONE)
	_, usable, choice_active, choice_usable = move_availability(state, engine_tables)
	only_first = jnp.arange(4)[None, :] == 0
	usable |= only_first & ~jnp.any(usable, axis=1)[:, None]
	choice_has_pp = jnp.any(choice_usable, axis=1)
	choice_mask = jnp.where(choice_has_pp[:, None], choice_usable, only_first)
	forced = (
		(state.volatiles[:, TWO_TURN_MOVE] > 0)
		| (state.volatiles[:, LOCKED_MOVE] > 0)
		| (state.volatiles[:, MUST_RECHARGE] > 0)
	)
	move_mask = (request[:, None] == MOVE) & jnp.where(forced[:, None], only_first, usable)
	move_mask = jnp.where(
		(request[:, None] == MOVE) & choice_active[:, None] & ~forced[:, None],
		choice_mask,
		move_mask,
	)
	move_mask |= only_first & (request == NONE)[:, None]
	bench_alive = state.hp[SIDES[:, None], bench_slots(state.active)] > 0
	can_switch = (request == SWITCH) | ((request == MOVE) & ~forced)
	source_slot = state.volatiles[:, PARTIAL_TRAP_SOURCE].astype(jnp.int32) - 1
	foe = 1 - SIDES
	source_active = source_slot == state.active[foe]
	source_alive = state.hp[foe, state.active[foe]] > 0
	trapped = (state.volatiles[:, PARTIALLY_TRAPPED] > 0) & source_active & source_alive
	can_switch &= ~trapped
	trapped_by_ability = hidden_traps(state, engine_tables)
	# A forced switch is never refused: only a move request can be trapped.
	hidden_trap = trapped_by_ability & (request == MOVE)
	can_switch |= request == SWITCH
	switch_mask = can_switch[:, None] & bench_alive
	return jnp.concatenate((move_mask, switch_mask), axis=1), hidden_trap


def block_switches(mask, blocked):
	"""Remove the five switch actions where blocked; mask [..., 9], blocked [...]."""
	return mask & ~(blocked[..., None] & (jnp.arange(NUM_ACTIONS) >= MOVE_ACTIONS))


def legal_actions(state, engine_tables, hidden_traps=True):
	"""The engine's legal mask [2, 9]; hidden_traps=False leaves Shadow Tag and the like out."""
	mask, hidden_trap = action_masks(state, engine_tables)
	return block_switches(mask, hidden_trap) if hidden_traps else mask


def observed_actions(state, engine_tables):
	"""The mask a player's request shows: a hidden trap only once its side has found out."""
	mask, _ = action_masks(state, engine_tables)
	request = jnp.where(state.winner == ONGOING, state.request, NONE)
	return block_switches(mask, state.trap_known & (request == MOVE))
