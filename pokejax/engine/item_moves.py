"""Item transfer and removal: Trick, Switcheroo and Knock Off.

Showdown's takeItem asks the TakeItem event first: Sticky Hold refuses a take by
another Pokemon and any Knock Off take, and an Arceus refuses to lose its Plate.
Removing an item with Unburden starts the Unburden volatile even if the item is
given back. A Trick or Switcheroo swap clears both Choice locks, because setItem runs
the new item's Start (which removes choicelock). Knock Off removes the item only
after the hit lands on the target. Gen 5 never sets itemKnockedOff (that flag is Gen 4's).
"""

import jax.numpy as jnp

from .abilities import ITEM_GUARD, ITEM_GUARD_STICKY_HOLD, SPEED, SPEED_UNBURDEN
from .actions import SIDES
from .state import CHOICE_MOVE, TEAM_SLOTS, UNBURDEN


def _set_unburden(state, mask):
	"""Unburden's volatile on each side whose held item left it."""
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(mask[:, None] & (columns[None, :] == UNBURDEN), 1, state.volatiles)
	return state._replace(volatiles=volatiles.astype(state.volatiles.dtype))


def _clear_choice_locks(state, sides):
	columns = jnp.arange(state.volatiles.shape[1])
	mask = sides[:, None] & (columns[None, :] == CHOICE_MOVE)
	return state._replace(
		volatiles=jnp.where(mask, 0, state.volatiles).astype(state.volatiles.dtype)
	)


def _take_blocks(state, tables):
	"""Per side: the species that blocks taking the held item, and the held item IDs."""
	held = state.item[SIDES, state.active].astype(jnp.int32)
	species = tables.species_num[state.species[SIDES, state.active].astype(jnp.int32)]
	block = tables.item_take_species[held]
	return held, species, block


def apply_item_swap(state, side, enabled, tables, rows):
	"""Trick and Switcheroo: the user and the target trade held items.

	Fails when a take is refused (an Arceus Plate, or the target's Sticky Hold), or
	when neither Pokemon holds an item. Returns (state, swapped).
	"""
	foe = 1 - side
	held, species, block = _take_blocks(state, tables)
	sticky_foe = rows[foe, ITEM_GUARD] == ITEM_GUARD_STICKY_HOLD
	# The user takes its own item (taker = holder) and the target's item (taker = user).
	own_refused = (held[side] != 0) & (block[side] == species[side])
	foe_refused = (held[foe] != 0) & ((block[foe] == species[foe]) | (block[foe] == species[side]))
	refused = own_refused | foe_refused | ((held[foe] != 0) & sticky_foe)
	empty = (held[side] == 0) & (held[foe] == 0)
	swapped = enabled & ~refused & ~empty
	incoming = jnp.where(SIDES == side, held[foe], held[side])
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	write = slots & swapped
	receives = write & (incoming[:, None] != 0)
	state = state._replace(
		item=jnp.where(write, incoming[:, None].astype(state.item.dtype), state.item),
		item_revealed=state.item_revealed | receives,
		item_public=jnp.where(
			receives, incoming[:, None].astype(state.item_public.dtype), state.item_public
		),
	)
	removed = jnp.broadcast_to(swapped, (2,)) & (held != 0)
	state = _set_unburden(state, removed & (rows[:, SPEED] == SPEED_UNBURDEN))
	state = _clear_choice_locks(state, jnp.broadcast_to(swapped, (2,)))
	return state, swapped


def apply_knock_off(state, side, enabled, tables, rows):
	"""Knock Off after a landed hit: the target loses its item for the rest of the battle."""
	foe = 1 - side
	target = state.active[foe]
	item = state.item[foe, target].astype(jnp.int32)
	species = tables.species_num[state.species[foe, target].astype(jnp.int32)]
	plate = tables.item_take_species[item] == species
	sticky = rows[foe, ITEM_GUARD] == ITEM_GUARD_STICKY_HOLD
	removed = enabled & (item != 0) & ~plate & ~sticky
	sides = (SIDES == foe) & removed
	slots = jnp.arange(TEAM_SLOTS)[None, :] == target
	write = sides[:, None] & slots
	state = state._replace(
		item=jnp.where(write, 0, state.item).astype(state.item.dtype),
		item_revealed=state.item_revealed | write,
		item_public=jnp.where(write, item.astype(state.item_public.dtype), state.item_public),
	)
	return _set_unburden(state, sides & (rows[:, SPEED] == SPEED_UNBURDEN))
