"""Illusion: the disguise of a switched-in holder, its break, and what the opponent is shown.

Showdown: onBeforeSwitchIn makes the holder's disguise the last living member to its right
(none when the holder is the rightmost living member). A damaging hit on the holder ends the
disguise and names the real species (the 'replace' message); a switch-out or a faint clears it
without a message, so the opponent keeps the disguise species it was shown.
"""

import jax.numpy as jnp

from .abilities import ILLUSION, has_hook, reveal_abilities
from .actions import SIDES
from .forms import member_mask
from .state import ILLUSION_SLOT, TEAM_SLOTS


def enter_illusion(state, entering, tables):
	"""Set the disguise of each entering Illusion holder (slot plus one; zero for none)."""
	active = state.active
	ranks = jnp.arange(TEAM_SLOTS)
	rows = tables.ability_hooks[state.ability[SIDES, active].astype(jnp.int32)]
	holder = entering & has_hook(rows, ILLUSION)
	# BeforeSwitchIn runs after the swap, so the holder is party index 0 and the disguise is the
	# last living member at a higher party index (Showdown's side.pokemon order).
	party = state.party.astype(jnp.int32)
	living = jnp.take_along_axis(state.hp, party, axis=1) > 0
	rightmost_index = jnp.max(jnp.where(living & (ranks[None, :] > 0), ranks[None, :], -1), axis=1)
	disguised = holder & (rightmost_index >= 0)
	rightmost = jnp.take_along_axis(
		party, jnp.clip(rightmost_index, 0, TEAM_SLOTS - 1)[:, None], axis=1
	)[:, 0]
	columns = jnp.arange(state.volatiles.shape[1])
	disguise = jnp.where(disguised, rightmost + 1, 0)
	volatiles = jnp.where(
		entering[:, None] & (columns[None, :] == ILLUSION_SLOT),
		disguise[:, None].astype(state.volatiles.dtype),
		state.volatiles,
	)
	shown_slot = jnp.where(disguised, rightmost, active)
	clipped = jnp.clip(shown_slot, 0, TEAM_SLOTS - 1)
	shown = state.species[SIDES, clipped]
	shown_level = state.level[SIDES, clipped]
	entered = member_mask(state, entering)
	return state._replace(
		volatiles=volatiles,
		shown_species=jnp.where(
			entered, shown[:, None].astype(state.shown_species.dtype), state.shown_species
		),
		shown_level=jnp.where(
			entered, shown_level[:, None].astype(state.shown_level.dtype), state.shown_level
		),
	)


def break_illusion(state, broken, tables):
	"""A damaging hit ends the disguise: the real species is shown and Illusion is revealed."""
	del tables
	on = broken & (state.volatiles[:, ILLUSION_SLOT] > 0)
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(
		on[:, None] & (columns[None, :] == ILLUSION_SLOT), 0, state.volatiles
	).astype(state.volatiles.dtype)
	real = state.species[SIDES, state.active]
	real_level = state.level[SIDES, state.active]
	revealed = member_mask(state, on)
	shown = jnp.where(revealed, real[:, None], state.shown_species)
	shown_level = jnp.where(revealed, real_level[:, None], state.shown_level)
	state = state._replace(
		volatiles=volatiles,
		shown_species=shown.astype(state.shown_species.dtype),
		shown_level=shown_level.astype(state.shown_level.dtype),
	)
	return reveal_abilities(state, on)
