"""Declarative secondary rows, applied after damage and primary/self effects."""

import jax.numpy as jnp

from pokejax.tables import SECONDARY_SLOTS

from .randomness import SECONDARY, SECONDARY_STATUS, chance, integer
from .state import FLINCH
from .status import inflict_status
from .volatiles import confuse


def apply_secondaries(state, side, move, enabled, key, forced, tables):
	t = tables
	for row in range(SECONDARY_SLOTS):
		chance_ = t.move_secondary_chance[move, row]
		required_species = t.move_secondary_user_species[move, row]
		eligible = (required_species == 0) | (
			state.species[side, state.active[side]] == required_species
		)
		apply = enabled & eligible & chance(key, forced, SECONDARY, side, row, chance_, 100)
		target = jnp.where(t.move_secondary_self[move, row], side, 1 - side)
		slot = state.active[target]
		choices = t.move_secondary_status_choices[move, row]
		choice = integer(key, forced, SECONDARY_STATUS, side, row, 3)
		status = jnp.where(
			jnp.any(choices != 0), choices[choice], t.move_secondary_status[move, row]
		)
		state = inflict_status(state, target, status, apply, key, forced, t)
		# boost() checks hp; a self secondary still works when the foe has fainted.
		boost = jnp.clip(
			state.boosts[target].astype(jnp.int32) + t.move_secondary_boosts[move, row], -6, 6
		).astype(state.boosts.dtype)
		state = state._replace(
			boosts=state.boosts.at[target].set(
				jnp.where(apply & (state.hp[target, slot] > 0), boost, state.boosts[target])
			)
		)
		volatile = t.move_secondary_volatile[move, row]
		state = confuse(state, target, apply & (volatile == 1), key, forced)
		flinch = apply & (volatile == 2) & (state.hp[target, slot] > 0)
		state = state._replace(
			volatiles=state.volatiles.at[target, FLINCH].set(
				jnp.where(flinch, 1, state.volatiles[target, FLINCH])
			)
		)
	return state
