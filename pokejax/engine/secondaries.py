"""Declarative secondary rows, applied once after the hit sequence."""

import jax.numpy as jnp

from pokejax.tables import SECONDARY_SLOTS

from .randomness import MAX_OCCURRENCES, SECONDARY, SECONDARY_STATUS, chance, integer
from .state import FLINCH
from .status import inflict_status
from .volatiles import confuse


def apply_secondaries(state, side, move, first_hit, later_hits, key, forced, tables):
	"""Roll first-hit rows and combine later Twineedle poison chances once."""
	later_forced = forced[SECONDARY, side, 2:MAX_OCCURRENCES:SECONDARY_SLOTS]
	later_bits = key[SECONDARY, side, 2:MAX_OCCURRENCES:SECONDARY_SLOTS]
	later_passed = jnp.where(
		later_forced >= 0,
		later_forced != 0,
		(later_bits % 100) < tables.move_secondary_chance[move, 0],
	)
	any_later = jnp.any(later_passed & later_hits)
	for row in range(SECONDARY_SLOTS):
		chance_ = tables.move_secondary_chance[move, row]
		required_species = tables.move_secondary_user_species[move, row]
		eligible = (required_species == 0) | (
			state.species[side, state.active[side]] == required_species
		)
		first_pass = first_hit & chance(key, forced, SECONDARY, side, row, chance_, 100)
		apply = eligible & (first_pass | (any_later & (row == 0)))
		target = jnp.where(tables.move_secondary_self[move, row], side, 1 - side)
		slot = state.active[target]
		choices = tables.move_secondary_status_choices[move, row]
		choice = integer(key, forced, SECONDARY_STATUS, side, row, 3)
		status = jnp.where(
			jnp.any(choices != 0), choices[choice], tables.move_secondary_status[move, row]
		)
		state = inflict_status(state, target, status, apply, key, forced, tables)
		boost = jnp.clip(
			state.boosts[target].astype(jnp.int32) + tables.move_secondary_boosts[move, row], -6, 6
		).astype(state.boosts.dtype)
		state = state._replace(
			boosts=state.boosts.at[target].set(
				jnp.where(apply & (state.hp[target, slot] > 0), boost, state.boosts[target])
			)
		)
		volatile = tables.move_secondary_volatile[move, row]
		state = confuse(state, target, apply & (volatile == 1), key, forced)
		flinch = apply & (volatile == 2) & (state.hp[target, slot] > 0)
		state = state._replace(
			volatiles=state.volatiles.at[target, FLINCH].set(
				jnp.where(flinch, 1, state.volatiles[target, FLINCH])
			)
		)
	return state
