"""Each side's active member, gathered once per attack and reused by the damage stages."""

from typing import NamedTuple

import jax.numpy as jnp

from .actions import SIDES
from .state import HP


class ActiveFacts(NamedTuple):
	"""Active-member values read by the hit calculation; they hold until the HP write."""

	slot: jnp.ndarray  # int32 [2]: each side's active member
	hp: jnp.ndarray  # int32 [2]
	maximum: jnp.ndarray  # int32 [2]: stored max HP
	level: jnp.ndarray  # int32 [2]
	status: jnp.ndarray  # int8 [2]
	types: jnp.ndarray  # int8 [2, 2]
	stats: jnp.ndarray  # int32 [2, 6]: HP, then Atk..Spe


def active_facts(state):
	"""Gather the active rows once: one index per side, reused by every stage of the attack."""
	stats = state.stats[SIDES, state.active].astype(jnp.int32)
	return ActiveFacts(
		slot=state.active.astype(jnp.int32),
		hp=state.hp[SIDES, state.active].astype(jnp.int32),
		maximum=stats[:, HP],
		level=state.level[SIDES, state.active].astype(jnp.int32),
		status=state.status[SIDES, state.active],
		types=state.types[SIDES, state.active],
		stats=stats,
	)
