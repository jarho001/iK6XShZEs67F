"""Substitute: a holder's volatile HP pool that takes the damage aimed at its holder.

Showdown: the move's onTryHit refuses a second substitute, a user at or below a
quarter of its maximum HP, and 1-HP Shedinja. onHit pays max(1, maxhp / 4) HP, and
the condition starts with floor(maxhp / 4) HP. A hit that reaches a standing
substitute (BattleActions.spreadMoveHit, TryPrimaryHit) takes its damage instead of
the target's, so the target sees no damage, secondary, contact or drain effect.
"""

import jax.numpy as jnp

from .actions import SIDES
from .state import HP, PARTIAL_TRAP_SOURCE, PARTIALLY_TRAPPED, SUBSTITUTE, TEAM_SLOTS

# Move-table code of the self volatile Substitute.
SELF_SUBSTITUTE = 3


def substitute_allowed(state, side):
	"""onTryHit: no standing substitute, the user above a quarter of its HP, and not Shedinja."""
	user = state.active[side]
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	hp = state.hp[side, user].astype(jnp.int32)
	return (state.volatiles[side, SUBSTITUTE] == 0) & (4 * hp > maximum) & (maximum > 1)


def apply_substitute(state, side, enabled):
	"""The Substitute move for one side: HP cost, volatile HP and partial-trap removal."""
	user = state.active[side]
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	hp = state.hp[side, user].astype(jnp.int32)
	allowed = enabled & substitute_allowed(state, side)
	sub_hp = maximum // 4
	cost = jnp.maximum(sub_hp, 1)
	sides = SIDES == side
	member = sides[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	hp_after = jnp.where(member & allowed, (hp - cost).astype(state.hp.dtype), state.hp)
	columns = jnp.arange(state.volatiles.shape[1])
	gain = (sides & allowed)[:, None]
	volatiles = jnp.where(
		gain & (columns[None, :] == SUBSTITUTE),
		sub_hp.astype(state.volatiles.dtype),
		state.volatiles,
	)
	trapped = (columns == PARTIALLY_TRAPPED) | (columns == PARTIAL_TRAP_SOURCE)
	volatiles = jnp.where(gain & trapped[None, :], 0, volatiles)
	return state._replace(hp=hp_after, volatiles=volatiles)


def split_hits(substitute_hp, raw, connects):
	"""Which connecting hits reach a standing substitute, and the damage each one deals.

	A hit reaches the substitute while the raw damage of the earlier connecting hits
	is below its HP; a broken substitute passes later hits on to the target. Each
	absorbed hit's damage is capped at the HP still left. Zero-damage hits count.
	"""
	raw = jnp.where(connects, raw, 0)
	prior = jnp.cumsum(raw) - raw
	absorbed = connects & (prior < substitute_hp)
	dealt = jnp.where(absorbed, jnp.minimum(raw, substitute_hp - prior), 0)
	return absorbed, dealt
