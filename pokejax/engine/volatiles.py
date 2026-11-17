"""Active-only volatile columns; zero means absent, confusion stores its timer.

Gen 5 inherits confusion from gen6/conditions.ts and its Start from base.
BeforeMove order is sleep/freeze (10), flinch (8), confusion (3), paralysis (1).
"""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .randomness import (
	CONFUSION_DAMAGE_ROLL,
	CONFUSION_DURATION,
	CONFUSION_HIT,
	chance,
	integer,
)
from .state import ATK, BOOST_ATK, BOOST_DEF, CONFUSION, DEF, FLINCH, ROOST


def confuse(state, side, enabled, key, forced):
	"""addVolatile does not restart existing confusion and rejects zero HP."""
	current = state.volatiles[side, CONFUSION]
	apply = enabled & (current == 0) & (state.hp[side, state.active[side]] > 0)
	duration = integer(key, forced, CONFUSION_DURATION, side, 0, 4) + 2
	return state._replace(
		volatiles=state.volatiles.at[side, CONFUSION].set(
			jnp.where(apply, duration, current).astype(state.volatiles.dtype)
		)
	)


def confusion_damage(state, side, key, forced):
	"""40-power physical self-hit: stages and roll only, no normal damage hooks.

	BattleActions.getConfusionDamage truncates base damage to 16 bits BEFORE
	randomizer, and enforces minimum 1 afterwards. Burn does not halve it.
	"""
	slot = state.active[side]
	attack = scale_stat_stage(
		state.stats[side, slot, ATK].astype(jnp.int32),
		state.boosts[side, BOOST_ATK].astype(jnp.int32),
	)
	defense = scale_stat_stage(
		state.stats[side, slot, DEF].astype(jnp.int32),
		state.boosts[side, BOOST_DEF].astype(jnp.int32),
	)
	level = state.level[side, slot].astype(jnp.int32)
	base = (((2 * level // 5 + 2) * 40 * attack // jnp.maximum(defense, 1)) // 50 + 2) & 65535
	roll = integer(key, forced, CONFUSION_DAMAGE_ROLL, side, 0, 16)
	return jnp.maximum(1, base * (100 - roll) // 100)


def before_move(state, side, enabled, key, forced):
	enabled &= state.volatiles[side, FLINCH] == 0
	counter = state.volatiles[side, CONFUSION]
	confused = enabled & (counter > 0)
	counter = jnp.where(confused, counter - 1, counter)
	self_hit = confused & (counter > 0) & chance(key, forced, CONFUSION_HIT, side, 0, 1, 2)
	slot = state.active[side]
	damage = jnp.where(self_hit, confusion_damage(state, side, key, forced), 0)
	hp = jnp.maximum(state.hp[side, slot].astype(jnp.int32) - damage, 0)
	return state._replace(
		volatiles=state.volatiles.at[side, CONFUSION].set(counter),
		hp=state.hp.at[side, slot].set(hp.astype(state.hp.dtype)),
	), enabled & ~self_hit


def end_turn(state, enabled, expire_roost=None):
	"""Expire flinch and restore Roost's saved types after the turn."""
	if expire_roost is None:
		expire_roost = enabled
	active = state.active
	roost = (state.volatiles[:, ROOST] > 0) & expire_roost
	types = state.types.at[jnp.arange(2), active].set(
		jnp.where(
			roost[:, None],
			state.roost_types[jnp.arange(2), active],
			state.types[jnp.arange(2), active],
		)
	)
	volatiles = state.volatiles.at[:, FLINCH].set(jnp.where(enabled, 0, state.volatiles[:, FLINCH]))
	volatiles = volatiles.at[:, ROOST].set(jnp.where(expire_roost, 0, volatiles[:, ROOST]))
	return state._replace(types=types, volatiles=volatiles)
