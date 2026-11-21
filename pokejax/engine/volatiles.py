"""Active-only volatile columns; zero means absent, confusion stores its timer.

Gen 5 inherits confusion from gen6/conditions.ts and its Start from base.
BeforeMove order is sleep/freeze (10), flinch (8), confusion (3), paralysis (1).
"""

import jax.numpy as jnp

from pokejax.mechanics.stat_stage import scale_stat_stage

from .abilities import reveal_abilities
from .actions import SIDES
from .events import CANT_CONFUSION, CANT_FLINCH, CANT_NONE, first_cut
from .items import DAMAGE_TAKEN_KIND, damage_survival, update_item_state
from .randomness import (
	CONFUSION_DAMAGE_ROLL,
	CONFUSION_DURATION,
	CONFUSION_HIT,
	chance,
	integer,
)
from .state import (
	ATK,
	BOOST_ATK,
	BOOST_DEF,
	CONFUSION,
	DEF,
	ENDURE,
	FLINCH,
	NUM_VOLATILES,
	PROTECT,
	QUICK_GUARD,
	ROOST,
	STALL,
	STALL_USED,
	TEAM_SLOTS,
	TWO_TURN_MOVE,
	WIDE_GUARD,
)


def confuse(state, side, enabled, key, forced):
	"""addVolatile does not restart existing confusion and rejects zero HP."""
	sides_enabled = (SIDES == side) & enabled
	return confuse_sides(state, sides_enabled, key, forced)


def confuse_sides(state, enabled, key, forced):
	"""Apply confusion to both sides through one dense active-volatile write."""
	current = state.volatiles[:, CONFUSION]
	apply = enabled & (current == 0) & (state.hp[SIDES, state.active] > 0)
	duration = integer(key, forced, CONFUSION_DURATION, SIDES, 0, 4) + 2
	columns = jnp.arange(state.volatiles.shape[1])
	mask = apply[:, None] & (columns[None, :] == CONFUSION)
	return state._replace(
		volatiles=jnp.where(mask, duration[:, None], state.volatiles).astype(state.volatiles.dtype)
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


def advance_active_turns(state, enabled):
	"""Advance the hidden active-turn marker when the turn number advances."""
	active = state.active
	slots = jnp.arange(state.active_turns.shape[1])
	active_mask = slots[None, :] == active[:, None]
	active_hp = state.hp[jnp.arange(2), active] > 0
	values = state.active_turns.astype(jnp.int16)
	values = jnp.where(
		active_mask & enabled & active_hp[:, None], jnp.minimum(values + 1, 255), values
	)
	return state._replace(active_turns=values.astype(state.active_turns.dtype))


def before_move(state, side, enabled, key, forced, tables, ability_row):
	"""Flinch, then confusion's self-hit. Returns (state, enabled, cant reason)."""
	before = enabled
	enabled &= state.volatiles[side, FLINCH] == 0
	reason = first_cut(CANT_NONE, before, enabled, CANT_FLINCH)
	counter = state.volatiles[side, CONFUSION]
	confused = enabled & (counter > 0)
	counter = jnp.where(confused, counter - 1, counter)
	self_hit = confused & (counter > 0) & chance(key, forced, CONFUSION_HIT, side, 0, 1, 2)
	slot = state.active[side]
	damage = jnp.where(self_hit, confusion_damage(state, side, key, forced), 0)
	old_hp = state.hp[side, slot].astype(jnp.int32)
	maximum = state.stats[side, slot, 0].astype(jnp.int32)
	item = state.item[side, slot].astype(jnp.int32)
	item_kind = tables.item_hooks.rows[item, DAMAGE_TAKEN_KIND]
	lethal = self_hit & (damage >= old_hp)
	endure = state.volatiles[side, ENDURE] > 0
	sturdy, sash = damage_survival(lethal & (old_hp == maximum), endure, ability_row, item_kind)
	hp = jnp.where(sturdy | sash | (lethal & endure), 1, jnp.maximum(old_hp - damage, 0))
	consume = (SIDES == side) & sash
	state = update_item_state(state, consume, jnp.zeros((2,), jnp.bool_), tables)
	state = reveal_abilities(state, (SIDES == side) & sturdy)
	active_slot = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == slot)
	columns = jnp.arange(NUM_VOLATILES)
	confusion_column = (SIDES == side)[:, None] & (columns[None, :] == CONFUSION)
	after = enabled & ~self_hit
	reason = first_cut(reason, enabled, after, CANT_CONFUSION)
	return (
		state._replace(
			volatiles=jnp.where(confusion_column, counter, state.volatiles),
			hp=jnp.where(active_slot, hp.astype(state.hp.dtype), state.hp),
		),
		after,
		reason,
	)


def end_turn(state, enabled, key, forced, expire_roost=None):
	"""Expire turn guards, flinch and Roost restoration."""
	del key, forced
	if expire_roost is None:
		expire_roost = enabled
	active = state.active
	enabled_sides = jnp.broadcast_to(enabled, (2,))
	expire_roost_sides = jnp.broadcast_to(expire_roost, (2,))
	roost = (state.volatiles[:, ROOST] > 0) & expire_roost_sides
	active_slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	types = jnp.where((active_slots & roost[:, None])[:, :, None], state.roost_types, state.types)
	columns = jnp.arange(NUM_VOLATILES)
	turn_columns = (
		(columns == FLINCH)
		| (columns == PROTECT)
		| (columns == ENDURE)
		| (columns == QUICK_GUARD)
		| (columns == WIDE_GUARD)
	)
	clear = enabled_sides[:, None] & turn_columns[None, :]
	clear |= expire_roost_sides[:, None] & (columns[None, :] == ROOST)
	clear |= (enabled_sides & (state.volatiles[:, STALL_USED] == 0))[:, None] & (
		columns[None, :] == STALL
	)
	clear |= enabled_sides[:, None] & (columns[None, :] == STALL_USED)
	clear |= (enabled_sides & (state.volatiles[:, TWO_TURN_MOVE] < 0))[:, None] & (
		columns[None, :] == TWO_TURN_MOVE
	)
	volatiles = jnp.where(clear, 0, state.volatiles)
	state = state._replace(volatiles=volatiles)
	return state._replace(types=types)
