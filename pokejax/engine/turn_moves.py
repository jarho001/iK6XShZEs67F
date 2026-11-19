"""Generic protection, charge, recharge, lock and trapping move state."""

import jax.numpy as jnp

from .abilities import LEVITATE, NO_GUARD, apply_boost_delta, has_hook
from .actions import SIDES
from .move_data import read_move
from .randomness import LOCK_DURATION, STALL_SUCCESS, TRAP_DURATION, integer
from .state import (
	CURSE,
	ENDURE,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	MUST_RECHARGE,
	NUM_VOLATILES,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	PROTECT,
	QUICK_GUARD,
	SMACKDOWN,
	STALL,
	STALL_USED,
	TEAM_SLOTS,
	TWO_TURN_MOVE,
	WIDE_GUARD,
)


def set_active_volatile(state, side, column, value):
	"""Write one active-only volatile through a dense two-side mask."""
	columns = jnp.arange(NUM_VOLATILES)
	mask = (SIDES == side)[:, None] & (columns[None, :] == column)
	value = jnp.asarray(value, state.volatiles.dtype)
	volatiles = jnp.where(mask, value, state.volatiles)
	return state._replace(volatiles=volatiles)


def target_move_guards(state, side, move, tables, abilities):
	"""Protect, Quick Guard, Wide Guard and semi-invulnerability checks."""
	row = read_move(tables, move)
	foe = 1 - side
	user_types = state.types[side, state.active[side]]
	curse_self = (row.move_volatile_status == 5) & ~jnp.any(user_types == tables.ghost_type)
	targeted = row.move_target_foe & ~curse_self
	protects = row.move_protects
	priority = row.move_priority > 0
	guarded = (
		targeted
		& protects
		& (
			(state.volatiles[foe, PROTECT] > 0)
			| ((state.volatiles[foe, QUICK_GUARD] > 0) & priority)
			| ((state.volatiles[foe, WIDE_GUARD] > 0) & row.move_spread)
		)
	)
	charge = jnp.maximum(state.volatiles[foe, TWO_TURN_MOVE], 0)
	semi = tables.move_charge_semi[charge].astype(jnp.int32)
	invulnerable = targeted & (semi > 0) & ~row.move_hits_semi[semi]
	return guarded, invulnerable & ~jnp.any(has_hook(abilities, NO_GUARD))


def try_stalling_move(state, side, move, enabled, will_act, key, forced, tables):
	"""Resolve the Gen 5 shared Protect-family counter and guard volatile."""
	row = read_move(tables, move)
	kind = row.move_stall_kind
	counter = state.volatiles[side, STALL].astype(jnp.int32)
	# The engine initializes the stall counter at two and doubles it to 256.
	# A power-of-two denominator has exactly the same low-bit remainder.
	mask = jnp.maximum(counter - 1, 0).astype(jnp.uint32)
	bits = key[STALL_SUCCESS, side, 0]
	sample = jnp.where(counter >= 256, bits == 0, (bits & mask) == 0)
	forced_value = forced[STALL_SUCCESS, side, 0]
	passed = jnp.where(forced_value >= 0, forced_value != 0, sample)
	succeeded = enabled & will_act & (kind > 0) & ((counter == 0) | passed)
	failed = enabled & will_act & (kind > 0) & ~succeeded
	next_counter = jnp.where(counter == 0, 2, jnp.minimum(counter * 2, 256))
	guard_column = jnp.where(
		kind == 1,
		PROTECT,
		jnp.where(kind == 2, ENDURE, jnp.where(kind == 3, QUICK_GUARD, WIDE_GUARD)),
	)
	columns = jnp.arange(NUM_VOLATILES)
	side_mask = SIDES == side
	guard_mask = columns == guard_column
	volatiles = state.volatiles
	volatiles = jnp.where(
		side_mask[:, None] & guard_mask[None, :] & succeeded,
		jnp.asarray(1, volatiles.dtype),
		volatiles,
	)
	volatiles = jnp.where(
		side_mask[:, None] & (columns[None, :] == STALL),
		jnp.where(succeeded, next_counter, counter).astype(volatiles.dtype),
		volatiles,
	)
	volatiles = jnp.where(
		side_mask[:, None] & (columns[None, :] == STALL_USED),
		jnp.where(
			succeeded,
			jnp.asarray(1, volatiles.dtype),
			jnp.where(failed, 0, state.volatiles[:, STALL_USED]),
		).astype(volatiles.dtype)[:, None],
		volatiles,
	)
	return state._replace(volatiles=volatiles), succeeded


def clear_broken_guards(state, side, enabled):
	"""Feint and Shadow Force break active protection, but retain stall history."""
	foe = 1 - side
	volatiles = state.volatiles
	columns = jnp.arange(NUM_VOLATILES)
	guard_columns = (columns == PROTECT) | (columns == QUICK_GUARD) | (columns == WIDE_GUARD)
	mask = (SIDES == foe)[:, None] & guard_columns[None, :] & enabled
	volatiles = jnp.where(mask, 0, volatiles)
	return state._replace(volatiles=volatiles)


def clear_charge(state, side):
	"""End a two-turn move after its continuation is attempted or interrupted."""
	return set_active_volatile(state, side, TWO_TURN_MOVE, 0)


def clear_recharge(state, side):
	return set_active_volatile(state, side, MUST_RECHARGE, 0)


def clear_partial_trap(state, side, enabled=True):
	"""Clear the active member's partial-trap duration and recorded source."""
	columns = jnp.arange(NUM_VOLATILES)
	clear = (columns == PARTIALLY_TRAPPED) | (columns == PARTIAL_TRAP_SOURCE)
	values = state.volatiles
	values = jnp.where((SIDES == side)[:, None] & clear[None, :] & enabled, 0, values)
	return state._replace(volatiles=values)


def locked_move(state, side):
	return state.volatiles[side, LOCKED_MOVE]


def apply_charge_boosts(state, side, move, enabled, tables, abilities):
	"""Apply a charge-turn stat boost, such as Skull Bash's Defense rise."""
	row = read_move(tables, move)
	delta = row.move_charge_boosts.astype(jnp.int32)
	changed = apply_boost_delta(
		state.boosts, jnp.broadcast_to(delta, state.boosts.shape), abilities, False
	)
	boosts = jnp.where((SIDES == side)[:, None] & enabled, changed, state.boosts)
	return state._replace(boosts=boosts.astype(state.boosts.dtype))


def apply_move_volatiles(state, side, move, hit, key, forced, tables, abilities, connected=None):
	"""Apply table-declared trap, lock and recharge volatiles after a hit.

	Self volatiles (lock, recharge) follow the move connecting, which includes a hit that
	the target's substitute absorbs. Trap and Smackdown need the target itself.
	"""
	connected = hit if connected is None else connected
	row = read_move(tables, move)
	foe = 1 - side
	target = state.active[foe]
	volatiles = state.volatiles
	trap = (row.move_volatile_status == 3) & hit
	trap &= (volatiles[foe, PARTIALLY_TRAPPED] == 0) & (state.hp[foe, target] > 0)
	trap_time = (integer(key, forced, TRAP_DURATION, side, 0, 2) + 5).astype(volatiles.dtype)
	columns = jnp.arange(NUM_VOLATILES)
	trap_mask = (SIDES == foe)[:, None] & trap
	volatiles = jnp.where(trap_mask & (columns[None, :] == PARTIALLY_TRAPPED), trap_time, volatiles)
	trap_source = (state.active[side].astype(volatiles.dtype) + 1).astype(volatiles.dtype)
	volatiles = jnp.where(
		trap_mask & (columns[None, :] == PARTIAL_TRAP_SOURCE), trap_source, volatiles
	)
	user_slot = state.active[side]
	user_types = state.types[side, user_slot]
	ghost_curse = (row.move_volatile_status == 5) & jnp.any(user_types == tables.ghost_type)
	curse_new = ghost_curse & hit & (state.hp[foe, target] > 0) & (volatiles[foe, CURSE] == 0)
	volatiles = jnp.where(
		(SIDES == foe)[:, None] & (columns[None, :] == CURSE) & curse_new,
		jnp.asarray(1, volatiles.dtype),
		volatiles,
	)
	maximum = state.stats[side, user_slot, 0].astype(jnp.int32)
	user_hp = state.hp[side, user_slot].astype(jnp.int32)
	cost = jnp.maximum(maximum // 2, 1)
	user_hp = jnp.maximum(user_hp - jnp.where(curse_new, cost, 0), 0)
	member_mask = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user_slot)
	hp = jnp.where(member_mask, user_hp.astype(state.hp.dtype), state.hp)

	lock_move = (row.move_self_volatile == 1) & connected
	new_lock = lock_move & (volatiles[side, LOCKED_MOVE] == 0)
	continue_lock = lock_move & (volatiles[side, LOCKED_MOVE] == row.id)
	true_duration = (integer(key, forced, LOCK_DURATION, side, 0, 2) + 2).astype(volatiles.dtype)
	lock_mask = (SIDES == side)[:, None] & new_lock
	volatiles = jnp.where(
		lock_mask & (columns[None, :] == LOCKED_MOVE), row.id.astype(volatiles.dtype), volatiles
	)
	volatiles = jnp.where(
		lock_mask & (columns[None, :] == LOCKED_REMAINING),
		jnp.asarray(2, volatiles.dtype),
		volatiles,
	)
	volatiles = jnp.where(
		lock_mask & (columns[None, :] == LOCKED_TRUE_DURATION), true_duration, volatiles
	)
	restart = continue_lock & (volatiles[side, LOCKED_TRUE_DURATION] >= 2)
	volatiles = jnp.where(
		(SIDES == side)[:, None] & restart & (columns[None, :] == LOCKED_REMAINING),
		jnp.asarray(2, volatiles.dtype),
		volatiles,
	)

	recharge = (row.move_self_volatile == 2) & connected
	volatiles = jnp.where(
		(SIDES == side)[:, None] & recharge & (columns[None, :] == MUST_RECHARGE),
		jnp.asarray(1, volatiles.dtype),
		volatiles,
	)
	target_types = state.types[foe, target]
	target_is_flying = jnp.any(target_types == tables.flying_type)
	target_charge = jnp.maximum(volatiles[foe, TWO_TURN_MOVE], 0).astype(jnp.int32)
	target_semi = tables.move_charge_semi[target_charge]
	smackdown = (row.move_volatile_status == 4) & hit
	smackdown &= (state.hp[foe, target] > 0) & (
		target_is_flying | (target_semi == 1) | has_hook(abilities[foe], LEVITATE)
	)
	volatiles = jnp.where(
		(SIDES == foe)[:, None] & smackdown & (columns[None, :] == SMACKDOWN),
		jnp.asarray(1, volatiles.dtype),
		volatiles,
	)
	volatiles = jnp.where(
		(SIDES == foe)[:, None] & smackdown & (columns[None, :] == TWO_TURN_MOVE),
		jnp.asarray(0, volatiles.dtype),
		volatiles,
	)
	return state._replace(hp=hp, volatiles=volatiles)
