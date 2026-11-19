"""Move-restricting volatiles: Encore, Taunt, Disable, Yawn and Leech Seed.

Showdown applies each volatile to the target after the move's accuracy and
immunity checks, and refuses one that already stands. Durations start at their
Gen 5 values and count down at the residual. The "one more" rules come from
BattleQueue.willMove: a target that still has to move this turn keeps the full
count, and one that has already moved gains a turn (Encore, Taunt, Disable use
the same rule; Disable gets one less because it starts at five).
"""

import jax.numpy as jnp

from .abilities import status_immunity

SIDES = jnp.arange(2)
from .state import (
	DISABLE_MOVE,
	DISABLE_REMAINING,
	ENCORE_MOVE,
	ENCORE_REMAINING,
	HEALTHY,
	LEECH_SEED,
	NUM_VOLATILES,
	SLEEP,
	TAUNT,
	YAWN,
)

# volatile_status codes in the move table (tables.py).
VOLATILE_ENCORE, VOLATILE_TAUNT, VOLATILE_DISABLE, VOLATILE_LEECH_SEED, VOLATILE_YAWN = range(6, 11)
ENCORE_TURNS, TAUNT_TURNS, YAWN_TURNS = 3, 3, 2
DISABLE_TURNS = 5
STATUS_CATEGORY = 2


def _write(volatiles, sides, condition, column, value):
	"""Write one volatile column on the selected sides, where condition holds."""
	columns = jnp.arange(NUM_VOLATILES)
	mask = (sides[:, None] & condition) & (columns[None, :] == column)
	return jnp.where(mask, jnp.asarray(value, volatiles.dtype), volatiles)


def foe_move_usable(state, foe, move):
	"""Showdown getMoveData(move) with pp > 0: the foe's slot for move exists and has PP."""
	target = state.active[foe]
	moves = state.moves[foe, target].astype(jnp.int32)
	pp = state.pp[foe, target]
	return (move != 0) & jnp.any((moves == move) & (pp > 0))


def apply_restrictions(state, side, row, hit, foe_pending, tables, rows, weather):
	"""Encore, Taunt, Disable, Yawn and Leech Seed on the foe, from this move's hit."""
	foe = 1 - side
	target = state.active[foe]
	foe_sides = SIDES == foe
	last = state.last_move[foe, target].astype(jnp.int32)
	usable_last = foe_move_usable(state, foe, last)
	volatiles = state.volatiles
	status = row.move_volatile_status.astype(jnp.int32)
	pending = foe_pending.astype(jnp.int32)

	encore = (
		hit
		& (status == VOLATILE_ENCORE)
		& (volatiles[foe, ENCORE_REMAINING] == 0)
		& usable_last
		& (tables.move_failencore[last] == 0)
	)
	volatiles = _write(volatiles, foe_sides, encore, ENCORE_MOVE, last)
	volatiles = _write(volatiles, foe_sides, encore, ENCORE_REMAINING, ENCORE_TURNS + 1 - pending)

	taunt_bonus = (state.active_turns[foe, target] > 0) & ~foe_pending
	taunt = hit & (status == VOLATILE_TAUNT) & (volatiles[foe, TAUNT] == 0)
	volatiles = _write(
		volatiles, foe_sides, taunt, TAUNT, TAUNT_TURNS + taunt_bonus.astype(jnp.int32)
	)

	disable = (
		hit
		& (status == VOLATILE_DISABLE)
		& (volatiles[foe, DISABLE_REMAINING] == 0)
		& usable_last
		& (last != tables.struggle_id)
	)
	volatiles = _write(volatiles, foe_sides, disable, DISABLE_MOVE, last)
	volatiles = _write(volatiles, foe_sides, disable, DISABLE_REMAINING, DISABLE_TURNS - pending)

	immune = status_immunity(rows[foe], SLEEP, weather)
	yawn = (
		hit
		& (status == VOLATILE_YAWN)
		& (volatiles[foe, YAWN] == 0)
		& (state.status[foe, target] == HEALTHY)
		& ~immune
	)
	volatiles = _write(volatiles, foe_sides, yawn, YAWN, YAWN_TURNS)

	grass = jnp.any(state.types[foe, target] == tables.grass_type)
	seed = hit & (status == VOLATILE_LEECH_SEED) & (volatiles[foe, LEECH_SEED] == 0) & ~grass
	volatiles = _write(volatiles, foe_sides, seed, LEECH_SEED, 1)
	return state._replace(volatiles=volatiles)


def encore_override(state, side, move_id, tables):
	"""Showdown OverrideAction: a non-Struggle action becomes the encored move.

	Returns (apply, encored_move, slot): slot is the encored move's moveset slot.
	"""
	user = state.active[side]
	encored = state.volatiles[side, ENCORE_MOVE].astype(jnp.int32)
	moves = state.moves[side, user].astype(jnp.int32)
	matches = moves == encored
	apply = (
		(state.volatiles[side, ENCORE_REMAINING] > 0)
		& jnp.any(matches)
		& (move_id != tables.struggle_id)
		& (move_id != encored)
	)
	return apply, encored, jnp.argmax(matches).astype(jnp.int32)


def restrict_legal_moves(state, moves, usable, tables):
	"""Encore keeps only its move (if the moveset still has it); Taunt removes status moves."""
	encored = state.volatiles[:, ENCORE_MOVE].astype(jnp.int32)[:, None]
	encore_on = (state.volatiles[:, ENCORE_REMAINING] > 0)[:, None]
	has_encored = encore_on & jnp.any(moves == encored, axis=1, keepdims=True)
	usable &= ~has_encored | (moves == encored)
	taunted = (state.volatiles[:, TAUNT] > 0)[:, None]
	status = tables.move_category[moves] == STATUS_CATEGORY
	return usable & ~(taunted & status)


def taunt_blocks(state, side, row):
	"""Taunt cancels a status move before it starts (BeforeMove, priority 5)."""
	return (state.volatiles[side, TAUNT] > 0) & (row.move_category == STATUS_CATEGORY)


def _tick(state, column, enabled):
	"""Count one residual turn of a duration volatile; returns (left, ended)."""
	remaining = state.volatiles[:, column].astype(jnp.int32)
	tick = enabled & (remaining > 0)
	left = jnp.where(tick, remaining - 1, remaining)
	return left, tick & (left == 0)


def encore_residual(state, enabled, key, forced, tables, rows):
	"""Encore (order 16): the duration counts down; the encore ends early without PP."""
	del key, forced, tables, rows
	left, ended = _tick(state, ENCORE_REMAINING, enabled)
	encored = state.volatiles[:, ENCORE_MOVE].astype(jnp.int32)
	moves = state.moves[SIDES, state.active].astype(jnp.int32)
	pp = state.pp[SIDES, state.active]
	has_pp = jnp.any((moves == encored[:, None]) & (pp > 0), axis=1)
	ticking = enabled & (state.volatiles[:, ENCORE_REMAINING] > 0)
	clear = ended | (ticking & ~has_pp)
	state = _set_duration(state, ENCORE_REMAINING, left)
	return clear_columns(state, clear, (ENCORE_REMAINING, ENCORE_MOVE))


def taunt_residual(state, enabled, key, forced, tables, rows):
	"""Taunt (order 15): the duration counts down."""
	del key, forced, tables, rows
	left, _ = _tick(state, TAUNT, enabled)
	return _set_duration(state, TAUNT, left)


def _set_duration(state, column, left):
	columns = jnp.arange(NUM_VOLATILES)
	mask = columns[None, :] == column
	volatiles = jnp.where(mask, left[:, None].astype(state.volatiles.dtype), state.volatiles)
	return state._replace(volatiles=volatiles)


def clear_columns(state, sides, columns):
	"""Zero the given volatile columns on the selected sides."""
	volatile_columns = jnp.arange(NUM_VOLATILES)
	mask = sides[:, None] & (
		(volatile_columns[None, :] == columns[0]) | (volatile_columns[None, :] == columns[1])
	)
	return state._replace(volatiles=jnp.where(mask, 0, state.volatiles))
