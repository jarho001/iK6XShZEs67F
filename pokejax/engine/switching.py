"""Dense fixed-slot switch writes and shared switch-in hazard handling."""

import jax.numpy as jnp

from .actions import SIDES, bench_slots
from .field import apply_switch_hazards
from .randomness import DRAG_IN, UNFORCED, integer
from .state import (
	NUM_VOLATILES,
	QUICK_GUARD,
	ROOST,
	SLEEP,
	TEAM_SLOTS,
	TOXIC,
	WIDE_GUARD,
)


def switch_members(state, switch_mask, targets):
	"""Replace selected active members, clearing active-only state densely."""
	outgoing = state.active
	active = jnp.where(switch_mask, targets, outgoing).astype(state.active.dtype)
	boosts = jnp.where(switch_mask[:, None], 0, state.boosts)
	columns = jnp.arange(NUM_VOLATILES)
	side_condition = (columns == QUICK_GUARD) | (columns == WIDE_GUARD)
	clear_volatile = switch_mask[:, None] & ~side_condition[None, :]
	volatiles = jnp.where(clear_volatile, 0, state.volatiles)

	slots = jnp.arange(TEAM_SLOTS)
	outgoing_mask = slots[None, :] == outgoing[:, None]
	clear_fainted = switch_mask & (state.hp[SIDES, outgoing] == 0)
	status_team = jnp.where(outgoing_mask & clear_fainted[:, None], 0, state.status)
	status_counter_team = jnp.where(outgoing_mask & clear_fainted[:, None], 0, state.status_counter)
	restore_roost = switch_mask & (state.volatiles[:, ROOST] > 0)
	types = jnp.where(
		(restore_roost[:, None] & outgoing_mask)[:, :, None], state.roost_types, state.types
	)
	target_seen = state.seen_order[SIDES, targets]
	next_seen = jnp.max(state.seen_order, axis=1) + 1
	seen_order = jnp.where(
		(switch_mask & (target_seen == 0))[:, None] & (slots[None, :] == targets[:, None]),
		next_seen[:, None],
		state.seen_order,
	)

	target_status = state.status[SIDES, targets]
	counter = state.status_counter[SIDES, targets]
	counter = jnp.where(target_status == TOXIC, 0, counter)
	counter = jnp.where(target_status == SLEEP, state.sleep_start[SIDES, targets], counter)
	target_mask = switch_mask[:, None] & (slots[None, :] == targets[:, None])
	status_team = jnp.where(target_mask, target_status[:, None], status_team)
	active_turns = jnp.where(target_mask, 0, state.active_turns)
	status_counter_team = jnp.where(target_mask, counter[:, None], status_counter_team)
	return state._replace(
		active=active,
		active_turns=active_turns,
		boosts=boosts,
		volatiles=volatiles,
		types=types,
		status=status_team,
		status_counter=status_counter_team,
		seen_order=seen_order,
	)


def switch_actions(state, actions, switch_mask):
	"""Translate action IDs 4..8 to fixed team slots and switch both sides."""
	choices = jnp.clip(actions - 4, 0, 4)
	targets = bench_slots(state.active)[SIDES, choices]
	return switch_members(state, switch_mask, targets)


def drag_in(state, side, enabled, key, forced, tables):
	"""Uniformly select a usable bench member and run its normal hazards."""
	slots = jnp.arange(TEAM_SLOTS)
	eligible = (state.hp[side] > 0) & (slots != state.active[side])
	count = jnp.sum(eligible.astype(jnp.int32))
	rank = integer(key, UNFORCED, DRAG_IN, side, 0, jnp.maximum(count, 1))
	cumulative = jnp.cumsum(eligible.astype(jnp.int32))
	sampled = jnp.argmax(eligible & (cumulative > rank))
	forced_slot = forced[DRAG_IN, side, 0].astype(jnp.int32)
	target = jnp.where(forced_slot >= 0, forced_slot, sampled)
	forced_valid = (forced_slot < 0) | eligible[jnp.clip(forced_slot, 0, TEAM_SLOTS - 1)]
	active = enabled & (count > 0) & forced_valid
	switch_mask = (SIDES == side) & active
	targets = jnp.where(SIDES == side, target, state.active)
	state = switch_members(state, switch_mask, targets)
	state = apply_switch_hazards(state, side, target, active, tables)
	return state, switch_mask
