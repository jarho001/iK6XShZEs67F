"""Pure decision-point transition for Gen 5 singles.

Actions must be legal. Random outcomes retain engine.randomness addresses;
masked arithmetic can evaluate unused outcomes without consuming randomness.
"""

import jax.numpy as jnp

from .actions import SIDES, bench_slots
from .boosts import apply_boosts
from .damage_effects import SELF_FAINT_ALWAYS, apply_damage_effects
from .field import (
	apply_move_field_effects,
	apply_switch_hazards_ordered,
)
from .order import action_order
from .randomness import random_bits
from .residual import residual
from .secondaries import apply_secondaries
from .state import MOVE, NONE, ONGOING, ROOST, SWITCH
from .status import after_hit, apply_move, before_move, switch_in
from .volatiles import confuse, end_turn


def _switch(state, side, action, enabled, tables):
	"""Switch and clear boosts (Pokemon.clearVolatile), keeping fixed slots."""
	choice = jnp.clip(action - 4, 0, 4)
	target = bench_slots(state.active)[side, choice]
	outgoing = state.active[side]
	active = state.active.at[side].set(
		jnp.where(enabled, target, state.active[side]).astype(state.active.dtype)
	)
	boosts = state.boosts.at[side].set(jnp.where(enabled, 0, state.boosts[side]))
	volatiles = state.volatiles.at[side].set(jnp.where(enabled, 0, state.volatiles[side]))
	old_types = state.types[side, outgoing]
	restore_roost = enabled & (state.volatiles[side, ROOST] > 0)
	types = state.types.at[side, outgoing].set(
		jnp.where(restore_roost, state.roost_types[side, outgoing], old_types)
	)
	order = state.seen_order[side, target]
	order = jnp.where(enabled & (order == 0), jnp.max(state.seen_order[side]) + 1, order)
	state = switch_in(state, side, target, enabled)
	state = state._replace(
		active=active,
		boosts=boosts,
		volatiles=volatiles,
		types=types,
		seen_order=state.seen_order.at[side, target].set(order),
	)
	return state


def _move(state, side, action, move, enabled, key, forced, tables):
	user, foe = state.active[side], 1 - side
	enabled &= (state.hp[side, user] > 0) & jnp.any(state.hp[foe] > 0) & (state.winner == ONGOING)
	state, enabled = before_move(state, side, move, enabled, key, forced, tables)
	struggle = move == tables.struggle_id
	slot = jnp.minimum(action, 3)
	pp = state.pp[side, user, slot].astype(jnp.int32)
	new_pp = jnp.maximum(pp - (enabled & ~struggle), 0).astype(state.pp.dtype)
	revealed = state.revealed_moves[side, user, slot] | (enabled & ~struggle)
	state = state._replace(
		pp=state.pp.at[side, user, slot].set(new_pp),
		revealed_moves=state.revealed_moves.at[side, user, slot].set(revealed),
	)
	state, success, hit, hit_landed, field_hit = apply_damage_effects(
		state, side, move, enabled, key, forced, tables
	)
	state = apply_move(state, side, move, enabled, hit, key, forced, tables)
	state = apply_boosts(state, side, move, enabled, hit, success, tables)
	state = confuse(state, foe, enabled & hit & tables.move_confusion[move], key, forced)
	state = apply_secondaries(state, side, move, hit_landed[0], hit_landed[1:], key, forced, tables)
	state = after_hit(state, side, move, enabled & success, tables)
	state = apply_move_field_effects(
		state, side, move, enabled, field_hit, success, key, forced, tables
	)
	hp = state.hp.astype(jnp.int32)
	boosts = jnp.where((hp[SIDES, state.active] > 0)[:, None], state.boosts, 0)
	return state._replace(boosts=boosts)


def step(state, actions, key, forced, engine_tables):
	"""Run both legal choices to the next move, replacement, or end request.

	Faint replacements precede Battle.nextTurn, so their request retains the
	old turn number. Finishing the replacement step advances it; a win does not.
	"""
	t = engine_tables
	key = random_bits(key)
	ongoing = state.winner == ONGOING
	normal = jnp.any(state.request == MOVE) & ongoing
	replacement = ~normal & jnp.any(state.request == SWITCH) & ongoing
	moves = state.moves[SIDES, state.active, jnp.minimum(actions, 3)]
	usable = (state.moves[SIDES, state.active] != 0) & (state.pp[SIDES, state.active] > 0)
	moves = jnp.where(jnp.any(usable, axis=1), moves, t.struggle_id)
	first = action_order(state, actions, moves, replacement, key, forced, t)
	requests = state.request
	will_switch = (actions >= 4) & (requests != NONE) & ongoing
	last_faint = jnp.asarray(0, state.winner.dtype)
	alive = jnp.any(state.hp > 0, axis=1)
	# Switches have higher queue priority than every move in this pool. Showdown
	# resolves the grouped SwitchIn handlers after the switch actions and before
	# the first move, so compute their shared speed order once here.
	for side in (first, 1 - first):
		action = actions[side]
		switch = will_switch[side]
		state = _switch(state, side, action, switch, t)
	alive_after = jnp.any(state.hp > 0, axis=1)
	newly_dead = alive & ~alive_after
	last_faint = jnp.where(
		jnp.any(newly_dead),
		jnp.where(jnp.sum(newly_dead) == 1, jnp.argmax(newly_dead), 1 - first),
		last_faint,
	)
	alive = alive_after
	state = apply_switch_hazards_ordered(state, will_switch, jnp.any(will_switch), key, forced, t)
	alive_after = jnp.any(state.hp > 0, axis=1)
	newly_dead = alive & ~alive_after
	last_faint = jnp.where(
		jnp.any(newly_dead),
		jnp.where(jnp.sum(newly_dead) == 1, jnp.argmax(newly_dead), 1 - first),
		last_faint,
	)
	alive = alive_after
	for side in (first, 1 - first):
		action = actions[side]
		move = normal & (requests[side] == MOVE) & (action < 4)
		state = _move(state, side, action, moves[side], move, key, forced, t)
		alive_after = jnp.any(state.hp > 0, axis=1)
		newly_dead = alive & ~alive_after
		faint_order = jnp.where(t.move_self_faint[moves[side]] == SELF_FAINT_ALWAYS, 1 - side, side)
		faint_order = jnp.where(
			jnp.any(newly_dead) & (jnp.sum(newly_dead) == 1),
			jnp.argmax(newly_dead.astype(jnp.int32)),
			faint_order,
		)
		last_faint = jnp.where(jnp.any(newly_dead), faint_order, last_faint)
		alive = alive_after
	winner = jnp.where(~alive[0], 1, 0)
	winner = jnp.where(~jnp.any(alive), last_faint, winner)
	winner = jnp.where(jnp.all(alive), ONGOING, winner).astype(state.winner.dtype)
	winner = jnp.where(state.winner == ONGOING, winner, state.winner)
	state = state._replace(winner=winner)
	state = residual(state, normal & (state.winner == ONGOING), key, forced, t)
	ended = state.winner != ONGOING
	fainted = state.hp[SIDES, state.active] == 0
	needs_switch = fainted & jnp.any(state.hp > 0, axis=1)
	clear_fainted = needs_switch & ~ended
	roost = (state.volatiles[SIDES, ROOST] > 0) & fainted
	types = state.types.at[SIDES, state.active].set(
		jnp.where(
			roost[:, None], state.roost_types[SIDES, state.active], state.types[SIDES, state.active]
		)
	)
	state = state._replace(
		types=types,
		volatiles=jnp.where(fainted[:, None], 0, state.volatiles),
		status=state.status.at[SIDES, state.active].set(
			jnp.where(clear_fainted, 0, state.status[SIDES, state.active])
		),
		status_counter=state.status_counter.at[SIDES, state.active].set(
			jnp.where(clear_fainted, 0, state.status_counter[SIDES, state.active])
		),
	)
	waiting = jnp.any(needs_switch)
	request = jnp.where(waiting, jnp.where(needs_switch, SWITCH, NONE), MOVE)
	request = jnp.where(ended, NONE, request).astype(state.request.dtype)
	advance = (normal | replacement) & ~waiting & ~ended
	state = end_turn(state, normal & ~ended)
	return state._replace(request=request, turn=state.turn + advance.astype(jnp.int32))
