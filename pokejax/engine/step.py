"""Pure decision-point transition for Gen 5 singles.

Actions must be legal. Random outcomes retain engine.randomness addresses;
masked arithmetic can evaluate unused outcomes without consuming randomness.
"""

import jax.numpy as jnp

from .actions import SIDES, bench_slots
from .damage import move_damage
from .order import action_order
from .residual import residual
from .state import MOVE, NONE, ONGOING, SWITCH
from .status import apply_move, before_move, switch_in


def _switch(state, side, action, enabled):
	"""Switch and clear boosts (Pokemon.clearVolatile), keeping fixed slots."""
	choice = jnp.clip(action - 4, 0, 4)
	target = bench_slots(state.active)[side, choice]
	active = state.active.at[side].set(
		jnp.where(enabled, target, state.active[side]).astype(state.active.dtype)
	)
	boosts = state.boosts.at[side].set(jnp.where(enabled, 0, state.boosts[side]))
	order = state.seen_order[side, target]
	order = jnp.where(enabled & (order == 0), jnp.max(state.seen_order[side]) + 1, order)
	state = switch_in(state, side, target, enabled)
	return state._replace(
		active=active, boosts=boosts, seen_order=state.seen_order.at[side, target].set(order)
	)


def _move(state, side, action, move, enabled, key, forced, tables):
	user, foe = state.active[side], 1 - side
	target = state.active[foe]
	enabled &= (state.hp[side, user] > 0) & (state.winner == ONGOING)
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
	damage, success, hit = move_damage(state, side, move, key, forced, tables)
	state = apply_move(state, side, move, enabled, hit, success, damage, key, forced, tables)
	damage = jnp.where(enabled, damage, 0)
	hp = state.hp.astype(jnp.int32)
	hp = hp.at[foe, target].set(jnp.maximum(hp[foe, target] - damage, 0))
	# BattleActions.applyRecoilDamage: Math.round(baseMaxhp / 4), minimum 1.
	recoil = jnp.maximum((state.stats[side, user, 0].astype(jnp.int32) + 2) // 4, 1)
	recoil = jnp.where(enabled & struggle & success & (damage > 0), recoil, 0)
	hp = hp.at[side, user].set(jnp.maximum(hp[side, user] - recoil, 0))
	alive = jnp.any(hp > 0, axis=1)
	# resolveFaints drains the queue before checkWin. Struggle queues the target
	# first, then its user; Gen 5 gives a simultaneous wipe to the last faint's side.
	winner = jnp.where(~alive[0], 1, 0)
	winner = jnp.where(~jnp.any(alive), side, winner)
	winner = jnp.where(jnp.all(alive), ONGOING, winner).astype(state.winner.dtype)
	winner = jnp.where(enabled, winner, state.winner)
	boosts = jnp.where((hp[SIDES, state.active] > 0)[:, None], state.boosts, 0)
	return state._replace(hp=hp.astype(state.hp.dtype), boosts=boosts, winner=winner)


def step(state, actions, key, forced, engine_tables):
	"""Run both legal choices to the next move, replacement, or end request.

	Faint replacements precede Battle.nextTurn, so their request retains the
	old turn number. Finishing the replacement step advances it; a win does not.
	"""
	t = engine_tables
	ongoing = state.winner == ONGOING
	normal = jnp.any(state.request == MOVE) & ongoing
	replacement = ~normal & jnp.any(state.request == SWITCH) & ongoing
	moves = state.moves[SIDES, state.active, jnp.minimum(actions, 3)]
	usable = (state.moves[SIDES, state.active] != 0) & (state.pp[SIDES, state.active] > 0)
	moves = jnp.where(jnp.any(usable, axis=1), moves, t.struggle_id)
	first = action_order(state, actions, moves, replacement, key, forced, t)
	requests = state.request
	for side in (first, 1 - first):
		action = actions[side]
		switch = (action >= 4) & (requests[side] != NONE) & ongoing
		state = _switch(state, side, action, switch)
		move = normal & (requests[side] == MOVE) & (action < 4)
		state = _move(state, side, action, moves[side], move, key, forced, t)
	state = residual(state, normal & (state.winner == ONGOING), key, forced)
	ended = state.winner != ONGOING
	fainted = state.hp[SIDES, state.active] == 0
	needs_switch = fainted & jnp.any(state.hp > 0, axis=1)
	clear_fainted = needs_switch & ~ended
	state = state._replace(
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
	return state._replace(request=request, turn=state.turn + advance.astype(jnp.int32))
