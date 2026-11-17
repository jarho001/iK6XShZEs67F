"""Table-driven stage changes and the three boost-reset/HP-cost effects."""

import jax.numpy as jnp

STATUS_BOOST, ATTACK_DROP = 1, 2
HAZE, CLEAR_SMOG, BELLY_DRUM = 1, 2, 3


def apply_boosts(state, side, move, enabled, hit, success, tables):
	foe = 1 - side
	user = state.active[side]
	kind = tables.move_boost_effect[move]
	when = tables.move_boost_when[move]
	target = jnp.where(tables.move_boost_target[move] == 0, side, foe)
	status_hit = (target == side) | hit
	apply = enabled & (((when == STATUS_BOOST) & status_hit) | ((when == ATTACK_DROP) & success))
	current = state.boosts[target]
	changed = jnp.clip(current.astype(jnp.int32) + tables.move_boosts[move], -6, 6).astype(
		current.dtype
	)
	boosts = state.boosts.at[target].set(jnp.where(apply, changed, current))
	boosts = jnp.where(enabled & (kind == HAZE), 0, boosts)
	boosts = boosts.at[foe].set(jnp.where(enabled & success & (kind == CLEAR_SMOG), 0, boosts[foe]))
	maximum = state.stats[side, user, 0].astype(jnp.int32)
	hp = state.hp[side, user].astype(jnp.int32)
	drum = enabled & (kind == BELLY_DRUM) & (2 * hp > maximum) & (maximum > 1)
	drum &= boosts[side, 0] < 6
	boosts = boosts.at[side, 0].set(jnp.where(drum, 6, boosts[side, 0]))
	hp = state.hp.at[side, user].set(jnp.where(drum, hp - maximum // 2, hp).astype(state.hp.dtype))
	return state._replace(boosts=boosts, hp=hp)
