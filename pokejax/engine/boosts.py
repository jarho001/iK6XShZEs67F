"""Table-driven stage changes and the three boost-reset/HP-cost effects."""

import jax.numpy as jnp

from .actions import SIDES
from .state import TEAM_SLOTS

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
	changed = jnp.clip(
		state.boosts.astype(jnp.int32) + tables.move_boosts[move][None, :], -6, 6
	).astype(state.boosts.dtype)
	boosts = jnp.where((SIDES == target)[:, None] & apply, changed, state.boosts)
	boosts = jnp.where(enabled & (kind == HAZE), 0, boosts)
	boosts = jnp.where(
		((SIDES == foe) & (enabled & success & (kind == CLEAR_SMOG)))[:, None], 0, boosts
	)
	maximum = state.stats[side, user, 0].astype(jnp.int32)
	hp = state.hp[side, user].astype(jnp.int32)
	drum = enabled & (kind == BELLY_DRUM) & (2 * hp > maximum) & (maximum > 1)
	drum &= boosts[side, 0] < 6
	boosts = jnp.where(
		(SIDES == side)[:, None] & (jnp.arange(boosts.shape[1]) == 0)[None, :] & drum,
		6,
		boosts,
	)
	hp = jnp.where(drum, hp - maximum // 2, hp).astype(state.hp.dtype)
	member_mask = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	hp = jnp.where(member_mask, hp, state.hp)
	return state._replace(boosts=boosts, hp=hp)
