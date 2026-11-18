"""Table-driven stage changes and the three boost-reset/HP-cost effects."""

import jax.numpy as jnp

from .abilities import active_abilities, apply_boost_delta, move_abilities
from .actions import SIDES
from .move_data import read_move
from .state import BOOST_ATK, TEAM_SLOTS

STATUS_BOOST, ATTACK_DROP = 1, 2
HAZE, CLEAR_SMOG, BELLY_DRUM = 1, 2, 3


def apply_boosts(state, side, move, enabled, hit, success, tables, rows=None):
	row = read_move(tables, move)
	if rows is None:
		rows = move_abilities(active_abilities(state, tables), side)
	foe = 1 - side
	user = state.active[side]
	kind = row.move_boost_effect
	when = row.move_boost_when
	target = jnp.where(row.move_boost_target == 0, side, foe)
	status_hit = (target == side) | hit
	apply = enabled & (((when == STATUS_BOOST) & status_hit) | ((when == ATTACK_DROP) & success))
	delta = jnp.where(((SIDES == target) & apply)[:, None], row.move_boosts, 0)
	boosts = apply_boost_delta(state.boosts, delta, rows, SIDES != side)
	boosts = jnp.where(enabled & (kind == HAZE), 0, boosts)
	boosts = jnp.where(
		((SIDES == foe) & (enabled & success & (kind == CLEAR_SMOG)))[:, None], 0, boosts
	)
	maximum = state.stats[side, user, 0].astype(jnp.int32)
	hp = state.hp[side, user].astype(jnp.int32)
	drum = enabled & (kind == BELLY_DRUM) & (2 * hp > maximum) & (maximum > 1)
	drum &= boosts[side, 0] < 6
	drum_delta = 12 * (
		((SIDES == side) & drum)[:, None] & (jnp.arange(boosts.shape[1]) == BOOST_ATK)
	).astype(jnp.int8)
	boosts = apply_boost_delta(boosts, drum_delta, rows, False)
	hp = jnp.where(drum, hp - maximum // 2, hp).astype(state.hp.dtype)
	member_mask = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	hp = jnp.where(member_mask, hp, state.hp)
	return state._replace(boosts=boosts, hp=hp)
