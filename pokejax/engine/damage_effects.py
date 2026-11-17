"""Vectorized per-hit damage effects: drain, recoil, crash and self-faint."""

import jax.numpy as jnp

from .actions import SIDES
from .damage import accuracy_hit, damage_for_hits, move_damage_bases
from .move_effects import apply_healing_active
from .randomness import hit_count
from .state import HP

NO_SELF_FAINT, SELF_FAINT_ALWAYS, SELF_FAINT_ON_HIT = 0, 1, 2
MAX_HITS = 5


def _round_fraction(value, numerator, denominator):
	"""Showdown's positive Math.round(value * numerator / denominator)."""
	denominator = jnp.maximum(denominator.astype(jnp.int32), 1)
	return (2 * value * numerator.astype(jnp.int32) + denominator) // (2 * denominator)


def apply_damage_effects(state, side, move, enabled, key, forced, tables):
	"""Resolve a fixed five-hit axis with one move-level damage calculation."""
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	target_hp_before = state.hp[foe, target].astype(jnp.int32)
	self_faint_kind = t.move_self_faint[move]
	pre_faint = enabled & (self_faint_kind == SELF_FAINT_ALWAYS)

	checked_hit = accuracy_hit(state, side, move, key, forced, t)
	count = hit_count(
		key,
		forced,
		side,
		t.move_hit_min[move].astype(jnp.int32),
		t.move_hit_max[move].astype(jnp.int32),
	)
	bases = move_damage_bases(state, side, move, t)
	hit_indices = jnp.arange(MAX_HITS, dtype=jnp.int32)
	damage, can_damage = damage_for_hits(state, side, move, hit_indices, bases, key, forced, t)
	move_success = enabled & checked_hit & can_damage
	active_hits = (hit_indices < count) & move_success
	raw_active_damage = jnp.where(active_hits, damage, 0)
	r0, r1, r2, r3, _ = raw_active_damage
	damage_before = jnp.stack((jnp.zeros_like(r0), r0, r0 + r1, r0 + r1 + r2, r0 + r1 + r2 + r3))
	hp_before_hits = jnp.maximum(target_hp_before - damage_before, 0)
	dealt = jnp.where(
		active_hits & (hp_before_hits > 0),
		jnp.minimum(damage, hp_before_hits),
		0,
	)
	hit_landed = active_hits & (dealt > 0)
	total_damage = jnp.sum(dealt)
	success = jnp.any(hit_landed)
	drain_fraction = t.move_drain[move]
	drain = _round_fraction(dealt, drain_fraction[0], drain_fraction[1])
	drain = jnp.where((dealt > 0) & (drain_fraction[0] > 0), jnp.maximum(drain, 1), 0)
	drain = jnp.sum(drain)
	user_hp = state.hp[side, user].astype(jnp.int32)
	user_max = state.stats[side, user, HP].astype(jnp.int32)
	healed_hp = jnp.minimum(user_max, user_hp + jnp.where(user_hp > 0, drain, 0))
	user_hp = jnp.where(enabled & ~pre_faint & (drain > 0), healed_hp, user_hp)

	self_heal = (t.move_heal[move, 0] > 0) | t.move_pain_split[move]
	effect_hit = enabled & checked_hit & ((target_hp_before > 0) & bases[5] | self_heal)

	recoil_fraction = t.move_recoil[move]
	base_hp = state.stats[side, user, HP].astype(jnp.int32)
	base_recoil = _round_fraction(base_hp, recoil_fraction[0], recoil_fraction[1])
	dealt_recoil = _round_fraction(total_damage, recoil_fraction[0], recoil_fraction[1])
	recoil = jnp.where(t.move_recoil_basehp[move], base_recoil, dealt_recoil)
	recoil = jnp.where((total_damage > 0) & (recoil_fraction[0] > 0), jnp.maximum(recoil, 1), 0)
	user_hp = jnp.maximum(user_hp - jnp.where(enabled, recoil, 0), 0)

	crash = enabled & t.move_crash_damage[move] & ~success & (target_hp_before > 0)
	crash_damage = jnp.maximum(1, base_hp // 2)
	user_hp = jnp.maximum(user_hp - jnp.where(crash, crash_damage, 0), 0)
	post_faint = enabled & (self_faint_kind == SELF_FAINT_ON_HIT) & success
	user_hp = jnp.where(post_faint, 0, user_hp)
	user_hp = jnp.where(pre_faint, 0, user_hp)
	active_hp = state.hp[SIDES, state.active].astype(jnp.int32)
	active_hp = jnp.where(SIDES == foe, jnp.maximum(target_hp_before - total_damage, 0), active_hp)
	active_hp = jnp.where(SIDES == side, user_hp, active_hp)
	# These move-table effects are mutually exclusive with primary move effects.
	# Resolve them in this common HP write so every action scatters active HP once.
	state, active_hp, _ = apply_healing_active(
		state, side, move, enabled, checked_hit, active_hp, t
	)
	hp = state.hp.at[SIDES, state.active].set(active_hp.astype(state.hp.dtype))
	state = state._replace(hp=hp)
	return state, success, effect_hit, hit_landed, enabled & checked_hit
