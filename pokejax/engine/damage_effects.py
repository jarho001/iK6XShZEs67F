"""Vectorized per-hit damage effects: drain, recoil, crash and self-faint."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .abilities import (
	OBLIVIOUS,
	ROCK_HEAD,
	SKILL_LINK,
	absorb_hit,
	absorption,
	has_hook,
)
from .actions import SIDES
from .damage import accuracy_hit, damage_for_hits, move_damage_bases, type_matchup
from .field import effective_move_type
from .items import CONTACT_ROCKY_HELMET, DAMAGE_TAKEN_FOCUS_SASH
from .move_data import read_move
from .move_effects import apply_healing_active
from .randomness import hit_count
from .state import ENDURE, HP

NO_SELF_FAINT, SELF_FAINT_ALWAYS, SELF_FAINT_ON_HIT = 0, 1, 2
MAX_HITS = 5


class HitResult(NamedTuple):
	success: jax.Array
	hit: jax.Array
	landed: jax.Array
	field_hit: jax.Array
	item_consumptions: jax.Array
	item_reveals: jax.Array
	ability_reveals: jax.Array


def _round_fraction(value, numerator, denominator):
	"""Showdown's positive Math.round(value * numerator / denominator)."""
	denominator = jnp.maximum(denominator.astype(jnp.int32), 1)
	return lax.div(value * numerator.astype(jnp.int32) + denominator // 2, denominator)


def apply_damage_effects(
	state,
	side,
	move,
	enabled,
	guarded,
	invulnerable,
	key,
	forced,
	tables,
	item_rows,
	rows,
	weather,
	sheer,
	analytic,
):
	"""Resolve a fixed five-hit axis with one move-level damage calculation."""
	row = read_move(tables, move)
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	target_hp_before = state.hp[foe, target].astype(jnp.int32)
	self_faint_kind = row.move_self_faint
	pre_faint = enabled & (self_faint_kind == SELF_FAINT_ALWAYS)

	blocked = guarded | invulnerable
	checked_hit = accuracy_hit(state, side, row, key, forced, t, rows, weather)
	count = hit_count(
		key,
		forced,
		side,
		row.move_hit_min.astype(jnp.int32),
		row.move_hit_max.astype(jnp.int32),
	)
	count = jnp.where(has_hook(rows[side], SKILL_LINK), row.move_hit_max, count)
	type_ = effective_move_type(state, row, t, weather)
	immune, exponent, levitate = type_matchup(state, side, row.id, type_, t, rows)
	immune_for_move = immune & ((row.move_category != 2) | row.move_checks_immunity)
	absorbed = absorption(
		state,
		side,
		rows[foe],
		type_,
		exponent,
		immune_for_move,
		enabled & ~blocked,
		row,
		t,
	)
	oblivious = enabled & ~blocked & (row.id == t.captivate_id) & has_hook(rows[foe], OBLIVIOUS)
	reveal_ability = absorbed | oblivious | (enabled & levitate & immune_for_move & ~invulnerable)
	blocked |= absorbed | oblivious
	gem_trigger = enabled & ~blocked & checked_hit & (row.move_category != 2)
	gem_trigger &= (row.id != t.struggle_id) & (item_rows.gem_type[side] == type_)
	gem_trigger &= (target_hp_before > 0) & ~immune
	bases = move_damage_bases(
		state,
		side,
		row,
		key,
		forced,
		t,
		item_rows,
		gem_trigger,
		rows,
		weather,
		type_,
		immune,
		exponent,
		sheer,
		analytic,
	)
	hit_indices = jnp.arange(MAX_HITS, dtype=jnp.int32)
	damage, can_damage = damage_for_hits(
		state, side, row, hit_indices, bases, key, forced, t, item_rows, rows, type_
	)
	move_success = enabled & ~blocked & checked_hit & can_damage
	active_hits = (hit_indices < count) & move_success
	raw_active_damage = jnp.where(active_hits, damage, 0)
	endure = state.volatiles[foe, ENDURE] > 0
	target_maximum = state.stats[foe, target, HP].astype(jnp.int32)
	sash = item_rows.damage_taken_kind[foe] == DAMAGE_TAKEN_FOCUS_SASH
	sash_trigger = active_hits[0] & (damage[0] >= target_hp_before)
	sash_trigger &= (target_hp_before == target_maximum) & ~endure & sash
	raw_active_damage = jnp.where(
		(hit_indices == 0) & sash_trigger,
		jnp.maximum(target_hp_before - 1, 0),
		raw_active_damage,
	)
	cumulative = jnp.cumsum(raw_active_damage)
	prior_damage = cumulative - raw_active_damage
	# Endure retains one HP even after the raw prefix reaches the target's HP.
	# Those later hits still land and can trigger contact effects.
	minimum_hp = endure.astype(jnp.int32)
	hp_before_hits = jnp.maximum(target_hp_before - prior_damage, minimum_hp)
	dealt = jnp.minimum(raw_active_damage, hp_before_hits - minimum_hp)
	hit_landed = active_hits & (hp_before_hits > 0)
	total_damage = jnp.minimum(cumulative[-1], jnp.maximum(target_hp_before - minimum_hp, 0))
	success = total_damage > 0
	drain_fraction = row.move_drain
	drain = _round_fraction(dealt, drain_fraction[0], drain_fraction[1])
	drain = jnp.where((dealt > 0) & (drain_fraction[0] > 0), jnp.maximum(drain, 1), 0)
	drain = jnp.sum(drain)
	user_hp = state.hp[side, user].astype(jnp.int32)
	user_max = state.stats[side, user, HP].astype(jnp.int32)
	contact_hits = jnp.sum(hit_landed & row.move_contact).astype(jnp.int32)
	helm_damage = jnp.maximum(user_max // 6, 1) * contact_hits
	helm_trigger = (item_rows.contact_kind[foe] == CONTACT_ROCKY_HELMET) & (helm_damage > 0)
	healed_hp = jnp.minimum(user_max, user_hp + jnp.where(user_hp > 0, drain, 0))
	user_hp = jnp.where(enabled & ~pre_faint & (drain > 0), healed_hp, user_hp)
	# Showdown resolves drain before the per-hit Rocky Helmet effect, so a
	# Drain Punch cannot immediately heal back the Helmet damage.
	user_hp = jnp.maximum(user_hp - jnp.where(helm_trigger, helm_damage, 0), 0)

	self_heal = (row.move_heal[0] > 0) | row.move_pain_split
	target_effect = jnp.where(row.move_category == 2, bases[5], bases[4])
	effect_hit = (
		enabled & ~blocked & checked_hit & (((target_hp_before > 0) & target_effect) | self_heal)
	)
	field_move = (row.move_hazard > 0) | (row.move_weather > 0)
	field_hit = effect_hit | (enabled & ~blocked & checked_hit & field_move)

	recoil_fraction = row.move_recoil
	recoil_base = jnp.where(row.move_recoil_basehp, user_max, total_damage)
	recoil = _round_fraction(recoil_base, recoil_fraction[0], recoil_fraction[1])
	recoil_blocked = has_hook(rows[side], ROCK_HEAD) & (row.id != t.struggle_id)
	recoil = jnp.where(
		~recoil_blocked & (total_damage > 0) & (recoil_fraction[0] > 0), jnp.maximum(recoil, 1), 0
	)
	user_hp = jnp.maximum(user_hp - jnp.where(enabled, recoil, 0), 0)

	crash = enabled & row.move_crash_damage & ~effect_hit & (target_hp_before > 0)
	crash_damage = jnp.maximum(1, user_max // 2)
	user_hp = jnp.maximum(user_hp - jnp.where(crash, crash_damage, 0), 0)
	post_faint = enabled & (self_faint_kind == SELF_FAINT_ON_HIT) & effect_hit
	user_hp = jnp.where(post_faint, 0, user_hp)
	user_hp = jnp.where(pre_faint, 0, user_hp)
	state, absorbed_hp = absorb_hit(state, foe, rows[foe], absorbed, target_hp_before)
	active_hp = state.hp[SIDES, state.active].astype(jnp.int32)
	active_hp = jnp.where(SIDES == foe, jnp.maximum(absorbed_hp - total_damage, 0), active_hp)
	active_hp = jnp.where(SIDES == side, user_hp, active_hp)
	# These move-table effects are mutually exclusive with primary move effects.
	# Resolve them in this common HP write so every action scatters active HP once.
	state, active_hp, _ = apply_healing_active(
		state, side, row, enabled & ~blocked, checked_hit, active_hp, t, weather
	)
	active_slots = jnp.arange(state.hp.shape[1])[None, :] == state.active[:, None]
	hp = jnp.where(active_slots, active_hp[:, None].astype(state.hp.dtype), state.hp)
	state = state._replace(hp=hp)
	item_consumptions = (SIDES == side) & gem_trigger
	item_consumptions |= (SIDES == foe) & sash_trigger
	return state, HitResult(
		success,
		effect_hit,
		hit_landed,
		field_hit,
		item_consumptions,
		(SIDES == foe) & helm_trigger,
		(SIDES == foe) & reveal_ability,
	)
