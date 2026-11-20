"""Vectorized per-hit damage effects: drain, recoil, crash and self-faint."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .abilities import (
	BOOST_EVENT,
	BOOST_EVENT_MOXIE,
	CONTACT,
	CONTACT_AFTERMATH,
	CONTACT_CURSED_BODY,
	CONTACT_EFFECT_SPORE,
	CONTACT_FLAME_BODY,
	CONTACT_IRON_BARBS,
	CONTACT_JUSTIFIED,
	CONTACT_LIQUID_OOZE,
	CONTACT_MUMMY,
	CONTACT_POISON_TOUCH,
	CONTACT_ROUGH_SKIN,
	CONTACT_STATIC,
	CONTACT_WEAK_ARMOR,
	MAGIC_GUARD,
	MUMMY_IMMUNE,
	OBLIVIOUS,
	POWDER_IMMUNITY,
	ROCK_HEAD,
	SHIELD_DUST,
	SKILL_LINK,
	STATUS,
	STATUS_SUN,
	absorb_hit,
	absorption,
	apply_boost_delta,
	effective_weather,
	has_hook,
	reveal_abilities,
)
from .actions import SIDES
from .damage import accuracy_hit, damage_for_hits, move_damage_bases, type_matchup
from .facts import active_facts
from .field import effective_move_type
from .items import CONTACT_ROCKY_HELMET, damage_survival
from .move_data import read_move
from .move_effects import apply_healing_active
from .randomness import (
	ABILITY_CHANCE,
	EFFECT_SPORE,
	chance,
	hit_count,
	integer,
)
from .state import (
	ATK,
	BURN,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	ENDURE,
	HEALTHY,
	HP,
	PARALYSIS,
	POISON,
	SLEEP,
	SUBSTITUTE,
	SUN,
	TEAM_SLOTS,
)
from .status import inflict_status, synchronize_status
from .substitute import split_hits
from .variable_moves import record_damage_taken

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
	sub_hit: jax.Array
	instafaint_kill: jax.Array


def _round_fraction(value, numerator, denominator):
	"""Showdown's positive Math.round(value * numerator / denominator)."""
	denominator = jnp.maximum(denominator.astype(jnp.int32), 1)
	return lax.div(value * numerator.astype(jnp.int32) + denominator // 2, denominator)


def _apply_contact_statuses(state, side, foe, row, hit_landed, key, forced, tables, rows):
	"""Resolve contact statuses and Disable across the fixed five-hit axis."""
	target_hook = rows[foe, CONTACT]
	powder_allowed = ~has_hook(rows[side], POWDER_IMMUNITY)
	indices = jnp.arange(MAX_HITS, dtype=jnp.int32)
	landed = hit_landed & row.move_contact
	defender_roll = chance(key, forced, ABILITY_CHANCE, foe, 2 * indices, 3, 10)
	spore_roll = integer(key, forced, EFFECT_SPORE, foe, indices, 100)
	static = target_hook == CONTACT_STATIC
	flame = target_hook == CONTACT_FLAME_BODY
	spore = target_hook == CONTACT_EFFECT_SPORE
	candidates = jnp.where(
		static,
		PARALYSIS,
		jnp.where(
			flame,
			BURN,
			jnp.where(
				spore & (spore_roll < 11),
				SLEEP,
				jnp.where(
					spore & (spore_roll < 21),
					PARALYSIS,
					jnp.where(spore & (spore_roll < 30), POISON, HEALTHY),
				),
			),
		),
	).astype(jnp.int32)
	attempt = landed & (((static | flame) & defender_roll) | (spore & powder_allowed))
	attempt &= candidates != HEALTHY
	slot = state.active[side]
	active_status = state.status[side, slot]
	active_hp = state.hp[side, slot]
	current_types = state.types[side, slot]
	present_types = current_types >= 0
	safe_types = jnp.maximum(current_types, 0)
	type_immune = jnp.any(
		tables.status_immune[candidates[:, None], safe_types[None, :]] & present_types[None, :],
		axis=-1,
	)
	weather = effective_weather(state, rows)
	weather_immune = tables.weather_status_immune[weather, candidates]
	ability_mask = rows[side, STATUS]
	ability_immune = (ability_mask & (1 << candidates)) != 0
	ability_immune |= ((ability_mask & STATUS_SUN) != 0) & (weather == SUN)
	sleep_clause = jnp.any(
		(state.status[side] == SLEEP) & (state.sleep_from_foe[side] != 0) & (state.hp[side] > 0)
	)
	blocked = type_immune | weather_immune | ability_immune | ((candidates == SLEEP) & sleep_clause)
	eligible = attempt & (active_status == HEALTHY) & (active_hp > 0) & ~blocked
	first = jnp.argmax(eligible.astype(jnp.int32))
	inflicted = candidates[first]
	apply_status = jnp.any(eligible)
	before_source_status = active_status
	state = inflict_status(state, side, inflicted, apply_status, key, forced, tables, rows)
	source_status_applied = (before_source_status == HEALTHY) & (
		state.status[side, state.active[side]] == inflicted
	)
	source_status_applied &= apply_status
	state = synchronize_status(
		state, side, foe, inflicted, source_status_applied, key, forced, tables, rows
	)
	curse = target_hook == CONTACT_CURSED_BODY
	curse_trigger = jnp.any(hit_landed & curse & defender_roll)
	curse_trigger &= state.volatiles[side, DISABLE_REMAINING] == 0
	curse_trigger &= (row.id != tables.struggle_id) & ~row.move_future
	curse_trigger &= state.hp[side, state.active[side]] > 0
	move_available = jnp.any(
		(state.moves[side, state.active[side]] == row.id) & (state.pp[side, state.active[side]] > 0)
	)
	curse_trigger &= move_available
	columns = jnp.arange(state.volatiles.shape[-1])
	side_mask = SIDES == side
	volatiles = jnp.where(
		side_mask[:, None] & (columns[None, :] == DISABLE_MOVE) & curse_trigger,
		row.id.astype(state.volatiles.dtype),
		state.volatiles,
	)
	volatiles = jnp.where(
		side_mask[:, None] & (columns[None, :] == DISABLE_REMAINING) & curse_trigger,
		jnp.asarray(4, state.volatiles.dtype),
		volatiles,
	)
	state = state._replace(volatiles=volatiles)
	ability_reveal = (SIDES == foe) & (source_status_applied | curse_trigger)
	return reveal_abilities(state, ability_reveal), ability_reveal


def apply_poison_touch(state, side, row, hit_landed, key, forced, tables, rows):
	"""Poison Touch runs as onSourceDamagingHit after move secondaries."""
	foe = 1 - side
	poison_touch = rows[side, CONTACT] == CONTACT_POISON_TOUCH
	shield_dust = has_hook(rows[foe], SHIELD_DUST)
	indices = jnp.arange(MAX_HITS, dtype=jnp.int32)
	landed = hit_landed & row.move_contact
	rolls = chance(key, forced, ABILITY_CHANCE, side, 2 * indices + 1, 3, 10)
	attempt = jnp.any(landed & poison_touch & rolls & ~shield_dust)
	before = state.status[foe, state.active[foe]]
	state = inflict_status(
		state,
		foe,
		POISON,
		attempt & (state.hp[foe, state.active[foe]] > 0),
		key,
		forced,
		tables,
		rows,
	)
	applied = (before == HEALTHY) & (state.status[foe, state.active[foe]] == POISON)
	state = synchronize_status(state, side, foe, POISON, applied, key, forced, tables, rows)
	return reveal_abilities(state, (SIDES == side) & applied)


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
	switching_foe=False,
):
	"""Resolve a fixed five-hit axis with one move-level damage calculation."""
	row = read_move(tables, move)
	facts = active_facts(state)
	t = tables
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	target_hp_before = state.hp[foe, target].astype(jnp.int32)
	self_faint_kind = row.move_self_faint
	pre_faint = enabled & (self_faint_kind == SELF_FAINT_ALWAYS)

	sub_hp = state.volatiles[foe, SUBSTITUTE].astype(jnp.int32)
	sub_target = (sub_hp > 0) & row.move_target_foe & ~row.move_bypass_sub
	# A status move aimed at a substitute fails with no effect (TryPrimaryHit returns null).
	blocked = guarded | invulnerable | (sub_target & (row.move_category == 2))
	checked_hit = accuracy_hit(
		state, side, row, key, forced, t, rows, weather, switching_foe=switching_foe
	)
	count = hit_count(
		key,
		forced,
		side,
		row.move_hit_min.astype(jnp.int32),
		row.move_hit_max.astype(jnp.int32),
	)
	count = jnp.where(has_hook(rows[side], SKILL_LINK), row.move_hit_max, count)
	type_ = effective_move_type(state, side, row, t, weather)
	immune, exponent, levitate = type_matchup(state, side, row.id, type_, t, rows, facts)
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
	# A fainted target is retargeted with no effect and shows no immunity (Levitate included).
	reveal_ability = absorbed | oblivious
	reveal_ability |= enabled & levitate & immune_for_move & ~invulnerable & (target_hp_before > 0)
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
		switching_foe=switching_foe,
		facts=facts,
	)
	hit_indices = jnp.arange(MAX_HITS, dtype=jnp.int32)
	damage, can_damage = damage_for_hits(
		state, side, row, hit_indices, bases, key, forced, t, item_rows, rows, type_, facts
	)
	move_success = enabled & ~blocked & checked_hit & can_damage
	scheduled = (hit_indices < count) & move_success
	# Hits reach a standing substitute while the earlier hits leave it some HP.
	sub_hits, sub_dealt = split_hits(jnp.where(sub_target, sub_hp, 0), damage, scheduled)
	active_hits = scheduled & ~sub_hits
	first_target = (hit_indices == jnp.argmax(active_hits)) & jnp.any(active_hits)
	raw_active_damage = jnp.where(active_hits, damage, 0)
	endure = state.volatiles[foe, ENDURE] > 0
	target_maximum = state.stats[foe, target, HP].astype(jnp.int32)
	lethal = jnp.any(first_target & (damage >= target_hp_before))
	sturdy_trigger, sash_trigger = damage_survival(
		lethal & (target_hp_before == target_maximum),
		endure,
		rows[foe],
		item_rows.damage_taken_kind[foe],
	)
	reveal_ability |= (SIDES == foe) & sturdy_trigger
	raw_active_damage = jnp.where(
		first_target & (sash_trigger | sturdy_trigger),
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
	# A substitute's drain rounds up and is paid on the damage the substitute took.
	fraction_num = drain_fraction[0].astype(jnp.int32)
	fraction_den = jnp.maximum(drain_fraction[1].astype(jnp.int32), 1)
	sub_drain = jnp.where(
		(sub_dealt > 0) & (fraction_num > 0),
		(sub_dealt * fraction_num + fraction_den - 1) // fraction_den,
		0,
	)
	drain = drain + jnp.sum(sub_drain)
	user_hp = state.hp[side, user].astype(jnp.int32)
	user_max = state.stats[side, user, HP].astype(jnp.int32)
	contact_hits = jnp.sum(hit_landed & row.move_contact).astype(jnp.int32)
	helm_damage = jnp.maximum(user_max // 6, 1) * contact_hits
	helm_trigger = (item_rows.contact_kind[foe] == CONTACT_ROCKY_HELMET) & (helm_damage > 0)
	target_hook = rows[foe, CONTACT]
	liquid_ooze = (target_hook == CONTACT_LIQUID_OOZE) & (drain > 0)
	liquid_damage = jnp.where(liquid_ooze, drain, 0)
	contact_ability_damage = (target_hook == CONTACT_ROUGH_SKIN) | (
		target_hook == CONTACT_IRON_BARBS
	)
	contact_ability_damage &= contact_hits > 0
	contact_ability_damage = jnp.where(
		contact_ability_damage, jnp.maximum(user_max // 8, 1) * contact_hits, 0
	)
	target_hp_after_move = jnp.maximum(target_hp_before - total_damage, 0)
	aftermath = (target_hook == CONTACT_AFTERMATH) & row.move_contact & success
	aftermath &= (target_hp_before > 0) & (target_hp_after_move == 0)
	aftermath_damage = jnp.where(aftermath, jnp.maximum(user_max // 4, 1), 0)
	magic_guard = has_hook(rows[side], MAGIC_GUARD)
	helm_damage = jnp.where(magic_guard, 0, helm_damage)
	contact_ability_damage = jnp.where(magic_guard, 0, contact_ability_damage)
	liquid_damage = jnp.where(magic_guard, 0, liquid_damage)
	aftermath_damage = jnp.where(magic_guard, 0, aftermath_damage)
	healing_drain = jnp.where(liquid_ooze, 0, drain)
	healed_hp = jnp.minimum(user_max, user_hp + jnp.where(user_hp > 0, healing_drain, 0))
	user_hp = jnp.where(enabled & ~pre_faint & (healing_drain > 0), healed_hp, user_hp)
	# Showdown resolves drain before the per-hit Rocky Helmet effect, so a
	# Drain Punch cannot immediately heal back the Helmet damage.
	contact_self_damage = contact_ability_damage + jnp.where(helm_trigger, helm_damage, 0)
	contact_self_damage += aftermath_damage + liquid_damage
	user_hp = jnp.maximum(user_hp - contact_self_damage, 0)
	# Aftermath, Rough Skin, Iron Barbs and Rocky Helmet damage are instafaint in Gen 5: the
	# attacker is processed first in the faint queue, so the holder that fainted in this move
	# is the last entry (checkWin names the holder's side).
	instafaint_damage = contact_self_damage - liquid_damage
	instafaint_kill = (instafaint_damage > 0) & (user_hp == 0)
	# Moxie checks the user after contact damage; a later recoil faint still boosts.
	moxie_user_alive = user_hp > 0

	self_heal = (row.move_heal[0] > 0) | row.move_pain_split
	target_effect = jnp.where(row.move_category == 2, bases[5], bases[4])
	effect_hit = (
		enabled & ~blocked & checked_hit & (((target_hp_before > 0) & target_effect) | self_heal)
	)
	# A substitute absorbs the hit: the target gets no effect unless a later hit lands on it.
	effect_hit &= ~sub_target | jnp.any(active_hits)
	connected = effect_hit | jnp.any(sub_hits)
	field_move = (row.move_hazard > 0) | (row.move_weather > 0)
	curse_hit = enabled & ~blocked & checked_hit & (row.move_volatile_status == 5)
	field_hit = effect_hit | (enabled & ~blocked & checked_hit & field_move) | curse_hit

	recoil_fraction = row.move_recoil
	recoil_base = jnp.where(row.move_recoil_basehp, user_max, total_damage)
	recoil = _round_fraction(recoil_base, recoil_fraction[0], recoil_fraction[1])
	# DamagingHit runs before recoil in Showdown, so Mummy can suppress Rock
	# Head on the very hit that transfers the ability.
	mummy_disables_rock_head = (target_hook == CONTACT_MUMMY) & (contact_hits > 0)
	mummy_disables_rock_head &= ~has_hook(rows[side], MUMMY_IMMUNE)
	mummy_disables_rock_head &= state.ability[side, user] != t.mummy_id
	recoil_blocked = has_hook(rows[side], ROCK_HEAD) & ~mummy_disables_rock_head
	recoil_blocked &= row.id != t.struggle_id
	recoil_blocked |= magic_guard & (row.id != t.struggle_id)
	# Recoil from a substitute hit is worked out per hit on the damage it took.
	sub_recoil_base = jnp.where(row.move_recoil_basehp, user_max, sub_dealt)
	sub_recoil_each = jnp.maximum(
		_round_fraction(sub_recoil_base, recoil_fraction[0], recoil_fraction[1]), 1
	)
	sub_recoil = jnp.sum(
		jnp.where((sub_dealt > 0) & ~recoil_blocked & (recoil_fraction[0] > 0), sub_recoil_each, 0)
	)
	recoil = jnp.where(
		~recoil_blocked & (total_damage > 0) & (recoil_fraction[0] > 0), jnp.maximum(recoil, 1), 0
	)
	user_hp = jnp.maximum(user_hp - jnp.where(enabled, recoil + sub_recoil, 0), 0)

	crash = enabled & row.move_crash_damage & ~connected & (target_hp_before > 0)
	crash &= ~magic_guard
	crash_damage = jnp.maximum(1, user_max // 2)
	user_hp = jnp.maximum(user_hp - jnp.where(crash, crash_damage, 0), 0)
	post_faint = enabled & (self_faint_kind == SELF_FAINT_ON_HIT) & connected
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
	state, contact_reveals = _apply_contact_statuses(
		state, side, foe, row, hit_landed, key, forced, t, rows
	)
	weak_armor = (target_hook == CONTACT_WEAK_ARMOR) & (row.move_category == 0)
	target_alive_after_move = target_hp_after_move > 0
	weak_count = jnp.sum(hit_landed & (damage > 0)).astype(jnp.int32)
	weak_count = jnp.where(target_alive_after_move, weak_count, 0)
	weak_count = jnp.where(weak_armor, weak_count, 0)
	justified = (target_hook == CONTACT_JUSTIFIED) & (type_ == t.dark_type)
	justified &= target_alive_after_move
	justified_count = jnp.sum(hit_landed & justified).astype(jnp.int32)
	target_fainted = (target_hp_before > 0) & (target_hp_after_move == 0)
	moxie = (rows[side, BOOST_EVENT] == BOOST_EVENT_MOXIE) & target_fainted
	moxie &= moxie_user_alive
	delta = jnp.zeros((2, state.boosts.shape[-1]), state.boosts.dtype)
	stats = jnp.arange(state.boosts.shape[-1])
	target_delta = jnp.where(
		stats == ATK - 1,
		justified_count,
		jnp.where(stats == 1, -weak_count, jnp.where(stats == 4, weak_count, 0)),
	).astype(state.boosts.dtype)
	source_delta = (stats == ATK - 1).astype(state.boosts.dtype) * moxie.astype(state.boosts.dtype)
	delta = jnp.where((SIDES == foe)[:, None], target_delta[None, :], delta)
	delta = jnp.where((SIDES == side)[:, None], source_delta[None, :], delta)
	# Showdown names the ability when a boost is effective, i.e. the stage was below its cap
	# before the change (a boost from +5 to +6 still reveals), so the reveals read the old stages.
	stages_before = state.boosts
	state = state._replace(boosts=apply_boost_delta(state.boosts, delta, rows, False))
	weak_reveals = weak_armor & (weak_count > 0)
	weak_reveals &= (stages_before[foe, 1] > -6) | (stages_before[foe, 4] < 6)
	justified_reveals = justified & (justified_count > 0) & (stages_before[foe, 0] < 6)
	moxie_reveals = moxie & (stages_before[side, 0] < 6)
	mummy_trigger = (target_hook == CONTACT_MUMMY) & (contact_hits > 0)
	mummy_trigger &= ~has_hook(rows[side], MUMMY_IMMUNE)
	mummy_trigger &= state.ability[side, user] != t.mummy_id
	member_mask = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	state = state._replace(
		ability=jnp.where(
			member_mask & mummy_trigger, t.mummy_id.astype(state.ability.dtype), state.ability
		)
	)
	state = reveal_abilities(state, (SIDES == side) & mummy_trigger)
	contact_reveals |= (SIDES == foe) & (
		((target_hook == CONTACT_ROUGH_SKIN) | (target_hook == CONTACT_IRON_BARBS))
		& (contact_hits > 0)
	)
	contact_reveals |= (SIDES == foe) & ((aftermath & (aftermath_damage > 0)) | liquid_ooze)
	contact_reveals |= (SIDES == foe) & weak_reveals
	contact_reveals |= (SIDES == foe) & justified_reveals
	contact_reveals |= (SIDES == side) & moxie_reveals
	contact_reveals |= (SIDES == foe) & sturdy_trigger
	contact_reveals |= ((SIDES == side) | (SIDES == foe)) & mummy_trigger
	state = reveal_abilities(state, contact_reveals)
	item_consumptions = (SIDES == side) & gem_trigger
	item_consumptions |= (SIDES == foe) & sash_trigger
	sub_mask = (SIDES == foe)[:, None] & sub_target
	sub_mask &= jnp.arange(state.volatiles.shape[1])[None, :] == SUBSTITUTE
	sub_left = (sub_hp - jnp.sum(sub_dealt)).astype(state.volatiles.dtype)
	state = state._replace(volatiles=jnp.where(sub_mask, sub_left, state.volatiles))
	state = record_damage_taken(state, side, row, hit_landed, dealt)
	return state, HitResult(
		success,
		effect_hit,
		hit_landed,
		field_hit,
		item_consumptions,
		(SIDES == foe) & helm_trigger,
		contact_reveals | ((SIDES == foe) & reveal_ability),
		jnp.any(sub_hits),
		instafaint_kill,
	)
