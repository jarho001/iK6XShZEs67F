"""Formes and copies of the active Pokemon: Forecast, Flower Gift, Relic Song, Transform, Imposter.

Showdown's formeChange and transformInto both call setSpecies, which sets the species, the
types and Atk..Spe (storedStats). clearVolatile sets the species back to baseSpecies when a
Pokemon leaves the field, and a Transformed Pokemon also takes back its own moves and PP.
Forecast, Flower Gift and Relic Song act only on a holder that is not Transformed and whose
base species (species_base) is the ability's species. Every forme change is public: the
protocol names the new species, which is recorded in shown_species.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .abilities import (
	FLOWER_GIFT_NUM,
	FORECAST_NUM,
	FORM,
	MELOETTA_NUM,
	effective_weather,
	reveal_abilities,
	start_slow_start,
)
from .actions import SIDES
from .state import ILLUSION_SLOT, NUM_STATS, ROOST, SUBSTITUTE, SUN, TEAM_SLOTS, TRANSFORMED

TRANSFORM_PP_CAP = 5
STAT_SLICE = slice(1, NUM_STATS)


def member_mask(state, sides):
	"""[2, 6] mask of each side's active member, for the sides selected in a [2] mask."""
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	return slots & sides[:, None]


def displayed_species(state, species):
	"""What the opponent is shown for a species change: the disguise while Illusion holds."""
	disguise = state.volatiles[:, ILLUSION_SLOT].astype(jnp.int32)
	shown_slot = jnp.clip(disguise - 1, 0, TEAM_SLOTS - 1)
	return jnp.where(disguise > 0, state.species[SIDES, shown_slot], species)


def set_species(state, changed, species, tables):
	"""setSpecies for the active members of the changed sides: species, types and public form."""
	mask = member_mask(state, changed)
	target = species.astype(state.species.dtype)
	types = tables.species_types[species.astype(jnp.int32)]
	shown = displayed_species(state, species)
	return state._replace(
		species=jnp.where(mask, target[:, None], state.species),
		types=jnp.where(mask[..., None], types[:, None, :].astype(state.types.dtype), state.types),
		shown_species=jnp.where(
			mask, shown[:, None].astype(state.shown_species.dtype), state.shown_species
		),
	)


def reveal_ability_ids(state, sides, ids):
	"""Public ability for the masked active members: a protocol tag names it (Imposter)."""
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	mask = slots & sides[:, None]
	ability = jnp.broadcast_to(ids[:, None].astype(state.ability_public.dtype), mask.shape)
	return state._replace(ability_public=jnp.where(mask, ability, state.ability_public))


def write_stats(state, mask, stats5):
	"""Write Atk..Spe for the masked members; the HP stored on each member stays."""
	full = jnp.concatenate(
		(
			state.stats[..., :1],
			jnp.broadcast_to(stats5[:, None, :], state.stats[..., 1:].shape).astype(
				state.stats.dtype
			),
		),
		axis=-1,
	)
	return state._replace(stats=jnp.where(mask[..., None], full, state.stats))


def sync_weather_formes(state, tables, rows):
	"""Forecast and Flower Gift (onWeatherChange): the active holder follows the weather."""
	weather = effective_weather(state, rows)
	active = state.active
	free = (state.hp[SIDES, active] > 0) & (state.volatiles[:, TRANSFORMED] == 0)
	castform = free & (rows[:, FORM] == FORECAST_NUM)
	cherrim = free & (rows[:, FORM] == FLOWER_GIFT_NUM)
	castform_target = tables.castform_forms[weather]
	cherrim_target = tables.cherrim_forms[(weather == SUN).astype(jnp.int32)]
	target = jnp.where(castform, castform_target, cherrim_target)
	changed = (castform | cherrim) & (target != state.species[SIDES, active])
	# The protocol names the forme with the ability as its source, which reveals the ability.
	return reveal_abilities(set_species(state, changed, target, tables), changed)


def leaving_forme(state, old, old_rows, leaving, weather, tables):
	"""Forecast and Flower Gift of a Pokemon that is still on the field while a faster entrant's
	switch-in changes the weather: it changes forme (publicly) before it leaves in its own turn.

	old is the state before the switches and old_rows its active ability rows; only the shown
	species of the leaver's team slot changes, since leaving restores the base species anyway.
	"""
	slot = old.active
	free = (old.hp[SIDES, slot] > 0) & (old.volatiles[:, TRANSFORMED] == 0)
	castform = leaving & free & (old_rows[:, FORM] == FORECAST_NUM)
	cherrim = leaving & free & (old_rows[:, FORM] == FLOWER_GIFT_NUM)
	target = jnp.where(
		castform,
		tables.castform_forms[weather],
		tables.cherrim_forms[(weather == SUN).astype(jnp.int32)],
	)
	changed = (castform | cherrim) & (target != old.species[SIDES, slot])
	mask = (jnp.arange(TEAM_SLOTS)[None, :] == slot[:, None]) & changed[:, None]
	held = old.ability[SIDES, slot]
	return state._replace(
		shown_species=jnp.where(
			mask, target[:, None].astype(state.shown_species.dtype), state.shown_species
		),
		ability_public=jnp.where(
			mask, held[:, None].astype(state.ability_public.dtype), state.ability_public
		),
	)


def relic_song_forme(state, enabled, tables):
	"""Relic Song (onAfterMoveSecondarySelf): a Meloetta toggles between its two formes.

	Pirouette's Atk..Spe are precomputed at reset from the member's immutable inputs; the base
	Meloetta stats are the saved row, taken when the change first leaves the base forme.
	"""
	active = state.active
	base = tables.species_num[state.species_base[SIDES, active].astype(jnp.int32)]
	free = (state.hp[SIDES, active] > 0) & (state.volatiles[:, TRANSFORMED] == 0)
	meloetta = enabled & free & (base == MELOETTA_NUM)
	aria, pirouette = tables.meloetta_forms[0], tables.meloetta_forms[1]
	current = state.species[SIDES, active]
	target = jnp.where(current == aria, pirouette, aria)
	to_pirouette = target == pirouette
	saved = jnp.where(
		(meloetta & (current == aria))[:, None], state.stats[SIDES, active, 1:], state.saved_stats
	)
	state = state._replace(saved_stats=saved.astype(state.saved_stats.dtype))
	state = set_species(state, meloetta, target, tables)
	derived = state.pirouette_stats[SIDES, active]
	stats = jnp.where(to_pirouette[:, None], derived, state.saved_stats)
	return write_stats(state, member_mask(state, meloetta), stats)


def restore_form(state, leaving, tables):
	"""clearVolatile for a member that leaves the field: base species, and own stats and moves
	when a Transform or a Meloetta forme change had replaced them (Castform and Cherrim keep
	their stats across formes).

	Every write is team-wide under a mask of the leaving member: no gather keys on the active
	slot, so the restore costs no dependent kernel.
	"""
	mask = member_mask(state, leaving)
	transformed = state.volatiles[:, TRANSFORMED] > 0
	was = mask & transformed[:, None]
	base_is_meloetta = tables.species_num[state.species_base.astype(jnp.int32)] == MELOETTA_NUM
	meloetta_changed = (
		(state.species != state.species_base) & base_is_meloetta & ~transformed[:, None]
	)
	moved = was[..., None]
	# What the opponent was shown goes back to the base species too, unless an Illusion disguise
	# is up: that stays until the disguise breaks (a switch-out shows nothing new).
	disguised = (state.volatiles[:, ILLUSION_SLOT] > 0)[:, None]
	state = state._replace(
		shown_species=jnp.where(mask & ~disguised, state.species_base, state.shown_species),
		species=jnp.where(mask, state.species_base, state.species),
		types=jnp.where(mask[..., None], state.base_types, state.types),
		moves=jnp.where(moved, state.saved_moves[:, None, :], state.moves),
		pp=jnp.where(moved, state.saved_pp[:, None, :], state.pp),
		max_pp=jnp.where(moved, state.saved_max_pp[:, None, :], state.max_pp),
		revealed_moves=jnp.where(moved, state.saved_revealed[:, None, :], state.revealed_moves),
	)
	return write_stats(state, mask & (was | meloetta_changed), state.saved_stats)


class FoeView(NamedTuple):
	"""The foe of each side as that side's own switch-in or move sees it (per side, [2, ...])."""

	hp: jax.Array  # int16
	species: jax.Array  # int16
	ability: jax.Array  # int16
	types: jax.Array  # int8 [2, 2]: roost types while Roosting
	stats: jax.Array  # int16 [2, 6]
	moves: jax.Array  # int16 [2, 4]
	boosts: jax.Array  # int8 [2, 7]
	transformed: jax.Array  # bool
	illusion: jax.Array  # bool
	substitute: jax.Array  # bool


def foe_view(state, slot):
	"""The foe's member at `slot` (per side), with the foe's active-only volatiles."""
	foe = 1 - SIDES
	roost = (state.volatiles[foe, ROOST] > 0)[:, None]
	return FoeView(
		hp=state.hp[foe, slot],
		species=state.species[foe, slot],
		ability=state.ability[foe, slot],
		types=jnp.where(roost, state.roost_types[foe, slot], state.types[foe, slot]),
		stats=state.stats[foe, slot],
		moves=state.moves[foe, slot],
		boosts=state.boosts[foe],
		transformed=state.volatiles[foe, TRANSFORMED] > 0,
		illusion=state.volatiles[foe, ILLUSION_SLOT] > 0,
		substitute=state.volatiles[foe, SUBSTITUTE] > 0,
	)


def switch_foe_view(old, new, switching, first, replacement=False):
	"""Each side's view of its foe at the side's own switch-in.

	runSwitch runs the entering Pokemon in action order: a foe that enters after this side is
	still the old active when this side's switch-in effects run (Imposter, Trace). Faint
	replacements are different: both instaswitch actions run before any runSwitch, so each
	side already sees the new foe.
	"""
	foe = 1 - SIDES
	later = switching[foe] & (first == SIDES) & ~jnp.asarray(replacement)
	old_view = foe_view(old, old.active[foe])
	new_view = foe_view(new, new.active[foe])
	return jax.tree.map(
		lambda before, after: jnp.where(
			later.reshape((2,) + (1,) * (before.ndim - 1)), before, after
		),
		old_view,
		new_view,
	)


def transform_copy(state, enabled, tables, view):
	"""transformInto: each enabled active copies its foe (species, types, stats, moves at up to
	5 PP, boosts and ability), and keeps its own HP, level, status and item.

	Returns (state, changed): changed marks the sides whose ability became a new one.
	"""
	user = state.active
	transformed = state.volatiles[:, TRANSFORMED] > 0
	illusion = state.volatiles[:, ILLUSION_SLOT] > 0
	ok = (
		enabled
		& ~transformed
		& ~view.transformed
		& ~illusion
		& ~view.illusion
		& ~view.substitute
		& (view.hp > 0)
		& (state.hp[SIDES, user] > 0)
	)
	foe_types = view.types
	foe_species = view.species
	foe_moves = view.moves
	foe_pp = jnp.minimum(tables.move_base_pp[view.moves.astype(jnp.int32)], TRANSFORM_PP_CAP)
	foe_ability = view.ability
	old_ability = state.ability[SIDES, user]
	mask = member_mask(state, ok)
	mask3 = mask[..., None]
	own_moves = state.moves[SIDES, user]
	own_pp = state.pp[SIDES, user]
	own_max = state.max_pp[SIDES, user]
	own_revealed = state.revealed_moves[SIDES, user]
	ok_moves = ok[:, None]
	state = state._replace(
		saved_moves=jnp.where(ok_moves, own_moves, state.saved_moves),
		saved_pp=jnp.where(ok_moves, own_pp, state.saved_pp),
		saved_max_pp=jnp.where(ok_moves, own_max, state.saved_max_pp),
		saved_revealed=jnp.where(ok_moves, own_revealed, state.saved_revealed),
		species=jnp.where(mask, foe_species[:, None], state.species),
		types=jnp.where(mask3, foe_types[:, None, :], state.types),
		moves=jnp.where(mask3, foe_moves[:, None, :], state.moves),
		pp=jnp.where(mask3, foe_pp[:, None, :].astype(state.pp.dtype), state.pp),
		max_pp=jnp.where(mask3, foe_pp[:, None, :].astype(state.max_pp.dtype), state.max_pp),
		revealed_moves=jnp.where(mask3, 0, state.revealed_moves).astype(state.revealed_moves.dtype),
		boosts=jnp.where(ok_moves, view.boosts, state.boosts),
		ability=jnp.where(mask, foe_ability[:, None], state.ability),
		shown_species=jnp.where(mask, foe_species[:, None], state.shown_species),
	)
	state = state._replace(
		saved_stats=jnp.where(ok[:, None], state.stats[SIDES, user, STAT_SLICE], state.saved_stats)
	)
	state = write_stats(state, mask, view.stats[:, STAT_SLICE])
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(ok_moves & (columns[None, :] == TRANSFORMED), 1, state.volatiles)
	state = state._replace(volatiles=volatiles.astype(state.volatiles.dtype))
	changed = ok & (foe_ability != old_ability)
	# setAbility runs the new ability's Start (only when it differs from the old one).
	return start_slow_start(state, changed, tables), changed
