"""Table-driven Gen 5 item hooks used by moves, ordering, and residuals."""

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from pokejax.tables import Tables

from .abilities import (
	DAMAGE_TAKEN,
	DAMAGE_TAKEN_STURDY,
	MAGIC_GUARD,
	SPEED,
	SPEED_UNBURDEN,
	ability_cures,
	berry_allowed,
	has_hook,
	reveal_abilities,
	status_immunity,
)
from .modifiers import chain, modify
from .move_data import read_move
from .state import (
	BURN,
	CHOICE_MOVE,
	CONFUSION,
	HAIL,
	HEALTHY,
	HP,
	RAIN,
	SAND,
	SLEEP,
	SUN,
	TEAM_SLOTS,
	TOXIC,
	UNBURDEN,
)

SIDES = jnp.arange(2)

NO_HOOK = 0
RESIDUAL_LEFTOVERS, RESIDUAL_BLACK_SLUDGE, RESIDUAL_TOXIC_ORB, RESIDUAL_FLAME_ORB = range(1, 5)
BERRY_LUM, BERRY_SITRUS, BERRY_CHESTO, BERRY_CUSTAP = range(1, 5)
DAMAGE_NONE, DAMAGE_LIFE_ORB, DAMAGE_EXPERT_BELT = range(3)
DAMAGE_TAKEN_NONE, DAMAGE_TAKEN_FOCUS_SASH = range(2)
CONTACT_NONE, CONTACT_ROCKY_HELMET = range(2)


def damage_survival(full_hp_lethal, endure, ability_row, item_kind):
	"""Sturdy has an earlier damage priority than Focus Sash."""
	eligible = full_hp_lethal & ~endure
	sturdy = eligible & (ability_row[DAMAGE_TAKEN] == DAMAGE_TAKEN_STURDY)
	sash = eligible & ~sturdy & (item_kind == DAMAGE_TAKEN_FOCUS_SASH)
	return sturdy, sash


class ItemHookTable(NamedTuple):
	rows: jnp.ndarray  # [item, hook columns]; gather one row per active Pokemon


STAT_MODIFIER = slice(0, 6)
STAT_SPECIES_NUMBER = slice(6, 8)
STAT_SPECIES_RESTRICTED = 8
STAT_REQUIRES_NFE = 9
TYPE_BOOST = 10
TYPE_BOOST_SECONDARY = 11
TYPE_SPECIES_NUMBER = slice(12, 14)
TYPE_SPECIES_RESTRICTED = 14
DAMAGE_KIND = 15
GEM_TYPE = 16
RESIDUAL_KIND = 17
BERRY_KIND = 18
DAMAGE_TAKEN_KIND = 19
CONTACT_KIND = 20
BOOST_RESTORE = 21
CHOICE = 22
WEATHER_KIND = 23
WEATHER_DURATION = 24
CRIT_BONUS = 25
HARVEST_BERRY = 26


class ActiveItemRows(NamedTuple):
	ids: jnp.ndarray  # [2]
	stat_modifier: jnp.ndarray  # [2, 6]
	stat_species: jnp.ndarray  # [2]
	stat_requires_nfe: jnp.ndarray  # [2]
	species_nfe: jnp.ndarray  # [2], active species condition gathered with the item row
	type_boost: jnp.ndarray  # [2]
	type_boost_secondary: jnp.ndarray  # [2]
	type_species: jnp.ndarray  # [2]
	damage_kind: jnp.ndarray  # [2]
	gem_type: jnp.ndarray  # [2]
	residual_kind: jnp.ndarray  # [2]
	berry_kind: jnp.ndarray  # [2]
	damage_taken_kind: jnp.ndarray  # [2]
	contact_kind: jnp.ndarray  # [2]
	boost_restore: jnp.ndarray  # [2]
	choice: jnp.ndarray  # [2]
	weather_kind: jnp.ndarray  # [2]
	weather_duration: jnp.ndarray  # [2]
	crit_bonus: jnp.ndarray  # [2]


class ActiveItemUpdateHooks(NamedTuple):
	berry_kind: jnp.ndarray  # [2]
	boost_restore: jnp.ndarray  # [2]


PLATE_TYPES = {
	"dracoplate": "dragon",
	"dreadplate": "dark",
	"earthplate": "ground",
	"fistplate": "fighting",
	"flameplate": "fire",
	"icicleplate": "ice",
	"insectplate": "bug",
	"ironplate": "steel",
	"meadowplate": "grass",
	"mindplate": "psychic",
	"skyplate": "flying",
	"splashplate": "water",
	"stoneplate": "rock",
	"spookyplate": "ghost",
	"toxicplate": "poison",
	"zapplate": "electric",
}


def build_item_hook_table(tables: Tables) -> ItemHookTable:
	"""Resolve item names and type hooks into compact, per-item columns."""
	item_count = len(tables.items.ids)
	stat_modifier = np.full((item_count, 6), 4096, dtype=np.int16)
	stat_species_number = np.full((item_count, 2), -1, dtype=np.int16)
	stat_species_restricted = np.zeros(item_count, dtype=np.bool_)
	stat_requires_nfe = np.zeros(item_count, dtype=np.bool_)
	type_boost = np.full(item_count, -1, dtype=np.int8)
	type_boost_secondary = np.full(item_count, -1, dtype=np.int8)
	type_species_number = np.full((item_count, 2), -1, dtype=np.int16)
	type_species_restricted = np.zeros(item_count, dtype=np.bool_)
	damage_kind = np.zeros(item_count, dtype=np.uint8)
	gem_type = np.full(item_count, -1, dtype=np.int8)
	residual_kind = np.zeros(item_count, dtype=np.uint8)
	berry_kind = np.zeros(item_count, dtype=np.uint8)
	damage_taken_kind = np.zeros(item_count, dtype=np.uint8)
	contact_kind = np.zeros(item_count, dtype=np.uint8)
	boost_restore = np.zeros(item_count, dtype=np.bool_)
	choice = np.zeros(item_count, dtype=np.bool_)
	weather_kind = np.zeros(item_count, dtype=np.int8)
	weather_duration = np.zeros(item_count, dtype=np.uint8)
	crit_bonus = np.zeros(item_count, dtype=np.int8)
	harvest_berry = tables.items.is_berry.astype(np.bool_)

	def set_stat(item, stat, numerator):
		stat_modifier[tables.items.index[item], stat] = numerator

	set_stat("choiceband", 1, 6144)
	set_stat("choicespecs", 3, 6144)
	set_stat("choicescarf", 5, 6144)
	for item in ("choiceband", "choicespecs", "choicescarf"):
		choice[tables.items.index[item]] = True

	# These hooks use Showdown's baseSpecies national-dex identity, so every
	# exported form with the same number receives the same item effect.
	for item, species_numbers, stats, numerator in (
		("lightball", (25,), (1, 3), 8192),
		("thickclub", (104, 105), (1,), 8192),
	):
		row = tables.items.index[item]
		stat_species_number[row, : len(species_numbers)] = species_numbers
		stat_species_restricted[row] = True
		for stat in stats:
			set_stat(item, stat, numerator)

	eviolite = tables.items.index["eviolite"]
	stat_requires_nfe[eviolite] = True
	set_stat("eviolite", 2, 6144)
	set_stat("eviolite", 4, 6144)

	plate_types = PLATE_TYPES
	for item, type_name in {
		"silkscarf": "normal",
		"blackglasses": "dark",
		**plate_types,
	}.items():
		type_boost[tables.items.index[item]] = tables.types.index[type_name]
	for item, species_number, type_names in (
		("lustrousorb", (484,), ("water", "dragon")),
		("griseousorb", (487,), ("ghost", "dragon")),
	):
		row = tables.items.index[item]
		type_species_number[row, : len(species_number)] = species_number
		type_species_restricted[row] = True
		type_boost[row] = tables.types.index[type_names[0]]
		type_boost_secondary[row] = tables.types.index[type_names[1]]

	# Gen 5 inherits gen6's Soul Dew stat hook, which replaces the base
	# Psychic/Dragon BasePower hook. It boosts Latias/Latios SpA and SpD by 1.5.
	soul_dew = tables.items.index["souldew"]
	stat_species_number[soul_dew] = (380, 381)
	stat_species_restricted[soul_dew] = True
	set_stat("souldew", 3, 6144)
	set_stat("souldew", 4, 6144)

	gem_types = {
		"bug": "bug",
		"dark": "dark",
		"dragon": "dragon",
		"electric": "electric",
		"fighting": "fighting",
		"fire": "fire",
		"flying": "flying",
		"ghost": "ghost",
		"grass": "grass",
		"ground": "ground",
		"ice": "ice",
		"normal": "normal",
		"poison": "poison",
		"psychic": "psychic",
		"rock": "rock",
		"steel": "steel",
		"water": "water",
	}
	for type_name in gem_types:
		item = f"{type_name}gem"
		if item in tables.items.index:
			gem_type[tables.items.index[item]] = tables.types.index[type_name]

	for item, kind in (
		("leftovers", RESIDUAL_LEFTOVERS),
		("blacksludge", RESIDUAL_BLACK_SLUDGE),
		("toxicorb", RESIDUAL_TOXIC_ORB),
		("flameorb", RESIDUAL_FLAME_ORB),
	):
		residual_kind[tables.items.index[item]] = kind
	for item, kind in (
		("lumberry", BERRY_LUM),
		("sitrusberry", BERRY_SITRUS),
		("chestoberry", BERRY_CHESTO),
		("custapberry", BERRY_CUSTAP),
	):
		berry_kind[tables.items.index[item]] = kind
	damage_kind[tables.items.index["lifeorb"]] = DAMAGE_LIFE_ORB
	damage_kind[tables.items.index["expertbelt"]] = DAMAGE_EXPERT_BELT
	damage_taken_kind[tables.items.index["focussash"]] = DAMAGE_TAKEN_FOCUS_SASH
	contact_kind[tables.items.index["rockyhelmet"]] = CONTACT_ROCKY_HELMET
	boost_restore[tables.items.index["whiteherb"]] = True
	stick = tables.items.index["stick"]
	stat_species_number[stick, 0] = 83
	stat_species_restricted[stick] = True
	crit_bonus[stick] = 2

	for item, weather in (
		("damprock", RAIN),
		("heatrock", SUN),
		("smoothrock", SAND),
		("icyrock", HAIL),
	):
		if item in tables.items.index:
			row = tables.items.index[item]
			weather_kind[row] = weather
			weather_duration[row] = 8

	columns = np.concatenate(
		(
			stat_modifier,
			stat_species_number,
			stat_species_restricted[:, None],
			stat_requires_nfe[:, None],
			type_boost[:, None],
			type_boost_secondary[:, None],
			type_species_number,
			type_species_restricted[:, None],
			damage_kind[:, None],
			gem_type[:, None],
			residual_kind[:, None],
			berry_kind[:, None],
			damage_taken_kind[:, None],
			contact_kind[:, None],
			boost_restore[:, None],
			choice[:, None],
			weather_kind[:, None],
			weather_duration[:, None],
			crit_bonus[:, None],
			harvest_berry[:, None],
		),
		axis=1,
	).astype(np.int16)
	return ItemHookTable(jnp.asarray(columns))


def gather_active_items(state, tables) -> ActiveItemRows:
	"""Gather each active held-item row and species condition once per hook."""
	ids = state.item[SIDES, state.active].astype(jnp.int32)
	species = state.species[SIDES, state.active].astype(jnp.int32)
	species_number = tables.species_num[species].astype(jnp.int32)
	rows = tables.item_hooks.rows[ids]
	stat_species = (rows[:, STAT_SPECIES_RESTRICTED] == 0) | jnp.any(
		rows[:, STAT_SPECIES_NUMBER] == species_number[:, None], axis=1
	)
	type_species = (rows[:, TYPE_SPECIES_RESTRICTED] == 0) | jnp.any(
		rows[:, TYPE_SPECIES_NUMBER] == species_number[:, None], axis=1
	)
	return ActiveItemRows(
		ids,
		rows[:, STAT_MODIFIER],
		stat_species,
		(rows[:, STAT_REQUIRES_NFE] != 0),
		tables.species_nfe[species],
		rows[:, TYPE_BOOST],
		rows[:, TYPE_BOOST_SECONDARY],
		type_species,
		rows[:, DAMAGE_KIND],
		rows[:, GEM_TYPE],
		rows[:, RESIDUAL_KIND],
		rows[:, BERRY_KIND],
		rows[:, DAMAGE_TAKEN_KIND],
		rows[:, CONTACT_KIND],
		(rows[:, BOOST_RESTORE] != 0),
		(rows[:, CHOICE] != 0),
		rows[:, WEATHER_KIND],
		rows[:, WEATHER_DURATION],
		rows[:, CRIT_BONUS],
	)


def stat_modifier(stat, rows):
	active = rows.stat_species & (~rows.stat_requires_nfe | rows.species_nfe)
	return jnp.where(active, rows.stat_modifier[..., stat], 4096)


def apply_stat_modifier(value, side, stat, rows, modifier=4096):
	"""Apply ability and held-item modifiers together at the stat event boundary."""
	item = jnp.where(
		rows.stat_species[side] & (~rows.stat_requires_nfe[side] | rows.species_nfe[side]),
		rows.stat_modifier[side, stat],
		4096,
	)
	return modify(value, chain(modifier, item)).astype(value.dtype)


def apply_active_stat_modifiers(values, stat, rows, modifier=4096):
	return modify(values, chain(modifier, stat_modifier(stat, rows)))


def reveal_active_items(state, sides):
	"""Remember the item ID only when a public protocol event names it."""
	return update_item_state(state, jnp.zeros((2,), jnp.bool_), sides)


def update_item_state(state, consumed_sides, revealed_sides, tables=None):
	"""Write consumption and reveal events through one dense active-slot mask."""
	held = state.item[SIDES, state.active]
	consume = consumed_sides & (held != 0)
	reveal = (revealed_sides | consume) & (held != 0)
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	consume_mask = consume[:, None] & slots
	reveal_mask = reveal[:, None] & slots
	volatiles = state.volatiles
	if tables is not None:
		ability_rows = tables.ability_hooks[state.ability[SIDES, state.active].astype(jnp.int32)]
		unburden = (ability_rows[:, SPEED] == SPEED_UNBURDEN) & consume
		columns = jnp.arange(state.volatiles.shape[1])
		volatiles = jnp.where(unburden[:, None] & (columns[None, :] == UNBURDEN), 1, volatiles)
	return state._replace(
		item=jnp.where(consume_mask, 0, state.item),
		item_consumed=state.item_consumed | consume_mask,
		item_revealed=state.item_revealed | reveal_mask,
		item_public=jnp.where(reveal_mask, held[:, None], state.item_public),
		volatiles=volatiles,
	)


def consume_active_items(state, sides, tables=None):
	"""Consume active items through one consolidated dense member write."""
	return update_item_state(state, sides, jnp.zeros((2,), jnp.bool_), tables)


def item_type_matches(type_, side, rows):
	"""Return whether the active item row boosts this move type."""
	return ((rows.type_boost[side] >= 0) & (rows.type_boost[side] == type_)) | (
		(rows.type_boost_secondary[side] >= 0) & (rows.type_boost_secondary[side] == type_)
	)


def item_base_power_modifier(side, type_, gem_trigger, damage_kind, rows):
	"""Combine the active type-item or Gen 5 gem BasePower hook."""
	type_hook = item_type_matches(type_, side, rows) & rows.type_species[side]
	type_hook &= damage_kind < 1  # fixed-damage moves return before BasePower in Showdown
	gem_hook = gem_trigger & (damage_kind == 0)
	return jnp.where(gem_hook, 6144, jnp.where(type_hook, 4915, 4096)).astype(jnp.int32)


def apply_after_move_items(
	state, side, move, enabled, move_hit, rows, tables, sheer=False, abilities=None
):
	"""Resolve White Herb and Life Orb after a move's normal hit effects."""
	row = read_move(tables, move)
	negative = jnp.any(state.boosts < 0, axis=1)
	white_herb = enabled & rows.boost_restore & negative & (state.hp[SIDES, state.active] > 0)
	cleared = jnp.where(state.boosts < 0, 0, state.boosts)
	boosts = jnp.where(white_herb[:, None], cleared, state.boosts)
	state = state._replace(boosts=boosts)

	# In Gen 5 Life Orb recoils after a successful non-Status move. A pivot's
	# self-switch effect does not set source.forceSwitchFlag before this hook.
	user_side = SIDES == side
	recoil = user_side & enabled & move_hit & (rows.damage_kind == DAMAGE_LIFE_ORB)
	recoil &= ~jnp.asarray(sheer) & (row.move_category != 2)
	if abilities is not None:
		recoil &= ~has_hook(abilities[side], MAGIC_GUARD)
	user = state.active[side]
	maximum = state.stats[side, user, HP].astype(jnp.int32)
	hp = state.hp[side, user].astype(jnp.int32)
	damage = jnp.maximum(maximum // 10, 1)
	new_hp = jnp.maximum(hp - jnp.where(recoil[side], damage, 0), 0)
	active_mask = user_side[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == user)
	state = state._replace(hp=jnp.where(active_mask, new_hp.astype(state.hp.dtype), state.hp))
	return state, white_herb, recoil


def gather_active_update_hooks(state, tables):
	"""Gather berry and boost hooks after a drag changes the active member."""
	ids = state.item[SIDES, state.active].astype(jnp.int32)
	rows = tables.item_hooks.rows[ids]
	return ActiveItemUpdateHooks(rows[:, BERRY_KIND], rows[:, BOOST_RESTORE] != 0)


def update_active_items(state, enabled, rows, abilities=None, ignore_confusion=False, tables=None):
	"""Resolve active cures and held-item updates in event order."""
	active = state.active
	hp = state.hp[SIDES, active].astype(jnp.int32)
	maximum = state.stats[SIDES, active, HP].astype(jnp.int32)
	status = state.status[SIDES, active]
	confused = (state.volatiles[SIDES, CONFUSION] > 0) & ~jnp.asarray(ignore_confusion)
	status_cure = jnp.zeros((2,), jnp.bool_)
	confusion_cure = status_cure
	can_eat = True
	if abilities is not None:
		status_cure, confusion_cure = ability_cures(state, abilities)
		status_cure &= enabled
		confusion_cure &= enabled
		can_eat = berry_allowed(state, abilities)
		state = reveal_abilities(state, status_cure | confusion_cure)
	berry = rows.berry_kind
	lum = (berry == BERRY_LUM) & (
		((status != HEALTHY) & ~status_cure) | (confused & ~confusion_cure)
	)
	chesto = (berry == BERRY_CHESTO) & (status == SLEEP) & ~status_cure
	sitrus = (berry == BERRY_SITRUS) & (hp > 0) & (2 * hp <= maximum) & (hp < maximum)
	lum &= can_eat & (hp > 0)
	chesto &= can_eat & (hp > 0)
	sitrus &= can_eat
	consume = enabled & (lum | chesto | sitrus)
	clear_status = (enabled & (lum | chesto)) | status_cure
	clear_confusion = (enabled & lum) | confusion_cure
	white_herb = enabled & rows.boost_restore & jnp.any(state.boosts < 0, axis=1) & (hp > 0)
	boosts = jnp.where(
		white_herb[:, None], jnp.where(state.boosts < 0, 0, state.boosts), state.boosts
	)
	healed = jnp.minimum(maximum, hp + jnp.maximum(maximum // 4, 1))
	new_hp = jnp.where(sitrus & enabled, healed, hp)
	member_mask = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	status_team = jnp.where(member_mask & clear_status[:, None], HEALTHY, state.status)
	counter_team = jnp.where(member_mask & clear_status[:, None], 0, state.status_counter)
	columns = jnp.arange(state.volatiles.shape[1])
	confusion_mask = (columns[None, :] == CONFUSION) & clear_confusion[:, None]
	volatiles = jnp.where(confusion_mask, 0, state.volatiles)
	hp_team = jnp.where(member_mask, new_hp[:, None].astype(state.hp.dtype), state.hp)
	state = state._replace(
		status=status_team,
		status_counter=counter_team,
		volatiles=volatiles,
		hp=hp_team,
		boosts=boosts,
	)
	return consume_active_items(state, consume | white_herb, tables)


def clear_broken_choice_locks(state, tables):
	"""Remove the soft lock when its Choice item or source move no longer exists."""
	active_moves = state.moves[SIDES, state.active]
	choice_move = state.volatiles[:, CHOICE_MOVE]
	item = state.item[SIDES, state.active].astype(jnp.int32)
	has_move = jnp.any(active_moves == choice_move[:, None], axis=1)
	choice_item = tables.item_hooks.rows[item, CHOICE] != 0
	stale = (choice_move > 0) & (~choice_item | ~has_move)
	columns = jnp.arange(state.volatiles.shape[1])
	mask = stale[:, None] & (columns[None, :] == CHOICE_MOVE)
	return state._replace(volatiles=jnp.where(mask, 0, state.volatiles))


def residual_hp_item(state, enabled, tables, rows, abilities=None):
	"""Calculate Leftovers and Black Sludge effects for both holders."""
	hp = state.hp[SIDES, state.active].astype(jnp.int32)
	maximum = state.stats[SIDES, state.active, HP].astype(jnp.int32)
	types = state.types[SIDES, state.active]
	poison_type = jnp.any(types == tables.poison_type, axis=1)
	kind = rows.residual_kind
	active = enabled & (hp > 0)
	sludge = kind == RESIDUAL_BLACK_SLUDGE
	heal_hook = (kind == RESIDUAL_LEFTOVERS) | (sludge & poison_type)
	heal = active & heal_hook & (hp < maximum)
	damage = jnp.where(active & sludge & ~poison_type, jnp.maximum(maximum // 8, 1), 0)
	if abilities is not None:
		damage = jnp.where(has_hook(abilities, MAGIC_GUARD), 0, damage)
	healing = jnp.minimum(maximum - hp, jnp.maximum(maximum // 16, 1))
	return jnp.where(heal, healing, 0), damage


def apply_residual_white_herb(state, enabled, rows, tables=None):
	active = enabled & (state.hp[SIDES, state.active] > 0)
	trigger = active & rows.boost_restore & jnp.any(state.boosts < 0, axis=1)
	cleared = jnp.where(state.boosts < 0, 0, state.boosts)
	state = state._replace(boosts=jnp.where(trigger[:, None], cleared, state.boosts))
	return consume_active_items(state, trigger, tables)


def apply_residual_status_items(state, enabled, key, forced, tables, rows, abilities, weather):
	"""Apply Toxic and Flame Orbs together; their effects touch only the holder."""
	del key, forced
	kind = rows.residual_kind
	toxic = kind == RESIDUAL_TOXIC_ORB
	status = jnp.where(toxic, TOXIC, BURN).astype(state.status.dtype)
	types = state.types[SIDES, state.active]
	immune = jnp.any(
		(types >= 0) & tables.status_immune[status[:, None], jnp.maximum(types, 0)], axis=1
	)
	immune |= tables.weather_status_immune[weather, status] | status_immunity(
		abilities, status, weather
	)
	attempt = enabled & (state.hp[SIDES, state.active] > 0) & (toxic | (kind == RESIDUAL_FLAME_ORB))
	applied = attempt & (state.status[SIDES, state.active] == HEALTHY) & ~immune
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	write = slots & applied[:, None]
	state = state._replace(
		status=jnp.where(write, status[:, None], state.status),
		status_counter=jnp.where(write, 0, state.status_counter),
		sleep_from_foe=jnp.where(write, 0, state.sleep_from_foe),
	)
	return reveal_active_items(state, applied)
