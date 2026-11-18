"""Build compact NumPy tables from the resolved Showdown export.

Rows are ordered by Showdown id, except types retain the export's order.
The four entity tables reserve an entirely zero row for the empty id.
Arrays stay on the CPU; the engine will convert them to JAX when needed.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gen5-data-v1.json"
STAT_NAMES = ("hp", "atk", "def", "spa", "spd", "spe")
ABILITY_SLOTS = ("0", "1", "H")
SPECIES_TYPE_SLOTS = 2
NO_SECOND_TYPE = -1
NONE_ID = ""
NONE_ROW = 0
SECONDARY_SLOTS = 2
BOOST_NAMES = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
STATUS_NAMES = ("", "brn", "psn", "tox", "par", "slp", "frz")
VOLATILE_NAMES = ("", "confusion", "flinch")
CATEGORY_INDEX = {"Physical": 0, "Special": 1, "Status": 2}
NATURE_NEUTRAL = 10
NATURE_PLUS = 11
NATURE_MINUS = 9
TYPE_NEUTRAL = 0
TYPE_WEAK = 1
TYPE_RESIST = 2
TYPE_IMMUNE = 3
TYPE_EXPONENT = {TYPE_NEUTRAL: 0, TYPE_WEAK: 1, TYPE_RESIST: -1, TYPE_IMMUNE: 0}


def _read_data(path):
	with Path(path).open(encoding="utf-8") as source:
		return json.load(source)


@dataclass(frozen=True)
class RowTable:
	ids: tuple[str, ...]
	index: dict[str, int]
	names: tuple[str, ...]


@dataclass(frozen=True)
class SpeciesTable(RowTable):
	cosmetic_aliases: dict[str, str]
	base_stats: NDArray[np.uint8]
	types: NDArray[np.int8]
	abilities: NDArray[np.int16]
	weight_hg: NDArray[np.int16]
	num: NDArray[np.int16]


@dataclass(frozen=True)
class MoveTable(RowTable):
	num: NDArray[np.int16]
	type: NDArray[np.int8]
	category: NDArray[np.uint8]
	base_power: NDArray[np.int16]
	accuracy: NDArray[np.uint8]
	always_hits: NDArray[np.bool_]
	priority: NDArray[np.int8]
	pp: NDArray[np.uint8]
	no_pp_boosts: NDArray[np.bool_]
	crit_ratio: NDArray[np.uint8]
	crit_always: NDArray[np.bool_]
	drain: NDArray[np.uint8]
	recoil: NDArray[np.uint8]
	recoil_basehp: NDArray[np.bool_]
	heal: NDArray[np.uint8]
	heal_weather: NDArray[np.uint8]
	damage_kind: NDArray[np.uint8]
	fixed_damage: NDArray[np.int16]
	hit_min: NDArray[np.uint8]
	hit_max: NDArray[np.uint8]
	self_faint: NDArray[np.uint8]
	crash_damage: NDArray[np.bool_]
	pain_split: NDArray[np.bool_]
	roost: NDArray[np.bool_]
	target: NDArray[np.uint8]
	flags: NDArray[np.bool_]
	boosts: NDArray[np.int8]
	self_boosts: NDArray[np.int8]
	secondary_chance: NDArray[np.uint8]
	secondary_self: NDArray[np.bool_]
	secondary_status: NDArray[np.int8]
	secondary_status_choices: NDArray[np.int8]
	secondary_boosts: NDArray[np.int8]
	secondary_volatile: NDArray[np.int8]
	confusion: NDArray[np.bool_]
	stalling_move: NDArray[np.bool_]
	volatile_status: NDArray[np.uint8]
	self_volatile: NDArray[np.uint8]
	breaks_protect: NDArray[np.bool_]
	target_names: tuple[str, ...]
	flag_names: tuple[str, ...]


@dataclass(frozen=True)
class AbilityTable(RowTable):
	num: NDArray[np.int16]


@dataclass(frozen=True)
class ItemTable(RowTable):
	num: NDArray[np.int16]
	is_berry: NDArray[np.bool_]


@dataclass(frozen=True)
class NatureTable(RowTable):
	multipliers: NDArray[np.int16]


@dataclass(frozen=True)
class TypeTable(RowTable):
	exponent: NDArray[np.int8]
	immune: NDArray[np.bool_]


@dataclass(frozen=True)
class Tables:
	species: SpeciesTable
	moves: MoveTable
	abilities: AbilityTable
	items: ItemTable
	natures: NatureTable
	types: TypeTable


def _identity(rows, *, reserve_none=False):
	ids = tuple(row["id"] for row in rows)
	names = tuple(row["name"] for row in rows)
	if reserve_none:
		ids = (NONE_ID,) + ids
		names = ("",) + names
	index = {id_: row for row, id_ in enumerate(ids)}
	if len(index) != len(ids):
		raise ValueError("Duplicate Showdown ids in table")
	return ids, index, names


def _showdown_id(name):
	return "".join(char for char in name.lower() if char.isascii() and char.isalnum())


def _build_types(rows):
	ids, index, names = _identity(rows)
	shape = (len(rows), len(rows))
	exponent = np.zeros(shape, dtype=np.int8)
	immune = np.zeros(shape, dtype=np.bool_)
	for defender, defending in enumerate(rows):
		for attacker, attacking in enumerate(rows):
			code = defending["damageTaken"].get(attacking["name"], TYPE_NEUTRAL)
			exponent[attacker, defender] = TYPE_EXPONENT[code]
			immune[attacker, defender] = code == TYPE_IMMUNE
	return TypeTable(ids, index, names, exponent, immune)


def _build_natures(rows):
	ids, index, names = _identity(rows)
	multipliers = np.full((len(rows), len(STAT_NAMES)), NATURE_NEUTRAL, dtype=np.int16)
	for row, nature in enumerate(rows):
		if "plus" in nature:
			multipliers[row, STAT_NAMES.index(nature["plus"])] = NATURE_PLUS
		if "minus" in nature:
			multipliers[row, STAT_NAMES.index(nature["minus"])] = NATURE_MINUS
	return NatureTable(ids, index, names, multipliers)


def _build_species(rows, types, abilities):
	ids, index, names = _identity(rows, reserve_none=True)
	# Match generate-teams.cjs: preserve exported forms; collapse only cosmetic
	# forms explicitly marked by Showdown when they have no row in this export.
	cosmetic_aliases = {
		_showdown_id(form): species["id"]
		for species in rows
		for form in species.get("cosmeticFormes", ())
		if _showdown_id(form) not in index
	}
	count = len(ids)
	base_stats = np.zeros((count, len(STAT_NAMES)), dtype=np.uint8)
	type_rows = np.zeros((count, SPECIES_TYPE_SLOTS), dtype=np.int8)
	ability_rows = np.zeros((count, len(ABILITY_SLOTS)), dtype=np.int16)
	weight_hg = np.zeros(count, dtype=np.int16)
	num = np.zeros(count, dtype=np.int16)
	for row, species in enumerate(rows, start=1):
		base_stats[row] = [species["baseStats"][stat] for stat in STAT_NAMES]
		type_rows[row] = NO_SECOND_TYPE
		for slot, name in enumerate(species["types"]):
			type_rows[row, slot] = types.index[_showdown_id(name)]
		for slot, key in enumerate(ABILITY_SLOTS):
			ability = _showdown_id(species["abilities"].get(key, NONE_ID))
			# Gothita lists Competitive, which is absent from this Gen 5 ability export.
			# Unavailable abilities use the none row; do not add synthetic ability rows.
			ability_rows[row, slot] = abilities.index.get(ability, NONE_ROW)
		weight_hg[row] = species["weighthg"]
		num[row] = species["num"]
	return SpeciesTable(
		ids, index, names, cosmetic_aliases, base_stats, type_rows, ability_rows, weight_hg, num
	)


def _build_moves(rows, types, target_names, flag_names):
	target_index = {name: index for index, name in enumerate(target_names)}
	ids, index, names = _identity(rows, reserve_none=True)
	count = len(ids)
	columns = {
		"num": np.zeros(count, dtype=np.int16),
		"type": np.zeros(count, dtype=np.int8),
		"category": np.zeros(count, dtype=np.uint8),
		"base_power": np.zeros(count, dtype=np.int16),
		"accuracy": np.zeros(count, dtype=np.uint8),
		"always_hits": np.zeros(count, dtype=np.bool_),
		"priority": np.zeros(count, dtype=np.int8),
		"pp": np.zeros(count, dtype=np.uint8),
		"no_pp_boosts": np.zeros(count, dtype=np.bool_),
		"crit_ratio": np.zeros(count, dtype=np.uint8),
		"crit_always": np.zeros(count, dtype=np.bool_),
		"drain": np.zeros((count, 2), dtype=np.uint8),
		"recoil": np.zeros((count, 2), dtype=np.uint8),
		"recoil_basehp": np.zeros(count, dtype=np.bool_),
		"heal": np.zeros((count, 2), dtype=np.uint8),
		"heal_weather": np.zeros(count, dtype=np.uint8),
		"damage_kind": np.zeros(count, dtype=np.uint8),
		"fixed_damage": np.zeros(count, dtype=np.int16),
		"hit_min": np.zeros(count, dtype=np.uint8),
		"hit_max": np.zeros(count, dtype=np.uint8),
		"self_faint": np.zeros(count, dtype=np.uint8),
		"crash_damage": np.zeros(count, dtype=np.bool_),
		"pain_split": np.zeros(count, dtype=np.bool_),
		"roost": np.zeros(count, dtype=np.bool_),
		"target": np.zeros(count, dtype=np.uint8),
		"flags": np.zeros((count, len(flag_names)), dtype=np.bool_),
		"boosts": np.zeros((count, 7), dtype=np.int8),
		"self_boosts": np.zeros((count, 7), dtype=np.int8),
		"secondary_chance": np.zeros((count, SECONDARY_SLOTS), dtype=np.uint8),
		"secondary_self": np.zeros((count, SECONDARY_SLOTS), dtype=np.bool_),
		"secondary_status": np.zeros((count, SECONDARY_SLOTS), dtype=np.int8),
		"secondary_status_choices": np.zeros((count, SECONDARY_SLOTS, 3), dtype=np.int8),
		"secondary_boosts": np.zeros((count, SECONDARY_SLOTS, 7), dtype=np.int8),
		"secondary_volatile": np.zeros((count, SECONDARY_SLOTS), dtype=np.int8),
		"confusion": np.zeros(count, dtype=np.bool_),
		"stalling_move": np.zeros(count, dtype=np.bool_),
		"volatile_status": np.zeros(count, dtype=np.uint8),
		"self_volatile": np.zeros(count, dtype=np.uint8),
		"breaks_protect": np.zeros(count, dtype=np.bool_),
	}
	volatile_statuses = {"protect": 1, "endure": 2, "partiallytrapped": 3, "smackdown": 4}
	self_volatiles = {"lockedmove": 1, "mustrecharge": 2}
	for row, move in enumerate(rows, start=1):
		columns["confusion"][row] = move.get("volatileStatus") == "confusion"
		columns["crit_always"][row] = move.get("willCrit", False)
		for column, key in (("drain", "drain"), ("recoil", "recoil"), ("heal", "heal")):
			fraction = move.get(key)
			if fraction:
				columns[column][row] = fraction
		columns["recoil_basehp"][row] = move.get("struggleRecoil", False)
		if columns["recoil_basehp"][row]:
			columns["recoil"][row] = [1, 4]
		heal_callback = (move.get("onHit") or {}).get("$callback", "")
		# The three Gen 5 weather heals store their clear-weather fraction here;
		# Phase 6 supplies the weather-dependent override from this marker.
		columns["heal_weather"][row] = int(
			heal_callback
			in {
				"moves:moonlight:onHit",
				"moves:morningsun:onHit",
				"moves:synthesis:onHit",
			}
		)
		if columns["heal_weather"][row]:
			columns["heal"][row] = [1, 2]
		callback = (move.get("damageCallback") or {}).get("$callback", "")
		fixed_damage = move.get("damage")
		columns["damage_kind"][row] = (
			1
			if fixed_damage == "level"
			else 2
			if callback == "moves:superfang:damageCallback"
			else 3
			if isinstance(fixed_damage, int)
			else 0
		)
		if columns["damage_kind"][row] == 3:
			columns["fixed_damage"][row] = fixed_damage
		multihit = move.get("multihit", 1)
		if isinstance(multihit, list):
			columns["hit_min"][row], columns["hit_max"][row] = multihit
		else:
			columns["hit_min"][row] = columns["hit_max"][row] = multihit
		columns["self_faint"][row] = {"always": 1, "ifHit": 2}.get(move.get("selfdestruct"), 0)
		columns["crash_damage"][row] = move.get("hasCrashDamage", False)
		columns["pain_split"][row] = heal_callback == "moves:painsplit:onHit"
		columns["roost"][row] = move.get("self", {}).get("volatileStatus") == "roost"
		for slot, secondary in enumerate(move.get("secondaries") or ()):
			if slot >= SECONDARY_SLOTS:
				raise ValueError("Increase SECONDARY_SLOTS for this export")
			columns["secondary_chance"][row, slot] = secondary.get("chance", 100)
			columns["secondary_self"][row, slot] = "self" in secondary
			effect = secondary.get("self", secondary)
			columns["secondary_status"][row, slot] = STATUS_NAMES.index(effect.get("status", ""))
			# Tri Attack's sole callback samples these three statuses uniformly
			# (data/moves.ts). Keep the choice itself in the move table.
			if move["id"] == "triattack":
				columns["secondary_status_choices"][row, slot] = [1, 4, 6]
			volatile = effect.get("volatileStatus", "")
			if volatile in VOLATILE_NAMES:
				columns["secondary_volatile"][row, slot] = VOLATILE_NAMES.index(volatile)
			for stat, amount in effect.get("boosts", {}).items():
				columns["secondary_boosts"][row, slot, BOOST_NAMES.index(stat)] = amount
		for column, key in (
			("num", "num"),
			("base_power", "basePower"),
			("priority", "priority"),
			("pp", "pp"),
			("crit_ratio", "critRatio"),
		):
			columns[column][row] = move[key]
		for column, values in (
			("boosts", move.get("boosts", {})),
			("self_boosts", move.get("self", {}).get("boosts", {})),
		):
			for stat, amount in values.items():
				columns[column][
					row, ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion").index(stat)
				] = amount
		columns["no_pp_boosts"][row] = move.get("noPPBoosts", False)
		columns["stalling_move"][row] = move.get("stallingMove", False)
		columns["volatile_status"][row] = volatile_statuses.get(move.get("volatileStatus"), 0)
		columns["self_volatile"][row] = self_volatiles.get(
			move.get("self", {}).get("volatileStatus"), 0
		)
		columns["breaks_protect"][row] = move.get("breaksProtect", False)
		columns["type"][row] = types.index[_showdown_id(move["type"])]
		columns["category"][row] = CATEGORY_INDEX[move["category"]]
		columns["always_hits"][row] = move["accuracy"] is True
		columns["accuracy"][row] = 0 if move["accuracy"] is True else move["accuracy"]
		columns["target"][row] = target_index[move["target"]]
		columns["flags"][row] = [bool(move["flags"].get(flag)) for flag in flag_names]
	return MoveTable(ids, index, names, **columns, target_names=target_names, flag_names=flag_names)


def load_tables(path=None) -> Tables:
	"""Load all entries; missing exported ability references map to the none row.

	Move target and flag names are sorted from the selected export and stored
	on MoveTable: target[row] indexes target_names, and flags[:, j] corresponds
	to flag_names[j]. An explicit path can select another export.
	The default data lives at the repository root, independent of the cwd.
	"""
	data = _read_data(DEFAULT_DATA_PATH if path is None else path)
	rows = {
		kind: sorted(data[kind], key=lambda entry: entry["id"])
		for kind in ("species", "moves", "abilities", "items", "natures")
	}
	target_names = tuple(sorted({move["target"] for move in data["moves"]}))
	flag_names = tuple(sorted({flag for move in data["moves"] for flag in move["flags"]}))
	types = _build_types(data["types"])
	ability_ids, ability_index, ability_names = _identity(rows["abilities"], reserve_none=True)
	abilities = AbilityTable(
		ability_ids,
		ability_index,
		ability_names,
		np.array([0] + [entry["num"] for entry in rows["abilities"]], dtype=np.int16),
	)
	item_ids, item_index, item_names = _identity(rows["items"], reserve_none=True)
	items = ItemTable(
		item_ids,
		item_index,
		item_names,
		np.array([0] + [entry["num"] for entry in rows["items"]], dtype=np.int16),
		np.array([False] + [entry["isBerry"] for entry in rows["items"]], dtype=np.bool_),
	)
	return Tables(
		species=_build_species(rows["species"], types, abilities),
		moves=_build_moves(rows["moves"], types, target_names, flag_names),
		abilities=abilities,
		items=items,
		natures=_build_natures(rows["natures"]),
		types=types,
	)
