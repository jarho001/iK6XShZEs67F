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
	target: NDArray[np.uint8]
	flags: NDArray[np.bool_]
	boosts: NDArray[np.int8]
	self_boosts: NDArray[np.int8]
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
		"target": np.zeros(count, dtype=np.uint8),
		"flags": np.zeros((count, len(flag_names)), dtype=np.bool_),
		"boosts": np.zeros((count, 7), dtype=np.int8),
		"self_boosts": np.zeros((count, 7), dtype=np.int8),
	}
	for row, move in enumerate(rows, start=1):
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
