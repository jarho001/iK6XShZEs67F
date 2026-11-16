"""Check the compact tables against the source export."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

from pokejax.tables import STAT_NAMES, load_tables

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "gen5-data-v1.json"


@pytest.fixture(scope="module")
def data():
	return json.loads(DATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tables():
	return load_tables()


def test_counts_ids_and_none_rows(tables, data):
	for kind in ("species", "moves", "abilities", "items", "natures", "types"):
		table = getattr(tables, kind)
		reserved = kind in ("species", "moves", "abilities", "items")
		expected = tuple(entry["id"] for entry in data[kind])
		if kind != "types":
			expected = tuple(sorted(expected))
		if reserved:
			expected = ("",) + expected
		assert table.ids == expected
		names = {entry["id"]: entry["name"] for entry in data[kind]}
		assert table.names == tuple(names.get(id_, "") for id_ in expected)
		assert len(table.ids) == len(data[kind]) + int(reserved)
		assert table.index == {id_: row for row, id_ in enumerate(expected)}
		if reserved:
			for field in fields(table):
				column = getattr(table, field.name)
				if isinstance(column, np.ndarray):
					assert np.all(column[0] == 0), (kind, field.name)
	with pytest.raises(FrozenInstanceError):
		tables.moves = None


def test_shapes_and_dtypes(tables, data):
	shapes = {
		"species": {
			"base_stats": (np.uint8, (6,)),
			"types": (np.int8, (2,)),
			"abilities": (np.int16, (3,)),
			"weight_hg": (np.int16, ()),
			"num": (np.int16, ()),
		},
		"moves": {
			"num": (np.int16, ()),
			"type": (np.int8, ()),
			"category": (np.uint8, ()),
			"base_power": (np.int16, ()),
			"accuracy": (np.uint8, ()),
			"always_hits": (np.bool_, ()),
			"priority": (np.int8, ()),
			"pp": (np.uint8, ()),
			"crit_ratio": (np.uint8, ()),
			"target": (np.uint8, ()),
			"flags": (np.bool_, (len(tables.moves.flag_names),)),
		},
		"abilities": {"num": (np.int16, ())},
		"items": {"num": (np.int16, ()), "is_berry": (np.bool_, ())},
		"natures": {"multipliers": (np.int16, (6,))},
		"types": {"exponent": (np.int8, (17,)), "immune": (np.bool_, (17,))},
	}
	for kind, columns in shapes.items():
		table = getattr(tables, kind)
		count = len(data[kind]) + int(kind in ("species", "moves", "abilities", "items"))
		for name, (dtype, trailing) in columns.items():
			column = getattr(table, name)
			assert isinstance(column, np.ndarray)
			assert column.dtype == np.dtype(dtype), (kind, name)
			assert column.shape == (count, *trailing), (kind, name)


def test_species_round_trip(tables, data):
	# Include forms, missing ability references, and the extra event ability slot.
	for id_ in ("abomasnow", "abra", "gothita", "basculinbluestriped", "shedinja"):
		source = next(entry for entry in data["species"] if entry["id"] == id_)
		row = tables.species.index[id_]
		assert tables.species.ids[row] == source["id"]
		np.testing.assert_array_equal(
			tables.species.base_stats[row], [source["baseStats"][stat] for stat in STAT_NAMES]
		)
		expected_types = [tables.types.index[name.lower()] for name in source["types"]]
		if len(expected_types) == 1:
			expected_types.append(-1)
		np.testing.assert_array_equal(tables.species.types[row], expected_types)
		for column, slot in enumerate(("0", "1", "H")):
			ability_name = source["abilities"].get(slot, "")
			ability = next((a for a in data["abilities"] if a["name"] == ability_name), None)
			expected = tables.abilities.index[ability["id"]] if ability else 0
			assert tables.species.abilities[row, column] == expected
		assert tables.species.weight_hg[row] == source["weighthg"]
		assert tables.species.num[row] == source["num"]


def test_moves_round_trip(tables, data):
	# All rows are cheap to check and cover always-hit moves and every target.
	for source in data["moves"]:
		row = tables.moves.index[source["id"]]
		assert tables.moves.ids[row] == source["id"]
		for column, key in (
			("num", "num"),
			("base_power", "basePower"),
			("priority", "priority"),
			("pp", "pp"),
			("crit_ratio", "critRatio"),
		):
			assert getattr(tables.moves, column)[row] == source[key]
		assert tables.types.ids[tables.moves.type[row]] == source["type"].lower()
		assert ("Physical", "Special", "Status")[tables.moves.category[row]] == source["category"]
		assert tables.moves.always_hits[row] == (source["accuracy"] is True)
		assert tables.moves.accuracy[row] == (
			0 if source["accuracy"] is True else source["accuracy"]
		)
		assert tables.moves.target_names[tables.moves.target[row]] == source["target"]
	assert tables.moves.target_names == tuple(sorted({move["target"] for move in data["moves"]}))
	assert tables.moves.flag_names == tuple(
		sorted({flag for move in data["moves"] for flag in move["flags"]})
	)


@pytest.mark.parametrize("move_id", ("tackle", "protect", "earthquake", "toxic"))
def test_move_flags_match_json(tables, data, move_id):
	source = next(move for move in data["moves"] if move["id"] == move_id)
	row = tables.moves.index[move_id]
	for j, flag_name in enumerate(tables.moves.flag_names):
		assert tables.moves.flags[row, j] == bool(source["flags"].get(flag_name))


def test_ability_and_item_values(tables, data):
	for kind in ("abilities", "items"):
		table = getattr(tables, kind)
		for source in data[kind]:
			row = table.index[source["id"]]
			assert table.num[row] == source["num"]
			if kind == "items":
				assert table.is_berry[row] == source["isBerry"]


def test_type_chart(tables, data):
	names = {entry["name"] for entry in data["types"]}
	expected_immunities = sum(
		code == 3
		for defender in data["types"]
		for name, code in defender["damageTaken"].items()
		if name in names
	)
	assert tables.types.immune.sum() == expected_immunities
	assert set(np.unique(tables.types.exponent)) <= {-1, 0, 1}
	for defender, source in enumerate(data["types"]):
		for attacker, attacking in enumerate(data["types"]):
			code = source["damageTaken"].get(attacking["name"], 0)
			assert tables.types.exponent[attacker, defender] == {0: 0, 1: 1, 2: -1, 3: 0}[code]
			assert tables.types.immune[attacker, defender] == (code == 3)


def test_natures(tables, data):
	multipliers = tables.natures.multipliers
	assert multipliers.shape == (25, 6)
	assert np.all(multipliers[:, 0] == 10)
	for source in data["natures"]:
		row = multipliers[tables.natures.index[source["id"]]]
		assert np.all(row == 10) or (
			np.count_nonzero(row == 11) == 1
			and np.count_nonzero(row == 9) == 1
			and np.count_nonzero(row == 10) == 4
		)
		expected = [10] * len(STAT_NAMES)
		if "plus" in source:
			expected[STAT_NAMES.index(source["plus"])] = 11
		if "minus" in source:
			expected[STAT_NAMES.index(source["minus"])] = 9
		np.testing.assert_array_equal(row, expected)


def test_default_path_is_independent_of_cwd(tables, tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	loaded = load_tables()
	assert loaded.species.ids == tables.species.ids
	np.testing.assert_array_equal(loaded.moves.flags, tables.moves.flags)


def test_explicit_path_and_input_order(tables, data, tmp_path):
	# Type order is meaningful, while all entity ids must be sorted by the loader.
	for kind in ("species", "moves", "abilities", "items", "natures"):
		data[kind] = list(reversed(data[kind]))
	path = tmp_path / "export.json"
	path.write_text(json.dumps(data), encoding="utf-8")
	loaded = load_tables(path)
	assert loaded.species.ids == tables.species.ids
	assert loaded.types.ids == tables.types.ids
	np.testing.assert_array_equal(loaded.species.abilities, tables.species.abilities)
	np.testing.assert_array_equal(loaded.moves.flags, tables.moves.flags)


def test_move_names_come_from_selected_export(data, tmp_path):
	data = deepcopy(data)
	move = data["moves"][0]
	move["name"] = "Custom Display Name"
	move["target"] = "customTarget"
	move["flags"] = {"customFlag": 1, "disabledFlag": 0}
	data["moves"] = [move]
	path = tmp_path / "custom-export.json"
	path.write_text(json.dumps(data), encoding="utf-8")
	moves = load_tables(path).moves
	assert moves.names == ("", "Custom Display Name")
	assert moves.target_names == ("customTarget",)
	assert moves.flag_names == ("customFlag", "disabledFlag")
	assert moves.target[moves.index[move["id"]]] == 0
	np.testing.assert_array_equal(moves.flags, [[False, False], [True, False]])


def test_import_does_not_read_json():
	result = subprocess.run(
		[
			sys.executable,
			"-c",
			(
				"from unittest.mock import patch\n"
				"from pathlib import Path\n"
				"with patch.object(Path, 'open', side_effect=AssertionError('file read')):\n"
				"    import pokejax.tables\n"
			),
		],
		capture_output=True,
		text=True,
		check=False,
	)
	assert result.returncode == 0, result.stderr
