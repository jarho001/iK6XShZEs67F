"""Check reproducible Showdown teams and their compact array representation."""

import json
import subprocess
from copy import deepcopy
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest

from pokejax.tables import STAT_NAMES, load_tables
from pokejax.teams import (
	FEMALE,
	GENDERLESS,
	HIDDEN_POWER_TYPES,
	HIDDEN_POWER_WEIGHTS,
	MALE,
	load_npz,
	load_team_pool,
	save_npz,
)

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "reference/vendor/pokemon-showdown/dist").is_dir():
	pytest.skip(
		"Showdown dist is missing; run node reference/setup.cjs to enable team pool tests",
		allow_module_level=True,
	)


@pytest.fixture(scope="module")
def tables():
	return load_tables()


def generate(path, seed=1, count=50):
	result = subprocess.run(
		[
			"node",
			str(ROOT / "reference/generate-teams.cjs"),
			"--count",
			str(count),
			"--seed",
			str(seed),
			"--out",
			str(path),
		],
		cwd=ROOT,
		capture_output=True,
		text=True,
		check=True,
	)
	assert "Validation:" not in result.stderr
	return path


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
	return generate(tmp_path_factory.mktemp("teams") / "teams.json")


@pytest.fixture(scope="module")
def data(generated):
	return json.loads(generated.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pool(generated, tables):
	return load_team_pool(generated, tables)


def test_determinism(generated, data, tmp_path):
	repeated = generate(tmp_path / "repeated.json")
	assert repeated.read_bytes() == generated.read_bytes()
	other = generate(tmp_path / "other.json", seed=2)
	assert json.loads(other.read_text())["teams"] != data["teams"]
	# A team's seed is independent of the requested pool length.
	prefix = generate(tmp_path / "prefix.json", count=2)
	assert json.loads(prefix.read_text())["teams"] == data["teams"][:2]


def test_metadata_and_shapes(pool, data):
	assert pool.seed == data["seed"] == 1
	lock = json.loads((ROOT / "reference-lock.json").read_text())
	assert pool.showdown_commit == data["showdownCommit"] == lock["commit"]
	assert data["format"] == "gen5randombattle"
	assert data["count"] == 50
	columns = {
		"species": (np.int16, (50, 6)),
		"level": (np.uint8, (50, 6)),
		"moves": (np.int16, (50, 6, 4)),
		"ability": (np.int16, (50, 6)),
		"item": (np.int16, (50, 6)),
		"nature": (np.int8, (50, 6)),
		"gender": (np.int8, (50, 6)),
		"evs": (np.uint8, (50, 6, 6)),
		"ivs": (np.uint8, (50, 6, 6)),
		"happiness": (np.uint8, (50, 6)),
		"team_size": (np.uint8, (50,)),
	}
	for name, (dtype, shape) in columns.items():
		array = getattr(pool, name)
		assert isinstance(array, np.ndarray)
		assert array.dtype == np.dtype(dtype)
		assert array.shape == shape
	assert np.all(pool.species > 0)
	assert np.all(pool.ability > 0)
	assert np.all(pool.team_size == 6)
	assert np.all(np.count_nonzero(pool.moves, axis=-1) >= 1)
	assert np.all((pool.level >= 1) & (pool.level <= 100))
	assert np.all(pool.evs <= 255)
	assert np.all(pool.ivs <= 31)
	assert set(np.unique(pool.gender)) <= {GENDERLESS, MALE, FEMALE}
	with pytest.raises(FrozenInstanceError):
		pool.seed = 2


def test_npz_round_trip(pool, tmp_path):
	path = tmp_path / "teams.npz"
	save_npz(pool, path)
	loaded = load_npz(path)
	for field in fields(pool):
		original = getattr(pool, field.name)
		actual = getattr(loaded, field.name)
		if isinstance(original, np.ndarray):
			assert actual.dtype == original.dtype
			np.testing.assert_array_equal(actual, original)
		else:
			assert actual == original
			assert type(actual) is type(original)


def test_ids_and_stats_round_trip(pool, data, tables):
	hidden_power_count = 0
	for t, team in enumerate(data["teams"]):
		for slot, member in enumerate(team):
			for column, kind in (
				("species", "species"),
				("ability", "abilities"),
				("item", "items"),
				("nature", "natures"),
			):
				assert getattr(tables, kind).ids[getattr(pool, column)[t, slot]] == member[column]
			for m, move in enumerate(member["moves"]):
				expected = move
				if move.startswith("hiddenpower") and move not in tables.moves.index:
					hidden_power_count += 1
					expected = "hiddenpower"
					ivs = pool.ivs[t, slot].astype(np.int32)
					bits = int(np.sum((ivs & 1) * HIDDEN_POWER_WEIGHTS))
					assert HIDDEN_POWER_TYPES[bits * 15 // 63] == move[11:]
				assert tables.moves.ids[pool.moves[t, slot, m]] == expected
			for column in ("evs", "ivs"):
				np.testing.assert_array_equal(
					getattr(pool, column)[t, slot], [member[column][stat] for stat in STAT_NAMES]
				)
			assert pool.level[t, slot] == member["level"]
			assert pool.happiness[t, slot] == member["happiness"]
			assert (
				pool.gender[t, slot] == {"N": GENDERLESS, "M": MALE, "F": FEMALE}[member["gender"]]
			)
	assert hidden_power_count > 0


def write_data(tmp_path, data):
	path = tmp_path / "modified.json"
	path.write_text(json.dumps(data), encoding="utf-8")
	return path


def test_all_unresolved_ids_reported(data, tables, tmp_path):
	data = deepcopy(data)
	for member in data["teams"][0][:2]:
		member.update(species="unknownspecies", ability="unknownability", item="unknownitem")
		member["nature"] = "unknownnature"
		member["moves"] = ["unknownmove", "anothermove"]
	with pytest.raises(ValueError) as error:
		load_team_pool(write_data(tmp_path, data), tables)
	for kind, id_ in (
		("species", "unknownspecies"),
		("abilities", "unknownability"),
		("items", "unknownitem"),
		("natures", "unknownnature"),
		("moves", "unknownmove"),
		("moves", "anothermove"),
	):
		assert f"{kind}.{id_}: 2" in str(error.value)


def test_empty_item_and_padded_moves(data, tables, tmp_path):
	data = deepcopy(data)
	member = data["teams"][0][0]
	member["item"] = ""
	member["moves"] = ["tackle"]
	loaded = load_team_pool(write_data(tmp_path, data), tables)
	assert loaded.item[0, 0] == 0
	np.testing.assert_array_equal(loaded.moves[0, 0], [tables.moves.index["tackle"], 0, 0, 0])


@pytest.mark.parametrize("column,value", (("level", 256), ("happiness", -1)))
def test_reject_numeric_overflow(data, tables, tmp_path, column, value):
	data = deepcopy(data)
	data["teams"][0][0][column] = value
	with pytest.raises(ValueError, match=column):
		load_team_pool(write_data(tmp_path, data), tables)


def test_require_six_members(data, tables, tmp_path):
	data = deepcopy(data)
	data["teams"][0].pop()
	with pytest.raises(ValueError, match="expected 6"):
		load_team_pool(write_data(tmp_path, data), tables)


def test_hidden_power_type_mismatch(data, tables, tmp_path):
	data = deepcopy(data)
	member = data["teams"][0][0]
	member["moves"] = ["hiddenpowerfire"]
	member["ivs"] = dict.fromkeys(STAT_NAMES, 31)  # All odd IVs mean Dark.
	with pytest.raises(ValueError, match="Hidden Power IVs"):
		load_team_pool(write_data(tmp_path, data), tables)
