"""Pinned Showdown reset oracle, plus input and compact-state checks."""

import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.data import build_engine_tables
from pokejax.engine.reset import check_teams, reset, teams_from_pool
from pokejax.engine.state import FIRST_TURN, MOVE, ONGOING
from pokejax.tables import STAT_NAMES
from pokejax.teams import FORMAT, load_team_pool

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/reset.json"


def write_fixture_pool(fixture, path):
	"""Supply loader metadata without storing an entire pool header in the fixture."""
	path.write_text(
		json.dumps(
			{
				"format": FORMAT,
				"count": len(fixture["teamSets"]),
				"seed": 0,
				"showdownCommit": fixture["showdownCommit"],
				"teams": fixture["teamSets"],
			}
		)
	)


def test_showdown_reset(tables, tmp_path):
	fixture = json.loads(FIXTURE.read_text())
	assert (
		fixture["showdownCommit"]
		== json.loads((ROOT / "reference-lock.json").read_text())["commit"]
	)
	assert fixture["pairing"] == "consecutive"
	assert len(fixture["teamSets"]) == 100
	pool_path = tmp_path / "pool.json"
	write_fixture_pool(fixture, pool_path)
	pool = load_team_pool(pool_path, tables)
	pairs = np.asarray([battle["teams"] for battle in fixture["battles"]])
	teams = teams_from_pool(pool, pairs[:, 0], pairs[:, 1])
	check_teams(teams, tables)
	state = jax.jit(jax.vmap(reset, in_axes=(0, 0, None)))(
		jax.random.split(jax.random.key(0), len(pairs)), teams, build_engine_tables(tables)
	)
	state = jax.device_get(state)
	for b, battle in enumerate(fixture["battles"]):
		assert int(state.turn[b]) == battle["turn"] == FIRST_TURN
		for side, members in enumerate(battle["sides"]):
			for slot, member in enumerate(members):
				index = b, side, slot
				stats = [member["maxhp"], *(member["storedStats"][s] for s in STAT_NAMES[1:])]
				np.testing.assert_array_equal(state.stats[index], stats)
				assert int(state.hp[index]) == member["maxhp"]
				types = [tables.types.index[t.lower()] for t in member["types"]]
				np.testing.assert_array_equal(state.types[index], types + [-1] * (2 - len(types)))
				# Showdown normalizes typed Hidden Power slots to the plain move id.
				moves = [tables.moves.index[m] for m in member["moves"]]
				np.testing.assert_array_equal(state.moves[index], moves + [0] * (4 - len(moves)))
				pp = member["maxpp"] + [0] * (4 - len(moves))
				np.testing.assert_array_equal(state.max_pp[index], pp)
	np.testing.assert_array_equal(state.pp, state.max_pp)
	assert np.all(state.active == 0)
	assert np.all(state.boosts == 0)
	assert np.all(state.status == 0)
	assert np.all(state.status_counter == 0)
	assert np.all(state.request == MOVE)
	assert np.all(state.winner == ONGOING)
	for name in teams._fields:
		np.testing.assert_array_equal(getattr(state, name), getattr(teams, name))


def test_oracle_regeneration(tmp_path):
	if not (ROOT / "reference/vendor/pokemon-showdown/dist").is_dir():
		pytest.skip("Run node reference/setup.cjs to check oracle regeneration")
	fixture = json.loads(FIXTURE.read_text())
	pool = tmp_path / "pool.json"
	write_fixture_pool(fixture, pool)
	out = tmp_path / "reset.json"
	subprocess.run(
		["node", str(ROOT / "reference/dump-reset.cjs"), "--pool", str(pool), "--out", str(out)],
		check=True,
		capture_output=True,
		text=True,
	)
	assert out.read_bytes() == FIXTURE.read_bytes()


def test_compact_state_and_shedinja(synthetic_pool, tables):
	teams = teams_from_pool(synthetic_pool, 0, 1)
	teams = teams._replace(species=teams.species.at[0, 0].set(tables.species.index["shedinja"]))
	check_teams(teams, tables)
	t = build_engine_tables(tables)
	state = jax.jit(reset)(jax.random.key(0), teams, t)
	assert sum(leaf.nbytes for leaf in jax.tree.leaves(state)) < 4096
	assert int(state.hp[0, 0]) == 1
	assert int(state.stats[0, 0, 0]) == 1
	assert all(jnp.issubdtype(leaf.dtype, jnp.integer) for leaf in jax.tree.leaves(state))
	other = reset(jax.random.key(1), teams, t)
	for a, b in zip(state, other):
		np.testing.assert_array_equal(a, b)


def test_pool_selection(synthetic_pool, tables):
	batch = teams_from_pool(synthetic_pool, [0, 1, 2], 1)
	check_teams(batch, tables)
	for i in range(3):
		single = teams_from_pool(synthetic_pool, i, 1)
		for name in single._fields:
			np.testing.assert_array_equal(getattr(batch, name)[i], getattr(single, name))
	for a, b in ((-1, 0), (0, 3), (0.5, 0)):
		with pytest.raises(ValueError):
			teams_from_pool(synthetic_pool, a, b)


@pytest.mark.parametrize(
	"name,value",
	[
		("species", 0),
		("species", 10000),
		("level", 0),
		("level", 101),
		("moves", -1),
		("ability", -1),
		("item", 10000),
		("nature", 25),
		("gender", 3),
		("evs", 256),
		("ivs", 32),
		("happiness", -1),
	],
)
def test_check_teams_rejects_range(synthetic_pool, tables, name, value):
	teams = teams_from_pool(synthetic_pool, 0, 1)
	array = np.array(getattr(teams, name), dtype=np.int32)
	array.flat[0] = value
	with pytest.raises(ValueError, match=name):
		check_teams(teams._replace(**{name: array}), tables)


def test_check_teams_shapes_dtypes_and_moves(synthetic_pool, tables):
	teams = teams_from_pool(synthetic_pool, 0, 1)
	for bad in (
		teams._replace(species=teams.species[0]),
		teams._replace(evs=teams.evs[..., :5]),
		teams._replace(level=teams.level.astype(jnp.float32)),
		teams._replace(moves=jnp.zeros_like(teams.moves)),
		teams._replace(moves=teams.moves.at[0, 0, 2].set(1)),
	):
		with pytest.raises(ValueError):
			check_teams(bad, tables)


def test_engine_tables_and_pp(tables):
	t = build_engine_tables(tables)
	assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(t))
	for name, expected in (
		("species_base_stats", tables.species.base_stats),
		("species_types", tables.species.types),
		("move_type", tables.moves.type),
		("move_category", tables.moves.category),
		("move_base_power", tables.moves.base_power),
		("move_accuracy", tables.moves.accuracy),
		("move_always_hits", tables.moves.always_hits),
		("move_priority", tables.moves.priority),
		("move_crit_ratio", tables.moves.crit_ratio),
		("move_flags", tables.moves.flags),
		("type_exponent", tables.types.exponent),
		("type_immune", tables.types.immune),
		("nature_multipliers", tables.natures.multipliers),
	):
		np.testing.assert_array_equal(getattr(t, name), expected)
	max_pp = np.asarray(t.move_max_pp)
	for row, id_ in enumerate(tables.moves.ids):
		pp = int(tables.moves.pp[row])
		expected = pp if tables.moves.no_pp_boosts[row] or id_ == "trumpcard" else pp * 8 // 5
		assert int(max_pp[row]) == expected
	assert int(t.move_max_pp[tables.moves.index["trumpcard"]]) == 5
	assert int(t.move_max_pp[tables.moves.index["struggle"]]) == 1


@pytest.mark.parametrize("ability", ["noability", "", "levitate"])
def test_arceus_actual_types_without_multitype(synthetic_pool, tables, ability):
	# data/conditions.ts:856-869 and Pokemon.getTypes:2164: Arceus's Type
	# hook returns Normal without Multitype, regardless of the displayed form.
	teams = teams_from_pool(synthetic_pool, 0, 1)
	t = build_engine_tables(tables)
	forms = [id_ for id_ in tables.species.ids if id_.startswith("arceus")]
	for form in forms:
		teams = teams._replace(
			species=teams.species.at[0, 0].set(tables.species.index[form]),
			ability=teams.ability.at[0, 0].set(tables.abilities.index[ability]),
		)
		state = jax.jit(reset)(jax.random.key(0), teams, t)
		np.testing.assert_array_equal(state.types[0, 0], [tables.types.index["normal"], -1])
