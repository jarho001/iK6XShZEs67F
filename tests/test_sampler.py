"""Team sampler: only fully supported teams, every one of them reachable, reproducible from a key."""

import json
import subprocess
import sys
from dataclasses import fields, replace
from pathlib import Path

import jax
import numpy as np
import pytest

from pokejax import support
from pokejax.sampler import (
	excluded_teams,
	sample_team_indices,
	supported_team_indices,
)
from pokejax.teams import TeamPool, load_npz, save_npz

ROOT = Path(__file__).resolve().parents[1]


def tiled_pool(pool, count):
	"""The synthetic teams repeated (team i is synthetic team i % 3) to count teams."""
	arrays = {}
	for field in fields(TeamPool):
		value = getattr(pool, field.name)
		if isinstance(value, np.ndarray):
			repeats = -(-count // len(value))
			arrays[field.name] = np.concatenate([value] * repeats)[:count]
	return replace(pool, **arrays)


def draw(key, indices, batch=2000):
	return np.asarray(sample_team_indices(key, jax.numpy.asarray(indices), batch))


def test_excluded_team_is_never_drawn(synthetic_pool, tables, monkeypatch):
	monkeypatch.setattr(support, "SUPPORTED_ABILITIES", frozenset({"overgrow"}))
	pool = tiled_pool(synthetic_pool, 60)
	indices = supported_team_indices(pool, tables)
	assert indices.tolist() == [i for i in range(60) if i % 3 != 1]
	pairs = draw(jax.random.key(1), indices, 5000)
	assert set(np.unique(pairs).tolist()) <= set(indices.tolist())
	assert not np.any(pairs % 3 == 1)


def test_every_supported_team_is_drawn(synthetic_pool, tables, monkeypatch):
	monkeypatch.setattr(support, "SUPPORTED_ABILITIES", frozenset({"overgrow"}))
	pool = tiled_pool(synthetic_pool, 60)
	indices = supported_team_indices(pool, tables)
	pairs = draw(jax.random.key(2), indices, 4000)
	assert set(np.unique(pairs).tolist()) == set(indices.tolist())


def test_draws_are_deterministic_for_a_key(synthetic_pool, tables):
	indices = supported_team_indices(synthetic_pool, tables)
	first = draw(jax.random.key(3), indices, 256)
	np.testing.assert_array_equal(first, draw(jax.random.key(3), indices, 256))
	assert not np.array_equal(first, draw(jax.random.key(4), indices, 256))


def test_draw_shape_and_empty_index_array_are_checked(synthetic_pool, tables):
	indices = supported_team_indices(synthetic_pool, tables)
	assert draw(jax.random.key(5), indices, 16).shape == (16, 2)
	with pytest.raises(ValueError):
		sample_team_indices(jax.random.key(5), jax.numpy.zeros((0,), jax.numpy.int32), 4)


def test_excluded_teams_report_their_unsupported_ids(synthetic_pool, tables, monkeypatch):
	monkeypatch.setattr(support, "SUPPORTED_ABILITIES", frozenset({"overgrow"}))
	(excluded,) = excluded_teams(synthetic_pool, tables)
	assert (excluded.index, excluded.moves, excluded.abilities, excluded.items) == (
		1,
		(),
		("torrent",),
		(),
	)


def test_cli_prints_the_supported_indices(synthetic_pool, tables, tmp_path):
	path = tmp_path / "pool.npz"
	save_npz(synthetic_pool, path)
	output = subprocess.run(
		[sys.executable, "-m", "pokejax.sampler", str(path)],
		cwd=ROOT,
		check=True,
		capture_output=True,
		text=True,
	).stdout
	assert json.loads(output) == [0, 1, 2]


def test_checked_in_pool_is_fully_supported(tables):
	path = ROOT / "work/teams/gen5-teams.npz"
	if not path.exists():
		pytest.skip("Generate work/teams/gen5-teams.npz with pokejax.teams")
	pool = load_npz(path)
	assert len(supported_team_indices(pool, tables)) == len(pool.team_size)
	assert excluded_teams(pool, tables) == ()
	assert "Excluded teams: 0 (0.0%)" in (ROOT / "COVERAGE.md").read_text()
