"""Small table-backed team pools that do not require Showdown."""

import os
from pathlib import Path

os.environ.setdefault(
	"JAX_COMPILATION_CACHE_DIR", str(Path(__file__).resolve().parents[1] / "work/jax-cache")
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0.5")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
# Parallel workers share the GPU; allocate on demand instead of 75% each.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import ctypes
import gc

import jax
import numpy as np
import pytest

from pokejax.tables import STAT_NAMES, load_tables
from pokejax.teams import MOVE_SLOTS, TEAM_SLOTS, TeamPool

try:
	_LIBC = ctypes.CDLL("libc.so.6")
except OSError:  # not Linux: nothing to trim
	_LIBC = None


@pytest.fixture(scope="module", autouse=True)
def release_compiled_programs():
	"""Release the compiled programs of a finished module, and give the memory back.

	A worker otherwise keeps every program it compiled: four workers of the full suite grew to
	about 5 GB each, and the training modules alone add several programs each.
	"""
	yield
	jax.clear_caches()
	gc.collect()
	if _LIBC is not None:
		_LIBC.malloc_trim(0)


def pytest_addoption(parser):
	parser.addoption(
		"--large", action="store_true", help="also replay the 300-battle recordings in work/diff/"
	)


def pytest_configure(config):
	config.addinivalue_line("markers", "large: replays a 300-battle recording (needs --large)")


def pytest_collection_modifyitems(config, items):
	if config.getoption("--large"):
		return
	skip = pytest.mark.skip(reason="300-battle recording; run with --large")
	for item in items:
		if "large" in item.keywords:
			item.add_marker(skip)


@pytest.fixture(scope="module")
def cpu_only():
	"""The training tests run on the CPU: a bit-exact resume needs it (tests/test_train_resume.py)."""
	from train_support import on_cpu

	with on_cpu():
		yield


@pytest.fixture(scope="session")
def league_trainer(tables, tmp_path_factory):
	"""The trainer that the training tests share, on the CPU: its programs compile once per worker."""
	from train_support import league_config, on_cpu, write_pool

	from pokejax.train.ppo import Trainer

	with on_cpu():
		return Trainer(league_config(), write_pool(tmp_path_factory.mktemp("league"), tables))


@pytest.fixture(scope="session")
def tables():
	return load_tables()


@pytest.fixture
def synthetic_pool(tables):
	shape = (3, TEAM_SLOTS)
	moves = np.zeros((*shape, MOVE_SLOTS), dtype=np.int16)
	moves[:, :, 0] = tables.moves.index["tackle"]
	moves[1, :, 0] = tables.moves.index["surf"]
	moves[2, 0, 1] = tables.moves.index["protect"]
	ability = np.full(shape, tables.abilities.index["overgrow"], dtype=np.int16)
	ability[1] = tables.abilities.index["torrent"]
	item = np.full(shape, tables.items.index["leftovers"], dtype=np.int16)
	item[1] = 0
	item[2] = tables.items.index["lifeorb"]
	return TeamPool(
		species=np.full(shape, tables.species.index["bulbasaur"], dtype=np.int16),
		level=np.full(shape, 100, dtype=np.uint8),
		moves=moves,
		ability=ability,
		item=item,
		nature=np.zeros(shape, dtype=np.int8),
		gender=np.zeros(shape, dtype=np.int8),
		evs=np.zeros((*shape, len(STAT_NAMES)), dtype=np.uint8),
		ivs=np.full((*shape, len(STAT_NAMES)), 31, dtype=np.uint8),
		happiness=np.full(shape, 255, dtype=np.uint8),
		team_size=np.full(shape[0], TEAM_SLOTS, dtype=np.uint8),
		seed=7,
		showdown_commit="synthetic-commit",
	)
