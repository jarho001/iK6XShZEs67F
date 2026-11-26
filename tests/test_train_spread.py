"""Pool retention, checkpoint continuity, and Phase 7 tuning configurations."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.train.checkpoint import load_checkpoint, save_checkpoint
from pokejax.train.config import ConfigError, TrainConfig, load_config
from pokejax.train.league import PoolParams
from pokejax.train.ppo import Trainer


def exercise(retention, count, pool=None, start=0):
	trainer = object.__new__(Trainer)
	trainer.config = TrainConfig.from_dict(
		{"pool": {"size": 16, "every": 1, "retention": retention}}
	)
	if pool is None:
		stack = jnp.zeros((16,), jnp.int32)
		pool = PoolParams(stack, stack, stack if retention == "spread" else None)
	fresh = jnp.ones((0,), dtype=bool)
	insert = jax.jit(trainer._insert)
	for update in range(start + 1, count + 1):
		pool, fresh, _ = insert(pool, fresh, jnp.int32(update - 1), jnp.int32(update), update)
	return pool


def test_recent_round_robin_is_unchanged():
	pool = exercise("recent", 35)
	assert pool.inserted is None
	assert list(np.asarray(pool.current)) == [33, 34, 35, *range(20, 33)]


def test_spread_and_checkpoint(tmp_path):
	pool = exercise("spread", 200)
	ages = sorted(200 - np.asarray(pool.inserted))
	assert ages == [0, 1, 2, 3, 5, 7, 11, 15, 23, 31, 39, 55, 71, 103, 135, 199]
	assert 200 in np.asarray(pool.current)
	config = TrainConfig.from_dict({"pool": {"size": 16, "every": 1, "retention": "spread"}})
	path = tmp_path / "pool.npz"
	before = exercise("spread", 100)
	save_checkpoint(path, before, config, "team")
	restored = load_checkpoint(path, before, config, "team")
	resumed = exercise("spread", 200, restored, 100)
	for a, b in zip(jax.tree.leaves(pool), jax.tree.leaves(resumed), strict=True):
		np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", ["t-spread", "t-spread-nobots"])
def test_tuning_config(name):
	config = load_config(Path("configs/tune") / f"{name}.toml")
	assert config.run.name == name and config.pool.retention == "spread" and config.pool.size == 16
	assert config.groups["pool"] == (1536 if name.endswith("nobots") else 1024)


def test_spread_requires_battles_to_finish_between_insertions():
	config = load_config(Path("configs/tune/t-spread.toml"))
	with pytest.raises(ConfigError, match="spread requires"):
		replace(config, pool=replace(config.pool, every=1))
