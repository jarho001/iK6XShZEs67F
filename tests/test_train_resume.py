"""Checkpoints: a round trip keeps every leaf, and a resumed run continues bit for bit on the CPU."""

import jax
import numpy as np
import pytest
from train_support import league_config

from pokejax.train.checkpoint import CheckpointError, load_checkpoint, save_checkpoint

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


def as_numpy(tree):
	"""Every leaf as a numpy array; PRNG keys as their key data."""
	return [
		np.asarray(
			jax.random.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x
		)
		for x in jax.tree.leaves(tree)
	]


def test_resume_with_the_pool_continues_bit_for_bit(league_trainer, tmp_path):
	"""Pool insertions happen at every update here, so the resumed pool must match exactly."""
	straight = league_trainer.init()
	for _ in range(3):
		straight, _ = league_trainer.update(straight)

	halfway = league_trainer.init()
	halfway, _ = league_trainer.update(halfway)
	path = tmp_path / "checkpoint.npz"
	save_checkpoint(path, halfway, league_trainer.config)
	resumed = load_checkpoint(path, league_trainer.init(), league_trainer.config)
	assert int(resumed.update) == 1 and int(resumed.pool_inserts) == 1
	for _ in range(2):
		resumed, _ = league_trainer.update(resumed)

	assert int(resumed.update) == int(straight.update) == 3
	assert int(resumed.pool_inserts) == int(straight.pool_inserts) == 3
	for expected, actual in zip(as_numpy(straight), as_numpy(resumed)):
		np.testing.assert_array_equal(expected, actual)


def test_round_trip_keeps_every_leaf_and_checks_the_configuration(league_trainer, tmp_path):
	learner, _ = league_trainer.update(league_trainer.init())
	path = tmp_path / "checkpoint.npz"
	save_checkpoint(path, learner, league_trainer.config)
	loaded = load_checkpoint(path, league_trainer.init(), league_trainer.config)
	for expected, actual in zip(as_numpy(learner), as_numpy(loaded)):
		np.testing.assert_array_equal(expected, actual)
	assert [x.dtype for x in jax.tree.leaves(loaded)] == [x.dtype for x in jax.tree.leaves(learner)]

	# A longer run may resume; another learning rate may not.
	load_checkpoint(path, league_trainer.init(), league_config("run.updates=9"))
	with pytest.raises(CheckpointError, match="different configuration"):
		load_checkpoint(path, league_trainer.init(), league_config("ppo.learning_rate=1e-4"))


def test_the_pool_schedule_is_part_of_the_configuration(league_trainer, tmp_path):
	path = tmp_path / "pool.npz"
	save_checkpoint(path, league_trainer.init(), league_trainer.config)
	with pytest.raises(CheckpointError, match="different configuration"):
		load_checkpoint(path, league_trainer.init(), league_config("pool.every=2"))
