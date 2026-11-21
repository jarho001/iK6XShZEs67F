"""Checkpoints: a round trip keeps every leaf, and a resumed run continues bit for bit."""

import jax
import numpy as np
import pytest
from train_support import on_cpu, tiny_config, write_pool

from pokejax.train.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from pokejax.train.ppo import Trainer


@pytest.fixture(scope="module", autouse=True)
def cpu():
	with on_cpu():
		yield
	# Compiled executables of a module are not needed after it; a worker keeps them otherwise.
	jax.clear_caches()


def as_numpy(tree):
	"""Every leaf as a numpy array; PRNG keys as their key data."""
	return [
		np.asarray(
			jax.random.key_data(x) if jax.dtypes.issubdtype(x.dtype, jax.dtypes.prng_key) else x
		)
		for x in jax.tree.leaves(tree)
	]


@pytest.fixture(scope="module")
def trainer(tables, tmp_path_factory):
	return Trainer(tiny_config(), write_pool(tmp_path_factory.mktemp("resume"), tables))


def test_resume_continues_bit_for_bit(trainer, tmp_path):
	straight = trainer.init()
	for _ in range(4):
		straight, _ = trainer.update(straight)

	halfway = trainer.init()
	for _ in range(2):
		halfway, _ = trainer.update(halfway)
	path = tmp_path / "checkpoint.npz"
	save_checkpoint(path, halfway, trainer.config)
	resumed = load_checkpoint(path, trainer.init(), trainer.config)
	assert int(resumed.update) == 2
	for _ in range(2):
		resumed, _ = trainer.update(resumed)

	assert int(straight.update) == int(resumed.update) == 4
	for expected, actual in zip(jax.tree.leaves(straight.params), jax.tree.leaves(resumed.params)):
		np.testing.assert_array_equal(np.asarray(expected), np.asarray(actual))
	for expected, actual in zip(as_numpy(straight), as_numpy(resumed)):
		np.testing.assert_array_equal(expected, actual)


def test_round_trip_keeps_every_leaf_and_checks_the_configuration(trainer, tmp_path):
	learner, _ = trainer.update(trainer.init())
	path = tmp_path / "checkpoint.npz"
	save_checkpoint(path, learner, trainer.config)
	loaded = load_checkpoint(path, trainer.init(), trainer.config)
	for expected, actual in zip(as_numpy(learner), as_numpy(loaded)):
		np.testing.assert_array_equal(expected, actual)
	assert [x.dtype for x in jax.tree.leaves(loaded)] == [x.dtype for x in jax.tree.leaves(learner)]

	# A longer run may resume; another learning rate may not.
	longer = tiny_config("run.updates=9")
	load_checkpoint(path, trainer.init(), longer)
	with pytest.raises(CheckpointError, match="different configuration"):
		load_checkpoint(path, trainer.init(), tiny_config("ppo.learning_rate=1e-4"))
