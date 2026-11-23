"""Checkpoints: a round trip keeps every leaf, and a resumed run continues bit for bit on the CPU."""

import json

import jax
import numpy as np
import pytest
from train_support import league_config

from pokejax.train.checkpoint import (
	CONFIG_ENTRY,
	POOL_ENTRY,
	CheckpointError,
	load_checkpoint,
	pool_digest,
	resolve_decay_updates,
	save_checkpoint,
)
from pokejax.train.ppo import learning_rate_schedule

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
	save_checkpoint(path, halfway, league_trainer.config, league_trainer.team_digest)
	resumed = load_checkpoint(
		path, league_trainer.init(), league_trainer.config, league_trainer.team_digest
	)
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
	save_checkpoint(path, learner, league_trainer.config, league_trainer.team_digest)
	loaded = load_checkpoint(
		path, league_trainer.init(), league_trainer.config, league_trainer.team_digest
	)
	for expected, actual in zip(as_numpy(learner), as_numpy(loaded)):
		np.testing.assert_array_equal(expected, actual)
	assert [x.dtype for x in jax.tree.leaves(loaded)] == [x.dtype for x in jax.tree.leaves(learner)]

	# A longer run may resume; another learning rate may not.
	load_checkpoint(
		path, league_trainer.init(), league_config("run.updates=9"), league_trainer.team_digest
	)
	with pytest.raises(CheckpointError, match="different configuration"):
		load_checkpoint(
			path,
			league_trainer.init(),
			league_config("ppo.learning_rate=1e-4"),
			league_trainer.team_digest,
		)


def test_the_pool_schedule_is_part_of_the_configuration(league_trainer, tmp_path):
	path = tmp_path / "pool.npz"
	save_checkpoint(path, league_trainer.init(), league_trainer.config, league_trainer.team_digest)
	with pytest.raises(CheckpointError, match="different configuration"):
		load_checkpoint(
			path, league_trainer.init(), league_config("pool.every=2"), league_trainer.team_digest
		)


def test_a_checkpoint_written_before_feature_set_still_resumes(league_trainer, tmp_path):
	"""Checkpoints from before network.feature_set do not name the key; it reads as "full"."""
	learner, _ = league_trainer.update(league_trainer.init())
	path = tmp_path / "checkpoint.npz"
	save_checkpoint(path, learner, league_trainer.config, league_trainer.team_digest)
	with np.load(path, allow_pickle=False) as data:
		arrays = {name: data[name] for name in data.files}
	saved = json.loads(str(arrays[CONFIG_ENTRY]))
	del saved["network"]["feature_set"]
	arrays[CONFIG_ENTRY] = np.array(json.dumps(saved, sort_keys=True))
	old = tmp_path / "old.npz"
	np.savez(old, **arrays)
	loaded = load_checkpoint(
		old, league_trainer.init(), league_trainer.config, league_trainer.team_digest
	)
	for expected, actual in zip(as_numpy(learner), as_numpy(loaded), strict=True):
		np.testing.assert_array_equal(expected, actual)


def test_a_checkpoint_of_another_team_pool_is_refused(league_trainer, tmp_path):
	path = tmp_path / "pool.npz"
	learner = league_trainer.init()
	save_checkpoint(path, learner, league_trainer.config, "another-pool")
	with pytest.raises(CheckpointError, match="another team pool"):
		load_checkpoint(
			path, league_trainer.init(), league_trainer.config, league_trainer.team_digest
		)
	save_checkpoint(path, learner, league_trainer.config, league_trainer.team_digest)
	with np.load(path, allow_pickle=False) as data:
		arrays = {name: data[name] for name in data.files if name != POOL_ENTRY}
	np.savez(tmp_path / "unstamped.npz", **arrays)
	with pytest.raises(CheckpointError, match="no team pool digest"):
		load_checkpoint(
			tmp_path / "unstamped.npz",
			league_trainer.init(),
			league_trainer.config,
			league_trainer.team_digest,
		)


def test_the_team_digest_changes_with_any_team_of_the_pool(league_trainer):
	pool = league_trainer.env.pool
	changed = pool._replace(species=pool.species.at[0, 0].set(pool.species[0, 0] + 1))
	assert pool_digest(pool, league_trainer.env.team_indices) == league_trainer.team_digest
	assert pool_digest(changed, league_trainer.env.team_indices) != league_trainer.team_digest


def test_resume_keeps_the_decay_horizon_unless_the_configuration_asks_for_a_new_one(league_trainer):
	fresh = resolve_decay_updates(league_config("run.updates=2"), None)
	assert fresh.ppo.decay_updates == 2
	longer = resolve_decay_updates(league_config("run.updates=5"), fresh)
	assert longer.ppo.decay_updates == 2
	# The learning rate at the restored count (two updates) is the end of the decay, as it was.
	count = 2 * fresh.ppo.epochs * fresh.ppo.minibatches
	for config in (fresh, longer):
		schedule = learning_rate_schedule(
			config.ppo, config.ppo.decay_updates * config.ppo.epochs * config.ppo.minibatches
		)
		assert float(schedule(count)) == pytest.approx(config.ppo.learning_rate_end)
	restarted = resolve_decay_updates(league_config("run.updates=5", "ppo.decay_updates=5"), fresh)
	assert restarted.ppo.decay_updates == 5
	# A checkpoint from before the setting existed decays over its run.updates.
	legacy = fresh.__class__.from_dict(
		{**fresh.to_dict(), "ppo": {**fresh.to_dict()["ppo"], "decay_updates": 0}}
	)
	assert resolve_decay_updates(league_config("run.updates=7"), legacy).ppo.decay_updates == 2
