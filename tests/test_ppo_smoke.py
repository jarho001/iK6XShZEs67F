"""One small recurrent update on the environment, with no GPU-only operations."""

import jax
import numpy as np

from pokejax.ppo_smoke import Config, Trainer
from pokejax.teams import save_npz


def test_recurrent_ppo_update(tmp_path, synthetic_pool):
	path = tmp_path / "teams.npz"
	save_npz(synthetic_pool, path)
	with jax.default_device(jax.devices("cpu")[0]):
		trainer = Trainer(
			Config(
				batch_size=8,
				rollout_length=4,
				hidden_size=16,
				embedding_size=2,
				epochs=1,
				minibatches=2,
			),
			path,
		)
		train, carry, metrics = jax.block_until_ready(trainer.update(trainer.train, trainer.carry))
		assert all(np.isfinite(float(x)) for x in metrics.values())
		assert int(metrics["illegal_actions"]) == 0
		assert int(train.step) == 2
		assert carry.hidden.shape == (8, 16)
		# The carry holds the learner's observation of the battles the next update starts from.
		assert carry.obs.own.species.shape == (8, 6)
		assert carry.env.learner.shape == (8,)
		assert any(
			not np.array_equal(a, b)
			for a, b in zip(jax.tree.leaves(train.params), jax.tree.leaves(trainer.train.params))
		)
