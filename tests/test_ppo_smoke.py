"""One small recurrent update, with no GPU-only operations."""

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
		train, env, metrics = jax.block_until_ready(trainer.update(trainer.train, trainer.env))
		assert all(np.isfinite(float(x)) for x in metrics.values())
		assert int(metrics["illegal_actions"]) == 0
		assert int(train.step) == 2
		assert env.hidden.shape == (8, 16)
		assert any(
			not np.array_equal(a, b)
			for a, b in zip(jax.tree.leaves(train.params), jax.tree.leaves(trainer.train.params))
		)
