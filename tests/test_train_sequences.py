"""Chunks of whole sequences: their layout, and a chunk replay equal to the rollout's forward pass."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.train.ppo import chunk_time

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


def test_chunk_time_puts_chunk_c_of_row_b_at_c_times_batch_plus_b():
	steps, batch, length = 6, 2, 3
	x = np.arange(steps * batch).reshape(steps, batch)
	chunks = np.asarray(chunk_time(jnp.asarray(x), steps // length, length, batch))
	assert chunks.shape == (4, length)
	for c in range(steps // length):
		for b in range(batch):
			np.testing.assert_array_equal(
				chunks[c * batch + b], x[c * length : (c + 1) * length, b]
			)


@pytest.fixture(scope="module")
def rollout(league_trainer, cpu_only):
	"""The shared trainer's learner and one rollout of it, on the host."""
	learner = league_trainer.init()
	_, traj = league_trainer.rollout(learner.params, learner.pool, learner.league)
	return learner, jax.device_get(traj)


def _lane_chunks(trainer, x):
	"""A [T, L, ...] rollout array as time-major chunks [M, N, ...] (N chunks of the lanes)."""
	r = trainer.config.rollout
	chunks = r.rollout_length // r.sequence_length
	return jnp.swapaxes(
		chunk_time(jnp.asarray(x), chunks, r.sequence_length, trainer.layout.lanes), 0, 1
	)


def _chunk_starts(trainer, hidden):
	r = trainer.config.rollout
	chunks = r.rollout_length // r.sequence_length
	lanes = trainer.layout.lanes
	return jax.tree.map(
		lambda x: jnp.asarray(x[:: r.sequence_length].reshape((chunks * lanes,) + x.shape[2:])),
		hidden,
	)


def check_replay(trainer, learner, traj):
	"""The chunk replay of a rollout gives the rollout's own log-probabilities and values."""
	layout, r = trainer.layout, trainer.config.rollout
	ended = np.asarray(layout.lanes_in_time(traj.terminated | traj.truncated))
	# The rollout must hold episode boundaries inside chunks for this check to mean anything.
	inside = ended.reshape(-1, r.sequence_length, layout.lanes)[:, :-1]
	assert inside.any()

	obs = jax.tree.map(lambda x: _lane_chunks(trainer, x), traj.obs)
	start = _chunk_starts(trainer, traj.hidden)
	logits, values = jax.jit(trainer._replay)(
		learner.params, obs, _lane_chunks(trainer, ended), start
	)
	log_probs = np.asarray(jax.nn.log_softmax(np.asarray(logits)))
	actions = np.asarray(_lane_chunks(trainer, traj.action))
	chosen = np.take_along_axis(log_probs, actions[..., None], -1)[..., 0]
	np.testing.assert_allclose(chosen, np.asarray(_lane_chunks(trainer, traj.log_prob)), atol=1e-5)
	np.testing.assert_allclose(
		np.asarray(values), np.asarray(_lane_chunks(trainer, traj.value)), atol=1e-5
	)


def test_chunk_replay_equals_the_rollouts_own_forward_pass(league_trainer, rollout):
	learner, traj = rollout
	check_replay(league_trainer, learner, traj)


@pytest.fixture(scope="module")
def featured_trainer(tables, tmp_path_factory, cpu_only):
	"""A trainer with the public features and the factored action head, on the CPU."""
	from train_support import league_config, on_cpu, write_pool

	from pokejax.train.ppo import Trainer

	with on_cpu():
		config = league_config("network.features=true", "network.action_head=factored")
		return Trainer(config, write_pool(tmp_path_factory.mktemp("featured"), tables))


def test_chunk_replay_with_features_and_the_factored_head(featured_trainer, cpu_only):
	learner = featured_trainer.init()
	_, traj = featured_trainer.rollout(learner.params, learner.pool, learner.league)
	check_replay(featured_trainer, learner, jax.device_get(traj))


def test_skipping_the_reset_changes_the_replay(league_trainer, rollout):
	"""The reset matters: a replay that carries the state across battles disagrees with the rollout."""
	learner, traj = rollout
	obs = jax.tree.map(lambda x: _lane_chunks(league_trainer, x), traj.obs)
	start = _chunk_starts(league_trainer, traj.hidden)

	def without_reset(params, obs, start):
		def step(hidden, step_obs):
			after, _, value = league_trainer.model.apply({"params": params}, step_obs, hidden)
			return after, value

		_, values = jax.lax.scan(step, start, obs)
		return values

	values = np.asarray(jax.jit(without_reset)(learner.params, obs, start))
	assert np.abs(values - np.asarray(_lane_chunks(league_trainer, traj.value))).max() > 1e-4
