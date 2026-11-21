"""Chunks of whole sequences: their layout, and a chunk replay equal to the rollout's forward pass."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from train_support import on_cpu, tiny_config, write_pool

from pokejax.train.ppo import RolloutCarry, Trainer, chunk_time


@pytest.fixture(scope="module", autouse=True)
def cpu():
	with on_cpu():
		yield
	# Compiled executables of a module are not needed after it; a worker keeps them otherwise.
	jax.clear_caches()


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
def rollout(tables, tmp_path_factory):
	"""A tiny trainer and one rollout, whose battles are cut every three steps."""
	config = tiny_config("rollout.batch_size=4")
	trainer = Trainer(config, write_pool(tmp_path_factory.mktemp("sequences"), tables))
	learner = trainer.init()
	carry = RolloutCarry(learner.env, learner.obs, learner.hidden, learner.key)
	rollout_fn = jax.jit(lambda params, carry: trainer._rollout(params, carry))
	_, traj = rollout_fn(learner.params, carry)
	return trainer, learner, jax.device_get(traj)


def test_chunk_replay_equals_the_rollouts_own_forward_pass(rollout):
	trainer, learner, traj = rollout
	r = trainer.config.rollout
	chunks = r.rollout_length // r.sequence_length
	ended = traj.terminated | traj.truncated
	# The rollout must hold episode boundaries inside chunks for this check to mean anything.
	inside = ended.reshape(chunks, r.sequence_length, r.batch_size)[:, :-1]
	assert inside.any()

	def to_steps(x):
		return jnp.swapaxes(
			chunk_time(jnp.asarray(x), chunks, r.sequence_length, r.batch_size), 0, 1
		)

	obs = jax.tree.map(to_steps, traj.obs)
	start = jax.tree.map(
		lambda x: jnp.asarray(
			x[:: r.sequence_length].reshape((chunks * r.batch_size,) + x.shape[2:])
		),
		traj.hidden,
	)
	logits, values = jax.jit(trainer._replay)(learner.params, obs, to_steps(ended), start)
	logits, values = np.asarray(logits), np.asarray(values)
	log_probs = np.asarray(jax.nn.log_softmax(logits))
	chosen = np.take_along_axis(log_probs, np.asarray(to_steps(traj.action))[..., None], -1)[..., 0]
	np.testing.assert_allclose(chosen, np.asarray(to_steps(traj.log_prob)), atol=1e-5)
	np.testing.assert_allclose(values, np.asarray(to_steps(traj.value)), atol=1e-5)


def test_skipping_the_reset_changes_the_replay(rollout):
	"""The reset matters: a replay that carries the state across battles disagrees with the rollout."""
	trainer, learner, traj = rollout
	r = trainer.config.rollout
	chunks = r.rollout_length // r.sequence_length

	def to_steps(x):
		return jnp.swapaxes(
			chunk_time(jnp.asarray(x), chunks, r.sequence_length, r.batch_size), 0, 1
		)

	obs = jax.tree.map(to_steps, traj.obs)
	start = jax.tree.map(
		lambda x: jnp.asarray(
			x[:: r.sequence_length].reshape((chunks * r.batch_size,) + x.shape[2:])
		),
		traj.hidden,
	)

	def without_reset(params, obs, start):
		def step(hidden, step_obs):
			after, _, value = trainer.model.apply({"params": params}, step_obs, hidden)
			return after, value

		_, values = jax.lax.scan(step, start, obs)
		return values

	values = np.asarray(jax.jit(without_reset)(learner.params, obs, start))
	assert np.abs(values - np.asarray(to_steps(traj.value))).max() > 1e-4
