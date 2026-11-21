"""Generalised advantage estimates against a hand-computed case with ended battles."""

import jax.numpy as jnp
import numpy as np

from pokejax.train.ppo import gae

# Two battle rows, four steps, gamma 0.5 and lambda 0.5 (so gamma * lambda = 0.25).
GAMMA = 0.5
LAMBDA = 0.5


def test_gae_matches_the_hand_computed_case():
	rewards = jnp.array([[0.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 0.0]])
	values = jnp.array([[1.0, 0.5], [2.0, 0.5], [0.5, 0.5], [4.0, 0.5]])
	terminated = jnp.array([[False, False], [False, False], [True, False], [False, False]])
	truncated = jnp.array([[False, False], [False, False], [False, False], [True, False]])
	final_values = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
	last_value = jnp.array([3.0, 1.0])

	advantages, returns = gae(
		rewards, values, terminated, truncated, final_values, last_value, GAMMA, LAMBDA
	)

	# Row 0: step 3 is truncated, so it bootstraps its pre-reset value 2.0:
	#   delta3 = 0 + 0.5 * 2.0 - 4.0 = -3.0 (nothing carries in from the next battle);
	# step 2 is terminated, so it bootstraps 0 and does not take step 3's advantage:
	#   delta2 = 1.0 - 0.5 = 0.5;
	# step 1: delta1 = 0.5 * 0.5 - 2.0 = -1.75, advantage -1.75 + 0.25 * 0.5 = -1.625;
	# step 0: delta0 = 0.5 * 2.0 - 1.0 = 0.0, advantage 0.25 * -1.625 = -0.40625.
	# Row 1 has no end, so its last step bootstraps last_value = 1.0:
	#   delta3 = 0.5 * 1.0 - 0.5 = 0.0, advantage 0.0;
	#   delta2 = 0.5 * 0.5 - 0.5 + 0 (no reward) = -0.25, advantage -0.25 + 0.25 * 0.0 = -0.25;
	#   delta1 = -1.0 + 0.5 * 0.5 - 0.5 = -1.25, advantage -1.25 + 0.25 * -0.25 = -1.3125;
	#   delta0 = 0.5 * 0.5 - 0.5 = -0.25, advantage -0.25 + 0.25 * -1.3125 = -0.578125.
	expected_advantages = np.array(
		[[-0.40625, -0.578125], [-1.625, -1.3125], [0.5, -0.25], [-3.0, 0.0]]
	)
	np.testing.assert_allclose(advantages, expected_advantages, rtol=0, atol=1e-6)
	np.testing.assert_allclose(returns, expected_advantages + np.asarray(values), atol=1e-6)
	np.testing.assert_allclose(returns[0, 0], 0.59375, atol=1e-6)
	np.testing.assert_allclose(returns[2, 0], 1.0, atol=1e-6)


def test_terminated_steps_ignore_the_value_of_the_next_battle():
	values = jnp.array([[0.3], [9.0]])
	ones = jnp.ones((2, 1), jnp.float32)
	terminated = jnp.array([[True], [False]])
	no = jnp.zeros((2, 1), bool)
	advantages, _ = gae(ones, values, terminated, no, jnp.zeros((2, 1)), jnp.array([0.0]), 1.0, 1.0)
	# The first step bootstraps zero: delta = 1 - 0.3, and the next battle's 9.0 is not used.
	np.testing.assert_allclose(advantages[0, 0], 0.7, atol=1e-6)
