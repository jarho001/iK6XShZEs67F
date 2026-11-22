"""The network: padding embeds to zero, illegal actions get no probability, one legal action works."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from train_support import TINY_OVERRIDES, write_pool

from pokejax.env import Env, EnvConfig, OpponentEnv
from pokejax.train.config import load_config
from pokejax.train.network import (
	PokePolicy,
	count_parameters,
	initial_carry,
	padded_embed,
	vocabulary,
)

BATCH = 4

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


@pytest.fixture(scope="module")
def observed(tables, tmp_path_factory, cpu_only):
	"""A learner's observation of four battles (both sides' views) and its legal masks."""
	pool = write_pool(tmp_path_factory.mktemp("network"), tables)
	env = OpponentEnv(Env(EnvConfig(BATCH, observe="both"), pool, tables=tables))
	_, timestep = jax.jit(env.reset)(jax.random.key(3))
	return timestep.obs, timestep.legal


def tiny_network(core="gru"):
	config = load_config(None, [*TINY_OVERRIDES, f"network.core={core}"])
	return config.network


def test_padding_id_embeds_to_zero_and_other_ids_do_not():
	table = nn.Embed(8, 3)
	ids = jnp.array([[0, 2], [5, 0]], jnp.int16)
	params = table.init(jax.random.key(0), ids)
	embedded = padded_embed(table.bind(params), ids)
	np.testing.assert_array_equal(np.asarray(embedded[0, 0]), np.zeros(3))
	np.testing.assert_array_equal(np.asarray(embedded[1, 1]), np.zeros(3))
	assert np.abs(np.asarray(embedded[0, 1])).sum() > 0


def test_vocabulary_covers_the_observation_ids(tables):
	sizes = vocabulary(tables)
	assert sizes.species == len(tables.species.ids)
	assert sizes.types == len(tables.types.ids) + 1
	assert sizes.statuses == 7 and sizes.weathers == 5


@pytest.mark.parametrize("core", ["gru", "lstm"])
def test_forward_shapes_and_parameter_count(tables, observed, core):
	obs, _ = observed
	network = tiny_network(core)
	model = PokePolicy(vocabulary(tables), network)
	hidden = initial_carry(network, (BATCH,))
	params = model.init(jax.random.key(1), obs, hidden)["params"]
	carry, logits, value = model.apply({"params": params}, obs, hidden)
	assert logits.shape == (BATCH, 9) and value.shape == (BATCH,)
	assert jax.tree.map(jnp.shape, carry) == jax.tree.map(jnp.shape, hidden)
	assert count_parameters(params) > 0


def test_illegal_actions_have_zero_probability(tables, observed):
	obs, legal = observed
	network = tiny_network()
	model = PokePolicy(vocabulary(tables), network)
	hidden = initial_carry(network, (BATCH,))
	params = model.init(jax.random.key(1), obs, hidden)["params"]
	# Only the first two actions are legal in every battle.
	narrowed = obs._replace(legal_actions=jnp.zeros_like(legal).at[:, :2].set(True))
	_, logits, _ = model.apply({"params": params}, narrowed, hidden)
	probabilities = np.asarray(jax.nn.softmax(logits))
	assert np.all(probabilities[:, 2:] == 0.0)
	np.testing.assert_allclose(probabilities[:, :2].sum(axis=1), 1.0, atol=1e-6)
	assert np.all(np.isfinite(np.asarray(logits)))


def test_a_single_legal_action_is_chosen_with_no_nan(tables, observed):
	obs, _ = observed
	network = tiny_network()
	model = PokePolicy(vocabulary(tables), network)
	hidden = initial_carry(network, (BATCH,))
	params = model.init(jax.random.key(1), obs, hidden)["params"]
	only = obs._replace(legal_actions=jnp.zeros((BATCH, 9), bool).at[:, 5].set(True))

	def log_probability(params):
		_, logits, _ = model.apply({"params": params}, only, hidden)
		log_probs = jax.nn.log_softmax(logits)
		entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
		return log_probs[:, 5].sum(), entropy.sum()

	(logp, entropy), grads = jax.value_and_grad(log_probability, has_aux=True)(params)
	assert float(logp) == pytest.approx(0.0, abs=1e-6)
	assert float(entropy) == pytest.approx(0.0, abs=1e-6)
	assert all(np.all(np.isfinite(np.asarray(g))) for g in jax.tree.leaves(grads))
