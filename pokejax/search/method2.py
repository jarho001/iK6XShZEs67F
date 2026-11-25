"""One-sided, KL-regularized matrix search on the same player views as Method 1."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import NUM_ACTIONS
from pokejax.engine.state import NONE
from pokejax.search.method1 import Method1, RolloutLanes, _advance, _follow, _score, _value
from pokejax.search.oracle import Decision, SearchConfig, _auxiliary_request
from pokejax.search.transition import observed_mask


@dataclass(frozen=True)
class MatrixConfig(SearchConfig):
	lambda_us: float = 0.1
	lambda_foe: float = 0.3
	opponent_columns: int = 2
	iterations: int = 64
	greedy: bool = False

	def __post_init__(self):
		super().__post_init__()
		if not (self.lambda_us > 0 and self.lambda_foe > 0):
			raise ValueError("Both KL penalties must be positive")
		if not 2 <= self.opponent_columns <= NUM_ACTIONS or self.iterations < 1:
			raise ValueError("Expected 2–9 columns and a positive iteration count")


class MatrixSolution(NamedTuple):
	p: jax.Array
	q: jax.Array
	gap: jax.Array
	row_values: jax.Array


def _kl(p, log_p, log_prior):
	return (p * jnp.where(p > 0, log_p - log_prior, 0.0)).sum(-1)


def solve_matrix(matrix, weights, log_p0, log_q0, lambda_us, lambda_foe, iterations=64):
	"""Solve one [world, row, column] game with normalized priors and world weights.

	Illegal actions have log prior -inf. Each world has a nonempty column mask,
	including worlds with zero weight. The minimizer is eliminated analytically.
	"""
	p_mask = jnp.isfinite(log_p0)
	q_mask = jnp.isfinite(log_q0)
	matrix = jnp.where(p_mask[None, :, None] & q_mask[:, None, :], matrix, 0.0)

	# Subtract a reference row to remove column bias when bounding curvature.
	# If R is the largest legal column span of any resulting row, the
	# soft-min Hessian is bounded by R²/(4 lambda_foe) in the l1 norm.
	centered = matrix - matrix[:, jnp.argmax(p_mask), None, :]
	span = jnp.max(jnp.where(q_mask[:, None, :], centered, -jnp.inf), axis=-1)
	span -= jnp.min(jnp.where(q_mask[:, None, :], centered, jnp.inf), axis=-1)
	width = jnp.max(jnp.where(p_mask, span, 0.0), axis=-1)
	curvature = (weights * width**2).sum() / (4 * lambda_foe)
	mix = lambda_us / (lambda_us + curvature)

	def response(log_p):
		column_values = jnp.einsum("wab,a->wb", matrix, jnp.exp(log_p))
		log_q = jax.nn.log_softmax(log_q0 - column_values / lambda_foe, axis=-1)
		row_values = jnp.einsum("w,wab,wb->a", weights, matrix, jnp.exp(log_q))
		return log_q, row_values

	def update(_, log_p):
		_, values = response(log_p)
		tilt = log_p0 + values / lambda_us
		# Avoid 0 * -inf when the curvature bound is zero.
		new = (1 - mix) * jnp.where(p_mask, log_p, 0) + mix * tilt
		return jax.nn.log_softmax(new)

	log_p = jax.lax.fori_loop(0, iterations, update, log_p0)
	log_q, values = response(log_p)
	p = jnp.exp(log_p)
	# q is the exact response to p. The saddle gap is the remaining row
	# improvement, lambda_us * KL(p || best regularized response to q).
	log_best = jax.nn.log_softmax(log_p0 + values / lambda_us)
	gap = jnp.maximum(0.0, lambda_us * _kl(p, log_p, log_best))
	return MatrixSolution(p, jnp.exp(log_q), gap, values)


def _request_mask(position, side):
	one = jnp.arange(NUM_ACTIONS)
	mask = jnp.where(position.state.request[side] == NONE, one == 0, observed_mask(position)[side])
	return jnp.where(position.has_held[side], one == position.held[side], mask)


def _columns(position, side, head, key, cap):
	legal = _request_mask(position, side)
	logits = _auxiliary_request(position, side, head)
	logits = jnp.where(legal.sum() == 1, 0.0, logits)
	logits = jnp.where(legal, logits, -jnp.inf)
	# Invalid rebuilt requests can have no legal replacement. Keep an inert
	# column so their zero world weight never multiplies a NaN in the solver.
	logits = jnp.where(legal.any(), logits, jnp.where(jnp.arange(NUM_ACTIONS) == 0, 0, -jnp.inf))
	_, ranked = jax.lax.top_k(logits, cap)
	# Draw one uniformly from the legal tail on wide requests.
	tail = legal.at[ranked[: cap - 1]].set(False)
	extra = jax.random.categorical(key, jnp.where(tail, 0.0, -jnp.inf))
	columns = ranked.at[-1].set(jnp.where(legal.sum() > cap, extra, ranked[-1]))
	log_prior = jax.nn.log_softmax(logits[columns])
	return columns, log_prior


@partial(jax.jit, static_argnames=("cap", "draws"))
def _prepare_matrix(positions, sides, head, keys, cap, draws):
	def root(worlds, side, head, key):
		def world(position, key):
			tail, chance = jax.random.split(key)
			columns, log_prior = _columns(position, 1 - side, head, tail, cap)
			return columns, log_prior, jax.random.split(chance, draws)

		return jax.vmap(world)(worlds, jax.random.split(key, worlds.state.turn.size))

	return jax.vmap(root)(positions, sides, head, keys)


@partial(jax.jit, static_argnames=("size", "draws"))
def _first_lanes(size, draws, positions, pairs, sides, carry, columns, chance):
	batch, rows, worlds, cap = pairs.shape
	mask = jnp.broadcast_to(pairs[..., None], (*pairs.shape, draws)).reshape(-1)
	full = batch * rows * worlds * cap * draws
	ids = jnp.nonzero(mask, size=size, fill_value=full)[0]
	lane = jnp.minimum(ids, full - 1)
	root = lane // (rows * worlds * cap * draws)
	action = (lane // (worlds * cap * draws)) % rows
	world = (lane // (cap * draws)) % worlds
	column = (lane // draws) % cap
	side = sides[root]
	actions = jnp.where(
		jnp.arange(2) == side[:, None], action[:, None], columns[root, world, column, None]
	)
	return RolloutLanes(
		jax.tree.map(lambda x: x[root, world], positions),
		actions,
		chance[root, world, lane % draws],
		jax.tree.map(lambda x: x[root], carry),
		side,
	), ids


@partial(jax.jit, static_argnames=("iterations",))
def _finish(iterations, scores, valid, available, legal, logits, log_q0, noise, penalties, greedy):
	weights = valid / jnp.maximum(1, valid.sum(-1, keepdims=True))
	raw_logits = jnp.where(legal, logits, -jnp.inf)
	raw = jnp.argmax(raw_logits + noise, axis=-1).astype(jnp.int32)
	prior = 0.99 * jax.nn.softmax(raw_logits) + 0.01 / legal.sum(-1, keepdims=True)
	log_p0 = jnp.where(legal, jnp.log(prior), -jnp.inf)
	matrix = scores.mean(axis=-1).swapaxes(1, 2)
	solution = jax.vmap(solve_matrix, in_axes=(0, 0, 0, 0, None, None, None))(
		matrix, weights, log_p0, log_q0, *penalties, iterations
	)
	action = jnp.argmax(jnp.log(solution.p) + jnp.where(greedy, 0.0, noise), axis=-1)
	gains = solution.row_values - solution.row_values[jnp.arange(raw.size), raw, None]
	best_gain = jnp.max(jnp.where(legal, gains, -jnp.inf), axis=-1)
	chosen_gain = gains[jnp.arange(raw.size), action]
	return Decision(
		jnp.where(available, action, raw).astype(jnp.int32),
		raw,
		available,
		jnp.where(available, best_gain, 0.0),
		jnp.where(available & (action != raw), chosen_gain, 0.0),
	), solution


class Method2(Method1):
	"""Reuse the belief/deployment boundary, replacing only the world backup."""

	def __init__(self, model, tables, catalogue, config=None):
		if catalogue is None:
			raise ValueError("Method 2 needs a player-view set catalogue")
		super().__init__(model, tables, catalogue, config or MatrixConfig())
		self.last_solution = None

	def choose(
		self,
		params,
		views,
		beliefs,
		carry,
		logits,
		head,
		keys,
		noise,
		previous_actions,
		*,
		protocol=False,
	):
		self.last_solution = None
		decision, metrics = super().choose(
			params,
			views,
			beliefs,
			carry,
			logits,
			head,
			keys,
			noise,
			previous_actions,
			protocol=protocol,
		)
		# A committed action is a singleton row, even if the request's base
		# mask still displays its original choices.
		return decision._replace(
			action=jnp.where(views.has_held, views.held, decision.action),
			raw=jnp.where(views.has_held, views.held, decision.raw),
		), metrics

	def _choose_worlds(
		self, params, worlds, valid, legal, eligible, sides, carry, logits, head, keys, noise
	):
		config = self.config
		cap, draws = config.opponent_columns, config.draws
		columns, log_q0, chance = _prepare_matrix(worlds, sides, head, keys, cap, draws)
		n_valid = valid.sum(1)
		available = eligible & (n_valid > 0)
		pairs = (
			available[:, None, None, None]
			& legal[:, :, None, None]
			& valid[:, None, :, None]
			& jnp.isfinite(log_q0[:, None, :, :])
		)
		count, n_valid = jax.device_get((pairs.sum() * draws, n_valid))
		if count == 0:
			return self._raw(legal, logits, noise), n_valid

		replicates = config.worlds * cap * draws
		full = sides.size * NUM_ACTIONS * replicates
		# Match Method 1's lane shapes so both methods use the same rollout kernels.
		width = min(1024, full)
		size = min(full, (int(count) + width - 1) // width * width)
		lanes, ids = _first_lanes(size, draws, worlds, pairs, sides, carry, columns, chance)
		scores = jnp.zeros(full, jnp.float32)
		for ply in range(config.max_steps):
			successor = _advance(lanes, self.tables)
			after, values, wait = _value(
				self.model, params, successor, lanes.sides, lanes.carry, self.tables
			)
			scores = _score(scores, ids, successor, lanes.sides, values, ply, config.gamma)
			pending = int(jnp.sum(wait & (ids < full)))
			if not pending or ply + 1 == config.max_steps:
				break
			width = min(512, full)
			size = min(full, (pending + width - 1) // width * width)
			lanes, ids = _follow(
				size, replicates, successor, lanes, ids, after, wait, head, ply, full
			)
		decision, self.last_solution = _finish(
			config.iterations,
			scores.reshape((*pairs.shape, draws)),
			valid,
			available,
			legal,
			logits,
			log_q0,
			noise,
			jnp.asarray([config.lambda_us, config.lambda_foe], np.float32),
			config.greedy,
		)
		return decision, n_valid
