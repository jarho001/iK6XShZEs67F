"""Diagnostic Method 1 using the true state and both real player histories."""

from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp

from pokejax.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, bench_slots
from pokejax.engine.state import NONE, ONGOING
from pokejax.env import side_reward
from pokejax.search.transition import advance_batch, observe_both, observed_mask
from pokejax.train.auxiliary import RANK_OUTPUTS, SWITCH_KINDS


@dataclass(frozen=True)
class SearchConfig:
	worlds: int = 16
	draws: int = 2
	depth: int = 1
	margin: float = 0.0
	padding: int = 32
	opponent: str = "aux"
	sample_strength: float = 0.0  # zero uses greedy search with the raw greedy fallback
	gamma: float = 0.99
	max_steps: int = 1000

	def __post_init__(self):
		if min(self.worlds, self.draws, self.depth, self.padding, self.max_steps) < 1:
			raise ValueError("Worlds, draws, depth, padding and max_steps must be positive")
		if self.opponent not in ("aux", "mirror"):
			raise ValueError("Opponent must be aux or mirror")
		if self.margin < 0 or self.sample_strength < 0 or not 0 < self.gamma <= 1:
			raise ValueError("Expected nonnegative margin and sample strength, and 0 < gamma <= 1")

	@property
	def replicates(self):
		return self.worlds * self.draws


class Decision(NamedTuple):
	action: jax.Array
	raw: jax.Array
	searched: jax.Array
	best_advantage: jax.Array
	chosen_advantage: jax.Array


def eligible(positions, sides):
	rows = jnp.arange(sides.size)
	return (
		(positions.state.winner == ONGOING)
		& (positions.state.request[rows, sides] != NONE)
		& ~positions.has_held[rows, sides]
		& (observed_mask(positions)[rows, sides].sum(-1) > 1)
	)


def decide(scores, legal, logits, searched, config, noise):
	"""Reduce paired scores [B, action, replicate] with a strict override margin."""
	masked_logits = jnp.where(legal, logits, -jnp.inf)
	raw = jnp.argmax(masked_logits + noise, axis=-1).astype(jnp.int32)
	rows = jnp.arange(raw.size)
	advantages = (scores - scores[rows, raw, None, :]).mean(-1)
	advantages = jnp.where(legal, advantages, -jnp.inf)
	best = jnp.argmax(advantages, axis=-1).astype(jnp.int32)
	best_advantage = advantages[rows, best]
	candidate = best
	if config.sample_strength:
		# Shared Gumbel noise couples the reweighted sample to the raw policy sample.
		shift = jnp.clip(config.sample_strength * advantages, -2.0, 2.0)
		candidate = jnp.argmax(masked_logits + noise + shift, axis=-1).astype(jnp.int32)
	gain = advantages[rows, candidate]
	override = searched & (gain > config.margin)
	return Decision(
		jnp.where(override, candidate, raw),
		raw,
		searched,
		jnp.where(searched, best_advantage, 0.0),
		jnp.where(override, gain, 0.0),
	)


def auxiliary_logits(logits, move_ids, ranks, legal):
	"""Condition the head's kind and identity on the request, splitting shared labels."""
	kinds = logits[:SWITCH_KINDS]
	moves = logits[SWITCH_KINDS:-RANK_OUTPUTS]
	switches = logits[-RANK_OUTPUTS:]

	def weights(head, labels, allowed):
		duplicates = ((labels[:, None] == labels[None, :]) & allowed).sum(-1)
		conditional = head[labels] - jnp.log(jnp.maximum(duplicates, 1))
		# An absent kind contributes zero, with no all-minus-infinity softmax.
		valid = allowed.any()
		inputs = jnp.where(allowed | ~valid, conditional, -jnp.inf)
		return jnp.where(allowed, jax.nn.softmax(inputs), 0.0)

	move_legal = legal[:MOVE_ACTIONS]
	switch_legal = legal[MOVE_ACTIONS:]
	kind_legal = jnp.stack((move_legal.any(), switch_legal.any()))
	kind = jax.nn.softmax(jnp.where(kind_legal, kinds, -jnp.inf))
	probability = jnp.concatenate(
		(
			kind[0] * weights(moves, move_ids, move_legal),
			kind[1] * weights(switches, ranks, switch_legal),
		)
	)
	# One percent uniform floor for legal choices, including identity-ambiguous switches.
	return jnp.where(legal, jnp.log(0.99 * probability + 0.01 / legal.sum()), -jnp.inf)


def _auxiliary_request(position, side, head):
	state = position.state
	active = state.active[side].astype(jnp.int32)
	move_ids = state.moves[side, active].astype(jnp.int32)
	seen = state.seen_order[side, bench_slots(active)].astype(jnp.int32)
	ranks = jnp.where(seen > 0, seen - 1, RANK_OUTPUTS - 1)
	return auxiliary_logits(head, move_ids, ranks, observed_mask(position)[side])


@partial(jax.jit, static_argnames=("model", "auxiliary"))
def forward(model, params, positions, carry, tables, auxiliary=False):
	"""Run normal player-view inference, retaining the old carries for held requests."""
	obs = jax.vmap(observe_both, in_axes=(0, None))(positions, tables)
	if auxiliary:
		after, logits, values, heads = model.apply(
			{"params": params}, obs, carry, method=model.full
		)
		head = heads["next_action"]
	else:
		after, logits, values = model.apply({"params": params}, obs, carry)
		head = None
	after = jax.tree.map(
		lambda old, new: jnp.where(positions.has_held[..., None], old, new), carry, after
	)
	return after, logits, values, head


class Lanes(NamedTuple):
	position: object
	carry: object
	side: jax.Array
	keys: jax.Array
	actions: jax.Array
	active: jax.Array
	depth: jax.Array
	score: jax.Array


@partial(jax.jit, static_argnames=("config",))
def _prepare(config, positions, sides, carry, logits, head, keys):
	batch = sides.size
	rows = jnp.arange(batch)
	foes = 1 - sides
	if config.opponent == "aux":
		foe_logits = jax.vmap(_auxiliary_request)(positions, foes, head[rows, sides])
	else:
		foe_logits = logits[rows, foes]

	def sample_root(key, foe_logits):
		policy_key, chance_key = jax.random.split(key)
		draw = jax.random.categorical(policy_key, foe_logits, shape=(config.replicates,))
		return draw.astype(jnp.int32), jax.random.split(chance_key, config.replicates)

	foe_actions, chance = jax.vmap(sample_root)(keys, foe_logits)
	size = batch * NUM_ACTIONS * config.replicates
	padded = (size + config.padding - 1) // config.padding * config.padding
	lane = jnp.arange(padded)
	root = jnp.minimum(lane // (NUM_ACTIONS * config.replicates), batch - 1)
	action = (lane // config.replicates) % NUM_ACTIONS
	replicate = lane % config.replicates
	side = sides[root]
	legal = observed_mask(positions)[rows, sides]
	searched = eligible(positions, sides)
	actions = jnp.where(
		jnp.arange(2) == side[:, None], action[:, None], foe_actions[root, replicate, None]
	)
	active = (lane < size) & searched[root] & legal[root, action]
	return (
		Lanes(
			jax.tree.map(lambda x: x[root], positions),
			jax.tree.map(lambda x: x[root], carry),
			side,
			chance[root, replicate],
			actions,
			active,
			jnp.zeros(padded, jnp.int32),
			jnp.zeros(padded, jnp.float32),
		),
		searched,
		legal,
	)


@jax.jit
def _step(lanes, ply, tables):
	step_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(lanes.keys, 2 * ply)
	return advance_batch(lanes.position, lanes.actions, step_keys, None, tables)[0]


def backup(winner, side, value, ply, gamma):
	"""Discount an immediate transition reward or a successor bootstrap as in PPO."""
	terminal = side_reward(winner) * jnp.where(side == 0, 1.0, -1.0)
	return jnp.where(winner == ONGOING, gamma ** (ply + 1) * value, gamma**ply * terminal)


@partial(jax.jit, static_argnames=("config",))
def _successor(config, lanes, position, carry, logits, values, ply):
	rows = jnp.arange(lanes.side.size)
	winner = position.state.winner
	decision = position.state.request[rows, lanes.side] != NONE
	decision &= ~position.has_held[rows, lanes.side]
	depth = lanes.depth + decision.astype(jnp.int32)
	stop = (winner != ONGOING) | (depth >= config.depth) | (ply + 1 >= config.max_steps)
	value = backup(winner, lanes.side, values[rows, lanes.side], ply, config.gamma)
	score = jnp.where(lanes.active & stop, value, lanes.score)
	policy_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(lanes.keys, 2 * ply + 1)
	actions = jax.vmap(jax.random.categorical)(policy_keys, logits).astype(jnp.int32)
	return lanes._replace(
		position=position,
		carry=carry,
		actions=actions,
		active=lanes.active & ~stop,
		depth=depth,
		score=score,
	)


@partial(jax.jit, static_argnames=("config",))
def _finish(config, scores, logits, legal, searched, noise):
	batch = logits.shape[0]
	size = batch * NUM_ACTIONS * config.replicates
	scores = scores[:size].reshape(batch, NUM_ACTIONS, config.replicates)
	return decide(scores, legal, logits, searched, config, noise)


@partial(jax.jit, static_argnames=("size",))
def _compact(lanes, ids, size, sentinel):
	"""Keep active lanes in a fixed wave, with their original indices for score reduction."""
	stop = lanes.active.size
	indices = jnp.nonzero(lanes.active, size=size, fill_value=stop)[0]
	present = indices < stop
	indices = jnp.minimum(indices, stop - 1)
	compact = jax.tree.map(lambda x: x[indices], lanes)
	return compact._replace(active=present), jnp.where(present, ids[indices], sentinel)


@jax.jit
def _save(scores, ids, values):
	return scores.at[ids].set(values, mode="drop")


class DiagnosticOracle:
	"""Score fixed-shape rollout batches. The caller owns the real recurrent carries."""

	def __init__(self, model, tables, config=SearchConfig()):
		if config.opponent == "aux" and (
			model.network.arch == "dense" or model.auxiliary_targets.next_action <= 0
		):
			raise ValueError("The aux opponent requires a trained next-action head")
		self.model = model
		self.tables = tables
		self.config = config

	def root(self, params, positions, carry):
		return forward(
			self.model, params, positions, carry, self.tables, self.config.opponent == "aux"
		)

	def _layer(self, params, lanes, ply):
		position = _step(lanes, ply, self.tables)
		after, logits, values, _ = forward(self.model, params, position, lanes.carry, self.tables)
		return _successor(self.config, lanes, position, after, logits, values, ply)

	def choose(
		self, params, positions, sides, carry, logits, head, keys, noise=None, *, warm=False
	):
		"""Choose from already-processed roots. Simulated carry never leaves the search."""
		config = self.config
		lanes, searched, legal = _prepare(config, positions, sides, carry, logits, head, keys)
		own_logits = logits[jnp.arange(sides.size), sides]
		if noise is None:
			if config.sample_strength:
				keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, 1)
				noise = jax.vmap(lambda k: jax.random.gumbel(k, (NUM_ACTIONS,)))(keys)
			else:
				noise = jnp.zeros_like(own_logits)
		full = lanes.active.size
		if sides.size == 1:
			# A single request uses one executable shape through every rollout layer.
			ply = 0
			while bool(lanes.active.any()):
				lanes = self._layer(params, lanes, jnp.int32(ply))
				ply += 1
			return _finish(config, lanes.score, own_logits, legal, searched, noise)

		units = full // config.padding
		waves = sorted(
			{(units + fraction - 1) // fraction * config.padding for fraction in (16, 4, 1)}
		)
		ids = jnp.arange(full)
		scores = lanes.score
		if warm:
			for size in waves:
				padded, wave_ids = _compact(lanes, ids, size, full)
				layer = self._layer(params, padded, jnp.int32(0))
				jax.block_until_ready(_save(scores, wave_ids, layer.score))

		ply = 0
		while count := int(lanes.active.sum()):
			size = next(size for size in waves if count <= size)
			if size < lanes.active.size:
				scores = _save(scores, ids, lanes.score)
				lanes, ids = _compact(lanes, ids, size, full)
			lanes = self._layer(params, lanes, jnp.int32(ply))
			ply += 1
		scores = _save(scores, ids, lanes.score)
		return _finish(config, scores, own_logits, legal, searched, noise)

	def __call__(self, params, positions, sides, carry, keys, noise=None, *, warm=False):
		"""Process real roots once, then search and return only their real post-root carries."""
		after, logits, _, head = self.root(params, positions, carry)
		decision = self.choose(
			params, positions, sides, after, logits, head, keys, noise, warm=warm
		)
		return decision, after
