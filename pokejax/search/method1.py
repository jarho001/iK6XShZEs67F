"""One-decision Method 1. Deployment accepts only PlayerView and a battle-local belief."""

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import NUM_ACTIONS, action_masks
from pokejax.engine.observation import observe
from pokejax.engine.reset import prepare_teams
from pokejax.engine.state import LOCKED_MOVE, MOVE, MUST_RECHARGE, NONE, ONGOING, TWO_TURN_MOVE
from pokejax.search.belief import Belief
from pokejax.search.oracle import Decision, SearchConfig, _auxiliary_request, backup, decide
from pokejax.search.rebuild import SetData, rebuild
from pokejax.search.transition import advance_batch, observed_mask


class SearchMetrics(NamedTuple):
	eligible: np.ndarray
	no_particle: np.ndarray
	unrebuildable: np.ndarray
	particles: np.ndarray
	ess: np.ndarray
	valid_worlds: np.ndarray


@partial(jax.jit, static_argnames=("model",))
def infer(model, params, observations, carry, held):
	after, logits, _, heads = model.apply(
		{"params": params}, observations, carry, method=model.full
	)
	after = jax.tree.map(lambda old, new: jnp.where(held[..., None], old, new), carry, after)
	return after, logits, heads["next_action"]


def _queued(position, side, head, key, tables):
	"""Sample the unexecuted foe move at a pivot, conditional on it being a move."""
	foe = 1 - side
	state = position.state
	move_request = state._replace(request=state.request.at[foe].set(MOVE))
	base, blocked = action_masks(move_request, tables)
	base = base.at[foe, 4:].set(False)
	logits = _auxiliary_request(
		position._replace(state=move_request, base=base, blocked=blocked), foe, head
	)
	action = jax.random.categorical(key, logits).astype(jnp.int8)
	active = state.active[foe]
	moves = state.moves[foe, active]
	v = state.volatiles[foe]
	move = jnp.where(state.pp[foe, active, action] > 0, moves[action], tables.struggle_id)
	move = jnp.where(v[LOCKED_MOVE] > 0, v[LOCKED_MOVE], move)
	move = jnp.where(v[TWO_TURN_MOVE] > 0, v[TWO_TURN_MOVE], move)
	move = jnp.where(v[MUST_RECHARGE] > 0, 0, move)
	return position._replace(
		state=state._replace(
			turn_actions=state.turn_actions.at[foe].set(
				jnp.where(state.turn_pending[foe], action, 0)
			),
			turn_moves=state.turn_moves.at[foe].set(
				jnp.where(state.turn_pending[foe], move, 0).astype(state.turn_moves.dtype)
			),
		)
	)


@partial(jax.jit, static_argnames=("protocol",))
def _worlds(
	views, particles, histories, previous_actions, head, keys, data, tables, protocol=False
):
	def one(view, particles, history, previous_action, head, key):
		def make(roster, disguise, key):
			position, valid = rebuild(
				view, roster, disguise, history, previous_action, 0, data, tables, protocol
			)
			return _queued(position, view.side, head, key, tables), valid

		return jax.vmap(make)(*particles, jax.random.split(key, particles.sets.shape[0]))

	return jax.vmap(one)(views, particles, histories, previous_actions, head, keys)


class RolloutLanes(NamedTuple):
	positions: object
	actions: jax.Array
	chance: jax.Array
	carry: object
	sides: jax.Array


@partial(jax.jit, static_argnames=("draws",))
def _prepare(positions, sides, head, keys, draws):
	def one(worlds, side, head, key):
		def sample(position, key):
			policy, chance = jax.random.split(key)
			logits = _auxiliary_request(position, 1 - side, head)
			actions = jax.random.categorical(policy, logits, shape=(draws,)).astype(jnp.int32)
			return actions, jax.random.split(chance, draws)

		return jax.vmap(sample)(worlds, jax.random.split(key, worlds.state.turn.size))

	foe, chance = jax.vmap(one)(positions, sides, head, keys)
	return foe, chance


@partial(jax.jit, static_argnames=("size", "draws"))
def _lanes(size, draws, positions, valid, legal, eligible, sides, carry, actions, chance):
	full = valid.shape[0] * NUM_ACTIONS * valid.shape[1] * draws
	mask = eligible[:, None, None] & legal[:, :, None] & valid[:, None, :]
	mask = jnp.broadcast_to(mask[..., None], (*mask.shape, draws)).reshape(-1)
	ids = jnp.nonzero(mask, size=size, fill_value=full)[0]
	lane = jnp.minimum(ids, full - 1)
	samples = valid.shape[1] * draws
	root, action = lane // (NUM_ACTIONS * samples), (lane // samples) % NUM_ACTIONS
	world, draw = (lane // draws) % valid.shape[1], lane % draws
	side = sides[root]
	pairs = jnp.where(
		jnp.arange(2) == side[:, None], action[:, None], actions[root, world, draw, None]
	)
	return RolloutLanes(
		jax.tree.map(lambda x: x[root, world], positions),
		pairs,
		chance[root, world, draw],
		jax.tree.map(lambda x: x[root], carry),
		side,
	), ids


@jax.jit
def _advance(lanes, tables):
	return advance_batch(lanes.positions, lanes.actions, lanes.chance, None, tables)[0]


@partial(jax.jit, static_argnames=("model",))
def _value(model, params, position, sides, carry, tables):
	observation = jax.vmap(observe, in_axes=(0, 0, None))(position.state, sides, tables)
	after, _, values = model.apply({"params": params}, observation, carry)
	rows = jnp.arange(sides.size)
	held = position.has_held[rows, sides]
	after = jax.tree.map(lambda old, new: jnp.where(held[..., None], old, new), carry, after)
	wait = (position.state.winner == ONGOING) & (
		held | (position.state.request[rows, sides] == NONE)
	)
	return after, values, wait


@jax.jit
def _score(scores, ids, position, sides, values, ply, gamma):
	backed_up = backup(position.state.winner, sides, values, ply, gamma)
	return scores.at[ids].set(backed_up, mode="drop")


@partial(jax.jit, static_argnames=("size", "replicates"))
def _follow(size, replicates, positions, lanes, ids, after, wait, head, ply, full):
	# We have no new choice on these lanes. Advance an opponent-only or held
	# request with the same observation-conditioned head and fresh chance draws.
	indices = jnp.nonzero(wait & (ids < full), size=size, fill_value=ids.size)[0]
	present = indices < ids.size
	indices = jnp.minimum(indices, ids.size - 1)
	new_ids = jnp.where(present, ids[indices], full)
	root = jnp.minimum(new_ids, full - 1) // (NUM_ACTIONS * replicates)
	side = lanes.sides[indices]
	position = jax.tree.map(lambda x: x[indices], positions)
	keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(lanes.chance[indices], ply)
	policy, chance = jax.vmap(jax.random.split, out_axes=1)(keys)
	logits = jax.vmap(_auxiliary_request)(position, 1 - side, head[root])
	foe = jax.vmap(jax.random.categorical)(policy, logits).astype(jnp.int32)
	actions = jnp.where(jnp.arange(2) == side[:, None], 0, foe[:, None])
	return RolloutLanes(
		position, actions, chance, jax.tree.map(lambda x: x[indices], after), side
	), new_ids


@partial(jax.jit, static_argnames=("config",))
def _reduce(config, scores, valid, available, legal, logits, noise):
	scored = scores.reshape(logits.shape[0], NUM_ACTIONS, config.worlds, config.draws)
	# All actions share the same validity filter. Scale retained samples so
	# decide()'s mean is the conditional mean over the validated worlds.
	scored *= (config.worlds / jnp.maximum(1, valid.sum(1)))[:, None, None, None]
	return decide(
		scored.reshape(logits.shape[0], NUM_ACTIONS, -1), legal, logits, available, config, noise
	)


class Method1:
	def __init__(self, model, tables, catalogue, config=None):
		config = SearchConfig() if config is None else config
		if config.depth != 1 or config.opponent != "aux":
			raise ValueError("Method 1 requires depth 1 and the next-action opponent model")
		self.model = model
		self.tables = tables
		self.catalogue = catalogue
		self.config = config
		self._prewarm_follow = False
		if catalogue is not None:
			teams = jax.device_put(catalogue.sets)
			dex = catalogue
			self.data = SetData(
				teams,
				jax.jit(prepare_teams)(teams, tables),
				jnp.int16(dex.trace),
				jnp.int16(dex.regenerator),
			)

	def new_beliefs(self, seed, count):
		return [
			Belief(self.catalogue, np.random.default_rng(key))
			for key in np.random.SeedSequence(seed).spawn(count)
		]

	def request(self, params, views, beliefs, carry, keys, noise, previous_actions):
		"""Process one or more real requests; return only the real updated carry."""
		after, logits, head = infer(self.model, params, views.obs, carry, views.has_held)
		decision, metrics = self.choose(
			params, views, beliefs, after, logits, head, keys, noise, previous_actions
		)
		return decision, metrics, after

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
		sleep_origins=None,
	):
		"""Only PlayerView, own recurrent inference, keys and the prior enter deployment."""
		host_obs, refused, held, result = jax.device_get(
			(views.obs, views.refused, views.has_held, views.result)
		)
		eligible = (result == ONGOING) & ~held & (host_obs.legal_actions.sum(1) > 1)
		rows = zip(
			beliefs,
			(jax.tree.map(lambda x, i=i: x[i], host_obs) for i in range(len(beliefs))),
			refused | held,
			result,
			eligible,
			strict=True,
		)
		history_rows = []
		particle_rows = []
		for i, (belief, obs, was_refused, winner, search) in enumerate(rows):
			origin = None if sleep_origins is None else sleep_origins[i]
			history_rows.append(
				belief.update(obs, was_refused, propose=winner == ONGOING, sleep_origin=origin)
			)
			particle_rows.append(
				belief.sample(self.config.worlds) if search else belief.empty(self.config.worlds)
			)
		particles, counts, ess = zip(*particle_rows, strict=True)
		counts = np.asarray(counts)
		if not np.any(counts):
			decision = self._raw(views.obs.legal_actions, logits, noise)
			zeros = np.zeros_like(counts)
			return decision, SearchMetrics(
				eligible, eligible, zeros, counts, np.asarray(ess), zeros
			)
		histories = jax.tree.map(lambda *x: np.stack(x), *history_rows)
		particles = jax.tree.map(lambda *x: np.stack(x), *particles)
		queue_keys = jax.vmap(jax.random.fold_in, in_axes=(0, None))(keys, 19)
		worlds, valid = _worlds(
			views,
			particles,
			histories,
			previous_actions,
			head,
			queue_keys,
			self.data,
			self.tables,
			protocol,
		)
		valid &= jnp.asarray(counts[:, None] > 0)
		decision, valid_counts = self._choose_worlds(
			params,
			worlds,
			valid,
			views.obs.legal_actions,
			eligible,
			views.side,
			carry,
			logits,
			head,
			keys,
			noise,
		)
		return decision, SearchMetrics(
			eligible,
			eligible & (counts == 0),
			(counts > 0) & (valid_counts == 0),
			counts,
			np.asarray(ess),
			valid_counts,
		)

	def prewarm_single(self, params, view, belief, carry, keys):
		"""Compile the fixed single-battle rollout and waiting-request shapes."""
		if view.side.size != 1:
			raise ValueError("This prewarm covers single-battle concurrency only")
		after, logits, head = infer(self.model, params, view.obs, carry, view.has_held)
		previous = jnp.zeros(1, jnp.int32)
		self._prewarm_follow = True
		try:
			decision, metrics = self.choose(
				params,
				view,
				[belief],
				after,
				logits,
				head,
				keys,
				jnp.zeros_like(logits),
				previous,
				protocol=True,
			)
			jax.block_until_ready((decision, metrics.valid_worlds))
			jax.block_until_ready(self._raw(view.obs.legal_actions, logits, jnp.zeros_like(logits)))
			if not int(np.asarray(metrics.valid_worlds)[0]):
				raise RuntimeError("Prewarm had no valid rebuilt worlds")
		finally:
			self._prewarm_follow = False

	def diagnostic_oracle(self, params, truth, sides, carry, logits, head, keys, noise):
		"""Explicit diagnostic arm with true opposing state and queued actions."""
		batch = sides.size
		rows = jnp.arange(batch)
		legal = observed_mask(truth)[rows, sides]
		eligible = (
			(truth.state.winner == ONGOING) & (legal.sum(1) > 1) & ~truth.has_held[rows, sides]
		)
		worlds = jax.tree.map(
			lambda x: jnp.broadcast_to(x[:, None], (batch, self.config.worlds, *x.shape[1:])), truth
		)
		valid = jnp.broadcast_to(eligible[:, None], (batch, self.config.worlds))
		decision, counts = self._choose_worlds(
			params, worlds, valid, legal, eligible, sides, carry, logits, head, keys, noise
		)
		eligible = np.asarray(eligible)
		zeros = np.zeros(batch)
		return decision, SearchMetrics(eligible, zeros, zeros, zeros, zeros, counts)

	@staticmethod
	@jax.jit
	def _raw(legal, logits, noise):
		raw = jnp.argmax(jnp.where(legal, logits + noise, -jnp.inf), axis=-1).astype(jnp.int32)
		return Decision(
			raw,
			raw,
			jnp.zeros_like(raw, dtype=bool),
			jnp.zeros_like(raw, dtype=jnp.float32),
			jnp.zeros_like(raw, dtype=jnp.float32),
		)

	def _choose_worlds(
		self, params, worlds, valid, legal, eligible, sides, carry, logits, head, keys, noise
	):
		config = self.config
		n_valid, candidates = jax.device_get((valid.sum(1), legal.sum(1)))
		eligible = np.asarray(eligible)
		available = eligible & (n_valid > 0)
		count = int(np.sum(n_valid * candidates * available)) * config.draws
		if count == 0:
			return self._raw(legal, logits, noise), n_valid
		# Keep one fixed shape at live single-battle latency, and compact batched matches.
		full = sides.size * NUM_ACTIONS * config.replicates
		width = min(1024, full)
		size = min(full, (count + width - 1) // width * width)
		actions, chance = _prepare(worlds, sides, head, keys, config.draws)
		lanes, ids = _lanes(
			size, config.draws, worlds, valid, legal, available, sides, carry, actions, chance
		)
		scores = jnp.zeros(full, jnp.float32)
		for ply in range(config.max_steps):
			successor = _advance(lanes, self.tables)
			after, values, wait = _value(
				self.model, params, successor, lanes.sides, lanes.carry, self.tables
			)
			scores = _score(scores, ids, successor, lanes.sides, values, ply, config.gamma)
			if self._prewarm_follow and ply == 0:
				# Even a normal root can lead to a pivot or waiting request later.
				# At single-battle concurrency both compacted widths equal full.
				warm_lanes, warm_ids = _follow(
					full,
					config.replicates,
					successor,
					lanes,
					ids,
					after,
					jnp.ones_like(wait),
					head,
					ply,
					full,
				)
				jax.block_until_ready((warm_lanes, warm_ids))
			pending = int(jnp.sum(wait & (ids < full)))
			if not pending or ply + 1 == config.max_steps:
				break
			width = min(512, full)
			size = min(full, (pending + width - 1) // width * width)
			lanes, ids = _follow(
				size, config.replicates, successor, lanes, ids, after, wait, head, ply, full
			)
		decision = _reduce(config, scores, valid, available, legal, logits, noise)
		return decision, n_valid
