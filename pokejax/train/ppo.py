"""Recurrent PPO on the league of pokejax.train.league, with the whole update in one jitted function.

An update is: a rollout of rollout_length steps for batch_size battle rows, in one lax.scan that
samples the policies and steps the environment (League.play); the bootstrap values; GAE over the
lanes; and epochs of minibatch SGD over sequence chunks. The chunks are rollout_length /
sequence_length consecutive steps of one lane. Each chunk starts from the recurrent state the
rollout had at its first step, and the replay resets that state wherever a battle ended, as the
rollout did.

Advantages (gae): a terminated step bootstraps zero, a truncated one bootstraps the value of its
pre-reset observation (final_obs), and no advantage carries across an ended step, because the next
step starts a fresh battle in both cases.

A step whose action a hidden trap refused counts (the retry after it is a transition of its own). A
held step (league.py) and a substituted illegal action do not: they stay in the GAE recursion but get
zero weight in the critic loss. The policy's loss, entropy and their normalisation also leave out
steps with no real choice (a no-op or a single legal action).

The pool holds pool.size parameter sets stacked on a leading axis, all copies of the initial
parameters at first. Every pool.every updates the parameters after that update replace the set of
the next slot (round robin), and the replaced set is kept as the member's previous parameters: a
pool row whose battle began before the insertion keeps reading them until its battle ends
(league.PoolParams, pool_fresh). Pool members are never trained.

Evaluation plays the learner against one baseline on fixed battles (the same keys every call):
the learner takes side 0 in the even rows and side 1 in the odd ones, and it samples its policy or
takes the argmax.
"""

import statistics
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from pokejax.baselines import make_policy
from pokejax.engine.observation import Observation
from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.tables import load_tables
from pokejax.train.auxiliary import auxiliary_terms, finalize_auxiliary
from pokejax.train.checkpoint import pool_digest
from pokejax.train.config import PPOSection, TrainConfig
from pokejax.train.evaluation import summarize_battles
from pokejax.train.league import Layout, League, LeagueState, PoolParams, Transition, select_lanes
from pokejax.train.network import PokePolicy, count_parameters, reset_carry, vocabulary

# Seed offsets keep the evaluation and calibration keys apart from the training keys.
EVALUATION_SEED = 1_000_003
CALIBRATION_SEED = 2_000_003


class LearnerState(NamedTuple):
	params: Any
	opt_state: Any
	league: LeagueState  # the battles, lanes and carries; its key drives the rollout
	pool: Any  # K parameter sets stacked on a leading axis
	pool_inserts: jax.Array  # int32, insertions so far; the next one replaces slot inserts % K
	update: jax.Array  # int32 scalar, updates done


class Steps(NamedTuple):
	"""Sequence chunks, time-major after the gather: leaves are [L, M, ...] in a minibatch."""

	obs: Observation
	action: jax.Array  # int32
	log_prob: jax.Array  # float32, of the action under the rollout's policy
	value: jax.Array
	advantage: jax.Array
	returns: jax.Array
	weight: jax.Array  # float32, 0 for a step that did not count (held or substituted)
	actor: jax.Array  # float32, 1 for a counted decision with a real choice (policy and entropy)
	advance: jax.Array  # bool, the lane's carry advanced at this step (False where held)
	ended: jax.Array  # bool, the battle ended at this step (the carry resets after it)
	aux: Any = None  # AuxLabels of the lane's decision (None unless the auxiliary losses are on)


class Chunks(NamedTuple):
	steps: Steps  # leaves [N, L, ...]
	start: Any  # the carry at each chunk's first step, [N, ...]


class EvaluationCarry(NamedTuple):
	league: LeagueState
	finished: jax.Array  # bool [B], the row's first battle has ended
	wins: jax.Array  # bool [B], that battle was won
	truncated: jax.Array  # bool [B]
	length: jax.Array  # int32 [B], its environment steps
	count: jax.Array  # int32, environment steps taken


def merge_leading(tree, count: int):
	"""Leaves [T, B, ...] as [T * B, ...] (count leading axes merged)."""
	return jax.tree.map(lambda x: x.reshape((-1,) + x.shape[count:]), tree)


def chunk_time(x, chunks: int, length: int, batch: int):
	"""[T, B, ...] as [chunks * B, length, ...]: chunk c of row b is at index c * B + b."""
	x = x.reshape((chunks, length, batch) + x.shape[2:])
	x = jnp.swapaxes(x, 1, 2)
	return x.reshape((chunks * batch, length) + x.shape[3:])


def gae(rewards, values, terminated, truncated, final_values, last_value, gamma, lam):
	"""Generalised advantage estimates and returns, [T, B], with terminated and truncated apart.

	final_values[t] is the value of step t's pre-reset observation (used where truncated);
	last_value [B] is the value of the observation after the rollout.
	"""
	ended = terminated | truncated
	following = jnp.concatenate([values[1:], last_value[None]], axis=0)
	following = jnp.where(terminated, 0.0, jnp.where(truncated, final_values, following))
	deltas = rewards + gamma * following - values

	def backward(advantage, xs):
		delta, done = xs
		advantage = delta + gamma * lam * jnp.where(done, 0.0, advantage)
		return advantage, advantage

	_, advantages = jax.lax.scan(
		backward, jnp.zeros_like(last_value), (deltas, ended), reverse=True
	)
	return advantages, advantages + values


def explained_variance(returns, values, weight):
	"""1 - Var(returns - values) / Var(returns) over the weighted steps (0 when Var is 0)."""
	total = jnp.maximum(weight.sum(), 1.0)

	def variance(x):
		mean = (weight * x).sum() / total
		return (weight * jnp.square(x - mean)).sum() / total

	var_returns = variance(returns)
	var_residual = variance(returns - values)
	return jnp.where(var_returns > 1e-12, 1.0 - var_residual / jnp.maximum(var_returns, 1e-12), 0.0)


def learning_rate_schedule(ppo: PPOSection, gradient_steps: int):
	"""The learning rate by gradient step: constant, linear or cosine to learning_rate_end."""
	steps = max(gradient_steps, 1)
	if ppo.lr_schedule == "constant":
		return optax.constant_schedule(ppo.learning_rate)
	if ppo.lr_schedule == "linear":
		return optax.linear_schedule(ppo.learning_rate, ppo.learning_rate_end, steps)
	return optax.cosine_decay_schedule(
		ppo.learning_rate, steps, alpha=ppo.learning_rate_end / ppo.learning_rate
	)


def _normalize(values, weight, total):
	mean = (weight * values).sum() / total
	spread = jnp.sqrt((weight * jnp.square(values - mean)).sum() / total)
	return (values - mean) / (spread + 1e-8)


class Trainer:
	"""The configuration's pure functions and their host setup (tables, league, jit wrappers).

	Nothing here reads the run directory; run.py drives updates, evaluations and checkpoints.
	"""

	def __init__(self, config: TrainConfig, team_pool=DEFAULT_POOL):
		self.config = config
		self.tables = load_tables()
		r, p = config.rollout, config.ppo
		self.env = Env(
			EnvConfig(
				batch_size=r.batch_size,
				max_steps=config.env.max_steps,
				shaping=config.env.shaping,
				shaping_coef=config.env.shaping_coef,
				gamma=p.gamma,
				observe="both",
				keep_final=True,
			),
			team_pool,
			tables=self.tables,
		)
		self.layout = Layout(config.groups)
		# The digest a checkpoint of this run carries (checkpoint.pool_digest): resumes need the same pool.
		self.team_digest = pool_digest(self.env.pool, self.env.team_indices)
		self.model = PokePolicy(vocabulary(self.tables), config.network, config.auxiliary)
		self.auxiliary = config.auxiliary.enabled
		self.aux_width = self.model.aux_width
		policies = {
			name: make_policy(name, self.env.tables, self.tables)
			for name in self.layout.baseline_names
		}
		self.league = League(
			self.env,
			self.layout,
			self.model,
			config.network,
			config.pool.size,
			policies,
			labels=self.auxiliary,
		)
		decay = p.decay_updates or config.run.updates
		gradient_steps = decay * p.epochs * p.minibatches
		self.schedule = learning_rate_schedule(p, gradient_steps)
		self.tx = optax.chain(
			optax.clip_by_global_norm(p.max_grad_norm),
			optax.adam(self.schedule, eps=p.adam_eps),
		)
		self._init = jax.jit(self._init_state)
		self._update = jax.jit(self._update_step)
		self._rollout_jit = jax.jit(self._rollout)
		self._environment = jax.jit(self._environment_steps)
		self._evaluators = {}

	# State.

	def _init_state(self, key) -> LearnerState:
		key, init_key, reset_key = jax.random.split(key, 3)
		league = self.league.reset(reset_key)._replace(key=key)
		init_method = PokePolicy.full if self.auxiliary else None
		params = self.model.init(init_key, league.obs, league.hidden, method=init_method)["params"]
		return LearnerState(
			params=params,
			opt_state=self.tx.init(params),
			league=league,
			pool=None,
			pool_inserts=jnp.int32(0),
			update=jnp.int32(0),
		)

	def init(self, seed: int | None = None) -> LearnerState:
		"""A fresh learner: initial parameters, optimizer state, pool and battles (seeded)."""
		seed = self.config.run.seed if seed is None else seed
		learner = self._init(jax.random.key(seed))
		# The pool's copies are made outside the jitted init: inside it XLA may recompute the random
		# initial values for each copy, and the copies then differ in the last bits.
		stacked = jax.tree.map(
			lambda x: jnp.broadcast_to(x, (self.config.pool.size,) + x.shape), learner.params
		)
		return learner._replace(pool=PoolParams(current=stacked, previous=stacked))

	def parameter_count(self, learner: LearnerState) -> int:
		return count_parameters(learner.params)

	# Rollout.

	def _rollout(self, params, pool, league: LeagueState):
		def step(state, _):
			return self.league.play(params, pool, state)

		return jax.lax.scan(step, league, None, length=self.config.rollout.rollout_length)

	# Loss.

	def _replay(self, params, obs, ended, advance, start):
		"""The network over chunks from their start carries: logits [L, M, 9], values [L, M].

		The feed-forward encoder runs once over all L * M steps; only the recurrent core is
		scanned over time. A step where the lane's action was held keeps its carry (the rollout did
		not advance it), and the carry resets after each step where a battle ended.
		"""
		flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), obs)
		features = self.model.apply({"params": params}, flat, method=PokePolicy.encode)
		features = features.reshape(ended.shape + features.shape[1:])

		def step(hidden, xs):
			step_features, step_legal, step_ended, step_advance = xs
			after, logits, value, out = self.model.apply(
				{"params": params}, hidden, step_features, step_legal, method=PokePolicy.step_full
			)
			kept = select_lanes(step_advance, after, hidden)
			return reset_carry(kept, step_ended), (logits, value, out)

		_, (logits, values, outs) = jax.lax.scan(
			step, start, (features, obs.legal_actions, ended, advance)
		)
		# The auxiliary heads read the foe active vector, the last aux_width entries of each encoding.
		foe = features[..., features.shape[-1] - self.aux_width :] if self.aux_width else None
		return logits, values, outs, foe

	def _loss(self, params, steps: Steps, start):
		p = self.config.ppo
		logits, values, outs, foe = self._replay(
			params, steps.obs, steps.ended, steps.advance, start
		)
		log_probs = jax.nn.log_softmax(logits)
		chosen = jnp.take_along_axis(log_probs, steps.action[..., None], axis=-1)[..., 0]
		# The critic learns from every counted decision; the policy only from those with a choice,
		# so that no-op and single-action steps do not dilute its normalisation or metrics.
		weight = steps.weight
		total = jnp.maximum(weight.sum(), 1.0)
		actor = steps.actor
		actor_total = jnp.maximum(actor.sum(), 1.0)
		advantage = _normalize(steps.advantage, actor, actor_total)

		ratio = jnp.exp(chosen - steps.log_prob)
		clipped = jnp.clip(ratio, 1 - p.clip, 1 + p.clip)
		policy_loss = -jnp.minimum(ratio * advantage, clipped * advantage)

		value_error = jnp.square(values - steps.returns)
		if p.value_clip > 0:
			shifted = steps.value + jnp.clip(values - steps.value, -p.value_clip, p.value_clip)
			value_error = jnp.maximum(value_error, jnp.square(shifted - steps.returns))
		value_loss = 0.5 * value_error

		entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
		loss = (actor * (policy_loss - p.entropy_coef * entropy)).sum() / actor_total
		loss += p.value_coef * (weight * value_loss).sum() / total

		log_ratio = chosen - steps.log_prob
		metrics = {
			"policy_loss": (actor * policy_loss).sum() / actor_total,
			"value_loss": (weight * value_loss).sum() / total,
			"entropy": (actor * entropy).sum() / actor_total,
			"approx_kl": (actor * (ratio - 1 - log_ratio)).sum() / actor_total,
			"clip_fraction": (actor * (jnp.abs(ratio - 1) > p.clip)).sum() / actor_total,
		}
		if self.auxiliary and steps.aux is not None:
			flat = lambda x: x.reshape((-1,) + x.shape[2:])
			heads = self.model.apply(
				{"params": params}, flat(outs), flat(foe), method=PokePolicy.auxiliary
			)
			aux_loss, aux_metrics = auxiliary_terms(
				heads,
				jax.tree.map(flat, steps.aux),
				jax.tree.map(flat, steps.obs),
				self.config.auxiliary,
			)
			loss = loss + aux_loss
			metrics.update(aux_metrics)
		return loss, metrics

	def _learn(self, params, opt_state, chunks: Chunks, key):
		"""Epochs of minibatch SGD over shuffled chunks; returns metrics averaged over the steps."""
		p = self.config.ppo
		count = chunks.steps.weight.shape[0]

		def minibatch(carry, ids):
			params, opt_state = carry
			steps = jax.tree.map(lambda x: jnp.swapaxes(x[ids], 0, 1), chunks.steps)
			start = jax.tree.map(lambda x: x[ids], chunks.start)
			(_, metrics), grads = jax.value_and_grad(self._loss, has_aux=True)(params, steps, start)
			updates, opt_state = self.tx.update(grads, opt_state, params)
			params = optax.apply_updates(params, updates)
			return (params, opt_state), dict(metrics, grad_norm=optax.tree.norm(grads))

		def epoch(carry, epoch_key):
			order = jax.random.permutation(epoch_key, count).reshape(p.minibatches, -1)
			return jax.lax.scan(minibatch, carry, order)

		keys = jax.random.split(key, p.epochs)
		(params, opt_state), metrics = jax.lax.scan(epoch, (params, opt_state), keys)
		return params, opt_state, {name: value.mean() for name, value in metrics.items()}

	# The update.

	def _rollout_metrics(self, traj: Transition, returns, weight) -> dict:
		"""Battle and learner statistics of one rollout; win rates per group, from perspectives."""
		layout = self.layout
		ended = traj.terminated | traj.truncated
		completed = jnp.maximum(ended.sum().astype(jnp.float32), 1.0)
		lane_ended = layout.lanes_in_time(ended)
		lane_completed = jnp.maximum(lane_ended.sum().astype(jnp.float32), 1.0)
		metrics = {
			"explained_variance": explained_variance(
				returns.reshape(-1), traj.value.reshape(-1), weight.reshape(-1)
			),
			"completed_battles": ended.sum().astype(jnp.float32),
			"truncated_battles": traj.truncated.sum().astype(jnp.float32),
			"win_rate": traj.won.sum() / completed,
			"mean_battle_length": jnp.where(ended, traj.steps, 0).sum() / completed,
			"mean_battle_turns": jnp.where(ended, traj.turns, 0).sum() / completed,
			"mean_terminal_reward": (traj.reward * lane_ended).sum() / lane_completed,
			"illegal_actions": traj.illegal.sum().astype(jnp.float32),
			"refused_switches": traj.rejected.sum().astype(jnp.float32),
		}
		for name, (start, stop) in layout.spans.items():
			group_ended = ended[:, start:stop].sum().astype(jnp.float32)
			group_won = traj.won[:, start:stop].sum().astype(jnp.float32)
			metrics[f"completed_{name}"] = group_ended
			metrics[f"win_rate_{name}"] = group_won / jnp.maximum(group_ended, 1.0)
		return metrics

	def _insert(self, pool: PoolParams, fresh, inserts, params, update):
		"""The pool after update `update`, every pool.every updates, in its next slot.

		The slot's current parameters become its previous ones and the new parameters its current
		ones. The fresh pool rows of that member read the previous parameters from now on (their
		battles began on them), and the battles that start later read the new ones. Returns the
		pool, the fresh flags and the insertion count.
		"""
		every = self.config.pool.every
		if every == 0:
			return pool, fresh, inserts
		due = jnp.asarray(update) % every == 0
		slot = inserts % self.config.pool.size
		previous = jax.tree.map(
			lambda old, current: old.at[slot].set(jnp.where(due, current[slot], old[slot])),
			pool.previous,
			pool.current,
		)
		current = jax.tree.map(
			lambda stored, new: stored.at[slot].set(jnp.where(due, new, stored[slot])),
			pool.current,
			params,
		)
		if fresh.shape[0]:
			per_member = fresh.shape[0] // self.config.pool.size
			member = jnp.arange(fresh.shape[0]) // per_member
			fresh = jnp.where(due & (member == slot), False, fresh)
		return PoolParams(current, previous), fresh, inserts + due.astype(jnp.int32)

	def _update_step(self, learner: LearnerState):
		p, r = self.config.ppo, self.config.rollout
		layout = self.layout
		lanes, length, sequence = layout.lanes, r.rollout_length, r.sequence_length
		chunks_per_row = length // sequence
		league, traj = self._rollout(learner.params, learner.pool, learner.league)

		def apply(obs, hidden):
			return self.model.apply({"params": learner.params}, obs, hidden)

		_, _, last_value = apply(league.obs, league.hidden)

		final_value = self.final_values(learner.params, traj)
		terminated = layout.lanes_in_time(traj.terminated)
		truncated = layout.lanes_in_time(traj.truncated)
		advantage, returns = gae(
			traj.reward,
			traj.value,
			terminated,
			truncated,
			final_value,
			last_value,
			p.gamma,
			p.gae_lambda,
		)

		weight = traj.weight.astype(jnp.float32)
		actor = traj.actor.astype(jnp.float32)

		def chunked(x):
			return chunk_time(x, chunks_per_row, sequence, lanes)

		steps = Steps(
			obs=jax.tree.map(chunked, traj.obs),
			action=chunked(traj.action),
			log_prob=chunked(traj.log_prob),
			value=chunked(traj.value),
			advantage=chunked(advantage),
			returns=chunked(returns),
			weight=chunked(weight),
			actor=chunked(actor),
			advance=chunked(traj.advance),
			ended=chunked(terminated | truncated),
			aux=jax.tree.map(chunked, traj.aux),
		)
		start = jax.tree.map(
			lambda x: x[::sequence].reshape((chunks_per_row * lanes,) + x.shape[2:]), traj.hidden
		)
		key, shuffle_key = jax.random.split(league.key)
		params, opt_state, metrics = self._learn(
			learner.params, learner.opt_state, Chunks(steps, start), shuffle_key
		)
		metrics.update(self._rollout_metrics(traj, returns, weight))
		metrics = finalize_auxiliary(metrics)
		update = learner.update + 1
		metrics["learning_rate"] = self.schedule(learner.update * p.epochs * p.minibatches)
		pool, fresh, inserts = self._insert(
			learner.pool, league.pool_fresh, learner.pool_inserts, params, update
		)
		league = league._replace(key=key, pool_fresh=fresh)
		learner = LearnerState(params, opt_state, league, pool, inserts, update)
		return learner, metrics

	def final_values(self, params, traj: Transition):
		"""The value of each step's pre-reset observation [T, L], read only where a battle was truncated.

		gae uses these values only at truncated steps, so a rollout with no truncated battle skips the
		network over every final observation and returns zeros.
		"""
		length, lanes = traj.action.shape

		def truncation_values():
			def one_step(xs):
				obs, hidden = xs
				_, _, values = self.model.apply({"params": params}, obs, hidden)
				return values

			# One step at a time: the pair features of a whole rollout's final observations would not fit.
			return jax.lax.map(one_step, (traj.final_obs, traj.after))

		return jax.lax.cond(
			traj.truncated.any(),
			truncation_values,
			lambda: jnp.zeros((length, lanes), jnp.float32),
		)

	def update(self, learner: LearnerState):
		"""One update (rollout, GAE and the epochs), jitted; returns the new state and metrics."""
		return self._update(learner)

	def rollout(self, params, pool, league: LeagueState):
		"""The rollout of one update on its own, jitted: the league after it and Transition [T, ...].

		Tests and inspection use it; the update runs the same scan inside its own jitted function.
		"""
		return self._rollout_jit(params, pool, league)

	# Evaluation.

	def _evaluate_batch(self, league: League, params, key, greedy):
		batch = league.layout.batch
		learner = jnp.arange(batch, dtype=jnp.int32) % 2
		state = league.reset(key, learner)
		zeros = jnp.zeros(batch, jnp.bool_)
		start = EvaluationCarry(
			league=state,
			finished=zeros,
			wins=zeros,
			truncated=zeros,
			length=jnp.zeros(batch, jnp.int32),
			count=jnp.int32(0),
		)
		limit = self.config.env.max_steps

		def cond(c):
			return (~c.finished.all()) & (c.count < limit)

		def body(c):
			state, step = league.play(params, None, c.league, greedy)
			ended = step.terminated | step.truncated
			first = ended & ~c.finished
			return EvaluationCarry(
				league=state,
				finished=c.finished | ended,
				wins=jnp.where(first, step.won, c.wins),
				truncated=jnp.where(first, step.truncated, c.truncated),
				length=jnp.where(first, step.steps, c.length),
				count=c.count + 1,
			)

		final = jax.lax.while_loop(cond, body, start)
		return (
			final.wins.sum(),
			final.truncated.sum(),
			final.length.sum(),
			(~final.finished).sum(),
		)

	def _evaluator(self, name: str):
		if name not in self._evaluators:
			batch = self.config.evaluation.batch
			env = self.env.replace(batch_size=batch, keep_final=False)
			policy = make_policy(name, self.env.tables, self.tables)
			league = League(
				env,
				Layout({name: batch}),
				self.model,
				self.config.network,
				self.config.pool.size,
				{name: policy},
			)
			self._evaluators[name] = jax.jit(
				lambda params, key, greedy: self._evaluate_batch(league, params, key, greedy)
			)
		return self._evaluators[name]

	def evaluate(self, params, opponent: str, battles: int, greedy: bool = False) -> dict:
		"""The learner against one baseline over fixed battles, sampling or taking the argmax.

		The battles and their sides come from the same keys every call, so evaluations at
		different updates, and the sampled and greedy ones, play the same battles. battles must
		be a positive multiple of the evaluation batch: every battle counted is one that was played.
		"""
		batch = self.config.evaluation.batch
		if battles <= 0 or battles % batch:
			raise ValueError(
				f"evaluation needs a positive multiple of the evaluation batch ({batch}), "
				f"got {battles} battles"
			)
		run = self._evaluator(opponent)
		key = jax.random.key(self.config.run.seed + EVALUATION_SEED)
		wins = truncated = length = unfinished = 0
		for index in range(battles // batch):
			out = jax.device_get(run(params, jax.random.fold_in(key, index), jnp.bool_(greedy)))
			wins += int(out[0])
			truncated += int(out[1])
			length += int(out[2])
			unfinished += int(out[3])
		policy = "greedy" if greedy else "sampled"
		result = summarize_battles(
			opponent, battles, wins, truncated, length / battles, policy=policy
		)
		result["unfinished"] = unfinished
		return result

	def evaluate_all(self, params, battles: int) -> list[dict]:
		"""Every configured baseline, sampled and then (if configured) greedy."""
		modes = (False, True) if self.config.evaluation.greedy else (False,)
		return [
			self.evaluate(params, name, battles, greedy)
			for name in self.config.evaluation.opponents
			for greedy in modes
		]

	# Throughput.

	def _environment_steps(self, key):
		state = self.league.reset(key)

		def step(state, _):
			return self.league.random_play(state), None

		state, _ = jax.lax.scan(step, state, None, length=self.config.rollout.rollout_length)
		leaves = jax.tree.leaves(state.obs)
		return sum(jnp.sum(x.astype(jnp.float32)) for x in leaves)

	def environment_steps_per_second(self, repetitions: int = 3) -> float:
		"""Environment steps per second of one rollout's worth of steps, without the network.

		The environment alone (sampler draws, auto-reset, the baselines and observations) with
		uniform random legal actions on every side; the median of `repetitions` timed calls after
		one call that compiles.
		"""
		key = jax.random.key(self.config.run.seed + CALIBRATION_SEED)
		jax.block_until_ready(self._environment(key))
		durations = []
		for index in range(repetitions):
			start = time.perf_counter()
			jax.block_until_ready(self._environment(jax.random.fold_in(key, index)))
			durations.append(time.perf_counter() - start)
		steps = self.config.rollout.batch_size * self.config.rollout.rollout_length
		return steps / statistics.median(durations)

	def forward_cost(self, learner: LearnerState) -> dict:
		"""FLOPs of one policy forward pass for one lane (one observation), by XLA cost analysis.

		A training step runs more than one forward pass per battle row (both learner seats,
		and a pool or baseline opponent's network), so this is not a per-battle-step count.
		"""
		obs = jax.tree.map(lambda x: x[:1], learner.league.obs)
		hidden = jax.tree.map(lambda x: x[:1], learner.league.hidden)

		def forward(params, obs, hidden):
			return self.model.apply({"params": params}, obs, hidden)

		compiled = jax.jit(forward).lower(learner.params, obs, hidden).compile()
		analysis = compiled.cost_analysis()
		if isinstance(analysis, (list, tuple)):
			analysis = analysis[0] if analysis else {}
		return {"flops_per_policy_forward": float((analysis or {}).get("flops", float("nan")))}
