"""Recurrent PPO on the opponent environment, with the whole update in one jitted function.

An update is: a rollout of rollout_length environment steps for batch_size battles, in one
lax.scan that samples the policy and steps the environment; the bootstrap values; GAE; and
epochs of minibatch SGD over sequence chunks. The chunks are rollout_length / sequence_length
consecutive steps of one battle row. Each chunk starts from the recurrent state the rollout had
at its first step, and the replay resets that state wherever a battle ended, as the rollout did.

Advantages (gae): a terminated step bootstraps zero, a truncated one bootstraps the value of its
pre-reset observation (env keep_final), and no advantage carries across an ended step, because
the next step starts a fresh battle in both cases.

Steps whose learner switch a hidden trap refused did not advance the battle; they stay in the
GAE recursion but get zero weight in the losses and the normalisation.
"""

import statistics
import time
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

from pokejax.baselines import make_policy
from pokejax.engine.observation import Observation
from pokejax.env import DEFAULT_POOL, Env, EnvConfig, LearnerInfo, OpponentEnv, OpponentState
from pokejax.tables import load_tables
from pokejax.train.config import PPOSection, TrainConfig
from pokejax.train.evaluation import summarize_battles
from pokejax.train.network import (
	PokePolicy,
	count_parameters,
	initial_carry,
	reset_carry,
	vocabulary,
)

# Seed offsets keep the evaluation and calibration keys apart from the training keys.
EVALUATION_SEED = 1_000_003
CALIBRATION_SEED = 2_000_003


class LearnerState(NamedTuple):
	params: Any
	opt_state: Any
	env: OpponentState
	obs: Observation  # the learner's observation of each battle, [B, ...]
	hidden: Any  # the carry the next step starts from, [B, ...]
	key: jax.Array
	update: jax.Array  # int32 scalar, updates done


class RolloutCarry(NamedTuple):
	env: OpponentState
	obs: Observation
	hidden: Any
	key: jax.Array


class Transition(NamedTuple):
	"""One environment step of the rollout; after the scan every leaf is [T, B, ...]."""

	obs: Observation  # the learner's observation the action was chosen from
	hidden: Any  # the carry before the step
	action: jax.Array  # int32
	log_prob: jax.Array  # of the action under the rollout's policy
	value: jax.Array
	reward: jax.Array
	terminated: jax.Array
	truncated: jax.Array
	info: LearnerInfo
	final_obs: Observation  # the observation before an automatic reset
	after: Any  # the carry after the core for this observation, before any reset


class Steps(NamedTuple):
	"""Sequence chunks, time-major after the gather: leaves are [L, M, ...] in a minibatch."""

	obs: Observation
	action: jax.Array
	log_prob: jax.Array
	value: jax.Array
	advantage: jax.Array
	returns: jax.Array
	weight: jax.Array  # float32, 0 for a step whose switch was refused
	ended: jax.Array  # bool, the battle ended at this step (the carry resets after it)


class Chunks(NamedTuple):
	steps: Steps  # leaves [N, L, ...]
	start: Any  # the carry at each chunk's first step, [N, ...]


class EvaluationCarry(NamedTuple):
	state: OpponentState
	obs: Observation
	hidden: Any
	finished: jax.Array  # bool [B], the slot's first battle has ended
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
	"""The configuration's pure functions and their host setup (tables, pool, jit wrappers).

	Nothing here reads the run directory; run.py drives updates, evaluations and checkpoints.
	"""

	def __init__(self, config: TrainConfig, team_pool=DEFAULT_POOL):
		self.config = config
		self.tables = load_tables()
		r, p = config.rollout, config.ppo
		self.base = Env(
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
		baseline = make_policy(config.opponent.baseline, self.base.tables, self.tables)
		self.train_env = OpponentEnv(self.base, baseline)
		self.model = PokePolicy(vocabulary(self.tables), config.network)
		gradient_steps = config.run.updates * p.epochs * p.minibatches
		self.schedule = learning_rate_schedule(p, gradient_steps)
		self.tx = optax.chain(
			optax.clip_by_global_norm(p.max_grad_norm),
			optax.adam(self.schedule, eps=p.adam_eps),
		)
		self._init = jax.jit(self._init_state)
		self._update = jax.jit(self._update_step)
		self._environment = jax.jit(self._environment_steps)
		self._evaluators = {}

	# State.

	def _init_state(self, key) -> LearnerState:
		r = self.config.rollout
		key, init_key, reset_key = jax.random.split(key, 3)
		env, timestep = self.train_env.reset(reset_key)
		hidden = initial_carry(self.config.network, (r.batch_size,))
		params = self.model.init(init_key, timestep.obs, hidden)["params"]
		return LearnerState(
			params, self.tx.init(params), env, timestep.obs, hidden, key, jnp.int32(0)
		)

	def init(self, seed: int | None = None) -> LearnerState:
		"""A fresh learner: initial parameters, optimizer state and battles (seeded)."""
		seed = self.config.run.seed if seed is None else seed
		return self._init(jax.random.key(seed))

	def parameter_count(self, learner: LearnerState) -> int:
		return count_parameters(learner.params)

	# Rollout.

	def _rollout(self, params, carry: RolloutCarry):
		def step(carry, _):
			key, action_key = jax.random.split(carry.key)
			after, logits, value = self.model.apply({"params": params}, carry.obs, carry.hidden)
			action = jax.random.categorical(action_key, logits)
			log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=-1)[
				:, 0
			]
			env, timestep = self.train_env.step(carry.env, action)
			ended = timestep.terminated | timestep.truncated
			transition = Transition(
				obs=carry.obs,
				hidden=carry.hidden,
				action=action,
				log_prob=log_prob,
				value=value,
				reward=timestep.reward,
				terminated=timestep.terminated,
				truncated=timestep.truncated,
				info=timestep.info,
				final_obs=timestep.final_obs,
				after=after,
			)
			next_carry = RolloutCarry(env, timestep.obs, reset_carry(after, ended), key)
			return next_carry, transition

		return jax.lax.scan(step, carry, None, length=self.config.rollout.rollout_length)

	# Loss.

	def _replay(self, params, obs, ended, start):
		"""The network over chunks from their start carries: logits [L, M, 9], values [L, M].

		The feed-forward encoder runs once over all L * M steps; only the recurrent core is
		scanned over time, resetting the carry after each step where a battle ended.
		"""
		flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), obs)
		features = self.model.apply({"params": params}, flat, method=PokePolicy.encode)
		features = features.reshape(ended.shape + features.shape[1:])

		def step(hidden, xs):
			step_features, step_legal, step_ended = xs
			after, logits, value = self.model.apply(
				{"params": params}, hidden, step_features, step_legal, method=PokePolicy.step
			)
			return reset_carry(after, step_ended), (logits, value)

		_, (logits, values) = jax.lax.scan(step, start, (features, obs.legal_actions, ended))
		return logits, values

	def _loss(self, params, steps: Steps, start):
		p = self.config.ppo
		logits, values = self._replay(params, steps.obs, steps.ended, start)
		log_probs = jax.nn.log_softmax(logits)
		chosen = jnp.take_along_axis(log_probs, steps.action[..., None], axis=-1)[..., 0]
		weight = steps.weight
		total = jnp.maximum(weight.sum(), 1.0)
		advantage = _normalize(steps.advantage, weight, total)

		ratio = jnp.exp(chosen - steps.log_prob)
		clipped = jnp.clip(ratio, 1 - p.clip, 1 + p.clip)
		policy_loss = -jnp.minimum(ratio * advantage, clipped * advantage)

		value_error = jnp.square(values - steps.returns)
		if p.value_clip > 0:
			shifted = steps.value + jnp.clip(values - steps.value, -p.value_clip, p.value_clip)
			value_error = jnp.maximum(value_error, jnp.square(shifted - steps.returns))
		value_loss = 0.5 * value_error

		entropy = -jnp.sum(jnp.exp(log_probs) * log_probs, axis=-1)
		per_step = policy_loss + p.value_coef * value_loss - p.entropy_coef * entropy
		loss = (weight * per_step).sum() / total

		log_ratio = chosen - steps.log_prob
		metrics = {
			"policy_loss": (weight * policy_loss).sum() / total,
			"value_loss": (weight * value_loss).sum() / total,
			"entropy": (weight * entropy).sum() / total,
			"approx_kl": (weight * (ratio - 1 - log_ratio)).sum() / total,
			"clip_fraction": (weight * (jnp.abs(ratio - 1) > p.clip)).sum() / total,
		}
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

	def _update_step(self, learner: LearnerState):
		p, r = self.config.ppo, self.config.rollout
		batch, length, sequence = r.batch_size, r.rollout_length, r.sequence_length
		chunks_per_row = length // sequence
		carry = RolloutCarry(learner.env, learner.obs, learner.hidden, learner.key)
		carry, traj = self._rollout(learner.params, carry)

		def apply(obs, hidden):
			return self.model.apply({"params": learner.params}, obs, hidden)

		_, _, last_value = apply(carry.obs, carry.hidden)
		_, _, final_value = apply(merge_leading(traj.final_obs, 2), merge_leading(traj.after, 2))
		final_value = final_value.reshape(length, batch)
		advantage, returns = gae(
			traj.reward,
			traj.value,
			traj.terminated,
			traj.truncated,
			final_value,
			last_value,
			p.gamma,
			p.gae_lambda,
		)

		weight = (~traj.info.rejected).astype(jnp.float32)

		def chunked(x):
			return chunk_time(x, chunks_per_row, sequence, batch)

		steps = Steps(
			obs=jax.tree.map(chunked, traj.obs),
			action=chunked(traj.action),
			log_prob=chunked(traj.log_prob),
			value=chunked(traj.value),
			advantage=chunked(advantage),
			returns=chunked(returns),
			weight=chunked(weight),
			ended=chunked(traj.terminated | traj.truncated),
		)
		start = jax.tree.map(
			lambda x: x[::sequence].reshape((chunks_per_row * batch,) + x.shape[2:]), traj.hidden
		)
		key, shuffle_key = jax.random.split(carry.key)
		params, opt_state, metrics = self._learn(
			learner.params, learner.opt_state, Chunks(steps, start), shuffle_key
		)

		info = traj.info
		ended = info.ended
		completed = jnp.maximum(ended.sum().astype(jnp.float32), 1.0)
		wins = (ended & (info.winner == info.learner)).sum()
		metrics.update(
			explained_variance=explained_variance(
				returns.reshape(-1), traj.value.reshape(-1), weight.reshape(-1)
			),
			completed_battles=ended.sum().astype(jnp.float32),
			truncated_battles=traj.truncated.sum().astype(jnp.float32),
			win_rate=wins / completed,
			mean_battle_length=jnp.where(ended, info.steps, 0).sum() / completed,
			mean_battle_turns=jnp.where(ended, info.turns, 0).sum() / completed,
			mean_terminal_reward=(traj.reward * ended).sum() / completed,
			illegal_actions=(info.illegal | info.opponent_illegal).sum().astype(jnp.float32),
			refused_switches=info.rejected.sum().astype(jnp.float32),
			learning_rate=self.schedule(learner.update * p.epochs * p.minibatches),
		)
		learner = LearnerState(
			params, opt_state, carry.env, carry.obs, carry.hidden, key, learner.update + 1
		)
		return learner, metrics

	def update(self, learner: LearnerState):
		"""One update (rollout, GAE and the epochs), jitted; returns the new state and metrics."""
		return self._update(learner)

	# Evaluation.

	def _evaluate_batch(self, wrapper: OpponentEnv, params, key):
		batch = wrapper.config.batch_size
		state, timestep = wrapper.reset(key)
		zeros = jnp.zeros(batch, jnp.bool_)
		start = EvaluationCarry(
			state=state,
			obs=timestep.obs,
			hidden=initial_carry(self.config.network, (batch,)),
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
			action_key = jax.random.fold_in(key, c.count)
			after, logits, _ = self.model.apply({"params": params}, c.obs, c.hidden)
			action = jax.random.categorical(action_key, logits)
			state, timestep = wrapper.step(c.state, action)
			info = timestep.info
			first = info.ended & ~c.finished
			won = timestep.terminated & (info.winner == info.learner)
			return EvaluationCarry(
				state=state,
				obs=timestep.obs,
				hidden=reset_carry(after, info.ended),
				finished=c.finished | info.ended,
				wins=jnp.where(first, won, c.wins),
				truncated=jnp.where(first, timestep.truncated, c.truncated),
				length=jnp.where(first, info.steps, c.length),
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
			eval_env = self.base.replace(
				batch_size=self.config.evaluation.batch, observe="both", keep_final=False
			)
			wrapper = OpponentEnv(eval_env, make_policy(name, self.base.tables, self.tables))
			self._evaluators[name] = jax.jit(
				lambda params, key: self._evaluate_batch(wrapper, params, key)
			)
		return self._evaluators[name]

	def evaluate(self, params, opponent: str, battles: int) -> dict:
		"""The learner (sampling its policy) against one baseline, over fixed battles.

		The battles and their sides come from the same keys every call, so evaluations at
		different updates play the same battles.
		"""
		batch = self.config.evaluation.batch
		run = self._evaluator(opponent)
		key = jax.random.key(self.config.run.seed + EVALUATION_SEED)
		wins = truncated = length = unfinished = 0
		for index in range(battles // batch):
			out = jax.device_get(run(params, jax.random.fold_in(key, index)))
			wins += int(out[0])
			truncated += int(out[1])
			length += int(out[2])
			unfinished += int(out[3])
		result = summarize_battles(opponent, battles, wins, truncated, length / battles)
		result["unfinished"] = unfinished
		return result

	# Throughput.

	def _environment_steps(self, key):
		state, timestep = self.train_env.reset(key)

		def step(carry, _):
			state, timestep, key, checksum = carry
			key, action_key = jax.random.split(key)
			logits = jnp.where(timestep.legal, 0.0, -jnp.inf)
			action = jax.random.categorical(action_key, logits)
			state, timestep = self.train_env.step(state, action)
			leaves = jax.tree.leaves((timestep.obs, timestep.final_obs))
			checksum = checksum + sum(jnp.sum(x.astype(jnp.float32)) for x in leaves)
			return (state, timestep, key, checksum), None

		init = (state, timestep, key, jnp.float32(0.0))
		(_, _, _, checksum), _ = jax.lax.scan(
			step, init, None, length=self.config.rollout.rollout_length
		)
		return checksum

	def environment_steps_per_second(self, repetitions: int = 3) -> float:
		"""Environment steps per second of one rollout's worth of steps, without the network.

		The environment alone (sampler draws, auto-reset, the baseline on the other side, the
		observations) with random legal actions; the median of `repetitions` timed calls after
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
		"""FLOPs of one forward pass of one battle's step (XLA cost analysis) and its time."""
		obs = jax.tree.map(lambda x: x[:1], learner.obs)
		hidden = jax.tree.map(lambda x: x[:1], learner.hidden)

		def forward(params, obs, hidden):
			return self.model.apply({"params": params}, obs, hidden)

		compiled = jax.jit(forward).lower(learner.params, obs, hidden).compile()
		analysis = compiled.cost_analysis()
		if isinstance(analysis, (list, tuple)):
			analysis = analysis[0] if analysis else {}
		return {"flops_per_battle_step": float((analysis or {}).get("flops", float("nan")))}
