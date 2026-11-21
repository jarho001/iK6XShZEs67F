"""Small Flax Linen recurrent PPO against the random policy, on the environment of pokejax.env.

The learner plays one random side per battle (OpponentEnv); NONE requests are forced action-0
environment steps. A battle that the step cap truncates counts as finished, without a bootstrap.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from pokejax.env import DEFAULT_POOL, Env, EnvConfig, OpponentEnv
from pokejax.tables import load_tables


@dataclass(frozen=True)
class Config:
	batch_size: int = 256
	rollout_length: int = 64
	hidden_size: int = 128
	embedding_size: int = 8
	epochs: int = 4
	minibatches: int = 4
	learning_rate: float = 0.0003
	gamma: float = 0.99
	gae_lambda: float = 0.95
	clip: float = 0.2
	value_coef: float = 0.5
	entropy_coef: float = 0.01
	max_grad_norm: float = 0.5
	seed: int = 0
	max_steps: int = 1000  # environment steps before a battle is truncated
	updates: int = 600
	eval_battles: int = 4096
	eval_max_steps: int = 4096
	log_every: int = 10


# Observation fields that the policy embeds; every other field is read as a float by name.
EMBEDDED_FIELDS = ("species", "moves", "types")
# Top-level fields that are per-side or scalar and read as floats (events.py perspective order).
TOP_FIELDS = (
	"weather",
	"weather_age",
	"hazards",
	"trick_room",
	"trick_room_age",
	"wish",
	"wish_age",
	"healing_wish",
)


def flat_features(x, batch):
	return x.astype(jnp.float32).reshape(batch, -1)


class Policy(nn.Module):
	config: Config
	species_count: int
	move_count: int
	type_count: int

	@nn.compact
	def __call__(self, obs, hidden):
		c = self.config
		batch = hidden.shape[0]
		features = []
		for name, count in (
			("species", self.species_count),
			("moves", self.move_count),
			("types", self.type_count),
		):
			embed = nn.Embed(count, c.embedding_size, name=name)
			for side in (obs.own, obs.opponent):
				features.append(embed(getattr(side, name)).reshape(batch, -1))
		for side in (obs.own, obs.opponent):
			for name, x in zip(side._fields, side):
				if name not in EMBEDDED_FIELDS:
					features.append(flat_features(x, batch))
		for name, x in zip(obs.events._fields, obs.events):
			features.append(flat_features(x, batch))
		for name in TOP_FIELDS:
			features.append(flat_features(getattr(obs, name), batch))
		features.extend(
			(
				obs.turn[:, None],
				obs.mid_turn[:, None],
				obs.request,
				obs.legal_actions.astype(jnp.float32),
			)
		)
		x = nn.tanh(nn.Dense(c.hidden_size)(jnp.concatenate(features, axis=-1)))
		x = nn.tanh(nn.Dense(c.hidden_size)(x))
		hidden, x = nn.GRUCell(c.hidden_size)(hidden, x)
		logits = nn.Dense(9, kernel_init=nn.initializers.orthogonal(0.01))(x)
		value = nn.Dense(1)(x)[..., 0]
		return hidden, jnp.where(obs.legal_actions, logits, -1e9), value


class Carry(NamedTuple):
	env: object  # OpponentState
	obs: object  # the learner's observation of the current battles
	hidden: jax.Array
	key: jax.Array


class Transition(NamedTuple):
	obs: object
	action: jax.Array
	log_prob: jax.Array
	value: jax.Array
	reward: jax.Array
	done: jax.Array


def distribution(logits):
	logp = jax.nn.log_softmax(logits)
	return logp, -jnp.sum(jnp.exp(logp) * logp, axis=-1)


class Trainer:
	"""Host setup only; update includes device rollout, GAE, and sequence SGD."""

	def __init__(self, config=Config(), pool_path=DEFAULT_POOL):
		c = self.config = config
		if min(c.batch_size, c.rollout_length, c.epochs, c.minibatches, c.max_steps) < 1:
			raise ValueError(
				"Batch, rollout length, epochs, minibatches and steps must be positive"
			)
		if c.batch_size % c.minibatches:
			raise ValueError("Batch size must be divisible by minibatches")
		tables = load_tables()
		self.env = OpponentEnv(Env(EnvConfig(c.batch_size, c.max_steps), pool_path, tables=tables))
		self.model = Policy(
			c, len(tables.species.ids), len(tables.moves.ids), len(tables.types.ids) + 1
		)
		key, init_key = jax.random.split(jax.random.key(c.seed))
		self.carry = self.new_carry(key, self.env)
		params = self.model.init(init_key, self.carry.obs, self.carry.hidden)["params"]
		self.train = TrainState.create(
			apply_fn=self.model.apply,
			params=params,
			tx=optax.chain(optax.clip_by_global_norm(c.max_grad_norm), optax.adam(c.learning_rate)),
		)
		self.update = jax.jit(self._update)
		self.evaluate_batch = jax.jit(
			self._evaluate_batch, static_argnames=("random_policy", "batch")
		)

	def new_carry(self, key, env):
		key, reset_key = jax.random.split(key)
		state, timestep = env.reset(reset_key)
		batch = env.config.batch_size
		return Carry(state, timestep.obs, jnp.zeros((batch, self.config.hidden_size)), key)

	def _update(self, train, carry):
		c = self.config
		start_hidden = carry.hidden

		def collect(carry, _):
			key, action_key = jax.random.split(carry.key)
			hidden, logits, value = self.model.apply(
				{"params": train.params}, carry.obs, carry.hidden
			)
			action = jax.random.categorical(action_key, logits)
			logp, _ = distribution(logits)
			state, timestep = self.env.step(carry.env, action)
			done = timestep.terminated | timestep.truncated
			transition = Transition(
				carry.obs,
				action,
				jnp.take_along_axis(logp, action[:, None], -1)[:, 0],
				value,
				timestep.reward,
				done,
			)
			# The step already reset a finished battle: its observation is the new battle's.
			hidden = jnp.where(done[:, None], 0, hidden)
			info = timestep.info
			counts = (
				(info.illegal | info.opponent_illegal).sum(),
				info.rejected.sum(),
			)
			return Carry(state, timestep.obs, hidden, key), (transition, counts)

		carry, (trajectory, (illegal, refused)) = jax.lax.scan(
			collect, carry, None, length=c.rollout_length
		)
		_, _, bootstrap = self.model.apply({"params": train.params}, carry.obs, carry.hidden)

		def gae(carry, row):
			advantage, next_value = carry
			delta = row.reward + c.gamma * (1 - row.done) * next_value - row.value
			advantage = delta + c.gamma * c.gae_lambda * (1 - row.done) * advantage
			return (advantage, row.value), advantage

		_, advantages = jax.lax.scan(
			gae, (jnp.zeros_like(bootstrap), bootstrap), trajectory, reverse=True
		)
		targets = advantages + trajectory.value
		advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

		def loss(params, rows, hidden, adv, target):
			def replay(hidden, row):
				hidden, logits, value = self.model.apply({"params": params}, row.obs, hidden)
				return jnp.where(row.done[:, None], 0, hidden), (logits, value)

			_, (logits, values) = jax.lax.scan(replay, hidden, rows)
			logp, entropy = distribution(logits)
			chosen = jnp.take_along_axis(logp, rows.action[..., None], -1)[..., 0]
			log_ratio = chosen - rows.log_prob
			ratio = jnp.exp(log_ratio)
			policy_loss = -jnp.minimum(
				ratio * adv, jnp.clip(ratio, 1 - c.clip, 1 + c.clip) * adv
			).mean()
			value_loss = jnp.square(values - target).mean()
			entropy = entropy.mean()
			kl = (ratio - 1 - log_ratio).mean()
			return policy_loss + c.value_coef * value_loss - c.entropy_coef * entropy, jnp.stack(
				(policy_loss, value_loss, entropy, kl)
			)

		def epoch(carry, _):
			train, key = carry
			key, shuffle_key = jax.random.split(key)
			indices = jax.random.permutation(shuffle_key, c.batch_size).reshape(c.minibatches, -1)

			def minibatch(train, ids):
				rows = jax.tree.map(lambda x: x[:, ids], trajectory)
				(_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(
					train.params,
					rows,
					start_hidden[ids],
					advantages[:, ids],
					targets[:, ids],
				)
				return train.apply_gradients(grads=grads), metrics

			train, metrics = jax.lax.scan(minibatch, train, indices)
			return (train, key), metrics

		(train, key), metrics = jax.lax.scan(epoch, (train, carry.key), None, length=c.epochs)
		completed = trajectory.done.sum()
		metrics = dict(
			zip(("policy_loss", "value_loss", "entropy", "approx_kl"), metrics.mean((0, 1)))
		)
		metrics.update(
			completed=completed,
			mean_return=trajectory.reward.sum() / jnp.maximum(completed, 1),
			win_rate=(trajectory.reward > 0).sum() / jnp.maximum(completed, 1),
			illegal_actions=illegal.sum(),
			refused_switches=refused.sum(),
		)
		return train, carry._replace(key=key), metrics

	def _evaluate_batch(self, params, key, random_policy=False, batch=256):
		env = self.env.replace(batch_size=batch)
		reset_key, policy_key = jax.random.split(key)
		state, timestep = env.reset(reset_key)

		# Each slot plays exactly one fresh battle; finished slots cannot count twice.
		def cond(carry):
			_, _, _, finished, _, _, _, count, _ = carry
			return ~finished.all() & (count < self.config.eval_max_steps)

		def body(carry):
			state, obs, hidden, finished, wins, ties, truncated, count, illegal_total = carry
			action_key = jax.random.fold_in(policy_key, count)
			hidden, logits, _ = self.model.apply({"params": params}, obs, hidden)
			if random_policy:
				logits = jnp.where(obs.legal_actions, 0.0, -1e9)
			action = jax.random.categorical(action_key, logits)
			state, timestep = env.step(state, action)
			info = timestep.info
			new = info.ended & ~finished
			illegal = (info.illegal | info.opponent_illegal).sum()
			return (
				state,
				timestep.obs,
				jnp.where(info.ended[:, None], 0, hidden),
				finished | info.ended,
				wins + (new & (timestep.reward > 0)),
				ties + (new & timestep.terminated & (timestep.reward == 0)),
				truncated + (new & timestep.truncated),
				count + 1,
				illegal_total + illegal,
			)

		zeros = jnp.zeros(batch, jnp.int32)
		_, _, _, done, wins, ties, truncated, steps, illegal = jax.lax.while_loop(
			cond,
			body,
			(
				state,
				timestep.obs,
				jnp.zeros((batch, self.config.hidden_size)),
				jnp.zeros(batch, bool),
				zeros,
				zeros,
				zeros,
				jnp.int32(0),
				jnp.int32(0),
			),
		)
		return wins.sum(), ties.sum(), (~done).sum() + truncated.sum(), steps, illegal

	def evaluate(self, params, seed, battles, random_policy=False):
		wins = ties = timeouts = illegal = 0
		for start in range(0, battles, 256):
			result = jax.device_get(
				self.evaluate_batch(
					params,
					jax.random.fold_in(jax.random.key(seed), start),
					random_policy,
					min(256, battles - start),
				)
			)
			w, t, unfinished, _, invalid = map(int, result)
			wins += w
			ties += t
			timeouts += unfinished
			illegal += invalid
		# Wilson interval; unfinished battles remain in denominator (conservative).
		p, z = wins / battles, 1.96
		center = (p + z * z / (2 * battles)) / (1 + z * z / battles)
		half = (
			z
			* np.sqrt(p * (1 - p) / battles + z * z / (4 * battles * battles))
			/ (1 + z * z / battles)
		)
		return {
			"battles": battles,
			"wins": wins,
			"win_rate": p,
			"ci95": [center - half, center + half],
			"ties": ties,
			"timeouts": timeouts,
			"illegal_actions": illegal,
			"policy": "random" if random_policy else "sampled",
		}


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	for name, default in asdict(Config()).items():
		parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--output-dir", type=Path, default=Path("work/runs/ppo_smoke"))
	args = parser.parse_args()
	config = Config(**{name: getattr(args, name) for name in asdict(Config())})
	if min(config.updates, config.eval_battles, config.eval_max_steps, config.log_every) < 1:
		parser.error("Updates, evaluation sizes and log interval must be positive")
	if config.max_steps < 1:
		parser.error("The step cap must be positive")
	args.output_dir.mkdir(parents=True, exist_ok=True)
	(args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
	trainer = Trainer(config, args.team_pool)
	train, carry = trainer.train, trainer.carry
	start = time.perf_counter()
	with (args.output_dir / "metrics.jsonl").open("w") as log:
		for update in range(1, config.updates + 1):
			tick = time.perf_counter()
			train, carry, metrics = jax.block_until_ready(trainer.update(train, carry))
			metrics = {k: float(v) for k, v in metrics.items()}
			metrics.update(
				update=update,
				steps=update * config.batch_size * config.rollout_length,
				steps_per_second=config.batch_size
				* config.rollout_length
				/ (time.perf_counter() - tick),
			)
			log.write(json.dumps(metrics) + "\n")
			log.flush()
			if update == 1 or update % config.log_every == 0:
				print(
					f"{update}: win={metrics['win_rate']:.3f} return={metrics['mean_return']:.3f} "
					f"value={metrics['value_loss']:.3f} entropy={metrics['entropy']:.3f} "
					f"KL={metrics['approx_kl']:.4f} steps/s={metrics['steps_per_second']:.0f}",
					flush=True,
				)
	seconds = time.perf_counter() - start
	from flax.serialization import to_bytes

	(args.output_dir / "params.msgpack").write_bytes(to_bytes(train.params))
	result = {
		"device": str(jax.devices()[0]),
		"training_seconds": seconds,
		"environment_steps": config.updates * config.batch_size * config.rollout_length,
		"hyperparameters": asdict(config),
		"evaluation": trainer.evaluate(train.params, config.seed + 100000, config.eval_battles),
		"random_baseline": trainer.evaluate(
			train.params, config.seed + 200000, config.eval_battles, True
		),
	}
	(args.output_dir / "result.json").write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
	main()
