"""Small Flax Linen recurrent PPO; NONE requests are forced action-0 env steps."""

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

from pokejax.bench import DEFAULT_POOL, _new_states
from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.engine.randomness import UNFORCED
from pokejax.engine.reset import Teams, check_teams
from pokejax.engine.state import ONGOING
from pokejax.engine.step import step
from pokejax.sampler import supported_team_indices
from pokejax.tables import load_tables
from pokejax.teams import load_npz


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
	updates: int = 600
	eval_battles: int = 4096
	eval_max_steps: int = 4096
	log_every: int = 10


class Policy(nn.Module):
	config: Config
	species_count: int
	move_count: int
	type_count: int

	@nn.compact
	def __call__(self, obs, hidden):
		c = self.config
		features = []
		for name, count in (
			("species", self.species_count),
			("moves", self.move_count),
			("types", self.type_count),
		):
			embed = nn.Embed(count, c.embedding_size, name=name)
			for side in (obs.own, obs.opponent):
				features.append(embed(getattr(side, name)).reshape(hidden.shape[0], -1))
		for side in (obs.own, obs.opponent):
			features.extend(x.reshape(hidden.shape[0], -1) for x in side[3:])
		features.extend((obs.turn[:, None], obs.request, obs.legal_actions.astype(jnp.float32)))
		x = nn.tanh(nn.Dense(c.hidden_size)(jnp.concatenate(features, axis=-1)))
		x = nn.tanh(nn.Dense(c.hidden_size)(x))
		hidden, x = nn.GRUCell(c.hidden_size)(hidden, x)
		logits = nn.Dense(9, kernel_init=nn.initializers.orthogonal(0.01))(x)
		value = nn.Dense(1)(x)[..., 0]
		return hidden, jnp.where(obs.legal_actions, logits, -1e9), value


class Env(NamedTuple):
	state: object
	side: jax.Array
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


def select(mask, new, old):
	return jax.tree.map(
		lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (o.ndim - mask.ndim)), n, o),
		new,
		old,
	)


class Trainer:
	"""Host setup only; update includes device rollout, GAE, and sequence SGD."""

	def __init__(self, config=Config(), pool_path=DEFAULT_POOL):
		c = self.config = config
		if min(c.batch_size, c.rollout_length, c.epochs, c.minibatches) < 1:
			raise ValueError("Batch, rollout length, epochs and minibatches must be positive")
		if c.batch_size % c.minibatches:
			raise ValueError("Batch size must be divisible by minibatches")
		tables = load_tables()
		host_pool = load_npz(pool_path)
		if not len(host_pool.team_size) or np.any(host_pool.team_size != 6):
			raise ValueError("Expected complete six-member teams")
		check_teams(
			Teams(*(getattr(host_pool, n)[:, None].repeat(2, axis=1) for n in Teams._fields)),
			tables,
		)
		self.pool = Teams(*(jnp.asarray(getattr(host_pool, n)) for n in Teams._fields))
		# Only fully supported teams are drawn; the index array is computed once here.
		self.team_indices = jnp.asarray(supported_team_indices(host_pool, tables))
		self.tables = build_engine_tables(tables)
		self.model = Policy(
			c, len(tables.species.ids), len(tables.moves.ids), len(tables.types.ids) + 1
		)
		key, init_key = jax.random.split(jax.random.key(c.seed))
		self.env = self.new_env(key, c.batch_size)
		params = self.model.init(init_key, self.observation(self.env), self.env.hidden)["params"]
		self.train = TrainState.create(
			apply_fn=self.model.apply,
			params=params,
			tx=optax.chain(optax.clip_by_global_norm(c.max_grad_norm), optax.adam(c.learning_rate)),
		)
		self.update = jax.jit(self._update)
		self.evaluate_batch = jax.jit(
			self._evaluate_batch, static_argnames=("random_policy", "batch")
		)

	def new_env(self, key, batch):
		key, reset_key, side_key = jax.random.split(key, 3)
		return Env(
			_new_states(reset_key, self.pool, self.tables, batch, self.team_indices),
			jax.random.bernoulli(side_key, shape=(batch,)).astype(jnp.int32),
			jnp.zeros((batch, self.config.hidden_size)),
			key,
		)

	def observation(self, env):
		return jax.vmap(observe, in_axes=(0, 0, None))(env.state, env.side, self.tables)

	def advance(self, env, action):
		key, opponent_key, step_key = jax.random.split(env.key, 3)
		mask = jax.vmap(legal_actions, in_axes=(0, None))(env.state, self.tables)
		actions = jax.random.categorical(opponent_key, jnp.where(mask, 0.0, -1e9))
		actions = actions.at[jnp.arange(action.shape[0]), env.side].set(action)
		illegal = ~jnp.take_along_axis(mask, actions[..., None], axis=-1)[..., 0]
		state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
			env.state,
			actions,
			jax.random.split(step_key, action.shape[0]),
			UNFORCED,
			self.tables,
		)
		done = state.winner != ONGOING
		reward = jnp.where(done, jnp.where(state.winner == env.side, 1.0, -1.0), 0.0)
		# Only winner IDs 0/1 denote a win; any terminal tie receives zero.
		reward = jnp.where(done & (state.winner != 0) & (state.winner != 1), 0.0, reward)
		return env._replace(state=state, key=key), reward, done, jnp.sum(illegal)

	def _update(self, train, env):
		c = self.config
		start_hidden = env.hidden

		def collect(env, _):
			key, action_key, reset_key = jax.random.split(env.key, 3)
			obs = self.observation(env)
			hidden, logits, value = self.model.apply({"params": train.params}, obs, env.hidden)
			action = jax.random.categorical(action_key, logits)
			logp, _ = distribution(logits)
			env, reward, done, illegal = self.advance(env._replace(key=key, hidden=hidden), action)
			transition = Transition(
				obs,
				action,
				jnp.take_along_axis(logp, action[:, None], -1)[:, 0],
				value,
				reward,
				done,
			)
			fresh = self.new_env(reset_key, c.batch_size)
			env = Env(
				select(done, fresh.state, env.state),
				jnp.where(done, fresh.side, env.side),
				jnp.where(done[:, None], 0, hidden),
				env.key,
			)
			return env, (transition, illegal)

		env, (trajectory, illegal) = jax.lax.scan(collect, env, None, length=c.rollout_length)
		_, _, bootstrap = self.model.apply(
			{"params": train.params}, self.observation(env), env.hidden
		)

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

		(train, key), metrics = jax.lax.scan(epoch, (train, env.key), None, length=c.epochs)
		completed = trajectory.done.sum()
		metrics = dict(
			zip(("policy_loss", "value_loss", "entropy", "approx_kl"), metrics.mean((0, 1)))
		)
		metrics.update(
			completed=completed,
			mean_return=trajectory.reward.sum() / jnp.maximum(completed, 1),
			win_rate=(trajectory.reward > 0).sum() / jnp.maximum(completed, 1),
			illegal_actions=illegal.sum(),
		)
		return train, env._replace(key=key), metrics

	def _evaluate_batch(self, params, key, random_policy=False, batch=256):
		env = self.new_env(key, batch)

		# Each slot plays exactly one fresh battle; finished slots cannot count twice.
		def cond(carry):
			_, finished, _, _, count, _ = carry
			return ~finished.all() & (count < self.config.eval_max_steps)

		def body(carry):
			env, finished, wins, ties, count, illegal_total = carry
			key, action_key = jax.random.split(env.key)
			obs = self.observation(env)
			hidden, logits, _ = self.model.apply({"params": params}, obs, env.hidden)
			if random_policy:
				logits = jnp.where(obs.legal_actions, 0.0, -1e9)
			action = jax.random.categorical(action_key, logits)
			env, reward, done, illegal = self.advance(env._replace(key=key, hidden=hidden), action)
			new = done & ~finished
			return (
				env,
				finished | done,
				wins + (new & (reward > 0)),
				ties + (new & (reward == 0)),
				count + 1,
				illegal_total + illegal,
			)

		_, done, wins, ties, steps, illegal = jax.lax.while_loop(
			cond,
			body,
			(
				env,
				jnp.zeros(batch, bool),
				jnp.zeros(batch, jnp.int32),
				jnp.zeros(batch, jnp.int32),
				jnp.int32(0),
				jnp.int32(0),
			),
		)
		return wins.sum(), ties.sum(), (~done).sum(), steps, illegal

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
	args.output_dir.mkdir(parents=True, exist_ok=True)
	(args.output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))
	trainer = Trainer(config, args.team_pool)
	train, env = trainer.train, trainer.env
	start = time.perf_counter()
	with (args.output_dir / "metrics.jsonl").open("w") as log:
		for update in range(1, config.updates + 1):
			tick = time.perf_counter()
			train, env, metrics = jax.block_until_ready(trainer.update(train, env))
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
