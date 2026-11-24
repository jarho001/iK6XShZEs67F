"""Engine matchups for the head-to-head and panel evaluations of a trained network.

A player (a "seat") is a network, with its model and the carry it keeps per battle, or a baseline
policy of pokejax.baselines (as league.py plays them). MatchRunner plays two seats on one batch of
battles: the first on side seats_first[row], the second on the other side, each choosing its action
from its own observation, until every battle has ended or reached the step limit. For each battle it
reports the first battle that row ended: whether the first seat won, whether that battle was
truncated, and its length. A network seat samples its policy, or takes the argmax when greedy.

The battles of a batch are fixed by their inputs: the battle states (draw_battles), the seats and the
engine key. Two matchups given the same inputs play the same battles, so their outcomes differ only
through the players, which is what the paired comparisons of pokejax.train.headtohead and
pokejax.train.panel rely on. A network seat reads the same observation that league.py gives its
lanes, and a held action (a refused switch that the environment replays) keeps the carry, as it does
in training.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import make_policy
from pokejax.engine.observation import observe
from pokejax.engine.state import NUM_SIDES
from pokejax.env import Env, EnvConfig
from pokejax.train import probe
from pokejax.train.config import NetworkSection, TrainConfig
from pokejax.train.evaluation import wilson_interval
from pokejax.train.league import select_lanes
from pokejax.train.network import PokePolicy, initial_carry, reset_carry, vocabulary


@dataclass(frozen=True)
class Seat:
	"""One player of a matchup: a network (model, network section) or a baseline (policy)."""

	name: str
	cache_key: tuple
	model: PokePolicy | None = None
	network: NetworkSection | None = None
	policy: Callable | None = None

	@property
	def is_network(self) -> bool:
		return self.model is not None


@dataclass(frozen=True)
class Contestant:
	"""A labelled player: its seat and, for a network, its parameters."""

	label: str
	seat: Seat
	params: Any = None


class MatchState(NamedTuple):
	env: Any  # EnvState of the batch
	carries: tuple  # each network seat's carry [B, W]; None for a baseline seat
	finished: jax.Array  # bool [B], the row's first battle has ended
	won: jax.Array  # bool [B], that battle was won by the first seat
	truncated: jax.Array  # bool [B]
	length: jax.Array  # int32 [B], its environment steps
	count: jax.Array  # int32, environment steps taken


def baseline_contestant(name: str, env: Env, tables) -> Contestant:
	policy = make_policy(name, env.tables, tables)
	return Contestant(name, Seat(name, ("baseline", name), policy=policy))


def network_contestant(
	label: str, path: Path, pool: Path, tables
) -> tuple[Contestant, TrainConfig]:
	"""A checkpoint's parameters and configuration (run directory or checkpoint.npz)."""
	config = probe.read_config(path)
	model = PokePolicy(vocabulary(tables), config.network, config.auxiliary)
	one = Env(EnvConfig(batch_size=1, observe="both"), pool, tables=tables)
	example = jax.tree.map(lambda x: x[:, 0], one.observe_both(one.new_battles(jax.random.key(0))))
	carry = initial_carry(config.network, (1,))
	params = probe.restore(path, model, example, carry)
	seat = Seat(
		label,
		("network", config.network, config.auxiliary),
		model=model,
		network=config.network,
	)
	return Contestant(label, seat, params), config


def draw_battles(env: Env, key):
	"""The battle states of one batch (env.config.batch_size battles) drawn from key."""
	return env.new_battles(key)


class MatchRunner:
	"""Two seats on one batch of battles of env (its batch size is the number of rows), jitted."""

	def __init__(self, env: Env, first: Seat, second: Seat):
		self.env = env
		self.seats = (first, second)
		self._run = jax.jit(self._play)

	def __call__(self, params: tuple, state, seats_first, key, greedy: bool):
		"""The first seat's per-row outcome: won, truncated, length and finished (each [B])."""
		return self._run(params, state, jnp.asarray(seats_first, jnp.int32), key, jnp.bool_(greedy))

	def _play(self, params, state, seats_first, key, greedy):
		env, tables = self.env, self.env.tables
		batch = env.config.batch_size
		limit = env.config.max_steps
		rows = jnp.arange(batch, dtype=jnp.int32)
		seats = (seats_first, 1 - seats_first)
		step_key, sample_key = jax.random.split(key)
		zeros = jnp.zeros(batch, jnp.bool_)
		start = MatchState(
			env=env.init(state, step_key),
			carries=tuple(
				initial_carry(seat.network, (batch,)) if seat.is_network else None
				for seat in self.seats
			),
			finished=zeros,
			won=zeros,
			truncated=zeros,
			length=jnp.zeros(batch, jnp.int32),
			count=jnp.int32(0),
		)

		def cond(c):
			return (~c.finished.all()) & (c.count < limit)

		def body(c):
			keys = jax.random.split(jax.random.fold_in(sample_key, c.count), NUM_SIDES)
			actions, carries = [], []
			for index, seat in enumerate(self.seats):
				side = seats[index]
				if seat.is_network:
					obs = jax.vmap(observe, in_axes=(0, 0, None))(c.env.state, side, tables)
					after, logits, _ = seat.model.apply(
						{"params": params[index]}, obs, c.carries[index]
					)
					sampled = jax.random.categorical(keys[index], logits)
					action = jnp.where(greedy, jnp.argmax(logits, -1), sampled)
					# A held action (refused switch) is replayed by the environment: no advance.
					held = c.env.has_held[rows, side]
					carries.append(select_lanes(held, c.carries[index], after))
				else:
					keys_index = jax.random.split(keys[index], batch)
					action = jax.vmap(seat.policy)(c.env.state, side, keys_index)
					carries.append(None)
				actions.append(action.astype(jnp.int32))
			both = jnp.zeros((batch, NUM_SIDES), jnp.int32)
			both = both.at[rows, seats[0]].set(actions[0]).at[rows, seats[1]].set(actions[1])
			env_next, advance = env.advance(c.env, both)
			info = advance.info
			first_ended = info.ended & ~c.finished
			carries = tuple(
				None if carry is None else reset_carry(carry, info.ended) for carry in carries
			)
			return MatchState(
				env=env_next,
				carries=carries,
				finished=c.finished | info.ended,
				won=jnp.where(first_ended, advance.terminated & (info.winner == seats[0]), c.won),
				truncated=jnp.where(first_ended, advance.truncated, c.truncated),
				length=jnp.where(first_ended, info.steps, c.length),
				count=c.count + 1,
			)

		final = jax.lax.while_loop(cond, body, start)
		return final.won, final.truncated, final.length, final.finished


def summarize(
	won: np.ndarray, truncated: np.ndarray, length: np.ndarray, finished: np.ndarray
) -> dict:
	"""Wins, the win rate with its Wilson 95% interval, truncations and mean length of battles."""
	battles = int(won.size)
	wins = int(won.sum())
	low, high = wilson_interval(wins, battles)
	return {
		"battles": battles,
		"wins": wins,
		"win_rate": wins / battles,
		"ci95": [low, high],
		"truncated": int(truncated.sum()),
		"unfinished": int((~finished).sum()),
		"mean_battle_length": float(length.mean()),
	}


def paired_mean_interval(values: np.ndarray, resamples: int, seed: int) -> dict:
	"""The mean of per-battle values with a paired bootstrap 95% interval over the battles."""
	values = np.asarray(values, dtype=np.float64)
	rng = np.random.default_rng(seed)
	means = np.empty(resamples)
	for start in range(0, resamples, 500):
		stop = min(start + 500, resamples)
		index = rng.integers(0, values.size, size=(stop - start, values.size))
		means[start:stop] = values[index].mean(axis=1)

	low, high = np.percentile(means, [2.5, 97.5])
	return {"estimate": float(values.mean()), "ci95": [float(low), float(high)]}
