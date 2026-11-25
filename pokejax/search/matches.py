"""Paired engine matches for the diagnostic oracle and raw checkpoint."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.state import NUM_SIDES, ONGOING, TIE
from pokejax.search.oracle import DiagnosticOracle, eligible, forward
from pokejax.search.transition import advance_batch, init
from pokejax.train.matchup import paired_mean_interval
from pokejax.train.network import initial_carry


class MatchState(NamedTuple):
	position: object
	game_key: jax.Array
	length: jax.Array
	searched: jax.Array
	overrides: jax.Array
	best_advantage: jax.Array
	chosen_advantage: jax.Array


class MatchResult(NamedTuple):
	won: np.ndarray
	tied: np.ndarray
	truncated: np.ndarray
	length: np.ndarray
	searched: np.ndarray
	overrides: np.ndarray
	best_advantage: np.ndarray
	chosen_advantage: np.ndarray


@jax.jit
def start(state, tables, key):
	batch = state.winner.size
	count = jnp.zeros(batch, jnp.int32)
	score = jnp.zeros(batch, jnp.float32)
	return MatchState(
		jax.vmap(init, in_axes=(0, None))(state, tables), key, count, count, count, score, score
	)


@jax.jit
def choices(match, sides, logits, sample_key, ply, baseline, greedy):
	# MatchRunner samples by contestant, not physical seat, using the same setup key layout.
	our_key, foe_key = jax.random.split(jax.random.fold_in(sample_key, ply), NUM_SIDES)
	rows = jnp.arange(sides.size)
	ours = logits[rows, sides]
	noise = jax.random.gumbel(our_key, ours.shape, dtype=ours.dtype)
	own = jnp.argmax(ours + jnp.where(greedy, 0.0, noise), axis=-1)
	if baseline is None:
		foe = jax.random.categorical(foe_key, logits[rows, 1 - sides])
	else:
		foe = jax.vmap(baseline)(
			match.position.state, 1 - sides, jax.random.split(foe_key, sides.size)
		)
	return own.astype(jnp.int32), foe.astype(jnp.int32), noise


@jax.jit
def play(match, sides, own, foe, decision, tables):
	key, step_key, _ = jax.random.split(match.game_key, 3)
	actions = jnp.where(jnp.arange(NUM_SIDES) == sides[:, None], own[:, None], foe[:, None])
	position, _ = advance_batch(
		match.position, actions, jax.random.split(step_key, sides.size), None, tables
	)
	match = match._replace(
		position=position,
		game_key=key,
		length=match.length + (match.position.state.winner == ONGOING).astype(jnp.int32),
	)
	if decision is not None:
		match = match._replace(
			searched=match.searched + decision.searched,
			overrides=match.overrides + (decision.action != decision.raw),
			best_advantage=match.best_advantage + decision.best_advantage,
			chosen_advantage=match.chosen_advantage + decision.chosen_advantage,
		)
	return match


class DiagnosticMatch:
	"""Run batched matches with either oracle search or sampled/greedy raw play on the first seat."""

	def __init__(self, env, contestant, oracle: DiagnosticOracle | None, baseline=None):
		self.env = env
		self.contestant = contestant
		self.oracle = oracle
		self.baseline = baseline

	def run(self, state, sides, key, *, greedy=False):
		return self._run(state, sides, key, greedy=greedy)

	def warm_search(self, state, sides, key):
		"""Compile the fixed rollout shapes on a real opening request before playing games."""
		if self.oracle is not None:
			actor = self.contestant
			match = start(state, self.env.tables, key)
			carry = initial_carry(actor.seat.network, (sides.size, NUM_SIDES))
			keys = jax.random.split(key, sides.size)
			jax.block_until_ready(
				self.oracle(actor.params, match.position, sides, carry, keys, warm=True)
			)

	def _run(self, state, sides, key, *, greedy=False, root_samples=None, sample_count=0):
		tables = self.env.tables
		model, params = self.contestant.seat.model, self.contestant.params
		game_key, sample_key = jax.random.split(key)
		search_key = jax.random.fold_in(key, 0x5EA2C)
		match = start(state, tables, game_key)
		carry = initial_carry(self.contestant.seat.network, (sides.size, NUM_SIDES))
		for ply in range(self.env.config.max_steps):
			if self.oracle is None:
				after, logits, _, head = forward(model, params, match.position, carry, tables)
			else:
				after, logits, _, head = self.oracle.root(params, match.position, carry)
			own, foe, noise = choices(
				match, sides, logits, sample_key, jnp.int32(ply), self.baseline, greedy
			)
			decision = None
			if self.oracle is not None and bool(eligible(match.position, sides).any()):
				keys = jax.random.split(jax.random.fold_in(search_key, ply), sides.size)
				decision = self.oracle.choose(
					params,
					match.position,
					sides,
					after,
					logits,
					head,
					keys,
					noise if self.oracle.config.sample_strength else None,
				)
				own = decision.action
			if root_samples is not None and ply % 2 == 0 and len(root_samples) < sample_count:
				valid = np.flatnonzero(np.asarray(eligible(match.position, sides)))
				if valid.size:
					row = int(valid[(ply // 2) % valid.size])
					root_samples.append(
						(
							*jax.tree.map(
								lambda x, row=row: x[row : row + 1], (match.position, sides, carry)
							),
							ply,
						)
					)
			match = play(match, sides, own, foe, decision, tables)
			carry = after
			if bool((match.position.state.winner != ONGOING).all()):
				break
		winner = match.position.state.winner
		return MatchResult(
			*jax.device_get(
				(
					winner == sides,
					winner == TIE,
					winner == ONGOING,
					match.length,
					match.searched,
					match.overrides,
					match.best_advantage,
					match.chosen_advantage,
				)
			)
		)

	def timing_roots(self, setup, count):
		"""Collect real pre-observation carries at lead and mid-battle choice points."""
		samples = []
		self._run(*setup, root_samples=samples, sample_count=count)
		return samples


def combine(results):
	return MatchResult(*(np.concatenate(columns) for columns in zip(*results, strict=True)))


def report(result, seed, *, half=None, raw=None):
	def cluster(values):
		if half is None:
			return values
		return values.reshape(-1, 2, half).mean(axis=1).ravel()

	wins = result.won.astype(np.float64)
	searched = int(result.searched.sum())
	data = {
		"battles": wins.size,
		"wins": int(wins.sum()),
		"win_rate": paired_mean_interval(cluster(wins), 2000, seed),
		"ties": int(result.tied.sum()),
		"truncated": int(result.truncated.sum()),
		"mean_length": float(result.length.mean()),
		"searched": searched,
		"requests": int(result.length.sum()),
		"overrides": int(result.overrides.sum()),
		"override_rate": float(result.overrides.sum() / searched) if searched else None,
		"mean_best_advantage": float(result.best_advantage.sum() / searched) if searched else None,
		"mean_chosen_advantage": float(result.chosen_advantage.sum() / searched)
		if searched
		else None,
	}
	if half is not None:
		data["seats"] = wins.reshape(-1, 2, half).mean(axis=(0, 2)).tolist()
	if raw is not None:
		data["delta"] = paired_mean_interval(cluster(wins - raw.won), 2000, seed)
	return data
