"""Measure cold compilation and random-policy battle throughput."""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.engine.randomness import UNFORCED
from pokejax.engine.reset import Teams, check_teams, reset
from pokejax.engine.state import ONGOING, TEAM_SLOTS
from pokejax.engine.step import step
from pokejax.sampler import sample_team_indices
from pokejax.tables import load_tables
from pokejax.teams import load_npz

DEFAULT_POOL = Path("work/teams/gen5-teams.npz")
DEFAULT_BATCH = 4096
DEFAULT_STEPS = 256
DEFAULT_REPETITIONS = 3


def _new_states(key, pool, tables, batch_size, team_indices=None):
	"""Reset a batch on teams drawn from the whole pool, or from team_indices (pokejax.sampler)."""
	index_key, reset_key = jax.random.split(key)
	if team_indices is None:
		indices = jax.random.randint(index_key, (batch_size, 2), 0, pool.species.shape[0])
	else:
		indices = sample_team_indices(index_key, team_indices, batch_size)
	teams = jax.tree.map(lambda column: column[indices], pool)
	keys = jax.random.split(reset_key, batch_size)
	return jax.vmap(reset, in_axes=(0, 0, None))(keys, teams, tables)


def rollout(carry, pool, tables, steps, with_observation=False):
	"""Advance a batch and return completed count and total completed lengths."""

	def advance(carry, _):
		state, key, lengths = carry
		key, action_key, step_key, reset_key = jax.random.split(key, 4)
		mask = jax.vmap(legal_actions, in_axes=(0, None))(state, tables)
		logits = jnp.where(mask, 0.0, -jnp.inf)
		actions = jax.random.categorical(action_key, logits, axis=-1)
		keys = jax.random.split(step_key, state.winner.shape[0])
		state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
			state, actions, keys, UNFORCED, tables
		)
		# Consume every observation leaf so XLA cannot discard observation work.
		checksum = jnp.float32(0)
		if with_observation:
			observations = jax.vmap(
				lambda s: jax.vmap(observe, in_axes=(None, 0, None))(s, jnp.arange(2), tables)
			)(state)
			checksum = sum(jnp.sum(x.astype(jnp.float32)) for x in jax.tree.leaves(observations))
		finished = state.winner != ONGOING
		lengths = lengths + 1
		completed_length = jnp.sum(jnp.where(finished, lengths, 0))
		fresh = _new_states(reset_key, pool, tables, state.winner.shape[0])
		state = jax.tree.map(
			lambda new, old: jnp.where(
				finished.reshape((finished.shape[0],) + (1,) * (old.ndim - 1)), new, old
			),
			fresh,
			state,
		)
		lengths = jnp.where(finished, 0, lengths)
		stats = (jnp.sum(finished), completed_length)
		return (state, key, lengths), (*stats, checksum) if with_observation else stats

	carry, stats = jax.lax.scan(advance, carry, None, length=steps)
	return carry, tuple(jnp.sum(x) for x in stats)


def _benchmark(
	pool_path=DEFAULT_POOL,
	batch_size=DEFAULT_BATCH,
	steps=DEFAULT_STEPS,
	repetitions=DEFAULT_REPETITIONS,
	seed=0,
	with_observation=False,
):
	"""Benchmark on the default device; use a fresh process for a cold compile."""
	if min(batch_size, steps, repetitions) < 1:
		raise ValueError("batch size, steps and repetitions must be positive")
	host_pool = load_npz(pool_path)
	if not len(host_pool.team_size) or np.any(host_pool.team_size != TEAM_SLOTS):
		raise ValueError("The pool must contain only complete six-member teams")
	tables = load_tables()
	check_teams(
		Teams(*(getattr(host_pool, name)[:, None].repeat(2, axis=1) for name in Teams._fields)),
		tables,
	)
	pool = Teams(*(jnp.asarray(getattr(host_pool, name)) for name in Teams._fields))
	engine_tables = build_engine_tables(tables)
	key, init_key = jax.random.split(jax.random.key(seed))
	state = _new_states(init_key, pool, engine_tables, batch_size)
	carry = (state, key, jnp.zeros(batch_size, dtype=jnp.int32))
	jax.block_until_ready((carry, pool, engine_tables))
	run = jax.jit(rollout, static_argnums=(3, 4))
	start = time.perf_counter()
	carry, _ = jax.block_until_ready(run(carry, pool, engine_tables, steps, with_observation))
	compile_seconds = time.perf_counter() - start
	# The cold first call also warms up the executable.
	durations = []
	counts = []
	completed_lengths = []
	for _ in range(repetitions):
		start = time.perf_counter()
		carry, stats = jax.block_until_ready(
			run(carry, pool, engine_tables, steps, with_observation)
		)
		durations.append(time.perf_counter() - start)
		counts.append(int(stats[0]))
		completed_lengths.append(int(stats[1]))
	throughputs = [batch_size * steps / duration for duration in durations]
	battle_rates = [count / duration for count, duration in zip(counts, durations)]
	return {
		"device": str(jax.devices()[0].device_kind),
		"jax_version": jax.__version__,
		"batch_size": batch_size,
		"with_observation": with_observation,
		"observation_checksum": float(stats[2]) if with_observation else None,
		"steps_per_call": steps,
		"repetitions": repetitions,
		"compile_seconds": compile_seconds,
		"steps_per_second": statistics.median(throughputs),
		"steps_per_second_min": min(throughputs),
		"steps_per_second_max": max(throughputs),
		"battles_per_second": statistics.median(battle_rates),
		"battles_per_second_min": min(battle_rates),
		"battles_per_second_max": max(battle_rates),
		"completed_battles": sum(counts),
		"mean_battle_length": sum(completed_lengths) / sum(counts) if sum(counts) else None,
	}


def benchmark(*args, **kwargs):
	"""Run without the persistent cache and restore the caller's configuration."""
	previous = jax.config.jax_enable_compilation_cache
	jax.config.update("jax_enable_compilation_cache", False)
	try:
		return _benchmark(*args, **kwargs)
	finally:
		jax.config.update("jax_enable_compilation_cache", previous)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
	parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="steps per scan")
	parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--json", action="store_true")
	parser.add_argument(
		"--with-observation", action="store_true", help="observe both sides each step"
	)
	args = parser.parse_args()
	result = benchmark(
		args.team_pool,
		args.batch_size,
		args.steps,
		args.repetitions,
		args.seed,
		args.with_observation,
	)
	if args.json:
		print(json.dumps(result))
	else:
		length = result["mean_battle_length"]
		length_text = f"{length:.1f}" if length is not None else "n/a"
		print(
			f"{result['device']} | JAX {result['jax_version']} | batch {args.batch_size} | "
			f"K {args.steps} | compile {result['compile_seconds']:.2f}s | "
			f"steps/s {result['steps_per_second']:.0f} "
			f"({result['steps_per_second_min']:.0f}–{result['steps_per_second_max']:.0f}) | "
			f"battles/s {result['battles_per_second']:.2f} "
			f"({result['battles_per_second_min']:.2f}–{result['battles_per_second_max']:.2f}) | "
			f"mean length {length_text} steps | completed {result['completed_battles']}"
		)


if __name__ == "__main__":
	main()
