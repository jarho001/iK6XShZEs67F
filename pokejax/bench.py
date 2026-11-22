"""Measure cold compilation and random-policy battle throughput.

The default mode runs the engine with its own auto-reset loop. `--env` runs the environment of
pokejax.env instead (team draws from the sampler, auto-reset, rewards, done flags, step and
turn counts), with the random policy on both sides; add `--with-observation` to observe both
sides in either mode. `--env --policy NAME` has both sides play that baseline of
pokejax.baselines (random, random-move, max-damage, heuristics) from the state, without
observations unless `--with-observation` is given; `--policy1` gives side 1 another one. Without
`--policy` the random policy is one batched draw over the engine mask, the fastest form.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import POLICY_NAMES, make_policy
from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.engine.reset import Teams, check_teams, prepare_teams, reset
from pokejax.engine.state import ONGOING, TEAM_SLOTS
from pokejax.engine.step import step
from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.sampler import sample_team_indices
from pokejax.tables import load_tables
from pokejax.teams import load_npz

DEFAULT_BATCH = 4096
DEFAULT_STEPS = 256
DEFAULT_REPETITIONS = 3


def device_memory() -> dict:
	"""The default device's memory counters in bytes (peak and current use, limit); {} without any."""
	stats = jax.devices()[0].memory_stats() or {}
	keys = ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")
	return {key: int(stats[key]) for key in keys if key in stats}


def _new_states(key, pool, tables, batch_size, team_indices=None, prepared=None):
	"""Reset a batch on teams drawn from the whole pool, or from team_indices (pokejax.sampler).

	prepared is prepare_teams(pool, tables); a rollout passes it in once instead of per step.
	"""
	if prepared is None:
		prepared = prepare_teams(pool, tables)
	index_key, reset_key = jax.random.split(key)
	if team_indices is None:
		indices = jax.random.randint(index_key, (batch_size, 2), 0, pool.species.shape[0])
	else:
		indices = sample_team_indices(index_key, team_indices, batch_size)
	teams = jax.tree.map(lambda column: column[indices], pool)
	member_inputs = jax.tree.map(lambda column: column[indices], prepared)
	keys = jax.random.split(reset_key, batch_size)
	return jax.vmap(lambda k, t, p: reset(k, t, tables, None, p))(keys, teams, member_inputs)


def rollout(carry, pool, tables, steps, with_observation=False):
	"""Advance a batch and return completed count and total completed lengths."""
	prepared = prepare_teams(pool, tables)

	def advance(carry, _):
		state, key, lengths = carry
		key, action_key, step_key, reset_key = jax.random.split(key, 4)
		mask = jax.vmap(legal_actions, in_axes=(0, None))(state, tables)
		logits = jnp.where(mask, 0.0, -jnp.inf)
		actions = jax.random.categorical(action_key, logits, axis=-1)
		keys = jax.random.split(step_key, state.winner.shape[0])
		state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(state, actions, keys, None, tables)
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
		fresh = _new_states(reset_key, pool, tables, state.winner.shape[0], prepared=prepared)
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


def _policy_actions(env, env_state, key, policies):
	"""Both sides' baseline actions [B, 2] from the state and the engine's legal mask."""
	batch = env.config.batch_size
	mask = env.engine_mask(env_state)
	keys = jax.random.split(key, (len(policies), batch))
	return jnp.stack(
		[
			jax.vmap(lambda state, k, m, p=policy, s=side: p(state, s, k, m[s]))(
				env_state.state, keys[side], mask
			)
			for side, policy in enumerate(policies)
		],
		axis=1,
	)


def env_rollout(env, carry, steps, policies=None):
	"""Advance an environment, each side playing a baseline policy from the engine's legal mask.

	carry is (env_state, key); returns the carry and the sums of completed battles, of their
	lengths in steps and, with observations, of every observation leaf (so XLA keeps them).
	policies is a pair of pokejax.baselines policies, or None for the random policy as one
	batched draw over the mask.
	"""
	observing = env.config.observe == "both"

	def advance(carry, _):
		env_state, key = carry
		key, action_key = jax.random.split(key)
		if policies is None:
			logits = jnp.where(env.engine_mask(env_state), 0.0, -jnp.inf)
			actions = jax.random.categorical(action_key, logits, axis=-1)
		else:
			actions = _policy_actions(env, env_state, action_key, policies)
		env_state, timestep = env.step(env_state, actions)
		# Per-battle flags and lengths stack up like a training rollout's; they are summed after.
		stats = (timestep.info.ended, timestep.info.steps)
		if observing:
			stats += (sum(jnp.sum(x.astype(jnp.float32)) for x in jax.tree.leaves(timestep.obs)),)
		return (env_state, key), stats

	carry, stats = jax.lax.scan(advance, carry, None, length=steps)
	return carry, tuple(jnp.sum(x) for x in stats)


def _benchmark(
	pool_path=DEFAULT_POOL,
	batch_size=DEFAULT_BATCH,
	steps=DEFAULT_STEPS,
	repetitions=DEFAULT_REPETITIONS,
	seed=0,
	with_observation=False,
	environment=False,
	policy=None,
	policy1=None,
):
	"""Benchmark on the default device; use a fresh process for a cold compile.

	policy (a name of pokejax.baselines.POLICY_NAMES) is the baseline both sides play in the
	environment benchmark, policy1 side 1's if different; None is the batched random draw.
	"""
	if min(batch_size, steps, repetitions) < 1:
		raise ValueError("batch size, steps and repetitions must be positive")
	if (policy or policy1) and not environment:
		raise ValueError("A baseline policy needs the environment benchmark (--env)")
	if policy1 and not policy:
		raise ValueError("policy1 needs policy")
	if environment:
		env = Env(EnvConfig(batch_size, observe="both" if with_observation else "none"), pool_path)
		key, init_key = jax.random.split(jax.random.key(seed))
		carry = (env.reset(init_key)[0], key)
		side_policies = None
		if policy:
			host_tables = load_tables()
			side_policies = tuple(
				make_policy(name, env.tables, host_tables) for name in (policy, policy1 or policy)
			)
		jax.block_until_ready((carry, env, side_policies))
		jitted = jax.jit(env_rollout, static_argnums=(2,))

		def run(carry):
			return jitted(env, carry, steps, side_policies)
	else:
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
		jitted = jax.jit(rollout, static_argnums=(3, 4))

		def run(carry):
			return jitted(carry, pool, engine_tables, steps, with_observation)

	start = time.perf_counter()
	carry, _ = jax.block_until_ready(run(carry))
	compile_seconds = time.perf_counter() - start
	# The cold first call also warms up the executable.
	durations = []
	counts = []
	completed_lengths = []
	for _ in range(repetitions):
		start = time.perf_counter()
		carry, stats = jax.block_until_ready(run(carry))
		durations.append(time.perf_counter() - start)
		counts.append(int(stats[0]))
		completed_lengths.append(int(stats[1]))
	throughputs = [batch_size * steps / duration for duration in durations]
	battle_rates = [count / duration for count, duration in zip(counts, durations)]
	return {
		"device": str(jax.devices()[0].device_kind),
		"jax_version": jax.__version__,
		"batch_size": batch_size,
		"mode": "environment" if environment else "engine",
		"policy": [policy, policy1 or policy] if policy else None,
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
		"device_memory": device_memory(),
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
	parser.add_argument(
		"--env",
		action="store_true",
		help="run the environment (sampler draws, auto-reset, rewards, flags), not the bare engine",
	)
	parser.add_argument(
		"--policy",
		choices=POLICY_NAMES,
		help="with --env: the baseline both sides play (default: one batched random draw)",
	)
	parser.add_argument(
		"--policy1",
		choices=POLICY_NAMES,
		help="with --env --policy: side 1's baseline if different",
	)
	args = parser.parse_args()
	result = benchmark(
		args.team_pool,
		args.batch_size,
		args.steps,
		args.repetitions,
		args.seed,
		args.with_observation,
		args.env,
		args.policy,
		args.policy1,
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
