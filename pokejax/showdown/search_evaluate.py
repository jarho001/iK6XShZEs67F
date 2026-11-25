"""Resumable paired engine and pinned Showdown evaluation of frozen arm C."""

import argparse
import fcntl
import json
import os
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import jax
import numpy as np

from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.sampler import sample_team_indices
from pokejax.search.belief import Catalogue
from pokejax.search.evaluate import ARMS, OPPONENTS, EngineMatch, ensure_draws
from pokejax.search.matches import combine
from pokejax.search.method1 import Method1
from pokejax.search.oracle import SearchConfig
from pokejax.search.rebuild import OWN_FIELDS
from pokejax.showdown.player import battle_specs, engine_observations
from pokejax.showdown.search import Actor, play
from pokejax.tables import load_tables
from pokejax.train import headtohead, panel
from pokejax.train.matchup import baseline_contestant, network_contestant, paired_mean_interval

PILOT_SEED = 2299
EXIT_SEED = 2300
TEAM_SEED = 720025
RESAMPLES = 2000
MATCH_CONCURRENCY = 32


def cluster(values, batch, head):
	values = np.asarray(values, np.float64)
	return values.reshape(-1, 2, batch // 2).mean(1).ravel() if head else values


def interval(values, batch, head, seed):
	return paired_mean_interval(cluster(values, batch, head), RESAMPLES, seed)


def draw_setups(env, games, batch, seed, head, engine):
	"""Draw team pairs with the task 2 evaluator's reset keys, seats and ordering."""
	base = jax.random.key(seed + (headtohead.HEAD_SEED if head else panel.PANEL_SEED))
	half = batch // 2
	if engine:
		if head:
			setups = headtohead.battle_setups(env, games // 2, batch, seed)
		else:
			setups = panel.battle_setups(env, games, batch, seed)
	else:
		setups = None
	pairs, seats = [], []
	draw = jax.jit(sample_team_indices, static_argnums=(2,))
	for index in range(games // batch):
		keys = jax.random.split(jax.random.fold_in(base, index), 2 if head else 3)
		index_key, _ = jax.random.split(keys[0])
		indices = np.asarray(draw(index_key, env.team_indices, half if head else batch))
		if head:
			pairs.extend((indices, indices))
			seats.append(np.repeat([0, 1], half))
		else:
			pairs.append(indices)
			seats.append(np.asarray(jax.random.bernoulli(keys[1], shape=(batch,)), np.int32))
	specs = battle_specs(np.concatenate(pairs), np.concatenate(seats))
	if head:
		for start in range(0, games, batch):
			for offset in range(half):
				left, right = specs[start + offset], specs[start + half + offset]
				left["seedIndex"] = right["seedIndex"] = start + offset
	return specs, setups


class Journal:
	def __init__(self, path, signature):
		path.parent.mkdir(parents=True, exist_ok=True)
		self.file = path.open("a+", encoding="utf-8")
		fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
		self.file.seek(0)
		self.rows = {}
		while True:
			offset = self.file.tell()
			line = self.file.readline()
			if not line:
				break
			if not line.endswith("\n"):
				self.file.truncate(offset)
				break
			row = json.loads(line)
			if row["run"] != signature:
				raise ValueError("The existing journal belongs to another evaluation")
			self.rows[row["case"]] = row
		self.signature = signature

	def run(self, label, function):
		if label in self.rows:
			print(f"Skipping completed {label}", flush=True)
			return self.rows[label]
		row = {"case": label, "run": self.signature, **function()}
		self.file.write(json.dumps(row) + "\n")
		self.file.flush()
		os.fsync(self.file.fileno())
		self.rows[label] = row
		print(json.dumps({k: v for k, v in row.items() if k != "game_data"}), flush=True)
		return row


class Evaluation:
	def __init__(self, args):
		self.args = args
		tables = load_tables()
		ensure_draws(args.pool, 16_384, TEAM_SEED, tables)
		self.player, train = network_contestant(
			"480M checkpoint", args.checkpoint, DEFAULT_POOL, tables
		)
		self.env = Env(
			EnvConfig(args.batch, max_steps=train.env.max_steps, observe="none"),
			args.pool,
			tables=tables,
		)
		self.catalogue = Catalogue.load(
			DEFAULT_POOL,
			tables,
			extra=Path("work/s2-catalogue-draws.npz"),
			cache=Path("work/s2-catalogue.npz"),
		)
		self.config = replace(
			SearchConfig(gamma=train.ppo.gamma, max_steps=train.env.max_steps), **ARMS["C"]
		)
		self.method = Method1(self.player.seat.model, self.env.tables, self.catalogue, self.config)
		self.template = engine_observations(self.env, [[0, 0]], [0])
		state = self.env.new_battles(jax.random.key(0))
		self.dtypes = [getattr(state, name).dtype for name in OWN_FIELDS]
		self.baselines = {
			name: baseline_contestant(name, self.env, tables).seat.policy for name in OPPONENTS[1:]
		}

	def actor(self, arm, slots, seed):
		seat = self.player.seat
		return Actor(
			seat.model,
			self.player.params,
			seat.network,
			self.template,
			self.dtypes,
			slots,
			seed,
			self.method if arm == "C" else None,
		)

	def showdown(self, specs, opponent, arm, seed, concurrency, record=None):
		learner = self.actor(arm, concurrency, seed)
		head = opponent == OPPONENTS[0]
		foe = self.actor("raw", concurrency, seed + 1009) if head else None
		started = time.perf_counter()
		results, latency, rounds = play(
			learner,
			foe,
			specs,
			self.args.pool.with_suffix(".json"),
			"network" if head else opponent,
			seed,
			self.config.max_steps,
			record,
			audit=record is not None,
		)
		if len(results) != len(specs):
			raise RuntimeError(f"Showdown returned {len(results)} results for {len(specs)} games")
		won = [int(result["winner"] == result["learner"]) for result in results]
		# XLA compiles a new padded lane shape on demand. Keep its stalls in
		# elapsed time, but use steady-state round costs for the size projection.
		warm_times = rounds[1:]
		requests = sum(result["decisions"] for result in results)
		return {
			"games": len(results),
			"seconds": time.perf_counter() - started,
			"projectable_seconds": sum(
				round_seconds for round_seconds in warm_times if round_seconds < 2
			),
			"win_rate": interval(
				won, self.args.batch if record is None else len(specs), head, seed
			),
			"ties": sum(result["winner"] == "tie" for result in results),
			"timeouts": sum(result["truncated"] for result in results),
			"errors": sum(bool(result["errors"]) for result in results),
			"error_samples": [result["errors"] for result in results if result["errors"]][:4],
			"refusals": sum(result["refusals"] for result in results),
			"request_mask_mismatches": sum(result["maskMismatches"] for result in results),
			"request_mask_samples": [
				sample for result in results for sample in result["maskSamples"]
			][:4],
			"illegal_picks": sum(result["illegal"] for result in results),
			"requests": requests,
			"first_round_seconds": rounds[0],
			"warm_latency_samples": len(latency) - min(concurrency, len(specs)),
			"warm_latency_ms": np.percentile(
				latency[min(concurrency, len(specs)) :], [50, 95, 99]
			).tolist(),
			**learner.stats,
			"game_data": {"won": won},
		}

	def engine(self, setups, opponent, arm, seed):
		search = self.method if arm == "C" else None
		runner = EngineMatch(self.env, self.player, search, arm, self.baselines.get(opponent))
		started = time.perf_counter()
		results, metrics = [], []
		for setup in setups:
			result, counts = runner.run(*setup)
			results.append(result)
			metrics.append(counts)
		result = combine(results)
		eligible, empty, invalid, _, ess, valid = np.concatenate(metrics, axis=1).sum(axis=1)
		return {
			"games": result.won.size,
			"seconds": time.perf_counter() - started,
			"win_rate": interval(result.won, self.args.batch, opponent == OPPONENTS[0], seed),
			"ties": int(result.tied.sum()),
			"timeouts": int(result.truncated.sum()),
			"requests": int(result.length.sum()),
			"eligible": int(eligible),
			"no_particle": int(empty),
			"unrebuildable": int(invalid),
			"valid_worlds": int(valid),
			"ess": float(ess),
			"searched": int(result.searched.sum()),
			"overrides": int(result.overrides.sum()),
			"game_data": {"won": result.won.tolist()},
		}


def matches(evaluation, journal, stage, seed, games, head_games, engine=False):
	batch = evaluation.args.batch
	concurrency = evaluation.args.concurrency
	for opponent in OPPONENTS:
		head = opponent == OPPONENTS[0]
		specs, setups = draw_setups(
			evaluation.env,
			head_games if head else games,
			batch,
			seed,
			head,
			engine,
		)
		results = {}
		for simulator in ("showdown", "engine") if engine else ("showdown",):
			for arm in ("raw", "C"):
				label = f"{stage}-{simulator}-{arm}-{opponent}"
				print(f"Starting {label}, {len(specs)} games", flush=True)
				if simulator == "showdown":
					row = journal.run(
						label,
						partial(evaluation.showdown, specs, opponent, arm, seed, concurrency),
					)
				else:
					row = journal.run(
						label, partial(evaluation.engine, setups, opponent, arm, seed)
					)
				results[simulator, arm] = np.asarray(row["game_data"]["won"], np.float64)
		deltas = {
			sim: results[sim, "C"] - results[sim, "raw"]
			for sim in ("showdown", "engine")
			if (sim, "C") in results
		}
		journal.run(
			f"{stage}-paired-{opponent}",
			lambda deltas=deltas, head=head: {
				**{sim: interval(delta, batch, head, seed) for sim, delta in deltas.items()},
				**(
					{
						"showdown_minus_engine": interval(
							deltas["showdown"] - deltas["engine"],
							batch,
							head,
							seed,
						)
					}
					if engine
					else {}
				),
			},
		)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--pool", type=Path, default=Path("work/s3-eval-draws.npz"))
	parser.add_argument("--concurrency", type=int, default=MATCH_CONCURRENCY)
	parser.add_argument("--batch", type=int, default=32)
	parser.add_argument("--pilot-only", action="store_true")
	parser.add_argument("--timing-only", action="store_true")
	parser.add_argument("--audit", type=Path, default=Path("work/s3-records-validated.jsonl.gz"))
	args = parser.parse_args()
	if args.batch <= 0 or 512 % args.batch or args.batch % 2 or args.concurrency < 1:
		parser.error("batch must be an even divisor of 512 and concurrency must be positive")
	journal = Journal(
		args.out,
		{
			"checkpoint": str(args.checkpoint.resolve()),
			"pool": str(args.pool.resolve()),
			"team_seed": TEAM_SEED,
			"batch": args.batch,
			"concurrency": args.concurrency,
		},
	)
	print(f"{jax.devices()[0].device_kind}", flush=True)
	evaluation = Evaluation(args)
	specs, _ = draw_setups(evaluation.env, 512, args.batch, PILOT_SEED, True, False)
	journal.run(
		"timing-single-validated",
		lambda: evaluation.showdown(
			specs[:8],
			OPPONENTS[0],
			"C",
			PILOT_SEED,
			1,
			args.audit,
		),
	)
	if args.timing_only:
		return
	matches(evaluation, journal, "pilot", PILOT_SEED, 512, 512, engine=True)
	projection = journal.run(
		"exit-size",
		lambda: estimate_exit(journal),
	)
	if not args.pilot_only:
		games = projection["baseline_games"]
		matches(evaluation, journal, "exit", EXIT_SEED, games, 2 * games)


def estimate_exit(journal):
	seconds = sum(
		journal.rows[f"pilot-showdown-{arm}-{opponent}"]["projectable_seconds"]
		* (16 if opponent == OPPONENTS[0] else 8)
		for opponent in OPPONENTS
		for arm in ("raw", "C")
	)
	return {
		"estimated_full_seconds": seconds,
		"baseline_games": 2048 if seconds > 5 * 3600 else 4096,
	}


if __name__ == "__main__":
	main()
