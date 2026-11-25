"""Time and evaluate Method 2 against the frozen depth-one Method 1 arm C."""

import argparse
import fcntl
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.state import ONGOING
from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.search.belief import Catalogue
from pokejax.search.evaluate import EngineMatch, ensure_draws
from pokejax.search.matches import MatchResult, combine, report
from pokejax.search.method1 import Method1, infer
from pokejax.search.method2 import MatrixConfig, Method2
from pokejax.search.oracle import SearchConfig
from pokejax.search.signal import prefix
from pokejax.tables import load_tables
from pokejax.train import headtohead, panel
from pokejax.train.matchup import baseline_contestant, network_contestant

MATRIX_ARMS = {
	"M2-hi": (1.0, False),
	"M2-mid": (0.3, False),
	"M2-lo": (0.1, False),
	"M2-mid-greedy": (0.3, True),
}
BASELINES = ("max-damage", "heuristics")
POOL_SEED = 720026
TIMING_SEED, PILOT_SEED, CONFIRM_SEED = 31004, 31005, 31006


class Journal:
	def __init__(self, path):
		self.file = path.open("a+", encoding="utf-8")
		fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
		self.file.seek(0)
		self.rows = {}
		self.last_complete = 0
		while line := self.file.readline():
			if not line.endswith("\n"):
				# The last write was interrupted. Completed cells precede it.
				self.file.seek(self.last_complete)
				self.file.truncate()
				break
			row = json.loads(line)
			self.rows[row["case"]] = row
			self.last_complete = self.file.tell()

	def __enter__(self):
		return self

	def __exit__(self, *_):
		self.file.close()

	def write(self, case, **fields):
		row = {"case": case, **fields}
		self.file.write(json.dumps(row, allow_nan=False) + "\n")
		self.file.flush()
		os.fsync(self.file.fileno())
		self.rows[case] = row
		print(json.dumps({k: v for k, v in row.items() if k != "game_data"}), flush=True)
		return row


@jax.jit
def diagnostic_row(p, gap):
	entropy = -(p * jnp.log(jnp.maximum(p, jnp.finfo(p.dtype).tiny))).sum(-1)
	return jnp.stack((gap, entropy, p.max(-1)), axis=-1)


def play_cell(runner, setups, budget, label):
	began = time.monotonic()
	results, metrics, solutions = [], [], []
	batch_times = []
	limit = len(setups)

	def inspect(decision):
		solution = runner.method.last_solution
		if solution is not None:
			info, active = jax.device_get(
				(diagnostic_row(solution.p, solution.gap), decision.searched)
			)
			solutions.append(info[active])

	callback = inspect if isinstance(runner.method, Method2) else None
	for i, setup in enumerate(setups, 1):
		started = time.monotonic()
		result, counts = runner.run(*setup, on_decision=callback)
		results.append(result)
		metrics.append(counts)
		elapsed = time.monotonic() - began
		batch_times.append(time.monotonic() - started)
		# Estimate at warm batch boundaries and keep both seats of each setup.
		if (
			budget
			and 8 <= i <= len(setups) // 2
			and limit == len(setups)
			and elapsed + np.median(batch_times[-4:]) * (limit - i) > budget
		):
			limit //= 2
			print(f"{label} projected over budget, halving to {limit} batches", flush=True)
		if i <= 4 or i % 8 == 0 or i == limit:
			print(f"{label} {i}/{limit} batches, {elapsed:.1f}s", flush=True)
		if i == limit:
			break
	return (
		combine(results),
		np.concatenate(metrics, axis=1),
		np.concatenate(solutions) if solutions else None,
		time.monotonic() - began,
	)


def cell(journal, stage, arm, opponent, runner, setups, seed, half=None, control=None, budget=0):
	label = f"{stage}-{arm}-vs-{opponent}"
	if label in journal.rows:
		print(f"Skipping completed {label}", flush=True)
		return journal.rows[label]
	result, metrics, solutions, seconds = play_cell(runner, setups, budget, label)
	reference = None
	if control is not None:
		data = control["game_data"]
		reference = MatchResult(*(np.asarray(data[name]) for name in MatchResult._fields))
		count = min(result.won.size, reference.won.size)
		reference = prefix(reference, count)
	stat = report(result, seed, half=half)
	if reference is not None:
		matched = report(prefix(result, count), seed, half=half, raw=reference)
		stat.update(delta=matched["delta"], paired_battles=count)
	eligible, empty, invalid, *_ = metrics.sum(axis=1)
	if eligible:
		stat["fallback_rate"] = float((empty + invalid) / eligible)
	if solutions is not None:
		gap, entropy, max_p = solutions.T
		stat.update(
			solved=len(gap),
			gap_mean=float(gap.mean()),
			gap_p95=float(np.percentile(gap, 95)),
			gap_max=float(gap.max()),
			entropy_nats=float(entropy.mean()),
			mixed_share=float((max_p < 0.9).mean()),
		)
	return journal.write(
		label,
		stage=stage,
		arm=arm,
		opponent=opponent,
		seed=seed,
		planned_games=len(setups) * setups[0][1].size,
		seconds=seconds,
		paired_setup_size=2 if half is not None else 1,
		game_data={name: getattr(result, name).tolist() for name in result._fields},
		**stat,
	)


def timing(journal, env, player, methods, batch, samples):
	pending = [arm for arm in methods if f"timing-{arm}" not in journal.rows]
	if not pending:
		return
	setup = headtohead.battle_setups(env, batch // 2, batch, TIMING_SEED)[0]
	roots = []
	EngineMatch(env, player, None, "raw", None).run(*setup, roots=roots)
	model, params = player.seat.model, player.params
	for arm in pending:
		method = methods[arm]
		data, cold = [], None
		warmup = 50
		for repeat in range(batch + warmup + samples):
			row = repeat % batch
			seed = TIMING_SEED + repeat
			belief = method.new_beliefs(seed, 1)
			for ply, root in enumerate(roots):
				view, carry, prior = jax.tree.map(lambda x, row=row: x[row : row + 1], root)
				if int(view.result[0]) != ONGOING:
					break
				key = jax.random.fold_in(jax.random.key(seed), ply)
				started = time.perf_counter()
				after, logits, head = infer(model, params, view.obs, carry, view.has_held)
				if isinstance(method, Method2):
					policy = jax.random.fold_in(key, 1)
					noise = jax.random.gumbel(policy, logits.shape, dtype=logits.dtype)
				else:
					noise = jnp.zeros_like(logits)
				decision, metrics = method.choose(
					params, view, belief, after, logits, head, key[None], noise, prior
				)
				jax.block_until_ready(decision)
				seconds = time.perf_counter() - started
				if metrics.eligible[0]:
					if cold is None:
						cold = seconds
					data.append(seconds * 1000)
				if len(data) >= samples + warmup:
					break
			if len(data) >= samples + warmup:
				break
		journal.write(
			f"timing-{arm}",
			arm=arm,
			first_eligible_seconds=cold,
			samples=len(data) - warmup,
			warm_ms=np.percentile(data[warmup:], [50, 95, 99]).tolist(),
		)


def evaluate(journal, env, tables, player, methods, args):
	batch = args.batch
	baselines = {name: baseline_contestant(name, env, tables).seat.policy for name in BASELINES}

	def runner(arm, opponent):
		return EngineMatch(
			env,
			player,
			methods.get(arm),
			arm,
			baselines.get(opponent),
			foe_method=methods["C"] if opponent == "C" else None,
			sample_raw=arm in MATRIX_ARMS,
		)

	if "selected" not in journal.rows:
		head = headtohead.battle_setups(env, 256, batch, PILOT_SEED)
		panels = panel.battle_setups(env, 512, batch, PILOT_SEED)
		stats = {arm: {} for arm in methods}
		for opponent in ("sampled-raw", *BASELINES):
			setups = head if opponent == "sampled-raw" else panels
			half = batch // 2 if opponent == "sampled-raw" else None
			raw = cell(
				journal, "pilot", "raw", opponent, runner("raw", opponent), setups, PILOT_SEED, half
			)
			for arm in methods:
				stats[arm][opponent] = cell(
					journal,
					"pilot",
					arm,
					opponent,
					runner(arm, opponent),
					setups,
					PILOT_SEED,
					half,
					raw,
				)
		qualifiers = [
			arm
			for arm in MATRIX_ARMS
			if all(stats[arm][foe]["delta"]["estimate"] >= -0.02 for foe in BASELINES)
		]
		selected = max(
			qualifiers,
			key=lambda arm: stats[arm]["sampled-raw"]["win_rate"]["estimate"],
			default=None,
		)
		journal.write("selected", arm=selected, qualifiers=qualifiers)
	if args.stage == "pilot":
		return
	arm = journal.rows["selected"]["arm"]
	if arm is None:
		print("No matrix arm qualified for confirmation.", flush=True)
		return
	head = headtohead.battle_setups(env, 2048, batch, CONFIRM_SEED)
	panels = panel.battle_setups(env, 2048, batch, CONFIRM_SEED)
	cell(
		journal,
		"confirm",
		arm,
		"C",
		runner(arm, "C"),
		head,
		CONFIRM_SEED,
		batch // 2,
		budget=args.budget,
	)
	for opponent in BASELINES:
		control = cell(
			journal,
			"confirm",
			"C",
			opponent,
			runner("C", opponent),
			panels,
			CONFIRM_SEED,
			budget=args.budget,
		)
		cell(
			journal,
			"confirm",
			arm,
			opponent,
			runner(arm, opponent),
			panels,
			CONFIRM_SEED,
			control=control,
			budget=args.budget,
		)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("stage", choices=("all", "timing", "pilot"))
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument("--batch", type=int, default=32)
	parser.add_argument("--samples", type=int, default=200)
	parser.add_argument("--columns", type=int, default=2)
	parser.add_argument("--budget", type=float, default=5400)
	args = parser.parse_args()
	if args.batch < 2 or args.batch % 2 or 512 % args.batch:
		parser.error("--batch must be an even divisor of 512")
	if args.samples <= 0 or args.budget <= 0:
		parser.error("Sample count and budget must be positive")

	evaluation_pool = Path("work/s4-eval-draws.npz")
	tables = load_tables()
	ensure_draws(evaluation_pool, 16_384, POOL_SEED, tables)
	player, train = network_contestant("480M checkpoint", args.checkpoint, DEFAULT_POOL, tables)
	env = Env(
		EnvConfig(args.batch, max_steps=train.env.max_steps, observe="none"),
		evaluation_pool,
		tables=tables,
	)
	catalogue = Catalogue.load(
		DEFAULT_POOL,
		tables,
		extra=Path("work/s2-catalogue-draws.npz"),
		cache=Path("work/s2-catalogue.npz"),
	)
	base = {"gamma": train.ppo.gamma, "max_steps": train.env.max_steps}
	model = player.seat.model
	method_c = Method1(model, env.tables, catalogue, SearchConfig(margin=0.02, **base))
	methods = {"C": method_c}
	for arm, (penalty, greedy) in MATRIX_ARMS.items():
		config = MatrixConfig(
			lambda_foe=penalty, greedy=greedy, opponent_columns=args.columns, **base
		)
		method = Method2(model, env.tables, catalogue, config)
		method.data = method_c.data
		methods[arm] = method
	args.out.parent.mkdir(parents=True, exist_ok=True)
	config = {
		"checkpoint": str(args.checkpoint.resolve()),
		"pool_seed": POOL_SEED,
		"batch": args.batch,
		"columns": args.columns,
		"timing_samples": args.samples,
		"matrix_arms": {arm: list(settings) for arm, settings in MATRIX_ARMS.items()},
	}
	with Journal(args.out) as journal:
		if "config" not in journal.rows:
			journal.write("config", **config)
		elif journal.rows["config"] != {"case": "config", **config}:
			parser.error("Existing journal uses a different configuration")
		print(jax.devices()[0].device_kind, flush=True)
		timing(journal, env, player, methods, args.batch, args.samples)
		if args.stage != "timing":
			evaluate(journal, env, tables, player, methods, args)


if __name__ == "__main__":
	main()
