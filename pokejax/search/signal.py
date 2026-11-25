"""Run the true-state diagnostic with python -m pokejax.search.signal STAGE CHECKPOINT.

Use 'timing' for single-battle timings, 'pilot' to select a configuration, or 'confirm'
to evaluate a selection from --out. 'all' runs the pilot and confirmation matches.
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import jax
import numpy as np

from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.search.matches import DiagnosticMatch, combine, report
from pokejax.search.oracle import DiagnosticOracle, SearchConfig
from pokejax.tables import load_tables
from pokejax.train import headtohead, panel
from pokejax.train.matchup import baseline_contestant, network_contestant

PILOTS = ((16, 1, "aux"), (16, 1, "mirror"), (16, 3, "mirror"))
TIMINGS = tuple(
	(worlds, depth, model)
	for worlds, depth in ((16, 1), (32, 1), (16, 3))
	for model in ("aux", "mirror")
)


def emit(label, value, out):
	line = json.dumps({"case": label, "experiment": "diagnostic true-state oracle", **value})
	print(line, flush=True)
	if out is not None:
		with out.open("a", encoding="utf-8") as file:
			file.write(line + "\n")


def time_decisions(env, player, base, args):
	setup = headtohead.battle_setups(env, args.batch // 2, args.batch, args.seed)[0]
	raw = DiagnosticMatch(env, player, None)
	roots = raw.timing_roots(setup, 24)
	if not roots:
		raise RuntimeError("No eligible timing roots")
	keys = jax.random.split(jax.random.key(args.seed + 80), args.samples + 6)
	# Each timing row pays the compile cost for its own shapes.
	jax.config.update("jax_enable_compilation_cache", False)
	for index, spec in enumerate(TIMINGS):
		config = variant(base, spec)
		oracle = DiagnosticOracle(player.seat.model, env.tables, config)
		jax.clear_caches()

		def decision(number, oracle=oracle):
			position, side, carry, _ = roots[number % len(roots)]
			return jax.block_until_ready(
				oracle(player.params, position, side, carry, keys[number : number + 1])
			)

		started = time.perf_counter()
		decision(0)
		cold = time.perf_counter() - started
		for warmup in range(1, 6):
			decision(warmup)
		times = []
		for sample in range(6, args.samples + 6):
			started = time.perf_counter()
			decision(sample)
			times.append((time.perf_counter() - started) * 1000)
		emit(
			f"timing-{index + 1}",
			{
				"config": asdict(config),
				"cold_seconds": cold,
				"warm_ms": np.percentile(times, [50, 95, 99]).tolist(),
				"samples": len(times),
				"root_plies": [root[-1] for root in roots],
			},
			args.out,
		)


def variant(base, spec):
	worlds, depth, opponent = spec
	return replace(base, worlds=worlds, depth=depth, opponent=opponent)


def match(runner, setups, *, budget, paired_rows=None, greedy=False):
	start = time.monotonic()
	runner.warm_search(*setups[0])
	results = []
	limit = len(setups)
	for index, setup in enumerate(setups):
		now = time.monotonic()
		results.append(runner.run(*setup, greedy=greedy))
		execution = time.monotonic() - now
		elapsed = time.monotonic() - start
		if index == 2:
			# Allow for longer games within the same wall-clock budget.
			limit = min(limit, index + 1 + max(0, int((budget - elapsed) / (execution * 1.2))))
			print(
				f"Played 3/{limit} batches in {elapsed:.0f}s. Latest batch {execution:.1f}s.",
				flush=True,
			)
		if (index + 1) % 32 == 0:
			print(f"Played {index + 1}/{limit} batches in {elapsed:.0f}s", flush=True)
		if index + 1 >= limit or elapsed + execution > budget:
			break
	result = combine(results)
	return result, {
		"seconds": time.monotonic() - start,
		"batch": setups[0][1].size,
		"planned_games": len(setups) * setups[0][1].size,
		"setups": result.won.size if paired_rows is None else result.won.size // 2,
	}


def signal(env, player, tables, base, args):
	batch = args.batch
	half = batch // 2
	pilot_seed = args.seed
	confirm_seed = pilot_seed + 1
	if args.stage in ("pilot", "all"):
		pilot = headtohead.battle_setups(env, args.pilot_draws, batch, pilot_seed)
		ranking = []
		for spec in PILOTS:
			config = variant(base, spec)
			oracle = DiagnosticOracle(player.seat.model, env.tables, config)
			runner = DiagnosticMatch(env, player, oracle)
			result, meta = match(runner, pilot, budget=args.budget, paired_rows=half)
			stats = report(result, pilot_seed, half=half)
			ranking.append((stats["win_rate"]["estimate"], spec))
			emit(
				f"pilot-{spec[2]}-d{spec[1]}",
				{
					"config": asdict(config),
					"seed": pilot_seed,
					**meta,
					**stats,
				},
				args.out,
			)
		raw = DiagnosticMatch(env, player, None)
		result, meta = match(raw, pilot, budget=args.budget, paired_rows=half, greedy=True)
		emit(
			"pilot-raw-greedy-vs-raw-sampled",
			{"seed": pilot_seed, **meta, **report(result, pilot_seed, half=half)},
			args.out,
		)
		result, meta = match(raw, pilot, budget=args.budget, paired_rows=half)
		emit(
			"pilot-raw-sampled-vs-raw-sampled",
			{"seed": pilot_seed, **meta, **report(result, pilot_seed, half=half)},
			args.out,
		)
		best = max(ranking, key=lambda row: row[0])[1]
		emit("selected", {"variant": best}, args.out)
		if args.stage == "pilot":
			return
	else:
		if args.out is None or not args.out.is_file():
			raise ValueError("Confirmation needs --out containing the selected pilot variant")
		selected = [
			json.loads(line)
			for line in args.out.read_text().splitlines()
			if line.startswith('{"case": "selected"')
		]
		if not selected:
			raise ValueError(f"No completed pilot selection in {args.out}")
		best = selected[-1]["variant"]

	config = variant(base, best)
	oracle = DiagnosticOracle(player.seat.model, env.tables, config)
	setups = headtohead.battle_setups(env, args.confirm_draws, batch, confirm_seed)
	runner = DiagnosticMatch(env, player, oracle)
	searched, meta = match(runner, setups, budget=args.budget, paired_rows=half)
	emit(
		"confirm-vs-raw-sampled",
		{
			"config": asdict(config),
			"seed": confirm_seed,
			**meta,
			**report(searched, confirm_seed, half=half),
		},
		args.out,
	)
	raw = DiagnosticMatch(env, player, None)
	result, meta = match(
		raw, setups[: meta["setups"] // half], budget=args.budget, paired_rows=half, greedy=True
	)
	emit(
		"confirm-raw-greedy-vs-raw-sampled",
		{
			"seed": confirm_seed,
			**meta,
			**report(result, confirm_seed, half=half),
			"search_minus_greedy": paired_delta(searched, result, confirm_seed, half),
		},
		args.out,
	)
	setups = panel.battle_setups(env, args.baseline_games, batch, confirm_seed)
	for name in ("max-damage", "heuristics"):
		policy = baseline_contestant(name, env, tables).seat.policy
		result, meta = match(
			DiagnosticMatch(env, player, oracle, policy), setups, budget=args.budget
		)
		count = result.won.size // batch
		raw, _ = match(
			DiagnosticMatch(env, player, None, policy), setups[:count], budget=args.budget
		)
		matched = min(result.won.size, raw.won.size)
		result = prefix(result, matched)
		raw = prefix(raw, matched)
		emit(
			f"panel-vs-{name}",
			{
				"config": asdict(config),
				"seed": confirm_seed,
				**meta,
				"setups": matched,
				**report(result, confirm_seed, raw=raw),
				"raw": report(raw, confirm_seed),
			},
			args.out,
		)


def prefix(result, count):
	return type(result)(*(column[:count] for column in result))


def paired_delta(search, control, seed, half):
	count = min(search.won.size, control.won.size)
	return report(prefix(search, count), seed, half=half, raw=prefix(control, count))["delta"]


def main(argv=None):
	parser = argparse.ArgumentParser(description="Diagnostic oracle search using true battle state")
	parser.add_argument("stage", choices=("timing", "pilot", "confirm", "all"))
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--out", type=Path, help="append JSON lines here")
	parser.add_argument("--batch", type=int, default=16, help="parallel battles")
	parser.add_argument("--seed", type=int, default=771)
	parser.add_argument("--samples", type=int, default=100, help="warm decisions per timing case")
	parser.add_argument("--pilot-draws", type=int, default=256)
	parser.add_argument("--confirm-draws", type=int, default=4096)
	parser.add_argument("--baseline-games", type=int, default=1024)
	parser.add_argument("--budget", type=float, default=6900, help="seconds per match cell")
	parser.add_argument("--margin", type=float, default=0)
	parser.add_argument("--padding", type=int, default=32)
	parser.add_argument("--sample-strength", type=float, default=0)
	args = parser.parse_args(argv)
	if args.batch < 2 or args.batch % 2:
		parser.error("--batch must be a positive even number")
	if args.samples < 1 or args.budget <= 0:
		parser.error("--samples and --budget must be positive")
	if any(
		n < 1 or n % unit
		for n, unit in (
			(args.pilot_draws, args.batch // 2),
			(args.confirm_draws, args.batch // 2),
			(args.baseline_games, args.batch),
		)
	):
		parser.error(
			"draw counts must be positive multiples of half the batch, baseline of the batch"
		)
	tables = load_tables()
	player, train = network_contestant("480M checkpoint", args.checkpoint, args.team_pool, tables)
	env = Env(
		EnvConfig(args.batch, max_steps=train.env.max_steps, observe="none"),
		args.team_pool,
		tables=tables,
	)
	base = SearchConfig(
		margin=args.margin,
		padding=args.padding,
		sample_strength=args.sample_strength,
		gamma=train.ppo.gamma,
		max_steps=train.env.max_steps,
	)
	print(f"Diagnostic oracle only. {jax.devices()[0].device_kind}", flush=True)
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
	if args.stage == "timing":
		time_decisions(env, player, base, args)
	else:
		signal(env, player, tables, base, args)


if __name__ == "__main__":
	main()
