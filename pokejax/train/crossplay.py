"""python -m pokejax.train.crossplay: several trained networks against each other, every pair in the engine.

    python -m pokejax.train.crossplay [NAME=]CHECKPOINT [NAME=]CHECKPOINT... [--draws N] [--batch B]
        [--seed S] [--team-pool POOL] [--out FILE.json]

Each unordered pair of checkpoints plays the head-to-head of pokejax.train.headtohead: N team draws,
each played twice (each network on each side), with sampled and with greedy actions. Every pair meets
the same team draws (one seed), so the pairs are paired. A pair is played once, so the matrix has one
entry per pair: the FIRST checkpoint's win rate against the SECOND, with the Wilson interval, the
interval paired by draw and the rate of each seat. The JSON report (--out) holds every pair's
head-to-head report as headtohead.py writes it.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.tables import load_tables
from pokejax.train import probe
from pokejax.train.headtohead import MODES, HeadToHeadError, head_to_head, parse_checkpoint
from pokejax.train.matchup import network_contestant


class CrossPlayError(ValueError):
	"""A cross-play evaluation that cannot be run as asked."""


def pairs(count: int) -> list[tuple[int, int]]:
	"""Every unordered pair of `count` contestants as index pairs (i, j) with i < j, in order."""
	return [(first, second) for first in range(count) for second in range(first + 1, count)]


def cross_play(contestants: Sequence, env: Env, *, draws: int, batch: int, seed: int) -> list[dict]:
	"""The head-to-head report of every pair, the first of the pair being the earlier contestant."""
	reports = []
	for first, second in pairs(len(contestants)):
		a, b = contestants[first], contestants[second]
		modes = head_to_head(a, b, env, draws=draws, batch=batch, seed=seed)
		reports.append(
			{
				"first": a.label,
				"second": b.label,
				"draws": draws,
				"games": 2 * draws,
				"modes": modes,
			}
		)
	return reports


def run(args) -> dict:
	if args.batch % 2 or args.batch <= 0:
		raise CrossPlayError("--batch must be a positive even number")
	if args.draws <= 0 or args.draws % (args.batch // 2):
		raise CrossPlayError(
			f"--draws must be a positive multiple of batch / 2 = {args.batch // 2}"
		)
	tables = load_tables()
	contestants, configs, paths = [], [], {}
	for text in args.checkpoints:
		name, path = parse_checkpoint(text)
		contestant, config = network_contestant(name, path, args.team_pool, tables)
		contestants.append(contestant)
		configs.append(config)
		paths[name] = str(path)
	labels = [contestant.label for contestant in contestants]
	if len(contestants) < 2:
		raise CrossPlayError("cross-play needs at least two checkpoints")
	if len(set(labels)) != len(labels):
		raise CrossPlayError(f"labels must be distinct: {', '.join(labels)}")
	max_steps = configs[0].env.max_steps
	if any(config.env.max_steps != max_steps for config in configs):
		raise CrossPlayError("the checkpoints were trained with different env.max_steps")
	env = Env(
		EnvConfig(batch_size=args.batch, max_steps=max_steps, observe="none"),
		args.team_pool,
		tables=tables,
	)
	return {
		"labels": labels,
		"checkpoints": paths,
		"draws": args.draws,
		"batch": args.batch,
		"seed": args.seed,
		"max_steps": max_steps,
		"pairs": cross_play(contestants, env, draws=args.draws, batch=args.batch, seed=args.seed),
	}


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.train.crossplay", description=__doc__.split("\n")[0]
	)
	parser.add_argument("checkpoints", nargs="+", help="checkpoints: [NAME=]PATH")
	parser.add_argument(
		"--draws", type=int, default=2048, help="team draws per pair (2 games each)"
	)
	parser.add_argument("--batch", type=int, default=1024, help="battle rows per batch (even)")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
	return parser


def main(argv=None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	try:
		report = run(args)
	except (CrossPlayError, HeadToHeadError, probe.ProbeError, FileNotFoundError) as error:
		print(f"error: {error}", file=sys.stderr)
		return 2
	for pair in report["pairs"]:
		for mode, _ in MODES:
			row = pair["modes"][mode]
			low, high = row["ci95"]
			paired = row["paired_draws"]
			print(
				f"{mode}: {pair['first']} against {pair['second']} {100 * row['win_rate']:.2f}% "
				f"[{100 * low:.2f}, {100 * high:.2f}] over {row['battles']} games; side 0 "
				f"{100 * row['side0_win_rate']:.2f}%, side 1 {100 * row['side1_win_rate']:.2f}%; "
				f"paired by draw {100 * paired['estimate']:.2f}% "
				f"[{100 * paired['ci95'][0]:.2f}, {100 * paired['ci95'][1]:.2f}]; truncated "
				f"{row['truncated']}, mean length {row['mean_battle_length']:.1f}",
				flush=True,
			)
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
