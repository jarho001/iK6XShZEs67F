"""python -m pokejax.train.panel: learners against one fixed panel, with paired battle setups.

    python -m pokejax.train.panel [NAME=]CHECKPOINT... [--opponent [NAME=]CHECKPOINT]...
        [--baselines NAME...] [--battles N] [--batch B] [--seed S] [--difference A B]...
        [--team-pool POOL] [--out FILE.json]

Every checkpoint plays the same panel: the four baselines (random, random-move, max-damage and
heuristics, or those given with --baselines) and each --opponent checkpoint, which plays as a fixed
network with its own parameters and carry. A battle setup is one battle of the evaluation: its
team pair, the side each checkpoint takes and the engine's random keys. The setups are drawn once
from --seed in batches of --batch, and every checkpoint meets every opponent on the same N setups,
so the comparisons are paired. Each checkpoint is evaluated with sampled actions and with greedy
actions.

Per checkpoint and action mode: the win rate against each opponent with its Wilson 95% interval,
and the panel mean, the average of the opponents' win rates (equal weights), with a paired bootstrap
interval over the setups. A --difference A B is the panel mean of A minus that of B (the mean over
setups of their difference in panel score), with a paired bootstrap 95% interval over the setups, and
the same difference against each opponent. Labels are the NAME= given, or derived from the path.
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.policy_names import POLICY_NAMES
from pokejax.tables import load_tables
from pokejax.train import probe
from pokejax.train.headtohead import parse_checkpoint
from pokejax.train.matchup import (
	MatchRunner,
	baseline_contestant,
	draw_battles,
	network_contestant,
	paired_mean_interval,
	summarize,
)

PANEL_SEED = 3_000_009  # offset of the run seed, apart from the training and evaluation keys
BOOTSTRAP_RESAMPLES = 2000
MODES = (("sampled", False), ("greedy", True))


class PanelError(ValueError):
	"""A panel evaluation that cannot be run as asked."""


def battle_setups(env: Env, battles: int, batch: int, seed: int) -> list:
	"""The setups of the panel: (battle states, the evaluated side of each row, engine key) per batch.

	Battle states and sides come from keys derived from seed and the batch index only, so every
	checkpoint and opponent plays the same battles.
	"""
	draw = jax.jit(lambda key: draw_battles(env, key))
	base = jax.random.key(seed + PANEL_SEED)
	setups = []
	for index in range(battles // batch):
		draw_key, side_key, step_key = jax.random.split(jax.random.fold_in(base, index), 3)
		states = draw(draw_key)
		sides = jax.random.bernoulli(side_key, shape=(batch,)).astype(jnp.int32)
		setups.append((states, sides, step_key))
	return setups


def play_panel(candidates, opponents, env: Env, *, battles: int, batch: int, seed: int) -> dict:
	"""Outcomes of every candidate against every opponent: results[candidate][opponent][mode]."""
	setups = battle_setups(env.replace(batch_size=batch), battles, batch, seed)
	runners = {}
	results = {}
	for candidate in candidates:
		results[candidate.label] = {}
		for opponent in opponents:
			key = (candidate.seat.cache_key, opponent.seat.cache_key)
			if key not in runners:
				runners[key] = MatchRunner(
					env.replace(batch_size=batch), candidate.seat, opponent.seat
				)
			runner = runners[key]
			modes = {}
			for mode, greedy in MODES:
				outcomes = [
					jax.device_get(
						runner((candidate.params, opponent.params), states, sides, step_key, greedy)
					)
					for states, sides, step_key in setups
				]
				modes[mode] = {
					name: np.concatenate([np.asarray(out[index]) for out in outcomes])
					for index, name in enumerate(("won", "truncated", "length", "finished"))
				}
			results[candidate.label][opponent.label] = modes
	return results


def panel_scores(results: dict, labels: list[str], mode: str) -> dict[str, np.ndarray]:
	"""Each candidate's panel score per battle setup: the mean over opponents of its wins (0 or 1)."""
	scores = {}
	for label in labels:
		won = [results[label][name][mode]["won"].astype(np.float64) for name in results[label]]
		scores[label] = np.mean(won, axis=0)
	return scores


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.train.panel", description=__doc__.split("\n")[0]
	)
	parser.add_argument("checkpoints", nargs="+", help="checkpoints to evaluate: [NAME=]PATH")
	parser.add_argument(
		"--opponent",
		action="append",
		default=[],
		metavar="[NAME=]PATH",
		help="a fixed earlier checkpoint on the panel (repeatable)",
	)
	parser.add_argument(
		"--baselines",
		nargs="*",
		choices=POLICY_NAMES,
		default=list(POLICY_NAMES),
		help="the baselines on the panel (default: all four)",
	)
	parser.add_argument("--battles", type=int, default=4096, help="battle setups per opponent")
	parser.add_argument("--batch", type=int, default=1024, help="battles per batch")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument(
		"--difference",
		nargs=2,
		action="append",
		default=[],
		metavar=("A", "B"),
		help="the paired difference of the panel means, A minus B (repeatable)",
	)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
	return parser


def run(args) -> dict:
	tables = load_tables()
	candidates, configs = [], []
	for text in args.checkpoints:
		name, path = parse_checkpoint(text)
		contestant, config = network_contestant(name, path, args.team_pool, tables)
		candidates.append(contestant)
		configs.append(config)
	fixed = []
	for text in args.opponent:
		name, path = parse_checkpoint(text)
		contestant, config = network_contestant(name, path, args.team_pool, tables)
		fixed.append(contestant)
		configs.append(config)
	max_steps = configs[0].env.max_steps
	if any(config.env.max_steps != max_steps for config in configs):
		raise PanelError("the checkpoints were trained with different env.max_steps")
	if args.battles % args.batch:
		raise PanelError("--battles must be a multiple of --batch")
	env = Env(
		EnvConfig(batch_size=args.batch, max_steps=max_steps, observe="none"),
		args.team_pool,
		tables=tables,
	)
	opponents = [baseline_contestant(name, env, tables) for name in args.baselines] + fixed
	labels = [c.label for c in candidates]
	names = labels + [o.label for o in opponents]
	if len(set(names)) != len(names):
		raise PanelError(f"labels must be distinct: {', '.join(names)}")
	for first, second in args.difference:
		if first not in labels or second not in labels:
			raise PanelError(f"--difference names must be evaluated checkpoints: {first} {second}")
	results = play_panel(
		candidates, opponents, env, battles=args.battles, batch=args.batch, seed=args.seed
	)
	report = {
		"battles": args.battles,
		"batch": args.batch,
		"seed": args.seed,
		"max_steps": max_steps,
		"opponents": [o.label for o in opponents],
		"checkpoints": {
			label: str(parse_checkpoint(text)[1])
			for label, text in zip(labels, args.checkpoints, strict=True)
		},
		"results": {},
		"differences": [],
	}
	for mode, _ in MODES:
		scores = panel_scores(results, labels, mode)
		for label in labels:
			rows = {}
			for opponent in opponents:
				cell = results[label][opponent.label][mode]
				rows[opponent.label] = summarize(
					cell["won"], cell["truncated"], cell["length"], cell["finished"]
				)
			report["results"].setdefault(label, {})[mode] = {
				"opponents": rows,
				"panel_mean": float(np.mean([rows[o.label]["win_rate"] for o in opponents])),
				"panel_ci95": paired_mean_interval(scores[label], BOOTSTRAP_RESAMPLES, args.seed),
			}
	for first, second in args.difference:
		for mode, _ in MODES:
			scores = panel_scores(results, labels, mode)
			entry = {
				"a": first,
				"b": second,
				"mode": mode,
				"panel": paired_mean_interval(
					scores[first] - scores[second], BOOTSTRAP_RESAMPLES, args.seed
				),
				"opponents": {},
			}
			for opponent in opponents:
				wins_a = results[first][opponent.label][mode]["won"].astype(np.float64)
				wins_b = results[second][opponent.label][mode]["won"].astype(np.float64)
				entry["opponents"][opponent.label] = paired_mean_interval(
					wins_a - wins_b, BOOTSTRAP_RESAMPLES, args.seed
				)
			report["differences"].append(entry)
	return report


def main(argv=None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	try:
		report = run(args)
	except (PanelError, probe.ProbeError, FileNotFoundError) as error:
		print(f"error: {error}", file=sys.stderr)
		return 2
	for mode, _ in MODES:
		for label, row in report["results"].items():
			cells = row[mode]
			parts = [
				f"{name} {100 * cell['win_rate']:.2f}% [{100 * cell['ci95'][0]:.2f}, {100 * cell['ci95'][1]:.2f}]"
				for name, cell in cells["opponents"].items()
			]
			low, high = cells["panel_ci95"]["ci95"]
			print(
				f"{mode} {label}: panel mean {100 * cells['panel_mean']:.2f}% "
				f"[{100 * low:.2f}, {100 * high:.2f}]; " + "; ".join(parts),
				flush=True,
			)
	for entry in report["differences"]:
		low, high = entry["panel"]["ci95"]
		print(
			f"{entry['mode']} difference {entry['a']} minus {entry['b']}: "
			f"{100 * entry['panel']['estimate']:+.2f} points [{100 * low:+.2f}, {100 * high:+.2f}]",
			flush=True,
		)
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
