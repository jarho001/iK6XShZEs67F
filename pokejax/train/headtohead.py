"""python -m pokejax.train.headtohead FIRST SECOND: two trained networks against each other in the engine.

    python -m pokejax.train.headtohead [NAME=]FIRST [NAME=]SECOND [--draws N] [--batch B] [--seed S]
        [--team-pool POOL] [--out FILE.json]

FIRST and SECOND are checkpoints (run directories or checkpoint.npz files). Each team draw of
N draws is played twice on the same team pair and the same engine keys: FIRST on side 0, then
FIRST on side 1, so every draw counts both seats and the battles are paired. Both networks act with
their own parameters and recurrent carries. The evaluation runs twice: with sampled actions (each
network samples its policy) and with greedy actions (each takes the argmax). The win rate of FIRST
over the 2N games has a Wilson 95% interval (pokejax.train.evaluation); a paired bootstrap over the
N draws gives a second interval that accounts for the pairing, and the rate of each seat is reported
as well.
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.tables import load_tables
from pokejax.train import probe
from pokejax.train.matchup import (
	MatchRunner,
	draw_battles,
	network_contestant,
	paired_mean_interval,
	summarize,
)

HEAD_SEED = 1_000_005  # offset of the run seed, apart from the training and evaluation keys
BOOTSTRAP_RESAMPLES = 2000
MODES = (("sampled", False), ("greedy", True))


class HeadToHeadError(ValueError):
	"""A head-to-head evaluation that cannot be run as asked."""


def parse_checkpoint(text: str) -> tuple[str, Path]:
	"""NAME=PATH, or PATH with a name taken from its run directory (or its file name)."""
	name, separator, raw = text.partition("=")
	if not separator:
		path = Path(text)
		if path.is_dir():
			return path.name, path
		if path.name == "checkpoint.npz":
			return path.parent.name, path
		return path.stem, path
	return name, Path(raw)


def battle_setups(env: Env, draws: int, batch: int, seed: int) -> list:
	"""The batches of the evaluation: (states of 2 x half rows, seats of the first seat, engine key).

	Each batch holds half draws, twice: rows [0, half) have the first seat on side 0 and rows
	[half, batch) the same battles with it on side 1.
	"""
	half = batch // 2
	count = draws // half
	draw = jax.jit(lambda key: draw_battles(env.replace(batch_size=half), key))
	base = jax.random.key(seed + HEAD_SEED)
	setups = []
	for index in range(count):
		draw_key, step_key = jax.random.split(jax.random.fold_in(base, index))
		states = draw(draw_key)
		both = jax.tree.map(lambda x: jnp.concatenate([x, x], axis=0), states)
		seats = jnp.concatenate([jnp.zeros(half, jnp.int32), jnp.ones(half, jnp.int32)])
		setups.append((both, seats, step_key))
	return setups


def head_to_head(first, second, env: Env, *, draws: int, batch: int, seed: int) -> dict:
	"""FIRST against SECOND over 2 x draws games in each action mode; returns the report."""
	half = batch // 2
	if draws % half or draws <= 0:
		raise HeadToHeadError(f"draws must be a positive multiple of batch / 2 = {half}")
	runner = MatchRunner(env.replace(batch_size=batch), first.seat, second.seat)
	setups = battle_setups(env, draws, batch, seed)
	modes = {}
	for mode, greedy in MODES:
		outcomes = [
			jax.device_get(runner((first.params, second.params), states, seats, key, greedy))
			for states, seats, key in setups
		]
		won, truncated, length, finished = (
			np.concatenate([np.asarray(out[index]) for out in outcomes]) for index in range(4)
		)
		by_seat = won.astype(np.float64).reshape(-1, 2, half)
		draw_score = (by_seat[:, 0, :] + by_seat[:, 1, :]).reshape(-1) / 2.0
		report = summarize(won, truncated, length, finished)
		report["side0_win_rate"] = float(by_seat[:, 0, :].mean())
		report["side1_win_rate"] = float(by_seat[:, 1, :].mean())
		report["paired_draws"] = paired_mean_interval(draw_score, BOOTSTRAP_RESAMPLES, seed)
		modes[mode] = report
	return modes


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.train.headtohead", description=__doc__.split("\n")[0]
	)
	parser.add_argument("first", help="FIRST checkpoint: [NAME=]PATH")
	parser.add_argument("second", help="SECOND checkpoint: [NAME=]PATH")
	parser.add_argument("--draws", type=int, default=4096, help="team draws (2 games each)")
	parser.add_argument("--batch", type=int, default=1024, help="battle rows per batch (even)")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
	return parser


def main(argv=None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.batch % 2 or args.batch <= 0:
		parser.error("--batch must be a positive even number")
	try:
		tables = load_tables()
		first_name, first_path = parse_checkpoint(args.first)
		second_name, second_path = parse_checkpoint(args.second)
		first, first_config = network_contestant(first_name, first_path, args.team_pool, tables)
		second, second_config = network_contestant(second_name, second_path, args.team_pool, tables)
		if first_config.env.max_steps != second_config.env.max_steps:
			raise HeadToHeadError("the two checkpoints were trained with different env.max_steps")
		env = Env(
			EnvConfig(batch_size=args.batch, max_steps=first_config.env.max_steps, observe="none"),
			args.team_pool,
			tables=tables,
		)
		modes = head_to_head(first, second, env, draws=args.draws, batch=args.batch, seed=args.seed)
	except (HeadToHeadError, probe.ProbeError, FileNotFoundError) as error:
		print(f"error: {error}", file=sys.stderr)
		return 2
	report = {
		"first": {"label": first_name, "path": str(first_path)},
		"second": {"label": second_name, "path": str(second_path)},
		"draws": args.draws,
		"games": 2 * args.draws,
		"batch": args.batch,
		"seed": args.seed,
		"max_steps": first_config.env.max_steps,
		"modes": modes,
	}
	for mode, _ in MODES:
		row = modes[mode]
		low, high = row["ci95"]
		paired = row["paired_draws"]
		print(
			f"{mode}: {first_name} {100 * row['win_rate']:.2f}% [{100 * low:.2f}, {100 * high:.2f}] "
			f"over {row['battles']} games against {second_name}; side 0 "
			f"{100 * row['side0_win_rate']:.2f}%, side 1 {100 * row['side1_win_rate']:.2f}%; paired "
			f"by draw {100 * paired['estimate']:.2f}% [{100 * paired['ci95'][0]:.2f}, "
			f"{100 * paired['ci95'][1]:.2f}]; truncated {row['truncated']}, unfinished "
			f"{row['unfinished']}, mean length {row['mean_battle_length']:.1f}",
			flush=True,
		)
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
