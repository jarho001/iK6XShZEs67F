"""python -m pokejax.showdown: a trained checkpoint plays battles on Showdown against one baseline.

    python -m pokejax.showdown --checkpoint work/runs/rtx3060ti --opponent max-damage --battles 4096 \
        --policy greedy --seed 1 --record work/codex/runs/record.jsonl.gz --out work/codex/runs/out.json

The battles use the team pool's pairs drawn by pokejax.sampler from --seed (the learner takes side 0 in
even battles and side 1 in odd ones). The summary is printed, and written to --out as JSON when given.
The first observation of every battle is checked against the engine's (pokejax.showdown.player).
"""

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.policy_names import POLICY_NAMES
from pokejax.sampler import sample_team_indices
from pokejax.showdown.player import (
	DEFAULT_POOL_JSON,
	Policy,
	battle_specs,
	compare_first_observations,
	load_checkpoint,
	read_config,
	run_battles,
	summarize,
)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.showdown", description=__doc__.split("\n")[0]
	)
	parser.add_argument(
		"--checkpoint", type=Path, required=True, help="a run directory or checkpoint.npz"
	)
	parser.add_argument("--opponent", choices=POLICY_NAMES, required=True)
	parser.add_argument("--battles", type=int, required=True)
	parser.add_argument("--policy", choices=("sampled", "greedy"), default="sampled")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--concurrency", type=int, default=512, help="battles in progress at once")
	parser.add_argument(
		"--max-steps", type=int, default=None, help="default: the run's env.max_steps"
	)
	parser.add_argument(
		"--record", type=Path, default=None, help="per-battle JSON lines (.gz to compress)"
	)
	parser.add_argument("--out", type=Path, default=None, help="the summary as JSON")
	parser.add_argument("--pool", type=Path, default=DEFAULT_POOL, help="the team pool (npz)")
	parser.add_argument(
		"--pool-json", type=Path, default=DEFAULT_POOL_JSON, help="the same pool (JSON)"
	)
	return parser


def main(argv=None) -> int:
	args = build_parser().parse_args(argv)
	if args.battles < 1 or args.concurrency < 1:
		raise SystemExit("--battles and --concurrency must be positive")
	config = read_config(args.checkpoint)
	max_steps = config.env.max_steps if args.max_steps is None else args.max_steps
	env = Env(EnvConfig(batch_size=1, observe="none", max_steps=max_steps), args.pool)
	checkpoint = load_checkpoint(args.checkpoint, env)
	pairs = np.asarray(
		sample_team_indices(jax.random.key(args.seed), env.team_indices, args.battles)
	)
	learners = np.arange(args.battles) % 2
	policy = Policy(checkpoint, env, slots=args.concurrency, seed=args.seed)
	run = run_battles(
		policy,
		battle_specs(pairs, learners),
		pool_json=args.pool_json,
		opponent=args.opponent,
		seed=args.seed,
		max_steps=max_steps,
		concurrency=args.concurrency,
		greedy=args.policy == "greedy",
		record=args.record,
	)
	summary = summarize(run, args.opponent, args.policy)
	if len(run.first) != args.battles:
		raise SystemExit(f"{len(run.first)} first observations for {args.battles} battles")
	summary.first_check = compare_first_observations(env, pairs, learners, run.first)
	report = summary.as_dict()
	report.update(
		{
			"checkpoint": str(checkpoint.path),
			"seed": args.seed,
			"concurrency": args.concurrency,
			"max_steps": max_steps,
		}
	)
	print(json.dumps(report, indent=2, sort_keys=True))
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
