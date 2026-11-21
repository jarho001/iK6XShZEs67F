"""Command line of `python -m pokejax.train`: a configuration file, then flags for single values.

```
python -m pokejax.train                          # configs/default.toml, work/runs/default
python -m pokejax.train --run-name test --updates 20
python -m pokejax.train --set ppo.learning_rate=1e-4 --resume --run-name test
```
"""

import argparse
from pathlib import Path

from pokejax.env import DEFAULT_POOL
from pokejax.train.checkpoint import CheckpointError
from pokejax.train.config import ConfigError, load_config
from pokejax.train.run import RunError, run_training

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.toml"

# Flags that set one value each, as section.key overrides.
FLAG_OVERRIDES = (
	("run_name", "run.name"),
	("run_dir", "run.run_dir"),
	("updates", "run.updates"),
	("seed", "run.seed"),
	("baseline", "opponent.baseline"),
	("batch_size", "rollout.batch_size"),
	("rollout_length", "rollout.rollout_length"),
	("sequence_length", "rollout.sequence_length"),
	("core", "network.core"),
	("learning_rate", "ppo.learning_rate"),
	("eval_every", "evaluation.every"),
	("checkpoint_every", "checkpoint.every"),
	("log_every", "logging.every"),
)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.train", description="Recurrent maskable PPO on the environment."
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="a TOML file")
	parser.add_argument(
		"--resume", action="store_true", help="continue the run in its directory from checkpoint"
	)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	for flag, _ in FLAG_OVERRIDES:
		parser.add_argument("--" + flag.replace("_", "-"), dest=flag, default=None)
	parser.add_argument(
		"--set",
		action="append",
		default=[],
		metavar="SECTION.KEY=VALUE",
		help="override any value of the configuration (repeatable)",
	)
	return parser


def main(argv=None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	overrides = [
		f"{path}={getattr(args, flag)}"
		for flag, path in FLAG_OVERRIDES
		if getattr(args, flag) is not None
	]
	overrides += args.set
	try:
		config = load_config(args.config, overrides)
		run_training(config, team_pool=args.team_pool, resume=args.resume)
	except (ConfigError, RunError, CheckpointError, FileNotFoundError) as error:
		parser.error(str(error))
	return 0
