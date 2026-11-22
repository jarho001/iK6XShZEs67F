"""Training sizing: the self-play trainer over a grid of batch size, rollout length, width and core.

    python -m pokejax.sizing sweep --out work/lab/sweep.jsonl --grid "batch=1024,4096 core=gru,lstm"
    python -m pokejax.sizing summary work/lab/sweep*.jsonl

Each cell of the grid is measured in a process of its own (`python -m pokejax.sizing cell`): the
cell builds the configuration of --config (default configs/selfplay.toml) with its sizes, compiles
the update, runs --updates more and reports the environment and learner rates, the update and
compile times and the peak device memory of that process alone. A cell that fails (out of memory,
a timeout, a crash) is recorded as a failed cell and the sweep goes on. Cells already in --out are
skipped, so an interrupted sweep resumes where it stopped. --deadline (seconds since the epoch)
keeps the sweep from starting a cell that would not end by then; --shard I/N runs every N-th cell
of the ordered grid, one shard per GPU.

Axes: batch (rollout.batch_size), rollout (rollout.rollout_length), width (a multiplier of
network.core_width, trunk_width and member_width; the embeddings do not change) and core
(network.core). An axis left out of the grid keeps the value of the configuration. The cells run
in order of how many axes they change from the first value of each axis, so a cut sweep still
covers every axis before it covers the combinations.
"""

import argparse
import contextlib
import itertools
import json
import os
import resource
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pokejax.records import (
	DEFAULT_POOL,
	ENVIRONMENT_DEFAULTS,
	SWEEP_HEADERS,
	_finite,
	append_record,
	markdown_table,
	read_records,
	run_measurement,
	shard,
	sweep_rows,
)
from pokejax.train.config import CORE_KINDS, ConfigError, TrainConfig, load_config

AXES = ("batch", "rollout", "width", "core")
DEFAULT_CONFIG = Path("configs/selfplay.toml")
DEFAULT_GRID = "batch=1024,4096,16384 rollout=64,128 width=1,2,4 core=gru,lstm"
DEFAULT_UPDATES = 3
CELL_TIMEOUT_SECONDS = 1800


class GridError(ValueError):
	"""A grid that does not parse, or a cell that the configuration rejects."""


@dataclass(frozen=True)
class Cell:
	"""One point of the grid: the sizes that the sweep varies."""

	batch: int
	rollout: int
	width: int
	core: str

	@property
	def key(self) -> str:
		return f"batch{self.batch}-rollout{self.rollout}-width{self.width}-{self.core}"

	def values(self) -> dict[str, Any]:
		return {
			"batch": self.batch,
			"rollout": self.rollout,
			"width": self.width,
			"core": self.core,
		}

	def flags(self) -> list[str]:
		return [
			"--batch-size", str(self.batch),
			"--rollout-length", str(self.rollout),
			"--width", str(self.width),
			"--core", self.core,
		]  # fmt: skip


def parse_grid(text: str) -> dict[str, tuple]:
	"""The axes of a grid such as "batch=1024,4096 core=gru,lstm"; an axis not named is left out."""
	axes: dict[str, tuple] = {}
	for term in text.split():
		name, equals, body = term.partition("=")
		if not equals or name not in AXES or name in axes:
			raise GridError(f"grid term {term!r}: use axis=values, each axis once, axes {AXES}")
		items = [item for item in body.split(",") if item]
		if not items or len(set(items)) != len(items):
			raise GridError(f"grid axis {name} needs distinct values")
		if name == "core":
			unknown = [item for item in items if item not in CORE_KINDS]
			if unknown:
				raise GridError(f"grid axis core: {', '.join(unknown)} is not one of {CORE_KINDS}")
			axes[name] = tuple(items)
			continue
		numbers = []
		for item in items:
			if not item.isdigit() or int(item) < 1:
				raise GridError(f"grid axis {name}: {item!r} is not a positive integer")
			numbers.append(int(item))
		axes[name] = tuple(numbers)
	return axes


def grid_cells(axes: Mapping[str, Sequence], base: TrainConfig) -> list[Cell]:
	"""Every cell of the grid, in the order the sweep runs them (see the module docstring)."""
	defaults = {
		"batch": (base.rollout.batch_size,),
		"rollout": (base.rollout.rollout_length,),
		"width": (1,),
		"core": (base.network.core,),
	}
	values = {name: tuple(axes.get(name, defaults[name])) for name in AXES}
	cells = [Cell(*combination) for combination in itertools.product(*(values[n] for n in AXES))]
	first = {name: values[name][0] for name in AXES}

	def changed(cell: Cell) -> int:
		return sum(getattr(cell, name) != first[name] for name in AXES)

	return sorted(cells, key=changed)


def cell_overrides(base: TrainConfig, cell: Cell) -> list[str]:
	"""The section.key=value overrides that give the cell's sizes to the base configuration."""
	network = base.network
	return [
		f"rollout.batch_size={cell.batch}",
		f"rollout.rollout_length={cell.rollout}",
		f"network.core={cell.core}",
		f"network.core_width={network.core_width * cell.width}",
		f"network.trunk_width={network.trunk_width * cell.width}",
		f"network.member_width={network.member_width * cell.width}",
	]


def cell_config(config_path: Path, cell: Cell | None = None) -> TrainConfig:
	"""The configuration file with the cell's sizes (the file as it is when cell is None)."""
	if cell is None:
		return load_config(config_path)
	base = load_config(config_path)
	try:
		return load_config(config_path, cell_overrides(base, cell))
	except ConfigError as error:
		raise GridError(f"cell {cell.key} is not a valid configuration: {error}") from error


def classify_error(error: BaseException) -> str:
	text = f"{type(error).__name__}: {error}"
	lowered = text.lower()
	if "RESOURCE_EXHAUSTED" in text or "out of memory" in lowered or "out_of_memory" in lowered:
		return "oom"
	if isinstance(error, ConfigError | GridError):
		return "config"
	return "error"


def _first_line(text: str, limit: int = 300) -> str:
	"""The line of an error that says what happened: the out-of-memory line when there is one."""
	lines = [part.strip() for part in text.splitlines() if part.strip()]
	informative = [
		line for line in lines if "RESOURCE_EXHAUSTED" in line or "out of memory" in line.lower()
	]
	return (informative or lines or [""])[0][:limit]


def _device_memory() -> dict[str, Any]:
	"""The device's memory counters (imported here: the sweep parent must not initialize JAX)."""
	from pokejax.bench import device_memory

	return device_memory()


def measure_cell(
	config: TrainConfig, team_pool: Path, updates: int, environment: bool = True
) -> dict[str, Any]:
	"""Compile the update of the configuration, run `updates` more and time them (in this process)."""
	import jax

	from pokejax.train.ppo import Trainer

	if updates < 1:
		raise ValueError("a cell needs at least one update after the compiling one")
	trainer = Trainer(config, team_pool)
	learner = trainer.init()
	jax.block_until_ready(learner)
	parameters = trainer.parameter_count(learner)
	environment_rate = trainer.environment_steps_per_second() if environment else None
	timings = []
	last = None
	for _ in range(updates + 1):
		start = time.perf_counter()
		learner, last = trainer.update(learner)
		jax.block_until_ready((learner, last))
		timings.append(time.perf_counter() - start)
	warm = timings[1:]
	update_seconds = statistics.median(warm)
	rollout = config.rollout
	steps = rollout.batch_size * rollout.rollout_length
	return {
		"parameters": parameters,
		"lanes": trainer.layout.lanes,
		"environment_steps_per_update": steps,
		"environment_steps_per_second": environment_rate,
		"learner_steps_per_second": steps / update_seconds,
		"update_seconds": update_seconds,
		"update_seconds_runs": warm,
		"cold_update_seconds": timings[0],
		"compile_seconds": timings[0] - update_seconds,
		"updates": updates,
		"win_rate_last_update": float(last["win_rate"]),
		"device": str(jax.devices()[0].device_kind),
		"jax_version": jax.__version__,
		"device_memory": _device_memory(),
		"max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
		"config": config.to_dict(),
		"groups": config.groups,
	}


def cell_main(args: argparse.Namespace) -> int:
	"""Measure one cell (or the configuration as it is) and print its record as one JSON line."""
	cell = None
	if args.batch_size is not None:
		cell = Cell(args.batch_size, args.rollout_length, args.width, args.core)
	started = time.time()
	try:
		config = cell_config(args.config, cell)
		result = {"status": "ok", "error_class": None, "error": None}
		result.update(
			measure_cell(config, args.team_pool, args.updates, environment=not args.no_environment)
		)
	except (RuntimeError, ValueError, MemoryError) as error:  # a failed cell, not a failed sweep
		result = {
			"status": "failed",
			"error_class": classify_error(error),
			"error": _first_line(str(error)),
		}
		with contextlib.suppress(RuntimeError, ValueError):  # no device at all: nothing to add
			result["device_memory"] = _device_memory()
	result["key"] = cell.key if cell else args.key
	result["cell"] = cell.values() if cell else None
	result["config_file"] = str(args.config)
	result["xla"] = {name: os.environ.get(name) for name, _ in ENVIRONMENT_DEFAULTS}
	result["wall_seconds"] = time.time() - started
	print(json.dumps(_finite(result), sort_keys=True))
	return 0


def run_cell_process(
	*,
	key: str,
	cell: Cell | None,
	config_path: Path,
	team_pool: Path,
	updates: int,
	gpu: int | None,
	log_path: Path,
	timeout: float = CELL_TIMEOUT_SECONDS,
	environment: bool = True,
) -> dict[str, Any]:
	"""Measure one cell (or the configuration as it is, when cell is None) in a new process."""
	command = [
		sys.executable,
		"-m",
		"pokejax.sizing",
		"cell",
		"--config",
		str(config_path),
		"--team-pool",
		str(team_pool),
		"--updates",
		str(updates),
		"--key",
		key,
	]
	if cell is not None:
		command += cell.flags()
	if not environment:
		command.append("--no-environment")
	measured = run_measurement(command, gpu=gpu, log_path=log_path, timeout=timeout)
	record = measured["result"]
	if record is None or "status" not in record:
		reason = "timeout" if measured["timed_out"] else f"exit {measured['returncode']}, no result"
		record = {
			"status": "failed",
			"error_class": "timeout" if measured["timed_out"] else "crash",
			"error": f"{reason}; see {measured['log']}",
		}
	record.update(
		key=key,
		cell=cell.values() if cell else None,
		gpu=gpu,
		returncode=measured["returncode"],
		wall_seconds=measured["wall_seconds"],
		log=measured["log"],
	)
	return record


Runner = Callable[..., dict]


def run_sweep(
	cells: Sequence[Cell],
	*,
	config_path: Path,
	team_pool: Path,
	updates: int,
	out: Path,
	gpu: int | None = None,
	deadline: float | None = None,
	log_dir: Path | None = None,
	cell_timeout: float = CELL_TIMEOUT_SECONDS,
	runner: Runner = run_cell_process,
	clock: Callable[[], float] = time.time,
	log: Callable[[str], None] = print,
) -> dict[str, Any]:
	"""Measure the cells not yet in `out`, in order, appending each result to it.

	A cell starts only if the longest cell so far would still end before the deadline. Returns the
	counts: cells run now, cells already in `out` and the keys left for the deadline.
	"""
	existing = read_records(out)
	done = {record["key"] for record in existing if "key" in record}
	rated = {
		(record["cell"]["batch"], record["cell"]["rollout"])
		for record in existing
		if record.get("cell") and record.get("environment_steps_per_second") is not None
	}
	log_dir = log_dir or out.parent / "logs"
	ran = 0
	resumed = 0
	left: list[str] = []
	longest = 0.0
	for cell in cells:
		if cell.key in done:
			resumed += 1
			continue
		if deadline is not None and clock() + longest > deadline:
			left.append(cell.key)
			continue
		started = clock()
		pair = (cell.batch, cell.rollout)
		record = runner(
			key=cell.key,
			cell=cell,
			config_path=config_path,
			team_pool=team_pool,
			updates=updates,
			gpu=gpu,
			timeout=cell_timeout,
			log_path=log_dir / f"sweep-{cell.key}-gpu{gpu if gpu is not None else 'x'}.log",
			environment=pair not in rated,
		)
		if record.get("environment_steps_per_second") is not None:
			rated.add(pair)
		longest = max(longest, clock() - started)
		append_record(out, {"kind": "sweep", "step": "sweep", **record})
		log(f"sweep {cell.key} gpu {gpu}: {record.get('status')} {record.get('error') or ''}")
		ran += 1
	return {"ran": ran, "resumed": resumed, "left_for_deadline": left}


def sweep_main(args: argparse.Namespace) -> int:
	axes = parse_grid(args.grid)
	base = load_config(args.config)
	cells = grid_cells(axes, base)
	index, count = args.shard
	mine = shard(cells, index, count)
	for cell in mine:  # reject a bad size before any GPU time is spent
		cell_config(args.config, cell)
	deadline = args.deadline
	counts = run_sweep(
		mine,
		config_path=args.config,
		team_pool=args.team_pool,
		updates=args.updates,
		out=args.out,
		gpu=args.gpu,
		deadline=deadline,
		log_dir=args.log_dir,
		cell_timeout=args.cell_timeout,
	)
	print(json.dumps({"cells": len(mine), **counts}, sort_keys=True))
	return 0


def summary_main(args: argparse.Namespace) -> int:
	records = [record for path in args.files for record in read_records(Path(path))]
	print(markdown_table(SWEEP_HEADERS, sweep_rows(records)))
	return 0


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="python -m pokejax.sizing", description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)

	sweep = commands.add_parser("sweep", help="measure the grid, skipping cells already in --out")
	sweep.add_argument("--out", type=Path, required=True, help="JSON lines results file")
	sweep.add_argument("--grid", default=DEFAULT_GRID, help="axes, e.g. 'batch=1024,4096 core=gru'")
	sweep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	sweep.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	sweep.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
	sweep.add_argument("--gpu", type=int, default=None, help="physical GPU index for the cells")
	sweep.add_argument(
		"--shard", type=lambda text: tuple(map(int, text.split("/"))), default=(0, 1)
	)
	sweep.add_argument("--deadline", type=float, default=None, help="epoch seconds")
	sweep.add_argument("--log-dir", type=Path, default=None)
	sweep.add_argument("--cell-timeout", type=float, default=CELL_TIMEOUT_SECONDS)
	sweep.set_defaults(run=sweep_main)

	cells = commands.add_parser("cells", help="print the cells of a grid in the order they run")
	cells.add_argument("--grid", default=DEFAULT_GRID)
	cells.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	cells.set_defaults(run=lambda args: _print_cells(args))

	cell = commands.add_parser("cell", help="measure one cell in this process (used by sweep)")
	cell.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	cell.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	cell.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
	cell.add_argument("--key", default="configuration")
	cell.add_argument("--batch-size", type=int, default=None)
	cell.add_argument("--rollout-length", type=int, default=None)
	cell.add_argument("--width", type=int, default=1)
	cell.add_argument("--core", choices=CORE_KINDS, default="gru")
	cell.add_argument(
		"--no-environment",
		action="store_true",
		help="skip the environment rate (the sweep measures it once per batch and rollout)",
	)
	cell.set_defaults(run=cell_main)

	summary = commands.add_parser("summary", help="the sweep table of results files")
	summary.add_argument("files", nargs="+")
	summary.set_defaults(run=summary_main)
	return parser


def _print_cells(args: argparse.Namespace) -> int:
	base = load_config(args.config)
	for cell in grid_cells(parse_grid(args.grid), base):
		cell_config(args.config, cell)
		print(cell.key)
	return 0


def main(argv: Sequence[str] | None = None) -> int:
	argv = list(sys.argv[1:] if argv is None else argv)
	if argv and argv[0].startswith("-") and argv[0] not in ("-h", "--help"):
		argv.insert(0, "sweep")  # `python -m pokejax.sizing --grid ...` is the sweep
	# The allocator defaults of the lab's processes, for this process and its children (a value set
	# in the environment wins). XLA reads them when the backend starts, which is after this point.
	for name, value in ENVIRONMENT_DEFAULTS:
		os.environ.setdefault(name, value)
	args = build_parser().parse_args(argv)
	try:
		return args.run(args)
	except (GridError, ConfigError) as error:
		build_parser().error(str(error))
	return 2


if __name__ == "__main__":
	raise SystemExit(main())
