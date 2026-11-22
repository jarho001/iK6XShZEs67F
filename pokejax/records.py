"""Helpers of the sizing sweep and the lab that need no JAX: JSON lines results, the subprocess runner
that reads a measurement's last JSON line, and the markdown tables.

The lab's orchestrator (pokejax/lab.py) imports this module and nothing that imports the engine:
importing the engine initializes the JAX backend, and the preflight must still run when no GPU works.
"""

import json
import math
import os
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

GIB = 2**30

# The team pool the lab and pokejax.sizing read by default (pokejax.env.DEFAULT_POOL has the same path).
DEFAULT_POOL = Path("work/teams/gen5-teams.npz")

# The XLA allocator settings of every process that the lab and pokejax.sizing start. Preallocation
# reserves the share of the GPU's memory when the process starts, so an allocation is never refused
# because the allocator's free blocks are fragmented, and the lab runs one process per GPU at a time,
# so the reservations do not compete. A value the caller set in the environment is kept.
XLA_DEFAULTS = (
	("XLA_PYTHON_CLIENT_PREALLOCATE", "true"),
	("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90"),
)
# CUDA numbers the devices in PCI order, as nvidia-smi does (CUDA's default order can differ), so that
# the physical GPU index the lab records is the one CUDA_VISIBLE_DEVICES selects.
DEVICE_DEFAULTS = (("CUDA_DEVICE_ORDER", "PCI_BUS_ID"),)
ENVIRONMENT_DEFAULTS = XLA_DEFAULTS + DEVICE_DEFAULTS
GPU_IDLE_MIB = 500
GPU_IDLE_TIMEOUT = 60.0


def parse_memory_used(text: str) -> int | None:
	"""The MiB in use from one `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` line."""
	try:
		return int(text.strip().splitlines()[0])
	except (IndexError, ValueError):
		return None


def gpu_memory_used_mib(gpu: int) -> int | None:
	"""The MiB in use on the physical GPU `gpu` (nvidia-smi's index), or None when it cannot say."""
	command = [
		"nvidia-smi",
		"--query-gpu=memory.used",
		"--format=csv,noheader,nounits",
		"-i",
		str(gpu),
	]
	try:
		completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
	except (OSError, subprocess.SubprocessError):
		return None
	return parse_memory_used(completed.stdout) if completed.returncode == 0 else None


def wait_until_gpu_idle(gpu: int, timeout: float = GPU_IDLE_TIMEOUT) -> float:
	"""Wait until the GPU shows at most GPU_IDLE_MIB in use, so that the previous process has returned
	its memory before a new one preallocates; at most `timeout` seconds. Returns the seconds waited."""
	started = time.time()
	while True:
		used = gpu_memory_used_mib(gpu)
		if used is None or used <= GPU_IDLE_MIB or time.time() - started >= timeout:
			return time.time() - started
		time.sleep(1.0)


def xla_environment(
	base: Mapping[str, str] | None = None, gpu: int | None = None
) -> dict[str, str]:
	"""A copy of the environment with XLA_DEFAULTS for the names it does not set, and
	CUDA_VISIBLE_DEVICES set to the physical GPU index `gpu` when one is given."""
	environment = dict(os.environ if base is None else base)
	for name, value in ENVIRONMENT_DEFAULTS:
		environment.setdefault(name, value)
	if gpu is not None:
		environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
	return environment


def shard(items: Iterable, index: int, count: int) -> list:
	"""Every count-th item from index: shard(cells, 1, 2) are the odd positions of the grid."""
	if count < 1 or not 0 <= index < count:
		raise ValueError(f"shard {index} of {count} does not exist")
	return list(items)[index::count]


def _finite(value: Any) -> Any:
	"""The value with every non-finite float replaced by None, so that it is valid JSON."""
	if isinstance(value, float) and not math.isfinite(value):
		return None
	if isinstance(value, dict):
		return {key: _finite(item) for key, item in value.items()}
	if isinstance(value, (list, tuple)):
		return [_finite(item) for item in value]
	return value


def read_records(path: Path) -> list[dict]:
	"""The JSON lines of a results file; a line cut short by a kill is skipped."""
	if not path.exists():
		return []
	records = []
	for line in path.read_text(encoding="utf-8").splitlines():
		try:
			record = json.loads(line)
		except json.JSONDecodeError:
			continue
		if isinstance(record, dict):
			records.append(record)
	return records


def append_record(path: Path, record: dict) -> None:
	"""Append one JSON line and flush it; a line cut short by an earlier kill is ended first."""
	path.parent.mkdir(parents=True, exist_ok=True)
	prefix = ""
	if path.exists() and path.stat().st_size:
		with path.open("rb") as handle:
			handle.seek(-1, os.SEEK_END)
			prefix = "" if handle.read(1) == b"\n" else "\n"
	line = json.dumps(_finite(record), sort_keys=True, allow_nan=False)
	with path.open("a", encoding="utf-8") as handle:
		handle.write(prefix + line + "\n")
		handle.flush()
		os.fsync(handle.fileno())


def _last_json(text: str) -> dict | None:
	for line in reversed(text.splitlines()):
		try:
			record = json.loads(line)
		except json.JSONDecodeError:
			continue
		if isinstance(record, dict):
			return record
	return None


def run_measurement(
	command: Sequence[str], *, gpu: int | None, log_path: Path, timeout: float
) -> dict[str, Any]:
	"""Run a command on the GPU `gpu` (the default device when None) and parse its last JSON line.

	Returns the parsed record (None when there is none), the return code, whether the command timed
	out, the wall time and the log path; the command's stderr goes to log_path.
	"""
	if gpu is not None:
		wait_until_gpu_idle(gpu)
	environment = xla_environment(gpu=gpu)
	log_path.parent.mkdir(parents=True, exist_ok=True)
	started = time.time()
	timed_out = False
	output = ""
	returncode: int | None = None
	with log_path.open("w", encoding="utf-8") as log:
		try:
			completed = subprocess.run(
				list(command),
				stdout=subprocess.PIPE,
				stderr=log,
				text=True,
				timeout=timeout,
				env=environment,
				check=False,
			)
			output, returncode = completed.stdout, completed.returncode
		except subprocess.TimeoutExpired:
			timed_out = True
	return {
		"result": _last_json(output),
		"returncode": returncode,
		"timed_out": timed_out,
		"wall_seconds": time.time() - started,
		"log": str(log_path),
	}


def sweep_rows(records: Iterable[Mapping]) -> list[list[str]]:
	"""The table rows of sweep records, in the order of the grid (GPU, then the cell's sizes).

	The environment rate does not depend on the width or the core, so a cell that did not measure
	it shows the rate of a cell with the same batch and rollout.
	"""
	records = [record for record in records if record.get("kind") == "sweep"]
	rates: dict[tuple, float] = {}
	for record in records:
		cell = record.get("cell") or {}
		rate = record.get("environment_steps_per_second")
		if rate is not None and cell:
			rates.setdefault((cell.get("batch"), cell.get("rollout")), rate)
	rows = []
	for record in records:
		cell = record.get("cell") or {}
		memory = record.get("device_memory") or {}
		peak = memory.get("peak_bytes_in_use")
		ok = record.get("status") == "ok"
		environment = record.get("environment_steps_per_second")
		if environment is None and ok:
			environment = rates.get((cell.get("batch"), cell.get("rollout")))
		rows.append(
			[
				str(record.get("gpu", "")),
				str(cell.get("batch", "")),
				str(cell.get("rollout", "")),
				str(cell.get("width", "")),
				str(cell.get("core", "")),
				f"{record['parameters']:,}" if ok else "",
				f"{environment:,.0f}" if environment is not None else "",
				f"{record['learner_steps_per_second']:,.0f}" if ok else "",
				f"{record['update_seconds']:.3f}" if ok else "",
				f"{record['compile_seconds']:.1f}" if ok else "",
				f"{peak / GIB:.2f}" if peak is not None else "",
				"ok" if ok else f"failed ({record.get('error_class')})",
			]
		)
	rows.sort(key=lambda row: (row[0], int(row[1] or 0), int(row[2] or 0), row[3], row[4]))
	return rows


SWEEP_HEADERS = (
	"GPU",
	"Batch",
	"Rollout",
	"Width",
	"Core",
	"Parameters",
	"Env steps/s",
	"Learner steps/s",
	"Update s",
	"Compile s",
	"Peak GiB",
	"Status",
)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
	lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
	lines += ["| " + " | ".join(cell.replace("|", "/") for cell in row) + " |" for row in rows]
	return "\n".join(lines)
