"""The lab kit's steps: what lab/run.sh runs, one step per `python -m pokejax.lab step NAME`.

Steps, in order: preflight (the GPUs and JAX, first minute), facts (the system), bench (the engine and
the environment per GPU; only with LAB_BENCH=1), sweep (each arm's training for a few updates at
several batch sizes, one process per GPU), train (each arm on a GPU for a time budget, with kept
checkpoints at doubling step counts), panel (every kept checkpoint on the fixed panel, and the cross-play
matrix of the final checkpoints), and pack (summary.md, the training curves and the archive).
`python -m pokejax.lab is-done NAME` exits 0 when a step has its done marker.

Outputs go to LAB_DIR (default work/lab): results/*.jsonl (one JSON line per measurement), logs/,
runs/ (the training run directories), panel/ (the panel and cross-play reports), preflight/, facts.json,
pip-freeze.txt, progress.log (one JSON line per event) and done/NAME.json (the marker of a finished
step). A step is complete when all its measurements are in its results; rerunning skips complete steps
and the measurements already recorded. The time budget (LAB_HOURS) is set by make_plan: when the sweep
starts it may run for SWEEP_SHARE of the time left, and the training ends before the pack reserve. The
panel's share of the training window is planned from the number of checkpoints each run keeps (plan_runs).
The sweep and the training shrink to fit; the other steps have fixed sizes.

A training probe that runs out of device memory is repeated with half the batch (rollout.batch_size),
down to TRAIN_FLOOR; every attempt is recorded, and the batch a run used is in its plan (run_with_fallback).

Environment (set by lab/run.sh): LAB_HOURS (3), LAB_SMOKE (0, every step at minimal size), LAB_DIR
(work/lab), LAB_START (epoch seconds when this run started), LAB_TEAM_POOL (work/teams/gen5-teams.npz),
LAB_ARMS (comma-separated configuration files; default configs/p7-pairwise.toml and
configs/p7-pairwise-wide.toml), LAB_TRAIN_CONFIG (one configuration on every GPU), LAB_BENCH (0).
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, NamedTuple

from pokejax import lab_report
from pokejax.policy_names import POLICY_NAMES
from pokejax.records import (
	ENVIRONMENT_DEFAULTS,
	append_record,
	checkpoint_label,
	read_records,
	run_measurement,
	shard,
	xla_environment,
)
from pokejax.train.config import ConfigError, load_config

# The same path as pokejax.env.DEFAULT_POOL. Importing pokejax.env (or anything that imports the
# engine) initializes the JAX backend, and the orchestrator must run when no GPU works.
DEFAULT_POOL = Path("work/teams/gen5-teams.npz")
STEPS = ("preflight", "facts", "bench", "sweep", "train", "panel", "pack")
GPU_STEPS = ("bench", "sweep", "train", "panel")
PREFLIGHT_FAILED = 3  # exit status of a failed preflight (lab/run.sh stops the GPU steps)
PARTIAL_EXIT = 4  # a step that ran but is not complete yet (a rerun continues it)
SWEEP_SHARE = 0.55  # of the time left once the sweep starts, less the pack reserve
PACK_RESERVE_SECONDS = 300.0
STARTUP_SECONDS = 60.0  # process start, tables and pool before a training run's first update
OVERHEAD = 1.25  # per update: evaluations and checkpoints on top of the rollout and the learner
# The arms of a run when LAB_ARMS and LAB_TRAIN_CONFIG are not set: the pairwise network at 4,096 rows
# (configs/p7-pairwise.toml) and its wide variant (configs/p7-pairwise-wide.toml). The first arm is the
# base arm: its sweep also measures SWEEP_BASE_BATCH.
DEFAULT_ARMS = (Path("configs/p7-pairwise.toml"), Path("configs/p7-pairwise-wide.toml"))
SELFPLAY_CONFIG = Path("configs/selfplay.toml")
BENCH_BATCHES = (4096, 16384, 65536, 131072)
# The steps per scan and the repetitions of pokejax.bench's defaults, so that the lab's numbers compare
# with the README's (test_lab checks the equality). Each item compiles cold; the rollouts dominate
# at the large batches.
BENCH_STEPS, BENCH_REPETITIONS, BENCH_TIMEOUT = 256, 3, 1800
SMOKE_BENCH_BATCHES, SMOKE_BENCH_STEPS, SMOKE_BENCH_REPETITIONS = (4096,), 16, 1
SMOKE_BENCH_MODES = (("engine", None, False), ("env", None, True), ("env", "heuristics", False))
# The sweep: each arm's training run for SWEEP_UPDATES updates at each batch size (one process per
# GPU). Rows are the only size that changes; the base arm also runs SWEEP_BASE_BATCH, where OOM is a result.
SWEEP_BATCHES = (1024, 2048, 4096, 8192)
SWEEP_BASE_BATCH = 16384
SWEEP_UPDATES = 20
SMOKE_SWEEP_BATCHES = (512,)
SMOKE_SWEEP_UPDATES = 2
PROBE_UPDATES, SMOKE_PROBE_UPDATES = 3, 1
TRAIN_FLOOR = 1024  # a training probe halves its rows on out of memory, down to this many
# The smoke run on one RTX 3060 Ti: the pairwise update needs a 4.1 GiB block at 1,024 rows, which
# that card refuses (WSL2), so the smoke probes start at 512 rows and fall to 128.
SMOKE_TRAIN_BATCH, SMOKE_TRAIN_FLOOR = 512, 128
SMOKE_TRAIN_UPDATES = 4
SMOKE_TRAIN_SETTINGS = (
	"evaluation.batch=512",
	"evaluation.battles=512",
	"evaluation.final_battles=512",
	'evaluation.opponents=["random", "max-damage"]',
	"evaluation.greedy=false",
)
TRAIN_TIMEOUT_SLACK = 900.0
EVALUATIONS_PER_RUN = 12  # evaluation every twelfth of the run's updates (1,024 battles each)
KEEP_BASE_STEPS = (
	30_000_000  # kept checkpoints at 30M, 60M, 120M, ... environment steps, and the last
)
PLAN_ITERATIONS = 5  # the update counts and the panel reserve are fitted together this many times
# The panel (pokejax.train.panel) and the cross-play matrix (pokejax.train.crossplay). Measured on the
# RTX 3060 Ti with a Phase 7 checkpoint and the two references (lab/README.md): 262 s for 1,024 setups per
# opponent and 269 s for 2,048, so 0.52 ms per battle (6.3 s for twelve batches of 1,024 battles) and about
# 256 s per panel process for its start and compiles. A cross-play pair compiles its own runner (about
# 50 s each in the smoke run). The training leaves PANEL_SAFETY times that estimate for the GPU that
# runs the most panel work.
PANEL_BATTLES, FINAL_PANEL_BATTLES, PANEL_BATCH = 2048, 4096, 1024
MODE_NAMES = ("sampled", "greedy")  # the two action modes of every evaluation
CROSSPLAY_DRAWS, CROSSPLAY_BATCH = 2048, 1024
SMOKE_PANEL_BATTLES, SMOKE_FINAL_PANEL_BATTLES, SMOKE_PANEL_BATCH = 512, 512, 512
SMOKE_CROSSPLAY_DRAWS, SMOKE_CROSSPLAY_BATCH = 256, 512
PANEL_SECONDS_PER_BATTLE = 0.0006
PANEL_JOB_SECONDS = 300.0  # a panel process: start, compiles and the reports
CROSSPLAY_PAIR_SECONDS = 60.0  # a cross-play pair: its own compile
PANEL_SAFETY = 2.0
PANEL_TIMEOUT_SECONDS = 3600.0
PLOT_OPPONENTS = ("max-damage", "heuristics")
# The two Phase 6 reference checkpoints on every panel: the P2 network at 40M steps and the Phase 6
# run's 480M checkpoint. lab/make-kit.sh copies them into the kit; the paths are relative to the kit.
REFERENCE_CHECKPOINTS = (
	("ref-P2-40M", Path("work/reference-checkpoints/p6-p2-40M/checkpoint.npz")),
	("ref-P6-480M", Path("work/reference-checkpoints/p6-long-480M/checkpoint.npz")),
)
MINIMUM_DRIVER = {"cuda13": 580, "cuda12": 525, "cuda12-driver550": 525}
PACKAGES = (
	"jax",
	"jaxlib",
	"flax",
	"optax",
	"numpy",
	"jax-cuda12-plugin",
	"jax-cuda13-plugin",
	"nvidia-cuda-runtime-cu12",
	"nvidia-cuda-runtime",
	"nvidia-cublas-cu12",
	"nvidia-cublas",
	"nvidia-cudnn-cu12",
	"nvidia-cudnn-cu13",
	"nvidia-cuda-nvcc-cu12",
	"nvidia-cuda-nvcc",
)


@dataclass(frozen=True)
class Settings:
	lab_dir: Path
	hours: float
	smoke: bool
	start: float
	team_pool: Path
	arms: tuple[Path, ...]
	bench: bool

	@property
	def results(self) -> Path:
		return self.lab_dir / "results"

	@property
	def logs(self) -> Path:
		return self.lab_dir / "logs"

	@property
	def runs(self) -> Path:
		return self.lab_dir / "runs"

	@property
	def preflight_dir(self) -> Path:
		return self.lab_dir / "preflight"

	@property
	def done_dir(self) -> Path:
		return self.lab_dir / "done"

	@property
	def panel_dir(self) -> Path:
		return self.lab_dir / "panel"


@dataclass(frozen=True)
class Plan:
	start: float
	end: float
	sweep_deadline: float
	training_deadline: float


@dataclass(frozen=True)
class StepResult:
	status: str  # complete, partial or failed
	notes: tuple[str, ...] = ()
	exit_code: int = 0


def make_plan(hours: float, start: float, now: float) -> Plan:
	"""The deadlines of the budget, from the run's start and the time now.

	The sweep may start cells until SWEEP_SHARE of the time left has passed. The training ends before
	the pack reserve, which is at most a tenth of the budget. The panel's share of the training window
	is planned by the training step (it depends on how many checkpoints each run keeps).
	"""
	end = start + hours * 3600.0
	pack = min(PACK_RESERVE_SECONDS, 0.1 * hours * 3600.0)
	left = max(0.0, end - now - pack)
	return Plan(start, end, now + SWEEP_SHARE * left, end - pack)


def plan_updates(update_seconds: float, seconds: float, overhead: float = OVERHEAD) -> int:
	"""The updates that fit in `seconds` when each takes `update_seconds` plus the overhead."""
	if update_seconds <= 0 or seconds <= 0:
		return 1
	return max(1, int(seconds / (update_seconds * overhead)))


def arms_from_env(environ: Mapping[str, str]) -> tuple[Path, ...]:
	"""LAB_ARMS (comma-separated configuration files); else LAB_TRAIN_CONFIG (one configuration, on
	every GPU, as before); else DEFAULT_ARMS."""
	if environ.get("LAB_ARMS", "").strip():
		parts = [part.strip() for part in environ["LAB_ARMS"].split(",")]
		return tuple(Path(part) for part in parts if part)
	if environ.get("LAB_TRAIN_CONFIG", "").strip():
		return (Path(environ["LAB_TRAIN_CONFIG"].strip()),)
	return DEFAULT_ARMS


def settings_from_env(environ: Mapping[str, str] | None = None) -> Settings:
	environ = os.environ if environ is None else environ
	return Settings(
		lab_dir=Path(environ.get("LAB_DIR", "work/lab")),
		hours=float(environ.get("LAB_HOURS", "3")),
		smoke=environ.get("LAB_SMOKE", "0") == "1",
		start=float(environ.get("LAB_START", time.time())),
		team_pool=Path(environ.get("LAB_TEAM_POOL", str(DEFAULT_POOL))),
		arms=arms_from_env(environ),
		bench=environ.get("LAB_BENCH", "0") == "1",
	)


def iso_now() -> str:
	return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_progress(settings: Settings, step: str, event: str, **details: Any) -> None:
	record = {"time": iso_now(), "step": step, "event": event, **details}
	settings.lab_dir.mkdir(parents=True, exist_ok=True)
	with (settings.lab_dir / "progress.log").open("a", encoding="utf-8") as handle:
		handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
	print(f"[{record['time']}] {step} {event} {json.dumps(details, sort_keys=True, default=str)}")


def is_done(settings: Settings, step: str) -> bool:
	path = settings.done_dir / f"{step}.json"
	if not path.exists():
		return False
	return json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"


def mark_done(settings: Settings, step: str, result: StepResult, seconds: float) -> None:
	settings.done_dir.mkdir(parents=True, exist_ok=True)
	marker = {
		"step": step,
		"status": result.status,
		"finished": iso_now(),
		"seconds": seconds,
		"notes": list(result.notes),
	}
	(settings.done_dir / f"{step}.json").write_text(
		json.dumps(marker, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)


def load_gpus(settings: Settings) -> list[dict[str, Any]]:
	"""The GPUs of the passed preflight, as the physical indices and names it found."""
	path = settings.preflight_dir / "preflight.json"
	if not path.exists():
		raise RuntimeError("the preflight has not run: no GPU list yet")
	report = json.loads(path.read_text(encoding="utf-8"))
	if not report.get("ok"):
		raise RuntimeError("the preflight failed: the GPU steps cannot run")
	return list(report["gpus"])


def _run_quietly(
	command: Sequence[str], *, timeout: float = 120.0, env: Mapping[str, str] | None = None
) -> subprocess.CompletedProcess:
	return subprocess.run(
		list(command),
		capture_output=True,
		text=True,
		timeout=timeout,
		check=False,
		env=None if env is None else dict(env),
	)


def _jax_env(gpu: str | None = None) -> dict[str, str]:
	"""The environment of a JAX probe: CUDA only (no silent CPU fallback), one GPU if given."""
	environment = xla_environment(gpu=None if gpu is None else int(gpu))
	environment["JAX_PLATFORMS"] = "cuda"
	return environment


def xla_settings() -> dict[str, dict[str, str]]:
	"""The XLA allocator values the lab's processes run with, and whether the caller set them."""
	effective = xla_environment()
	return {
		name: {
			"value": effective[name],
			"source": "environment" if name in os.environ else "lab default",
		}
		for name, _ in ENVIRONMENT_DEFAULTS
	}


def _one_line(error: BaseException) -> str:
	text = f"{type(error).__name__}: {error}"
	return " ".join(text.split())[:500]


def _one_line_text(completed: subprocess.CompletedProcess) -> str:
	text = (completed.stderr or completed.stdout or f"exit {completed.returncode}").strip()
	return " ".join(text.splitlines()[-3:])[:500]


def _last_dict(text: str) -> dict | None:
	for line in reversed(text.splitlines()):
		try:
			record = json.loads(line)
		except json.JSONDecodeError:
			continue
		if isinstance(record, dict):
			return record
	return None


def image_variant() -> tuple[str, str]:
	"""The image variant and where it was read: POKEJAX_JAX_CUDA (the image sets it), else the
	installed JAX plugin (cuda12 and cuda13 are told apart; the driver-550 pins are not)."""
	declared = os.environ.get("POKEJAX_JAX_CUDA")
	if declared:
		return declared, "POKEJAX_JAX_CUDA"
	installed = _package_versions(("jax-cuda12-plugin", "jax-cuda13-plugin"))
	if installed["jax-cuda13-plugin"]:
		return "cuda13", "installed plugin"
	if installed["jax-cuda12-plugin"]:
		return "cuda12", "installed plugin (cuda12 or cuda12-driver550)"
	return "unknown", "no JAX CUDA plugin installed"


# Preflight.


def gpu_check(index: int, team_pool: Path) -> dict[str, Any]:
	"""Run inside a process that sees one GPU: a jitted product, then one cold compile of the engine.

	The JAX calls are inside a try: any failure is recorded in the result instead of raised.
	"""
	import jax
	import jax.numpy as jnp

	from pokejax.bench import benchmark, device_memory

	result: dict[str, Any] = {"index": index, "ok": False, "errors": []}
	try:
		device = jax.devices()[0]
		result["device_kind"] = str(device.device_kind)
		ones = jax.device_put(jnp.ones((256, 256), jnp.float32), device)
		start = time.perf_counter()
		product = jax.block_until_ready(jax.jit(lambda a: a @ a)(ones))
		value = float(product[0, 0])
		result["jitted_product"] = {
			"value": value,
			"ok": value == 256.0,
			"seconds": time.perf_counter() - start,
		}
	except (RuntimeError, ValueError) as error:
		result["errors"].append(f"jitted product: {_one_line(error)}")
	try:
		measured = benchmark(
			pool_path=team_pool,
			batch_size=4096,
			steps=8,
			repetitions=1,
			with_observation=False,
			environment=False,
		)
		result["engine_cold_compile"] = {
			"ok": True,
			"compile_seconds": measured["compile_seconds"],
			"steps_per_second": measured["steps_per_second"],
		}
		result["device_memory"] = device_memory()
	except (RuntimeError, ValueError, MemoryError) as error:
		result["errors"].append(f"engine cold compile: {_one_line(error)}")
	result["ok"] = not result["errors"] and bool(result.get("jitted_product", {}).get("ok"))
	return result


def gpu_probe_main(args: argparse.Namespace) -> int:
	print(json.dumps(gpu_check(args.index, args.team_pool), sort_keys=True))
	return 0


def jax_probe_main(args: argparse.Namespace) -> int:
	"""Every device JAX sees, and jax.print_environment_info() into --environment-out."""
	result: dict[str, Any] = {"ok": False, "error": None, "devices": []}
	buffer = io.StringIO()
	try:
		import jax

		result["jax_version"] = jax.__version__
		devices = jax.devices()
		result["devices"] = [
			{"platform": d.platform, "kind": str(d.device_kind), "id": int(d.id)} for d in devices
		]
		result["ok"] = bool(devices) and all(d.platform in ("gpu", "cuda") for d in devices)
		with contextlib.redirect_stdout(buffer):
			jax.print_environment_info()
	except (RuntimeError, ValueError, ImportError) as error:
		result["error"] = _one_line(error)
	args.environment_out.parent.mkdir(parents=True, exist_ok=True)
	args.environment_out.write_text(buffer.getvalue(), encoding="utf-8")
	print(json.dumps(result, sort_keys=True))
	return 0


GPU_QUERY_FIELDS = ("index", "name", "uuid", "driver", "memory_mib", "compute_capability")


def parse_gpu_query(text: str) -> list[dict[str, str]]:
	"""The GPUs of `nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,
	compute_cap --format=csv,noheader,nounits`, one dictionary per line that has six fields."""
	gpus = []
	for line in text.splitlines():
		fields = [field.strip() for field in line.split(",")]
		if len(fields) == len(GPU_QUERY_FIELDS):
			gpus.append(dict(zip(GPU_QUERY_FIELDS, fields, strict=True)))
	return gpus


def _nvidia_smi() -> tuple[list[dict[str, str]], str, str | None]:
	"""The GPUs nvidia-smi lists, the full nvidia-smi text and an error (None when it ran)."""
	query = [
		"nvidia-smi",
		"--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
		"--format=csv,noheader,nounits",
	]
	try:
		listed = _run_quietly(query)
		full = _run_quietly(["nvidia-smi"])
	except (OSError, subprocess.SubprocessError) as error:
		return [], "", _one_line(error)
	if listed.returncode != 0:
		return [], full.stdout + full.stderr, _one_line_text(listed)
	return parse_gpu_query(listed.stdout), full.stdout, None


def cuda_version(nvidia_smi_text: str) -> str | None:
	"""The CUDA version nvidia-smi prints: "CUDA Version" on Linux, "CUDA UMD Version" under WSL."""
	match = re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", nvidia_smi_text)
	return match.group(1) if match else None


def minimum_driver(variant: str) -> int | None:
	"""The lowest driver major version the image variant needs (JAX install guide: 525, 580)."""
	return MINIMUM_DRIVER.get(variant)


def preflight_checks(settings: Settings) -> dict[str, Any]:
	variant, source = image_variant()
	gpus, full_text, smi_error = _nvidia_smi()
	settings.preflight_dir.mkdir(parents=True, exist_ok=True)
	(settings.preflight_dir / "nvidia-smi.txt").write_text(full_text, encoding="utf-8")
	driver = gpus[0]["driver"] if gpus else None
	needed = minimum_driver(variant)
	checks: dict[str, Any] = {
		"image_variant": variant,
		"image_variant_source": source,
		"driver": driver,
		"cuda_version_nvidia_smi": cuda_version(full_text),
		"gpus_listed": len(gpus),
		"nvidia_smi_error": smi_error,
		"minimum_driver_for_variant": needed,
		"driver_ok_for_variant": (
			None if (driver is None or needed is None) else int(driver.split(".")[0]) >= needed
		),
	}
	jax_result = _run_quietly(
		[
			sys.executable,
			"-m",
			"pokejax.lab",
			"preflight-jax",
			"--environment-out",
			str(settings.preflight_dir / "jax-environment.txt"),
		],
		timeout=600,
		env=_jax_env(),
	)
	checks["jax"] = _last_dict(jax_result.stdout) or {
		"ok": False,
		"error": _one_line_text(jax_result),
	}
	return {"checks": checks, "gpus": gpus}


def step_preflight(settings: Settings) -> StepResult:
	found = preflight_checks(settings)
	checks, gpus = found["checks"], found["gpus"]
	jax_ok = bool(checks["jax"].get("ok"))
	count_ok = bool(gpus) and len(checks["jax"].get("devices", [])) == len(gpus)
	gpu_results: list[dict[str, Any]] = []
	if jax_ok and count_ok:

		def check(gpu: dict[str, str]) -> dict[str, Any]:
			completed = _run_quietly(
				[
					sys.executable,
					"-m",
					"pokejax.lab",
					"preflight-gpu",
					"--index",
					gpu["index"],
					"--team-pool",
					str(settings.team_pool),
				],
				timeout=1800,
				env=_jax_env(gpu["index"]),
			)
			return _last_dict(completed.stdout) or {
				"index": gpu["index"],
				"ok": False,
				"errors": [_one_line_text(completed)],
			}

		with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
			gpu_results = list(pool.map(check, gpus))
	gpus_ok = bool(gpu_results) and all(result.get("ok") for result in gpu_results)
	report = {
		"ok": jax_ok and count_ok and gpus_ok,
		"time": iso_now(),
		"host": socket.gethostname(),
		"checks": checks,
		"gpus": gpus,
		"per_gpu": gpu_results,
	}
	(settings.preflight_dir / "preflight.json").write_text(
		json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
	)
	text = lab_report.preflight_text(report)
	(settings.preflight_dir / "report.txt").write_text(text, encoding="utf-8")
	print(text)
	if report["ok"]:
		return StepResult("complete")
	return StepResult("failed", ("preflight failed; see preflight/report.txt",), PREFLIGHT_FAILED)


# Facts.


def _sha256(path: Path) -> str | None:
	if not path.exists():
		return None
	return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions(names: Sequence[str]) -> dict[str, str | None]:
	versions: dict[str, str | None] = {}
	for name in names:
		try:
			versions[name] = importlib.metadata.version(name)
		except importlib.metadata.PackageNotFoundError:
			versions[name] = None
	return versions


def _cpu_model() -> str:
	try:
		text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
	except OSError:
		return platform.processor() or "unknown"
	for line in text.splitlines():
		if line.startswith("model name"):
			return line.split(":", 1)[1].strip()
	return platform.processor() or "unknown"


def _ram_bytes() -> int | None:
	try:
		for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
			if line.startswith("MemTotal:"):
				return int(line.split()[1]) * 1024
	except (OSError, ValueError):
		return None
	return None


def _git(*args: str) -> str | None:
	try:
		completed = _run_quietly(["git", *args], timeout=30)
	except (OSError, subprocess.SubprocessError):
		return None
	return completed.stdout.strip() if completed.returncode == 0 else None


def _git_dirty() -> bool | None:
	status = _git("status", "--porcelain")
	return None if status is None else bool(status)


def _gpus_or_none(settings: Settings) -> list[dict[str, Any]] | None:
	try:
		return load_gpus(settings)
	except RuntimeError:
		return None


def step_facts(settings: Settings) -> StepResult:
	settings.logs.mkdir(parents=True, exist_ok=True)
	freeze = _run_quietly([sys.executable, "-m", "pip", "freeze"], timeout=300)
	(settings.lab_dir / "pip-freeze.txt").write_text(freeze.stdout, encoding="utf-8")
	_, full_text, _ = _nvidia_smi()
	(settings.logs / "nvidia-smi.txt").write_text(full_text, encoding="utf-8")
	kit_file = Path("KIT_COMMIT")
	variant, source = image_variant()
	facts = {
		"time_utc": iso_now(),
		"host": socket.gethostname(),
		"cpu": _cpu_model(),
		"cpu_count": os.cpu_count(),
		"ram_bytes": _ram_bytes(),
		"platform": platform.platform(),
		"python": platform.python_version(),
		"image_variant": variant,
		"image_variant_source": source,
		"image_agents": os.environ.get("POKEJAX_AGENTS"),
		"git_commit": _git("rev-parse", "HEAD"),
		"git_dirty": _git_dirty(),
		"kit_commit": kit_file.read_text(encoding="utf-8").strip() if kit_file.exists() else None,
		"packages": _package_versions(PACKAGES),
		"team_pool": str(settings.team_pool),
		"team_pool_sha256": _sha256(settings.team_pool),
		"lab_hours": settings.hours,
		"smoke": settings.smoke,
		"xla_settings": xla_settings(),
		"arms": [str(arm) for arm in settings.arms],
		"arm_sha256": {str(arm): _sha256(arm) for arm in settings.arms},
		"lab_bench": settings.bench,
		"panel_seconds_per_battle": PANEL_SECONDS_PER_BATTLE,
		"panel_safety": PANEL_SAFETY,
		"gpus": _gpus_or_none(settings),
	}
	(settings.lab_dir / "facts.json").write_text(
		json.dumps(facts, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
	)
	return StepResult("complete")


# Bench.


@dataclass(frozen=True)
class BenchItem:
	batch: int
	mode: str  # engine or env
	policy: str | None
	observation: bool
	steps: int
	repetitions: int

	@property
	def key(self) -> str:
		policy = self.policy or "random"
		observed = "obs" if self.observation else "noobs"
		return f"{self.mode}-{policy}-{observed}-b{self.batch}"

	def command(self, team_pool: Path) -> list[str]:
		command = [
			sys.executable,
			"-m",
			"pokejax.bench",
			"--json",
			"--batch-size",
			str(self.batch),
			"--steps",
			str(self.steps),
			"--repetitions",
			str(self.repetitions),
			"--team-pool",
			str(team_pool),
		]
		if self.mode == "env":
			command.append("--env")
		if self.policy:
			command += ["--policy", self.policy]
		if self.observation:
			command.append("--with-observation")
		return command


def bench_items(
	batches: Sequence[int], steps: int, repetitions: int, smoke: bool = False
) -> list[BenchItem]:
	"""The bench matrix: per batch, the engine and the environment (random and heuristics), each
	without and with observations. The smoke run keeps three of the six modes."""
	modes = (
		SMOKE_BENCH_MODES
		if smoke
		else tuple(
			(mode, policy, observed)
			for mode, policy in (("engine", None), ("env", None), ("env", "heuristics"))
			for observed in (False, True)
		)
	)
	return [
		BenchItem(batch, mode, policy, observed, steps, repetitions)
		for batch in batches
		for mode, policy, observed in modes
	]


def run_bench_shard(
	items: Sequence[BenchItem], *, gpu: int, out: Path, log_dir: Path, team_pool: Path, log
) -> None:
	done = {record["key"] for record in read_records(out) if "key" in record}
	for item in items:
		if item.key in done:
			continue
		measured = run_measurement(
			item.command(team_pool),
			gpu=gpu,
			log_path=log_dir / f"bench-{item.key}-gpu{gpu}.log",
			timeout=BENCH_TIMEOUT,
		)
		result = measured["result"]
		ok = measured["returncode"] == 0 and result is not None and "steps_per_second" in result
		record = {
			"kind": "bench",
			"step": "bench",
			"key": item.key,
			"gpu": gpu,
			"status": "ok" if ok else "failed",
			"batch": item.batch,
			"mode": item.mode,
			"policy": item.policy or "random",
			"observation": item.observation,
			"result": result,
			"returncode": measured["returncode"],
			"timed_out": measured["timed_out"],
			"wall_seconds": measured["wall_seconds"],
			"log": measured["log"],
		}
		if not ok:
			record["error"] = "timeout" if measured["timed_out"] else _last_error(measured["log"])
		append_record(out, record)
		log(f"bench gpu {gpu} {item.key}: {record['status']}")


def _last_error(log_path: str) -> str:
	try:
		lines = Path(log_path).read_text(encoding="utf-8").strip().splitlines()
	except OSError:
		return "no log"
	return _one_line(RuntimeError(lines[-1])) if lines else "empty log"


def _run_parallel(tasks: Sequence[Callable[[], Any]]) -> list[Any]:
	with ThreadPoolExecutor(max_workers=max(1, len(tasks))) as pool:
		futures = [pool.submit(task) for task in tasks]
		return [future.result() for future in futures]


def step_bench(settings: Settings) -> StepResult:
	gpus = load_gpus(settings)
	if settings.smoke:
		batches, steps, repetitions = (
			SMOKE_BENCH_BATCHES,
			SMOKE_BENCH_STEPS,
			SMOKE_BENCH_REPETITIONS,
		)
	else:
		batches, steps, repetitions = BENCH_BATCHES, BENCH_STEPS, BENCH_REPETITIONS
	items = bench_items(batches, steps, repetitions, smoke=settings.smoke)
	_run_parallel(
		[
			partial(
				run_bench_shard,
				shard(items, position, len(gpus)),
				gpu=int(gpu["index"]),
				out=settings.results / f"bench-gpu{gpu['index']}.jsonl",
				log_dir=settings.logs,
				team_pool=settings.team_pool,
				log=lambda text: log_progress(settings, "bench", "item", text=text),
			)
			for position, gpu in enumerate(gpus)
		]
	)
	records = [
		record
		for path in sorted(settings.results.glob("bench-gpu*.jsonl"))
		for record in read_records(path)
	]
	recorded = {record.get("key") for record in records}
	missing = [item.key for item in items if item.key not in recorded]
	failed = sorted({record["key"] for record in records if record.get("status") != "ok"})
	notes = [f"{len(records)} measurements recorded"]
	if failed:
		notes.append(f"failed: {', '.join(failed)}")
	return StepResult("complete" if not missing else "partial", tuple(notes))


# The batch fallback: a training probe that runs out of device memory is repeated with half the batch
# (rollout.batch_size), down to the floor of the caller. Every attempt is recorded.
BATCH_FLOOR = 4096
BATCH_LINE = re.compile(r"^batch_size = \d+[ 	]*(#.*)?$", re.MULTILINE)


class Fallback(NamedTuple):
	record: Mapping[str, Any]  # the last attempt's record: ok, or the failure that ended the ladder
	batch_size: int  # the batch of the last attempt: the batch the run used when its record is ok
	attempts: list[dict[str, Any]]  # one entry per attempt, in order


def batch_ladder(start: int, floor: int = BATCH_FLOOR) -> list[int]:
	"""The batches a fallback tries: start, then its halves, the last one clamped to the floor."""
	ladder = [start]
	while ladder[-1] > floor:
		ladder.append(max(floor, ladder[-1] // 2))
	return ladder


def is_out_of_memory(record: Mapping[str, Any]) -> bool:
	return record.get("status") != "ok" and record.get("error_class") == "oom"


def run_with_fallback(
	attempt: Callable[[int], Mapping[str, Any]], start: int, floor: int = BATCH_FLOOR
) -> Fallback:
	"""Run attempt(batch) down batch_ladder(start) while each attempt runs out of memory.

	The ladder ends at the first attempt that is not out of memory (ok, or another failure) and at
	the floor. The batch of the last attempt is the batch the run used.
	"""
	attempts: list[dict[str, Any]] = []
	for batch in batch_ladder(start, floor):
		record = attempt(batch)
		attempts.append(
			{
				"batch_size": batch,
				"status": record.get("status"),
				"error_class": record.get("error_class"),
				"error": record.get("error"),
			}
		)
		if not is_out_of_memory(record):
			break
	return Fallback(record, batch, attempts)


def write_batch_config(base: Path, batch: int, dest: Path) -> Path:
	"""Write the configuration base with rollout.batch_size set to batch, and check that it loads."""
	text = Path(base).read_text(encoding="utf-8")
	found = len(BATCH_LINE.findall(text))
	if found != 1:
		raise ConfigError(f"{base}: expected one batch_size line, found {found}")
	dest.parent.mkdir(parents=True, exist_ok=True)
	dest.write_text(BATCH_LINE.sub(f"batch_size = {batch}", text), encoding="utf-8")
	try:
		load_config(dest)  # a batch the trainer cannot use (not a multiple of 16, say) fails here
	except ConfigError:
		dest.unlink(missing_ok=True)
		raise
	return dest


# Sweep.


def sweep_cells(arm: Path, batches: Sequence[int]) -> list:
	"""The sizing cells of one arm: its own rollout and core, at each batch size of `batches`."""
	from pokejax.sizing import cell_config, grid_cells, parse_grid

	base = load_config(arm)
	cells = grid_cells(parse_grid("batch=" + ",".join(str(size) for size in batches)), base)
	for cell in cells:  # a size the arm rejects fails here, before any GPU time is spent
		cell_config(arm, cell)
	return cells


def sweep_jobs(arms: Sequence[Path], smoke: bool) -> list[tuple[Path, tuple[int, ...]]]:
	"""Each distinct arm with its batch sizes. The first arm (the base arm) also runs SWEEP_BASE_BATCH."""
	batches = SMOKE_SWEEP_BATCHES if smoke else SWEEP_BATCHES
	jobs = []
	for index, arm in enumerate(arms):
		sizes = batches + (SWEEP_BASE_BATCH,) if index == 0 and not smoke else batches
		jobs.append((arm, tuple(sizes)))
	return jobs


def _sweep_arms_on_gpu(settings: Settings, gpu: int, jobs, updates: int, deadline: float) -> None:
	from pokejax.sizing import run_sweep

	for arm, sizes in jobs:
		run_sweep(
			sweep_cells(arm, sizes),
			config_path=arm,
			team_pool=settings.team_pool,
			updates=updates,
			out=settings.results / f"sweep-{arm.stem}.jsonl",
			gpu=gpu,
			deadline=deadline,
			log_dir=settings.logs / f"sweep-{arm.stem}",
			log=lambda text: log_progress(settings, "sweep", "cell", text=text),
		)


def step_sweep(settings: Settings) -> StepResult:
	gpus = [int(gpu["index"]) for gpu in load_gpus(settings)]
	runs = assign_runs(settings.arms, gpus)
	gpu_of: dict[Path, int] = {}
	for run in runs:  # an arm is measured once, on the first GPU that trains it
		gpu_of.setdefault(run.arm, run.gpu)
	jobs = sweep_jobs(list(gpu_of), settings.smoke)
	updates = SMOKE_SWEEP_UPDATES if settings.smoke else SWEEP_UPDATES
	plan = make_plan(settings.hours, settings.start, time.time())
	log_progress(
		settings,
		"sweep",
		"deadline",
		seconds_from_now=round(plan.sweep_deadline - time.time()),
		arms=[str(arm) for arm, _ in jobs],
	)
	per_gpu: dict[int, list] = {gpu: [] for gpu in gpus}
	for arm, sizes in jobs:
		per_gpu[gpu_of[arm]].append((arm, sizes))
	_run_parallel(
		[
			partial(_sweep_arms_on_gpu, settings, gpu, items, updates, plan.sweep_deadline)
			for gpu, items in per_gpu.items()
			if items
		]
	)
	total = 0
	left: list[str] = []
	for arm, sizes in jobs:
		recorded = {
			r.get("key") for r in read_records(settings.results / f"sweep-{arm.stem}.jsonl")
		}
		for cell in sweep_cells(arm, sizes):
			total += 1
			if cell.key not in recorded:
				left.append(f"{arm.stem} {cell.key}")
	notes = [f"{total - len(left)} of {total} cells recorded (failed ones count)"]
	if left:
		notes.append(f"{len(left)} cells not measured yet (deadline or failure)")
		return StepResult("partial", tuple(notes))
	return StepResult("complete", tuple(notes))


# Training.


@dataclass(frozen=True)
class Run:
	"""One training run: an arm's configuration on one GPU. Its directory is named for both."""

	arm: Path
	gpu: int

	@property
	def name(self) -> str:
		return f"{self.arm.stem}-gpu{self.gpu}"


def assign_runs(arms: Sequence[Path], gpus: Sequence[int]) -> list[Run]:
	"""The runs of the arms on the GPUs.

	One configuration runs on every GPU (LAB_TRAIN_CONFIG, or a single LAB_ARMS entry). Several arms
	go round robin: arm i on GPU i modulo the number of GPUs, so with fewer GPUs than arms the arms of
	one GPU run one after another, each for an equal share of that GPU's time.
	"""
	if not arms or not gpus:
		raise ValueError("the lab needs at least one arm and one GPU")
	unique = list(dict.fromkeys(arms))
	if len(unique) == 1:
		return [Run(unique[0], gpu) for gpu in gpus]
	stems = [arm.stem for arm in unique]
	if len(set(stems)) != len(stems):
		raise ValueError(f"the arms need distinct file names: {', '.join(stems)}")
	return [Run(arm, gpus[index % len(gpus)]) for index, arm in enumerate(unique)]


def runs_on(runs: Sequence[Run], gpu: int) -> list[Run]:
	return [run for run in runs if run.gpu == gpu]


def unique_arms(runs: Sequence[Run]) -> list[Path]:
	return list(dict.fromkeys(run.arm for run in runs))


def doubling_checkpoints(
	updates: int, steps_per_update: int, base_steps: int = KEEP_BASE_STEPS
) -> list[int]:
	"""The updates whose checkpoints are kept: the first update at or after base_steps, then at each
	doubling of the environment steps below the last update, and the last update itself."""
	kept: list[int] = []
	steps = base_steps
	while True:
		update = math.ceil(steps / steps_per_update)
		if update >= updates:
			break
		kept.append(update)
		steps *= 2
	kept.append(updates)
	return kept


def panel_battles(kept: int, smoke: bool = False) -> int:
	"""The battles of one run's panel: each kept checkpoint against each opponent in both modes, and
	the final checkpoint again at the final size."""
	opponents = len(POLICY_NAMES) + len(REFERENCE_CHECKPOINTS)
	size = SMOKE_PANEL_BATTLES if smoke else PANEL_BATTLES
	final = SMOKE_FINAL_PANEL_BATTLES if smoke else FINAL_PANEL_BATTLES
	modes = len(MODE_NAMES)
	return kept * opponents * modes * size + opponents * modes * final


def pairs(contestants: int) -> int:
	"""The unordered pairs of a cross-play matrix (pokejax.train.crossplay plays each once)."""
	return contestants * (contestants - 1) // 2


def crossplay_battles(contestants: int, smoke: bool = False) -> int:
	"""The games of the cross-play matrix, as battles: each pair plays both modes, 2 games per draw."""
	draws = SMOKE_CROSSPLAY_DRAWS if smoke else CROSSPLAY_DRAWS
	return pairs(contestants) * len(MODE_NAMES) * 2 * draws


def panel_reserve_seconds(
	kept_by_run: Mapping[Run, int], gpus: Sequence[int], smoke: bool = False
) -> float:
	"""The seconds the training leaves for the panel: the GPU with the most panel work, the cross-play
	(run on the first GPU) included. Each run has two panel processes (its kept checkpoints, and its
	final one at the final size); the cross-play has one process and one compile per pair."""
	fixed = {gpu: 0.0 for gpu in gpus}
	battles = {gpu: 0 for gpu in gpus}
	for run, kept in kept_by_run.items():
		fixed[run.gpu] += 2 * PANEL_JOB_SECONDS
		battles[run.gpu] += panel_battles(kept, smoke)
	if gpus:
		contestants = len(kept_by_run) + len(REFERENCE_CHECKPOINTS)
		fixed[gpus[0]] += PANEL_JOB_SECONDS + pairs(contestants) * CROSSPLAY_PAIR_SECONDS
		battles[gpus[0]] += crossplay_battles(contestants, smoke)
	busiest = max(
		(fixed[gpu] + battles[gpu] * PANEL_SECONDS_PER_BATTLE for gpu in gpus), default=0.0
	)
	return PANEL_SAFETY * busiest


def probe_run(settings: Settings, run: Run) -> dict[str, Any]:
	"""A probe of the run's configuration on its GPU, at the first batch of the ladder that runs.

	A probe that runs is cached in the run directory (probe.json), so a rerun does not repeat it.
	"""
	from pokejax.sizing import run_cell_process

	run_dir = settings.runs / run.name
	cached = run_dir / "probe.json"
	if cached.exists():
		return json.loads(cached.read_text(encoding="utf-8"))
	start = load_config(run.arm).rollout.batch_size
	floor = TRAIN_FLOOR
	if settings.smoke:
		start, floor = min(start, SMOKE_TRAIN_BATCH), SMOKE_TRAIN_FLOOR
	probe_updates = SMOKE_PROBE_UPDATES if settings.smoke else PROBE_UPDATES

	def attempt(batch: int) -> dict[str, Any]:
		config = write_batch_config(run.arm, batch, run_dir / f"config-batch{batch}.toml")
		record = run_cell_process(
			key=f"train-probe-{run.name}-batch{batch}",
			cell=None,
			config_path=config,
			team_pool=settings.team_pool,
			updates=probe_updates,
			gpu=run.gpu,
			log_path=settings.logs / f"train-probe-{run.name}-batch{batch}.log",
		)
		append_record(
			settings.results / "train-probe.jsonl",
			{"kind": "probe", "run": run.name, "batch_size": batch, **record},
		)
		return record

	fallback = run_with_fallback(attempt, start, floor=floor)
	probe = fallback.record
	base = {"run": run.name, "gpu": run.gpu, "batch_size": fallback.batch_size}
	if probe.get("status") != "ok":
		return {
			**base,
			"status": "failed",
			"attempts": fallback.attempts,
			"error": probe.get("error"),
		}
	result = {
		**base,
		"status": "ok",
		"attempts": fallback.attempts,
		"config": str(run_dir / f"config-batch{fallback.batch_size}.toml"),
		"steps_per_update": fallback.batch_size * load_config(run.arm).rollout.rollout_length,
		"probe_update_seconds": probe["update_seconds"],
		"probe_cold_update_seconds": probe["cold_update_seconds"],
	}
	run_dir.mkdir(parents=True, exist_ok=True)
	cached.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return result


def plan_runs(
	settings: Settings,
	runs: Sequence[Run],
	probes: Mapping[str, Mapping[str, Any]],
	fixed: Mapping[str, Mapping[str, Any]],
	deadline: float,
	now: float,
) -> dict[str, dict[str, Any]]:
	"""The plan of each run that has a probe and no plan yet.

	A GPU's training window (from now until the deadline, less the panel reserve) is shared equally by
	the runs on that GPU, one after another. A run's updates fill its share after its cold update and
	its start-up. Its kept checkpoints follow from its updates, and the panel reserve from the kept
	checkpoints of every run, so the two are fitted together.
	"""
	gpus = sorted({run.gpu for run in runs})
	reserve = 0.0
	plans: dict[str, dict[str, Any]] = {}
	for _ in range(PLAN_ITERATIONS):
		plans = {}
		for gpu in gpus:
			active = [
				run
				for run in runs_on(runs, gpu)
				if run.name not in fixed and probes.get(run.name, {}).get("status") == "ok"
			]
			if not active:
				continue
			share = max(0.0, deadline - reserve - now) / len(active)
			for position, run in enumerate(active):
				probe = probes[run.name]
				if settings.smoke:
					updates = SMOKE_TRAIN_UPDATES
					base_steps = probe["steps_per_update"]  # smoke: a checkpoint at 1, 2, 4 updates
				else:
					available = share - probe["probe_cold_update_seconds"] - STARTUP_SECONDS
					updates = plan_updates(probe["probe_update_seconds"], available)
					base_steps = KEEP_BASE_STEPS
				kept = doubling_checkpoints(updates, probe["steps_per_update"], base_steps)
				plans[run.name] = {
					"run": run.name,
					"arm": str(run.arm),
					"gpu": run.gpu,
					"batch_size": probe["batch_size"],
					"attempts": probe["attempts"],
					"config": probe["config"],
					"steps_per_update": probe["steps_per_update"],
					"updates": updates,
					"decay_updates": updates,
					"keep": kept,
					"eval_every": 1 if settings.smoke else max(1, updates // EVALUATIONS_PER_RUN),
					"probe_update_seconds": probe["probe_update_seconds"],
					"probe_cold_update_seconds": probe["probe_cold_update_seconds"],
					"window_seconds": share,
					"window_end": now + share * (position + 1),
					"panel_reserve_seconds": None,
					"planned": iso_now(),
				}
		kept_by_run: dict[Run, int] = {}
		for run in runs:
			plan = plans.get(run.name) or fixed.get(run.name)
			if plan is not None:
				kept_by_run[run] = len(plan["keep"])
		new_reserve = panel_reserve_seconds(kept_by_run, gpus, settings.smoke)
		if abs(new_reserve - reserve) < 1.0:
			reserve = new_reserve
			break
		reserve = new_reserve
	for plan in plans.values():
		plan["panel_reserve_seconds"] = reserve
	return plans


def _plan_path(settings: Settings, run: Run) -> Path:
	return settings.runs / run.name / "lab-plan.json"


def _read_plan(settings: Settings, run: Run) -> dict[str, Any] | None:
	path = _plan_path(settings, run)
	return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _write_plan(settings: Settings, run: Run, plan: Mapping[str, Any]) -> None:
	path = _plan_path(settings, run)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _per_gpu(
	runs: Sequence[Run], gpus: Sequence[int], task: Callable[[Run], Any]
) -> dict[str, Any]:
	"""task(run) for every run: the runs of one GPU one after another, the GPUs in parallel."""
	groups = [runs_on(runs, gpu) for gpu in gpus]

	def group_task(group: Sequence[Run]) -> list[tuple[str, Any]]:
		return [(run.name, task(run)) for run in group]

	results = _run_parallel([partial(group_task, group) for group in groups if group])
	return {name: value for group in results for name, value in group}


def train_run(settings: Settings, run: Run, plan: Mapping[str, Any]) -> dict[str, Any]:
	"""A timed training run of one plan on its GPU, resumed when interrupted (same rows, same keep)."""
	run_dir = settings.runs / run.name
	run_dir.mkdir(parents=True, exist_ok=True)
	result_path = run_dir / "result.json"
	if result_path.exists():
		return {
			"status": "ok",
			"gpu": run.gpu,
			"run": run.name,
			"note": "already finished",
			"batch_size": plan.get("batch_size"),
			"attempts": plan.get("attempts", []),
		}
	resume = (run_dir / "checkpoint.npz").exists()
	if not resume:  # an interrupted start left no checkpoint: begin the run again
		for name in ("config.json", "metrics.jsonl"):
			(run_dir / name).unlink(missing_ok=True)
		for stale in run_dir.glob("checkpoint-*.npz"):
			stale.unlink()
	kept = ",".join(str(update) for update in plan["keep"])
	command = [
		sys.executable,
		"-m",
		"pokejax.train",
		"--config",
		plan["config"],
		"--team-pool",
		str(settings.team_pool),
		"--run-name",
		run.name,
		"--run-dir",
		str(run_dir),
		"--updates",
		str(plan["updates"]),
		"--eval-every",
		str(plan["eval_every"]),
		"--set",
		f"ppo.decay_updates={plan['decay_updates']}",
		"--set",
		f"checkpoint.keep=[{kept}]",
	]
	if settings.smoke:
		command += ["--checkpoint-every", "1"]
		for setting in SMOKE_TRAIN_SETTINGS:
			command += ["--set", setting]
	if resume:
		command.append("--resume")
	timeout = max(plan["window_end"] - time.time(), 60.0) + TRAIN_TIMEOUT_SLACK
	measured = run_measurement(
		command, gpu=run.gpu, log_path=settings.logs / f"train-{run.name}.log", timeout=timeout
	)
	ok = measured["returncode"] == 0 and result_path.exists()
	return {
		"status": "ok" if ok else "failed",
		"gpu": run.gpu,
		"run": run.name,
		"run_dir": str(run_dir),
		"plan": dict(plan),
		"batch_size": plan.get("batch_size"),
		"attempts": plan.get("attempts", []),
		"resumed": resume,
		"returncode": measured["returncode"],
		"timed_out": measured["timed_out"],
		"wall_seconds": measured["wall_seconds"],
		"log": measured["log"],
	}


def step_train(settings: Settings) -> StepResult:
	gpus = [int(gpu["index"]) for gpu in load_gpus(settings)]
	runs = assign_runs(settings.arms, gpus)
	deadline = make_plan(settings.hours, settings.start, time.time()).training_deadline
	out = settings.results / "train.jsonl"
	plans = {run.name: _read_plan(settings, run) for run in runs}
	pending = [run for run in runs if plans[run.name] is None]
	probes = _per_gpu(pending, gpus, lambda run: probe_run(settings, run))
	failures = []
	for run in pending:
		probe = probes[run.name]
		if probe["status"] != "ok":
			failures.append(run.name)
			append_record(out, {"kind": "training", "step": "train", "gpu": run.gpu, **probe})
	fixed = {name: plan for name, plan in plans.items() if plan is not None}
	new_plans = plan_runs(settings, runs, probes, fixed, deadline, time.time())
	for run in runs:
		if run.name in new_plans:
			plans[run.name] = new_plans[run.name]
			_write_plan(settings, run, new_plans[run.name])
	trainable = [run for run in runs if plans[run.name] is not None]
	records = _per_gpu(trainable, gpus, lambda run: train_run(settings, run, plans[run.name]))
	for run in trainable:
		append_record(out, {"kind": "training", "step": "train", **records[run.name]})
		if records[run.name]["status"] != "ok":
			failures.append(run.name)
	notes = tuple(
		f"{run.name}: batch {plans[run.name]['batch_size']}, {plans[run.name]['updates']} updates, "
		f"kept {len(plans[run.name]['keep'])} checkpoints ("
		f"{lab_report.attempts_text(plans[run.name]['attempts'])})"
		for run in runs
		if plans[run.name] is not None
	)
	if failures:
		return StepResult(
			"partial", (f"training failed or not planned: {', '.join(failures)}", *notes)
		)
	return StepResult("complete", (f"{len(runs)} training runs finished", *notes))


# Panel.


@dataclass(frozen=True)
class Job:
	"""One measurement of the panel step: a command on one GPU, done when its output file exists."""

	name: str
	gpu: int
	command: tuple[str, ...]
	out: Path
	timeout: float


def run_checkpoints(settings: Settings, run: Run, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
	"""The kept checkpoints of a run in update order, with their labels, steps and files."""
	run_dir = settings.runs / run.name
	return [
		{
			"update": update,
			"steps": update * plan["steps_per_update"],
			"label": checkpoint_label(run.name, update * plan["steps_per_update"]),
			"path": run_dir / f"checkpoint-{update}.npz",
		}
		for update in plan["keep"]
	]


def panel_command(
	settings: Settings,
	candidates: Sequence[tuple[str, Path]],
	*,
	battles: int,
	batch: int,
	out: Path,
	differences: Sequence[tuple[str, str]] = (),
) -> list[str]:
	command = [sys.executable, "-m", "pokejax.train.panel"]
	command += [f"{label}={path}" for label, path in candidates]
	for label, path in REFERENCE_CHECKPOINTS:
		command += ["--opponent", f"{label}={path}"]
	command += ["--battles", str(battles), "--batch", str(batch), "--seed", "0"]
	command += ["--team-pool", str(settings.team_pool), "--out", str(out)]
	for first, second in differences:
		command += ["--difference", first, second]
	return command


def crossplay_command(
	settings: Settings, contestants: Sequence[tuple[str, Path]], out: Path
) -> list[str]:
	draws = SMOKE_CROSSPLAY_DRAWS if settings.smoke else CROSSPLAY_DRAWS
	batch = SMOKE_CROSSPLAY_BATCH if settings.smoke else CROSSPLAY_BATCH
	command = [sys.executable, "-m", "pokejax.train.crossplay"]
	command += [f"{label}={path}" for label, path in [*contestants, *REFERENCE_CHECKPOINTS]]
	command += ["--draws", str(draws), "--batch", str(batch), "--seed", "0"]
	command += ["--team-pool", str(settings.team_pool), "--out", str(out)]
	return command


def _run_job(settings: Settings, job: Job) -> dict[str, Any]:
	if job.out.exists():
		return {"kind": "panel", "step": "panel", "job": job.name, "status": "ok", "note": "done"}
	job.out.parent.mkdir(parents=True, exist_ok=True)
	measured = run_measurement(
		list(job.command),
		gpu=job.gpu,
		log_path=settings.logs / f"panel-{job.name}.log",
		timeout=job.timeout,
	)
	ok = measured["returncode"] == 0 and job.out.exists()
	record = {
		"kind": "panel",
		"step": "panel",
		"job": job.name,
		"gpu": job.gpu,
		"status": "ok" if ok else "failed",
		"returncode": measured["returncode"],
		"timed_out": measured["timed_out"],
		"wall_seconds": measured["wall_seconds"],
		"log": measured["log"],
		"out": str(job.out),
	}
	append_record(settings.results / "panel.jsonl", record)
	log_progress(settings, "panel", "job", text=f"{job.name} gpu {job.gpu}: {record['status']}")
	return record


def step_panel(settings: Settings) -> StepResult:
	gpus = [int(gpu["index"]) for gpu in load_gpus(settings)]
	runs = assign_runs(settings.arms, gpus)
	missing = [str(path) for _, path in REFERENCE_CHECKPOINTS if not path.exists()]
	if missing:
		message = (
			f"missing reference checkpoint(s): {', '.join(missing)}; lab/make-kit.sh copies "
			"them into the kit (work/reference-checkpoints)"
		)
		return StepResult("failed", (message,), 1)
	notes: list[str] = []
	jobs: dict[int, list[Job]] = {gpu: [] for gpu in gpus}
	finals: list[tuple[str, Path]] = []
	for run in runs:
		plan = _read_plan(settings, run)
		if plan is None or not (settings.runs / run.name / "result.json").exists():
			notes.append(f"{run.name}: not trained, no panel")
			continue
		kept = run_checkpoints(settings, run, plan)
		if any(not checkpoint["path"].exists() for checkpoint in kept):
			notes.append(f"{run.name}: a kept checkpoint is missing, no panel")
			continue
		candidates = [(checkpoint["label"], checkpoint["path"]) for checkpoint in kept]
		differences = [(kept[i]["label"], kept[i - 1]["label"]) for i in range(1, len(kept))]
		battles = SMOKE_PANEL_BATTLES if settings.smoke else PANEL_BATTLES
		final_battles = SMOKE_FINAL_PANEL_BATTLES if settings.smoke else FINAL_PANEL_BATTLES
		batch = SMOKE_PANEL_BATCH if settings.smoke else PANEL_BATCH
		expected = panel_battles(len(kept), settings.smoke)
		timeout = PANEL_TIMEOUT_SECONDS + 2 * expected * PANEL_SECONDS_PER_BATTLE
		out = settings.panel_dir / f"{run.name}.json"
		command = panel_command(
			settings,
			candidates,
			battles=battles,
			batch=batch,
			out=out,
			differences=differences,
		)
		jobs[run.gpu].append(Job(run.name, run.gpu, tuple(command), out, timeout))
		final = kept[-1]
		final_out = settings.panel_dir / f"{run.name}-final.json"
		final_command = panel_command(
			settings,
			[(final["label"], final["path"])],
			battles=final_battles,
			batch=batch,
			out=final_out,
		)
		jobs[run.gpu].append(
			Job(f"{run.name}-final", run.gpu, tuple(final_command), final_out, timeout)
		)
		finals.append((final["label"], final["path"]))
	if len(finals) + len(REFERENCE_CHECKPOINTS) >= 2:
		cross_out = settings.panel_dir / "crossplay.json"
		jobs[gpus[0]].append(
			Job(
				"crossplay",
				gpus[0],
				tuple(crossplay_command(settings, finals, cross_out)),
				cross_out,
				PANEL_TIMEOUT_SECONDS
				+ 2
				* crossplay_battles(len(finals) + len(REFERENCE_CHECKPOINTS), settings.smoke)
				* PANEL_SECONDS_PER_BATTLE,
			)
		)
	if not finals:
		return StepResult("failed", ("no trained run to evaluate", *notes), 1)
	results = _per_gpu_jobs(settings, jobs)
	failed = [record["job"] for record in results if record["status"] != "ok"]
	if failed:
		return StepResult("partial", (f"panel jobs failed: {', '.join(failed)}", *notes))
	return StepResult("complete", (f"{len(results)} panel jobs done", *notes))


def _per_gpu_jobs(settings: Settings, jobs: Mapping[int, Sequence[Job]]) -> list[dict[str, Any]]:
	groups = [list(items) for items in jobs.values() if items]
	results = _run_parallel([partial(_run_job_list, settings, group) for group in groups])
	return [record for group in results for record in group]


def _run_job_list(settings: Settings, group: Sequence[Job]) -> list[dict[str, Any]]:
	return [_run_job(settings, job) for job in group]


# Pack.


def archive_name(settings: Settings, host: str) -> Path:
	date = datetime.fromtimestamp(settings.start, UTC).strftime("%Y-%m-%d")
	safe = "".join(ch if ch.isalnum() or ch == "-" else "-" for ch in host) or "host"
	folder = Path("work/lab-results")
	candidate = folder / f"{date}-{safe}.tar.gz"
	number = 2
	while candidate.exists():
		candidate = folder / f"{date}-{safe}-{number}.tar.gz"
		number += 1
	return candidate


def _archive_filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
	return None if info.name.endswith((".tmp", ".old")) else info


def make_plots(settings: Settings) -> list[str]:
	"""The training curves as SVG (pokejax.train.plot): each opponent the runs evaluated, by environment
	steps and by wall minutes, over every run that wrote evaluations. Returns the names that failed."""
	run_dirs = [
		path for path in sorted(settings.runs.glob("*-gpu*")) if (path / "metrics.jsonl").exists()
	]
	if not run_dirs:
		return []
	evaluated: set[str] | None = None
	for path in run_dirs:
		config = _load_json(path / "config.json")
		names = set(config.get("evaluation", {}).get("opponents", []))
		evaluated = names if evaluated is None else evaluated & names
	opponents = [name for name in PLOT_OPPONENTS if name in (evaluated or set())]
	failed = []
	for opponent in opponents:
		for x_mode in ("steps", "minutes"):
			name = f"{opponent}-by-{x_mode}.svg"
			command = [
				sys.executable,
				"-m",
				"pokejax.train.plot",
				*[str(path) for path in run_dirs],
				"--opponent",
				opponent,
				"--x",
				x_mode,
				"--out",
				str(settings.lab_dir / "plots" / name),
			]
			completed = _run_quietly(command, timeout=900)
			if completed.returncode != 0:
				failed.append(name)
	return failed


def _load_json(path: Path) -> dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}


def step_pack(settings: Settings) -> StepResult:
	failed_plots = make_plots(settings)
	summary = lab_report.summary_markdown(settings.lab_dir)
	(settings.lab_dir / "summary.md").write_text(summary, encoding="utf-8")
	archive = archive_name(settings, socket.gethostname())
	archive.parent.mkdir(parents=True, exist_ok=True)
	with tarfile.open(archive, "w:gz") as tar:
		tar.add(settings.lab_dir, arcname="lab", filter=_archive_filter)
		kit = Path("KIT_COMMIT")
		if kit.exists():
			tar.add(kit, arcname="KIT_COMMIT")
	notes = (f"archive {archive}",)
	if failed_plots:
		return StepResult("partial", (f"figures failed: {', '.join(failed_plots)}", *notes))
	return StepResult("complete", notes)


STEP_FUNCTIONS: dict[str, Callable[[Settings], StepResult]] = {
	"preflight": step_preflight,
	"facts": step_facts,
	"bench": step_bench,
	"sweep": step_sweep,
	"train": step_train,
	"panel": step_panel,
	"pack": step_pack,
}


def run_step(settings: Settings, name: str) -> int:
	started = time.time()
	log_progress(settings, name, "start")
	try:
		result = STEP_FUNCTIONS[name](settings)
	except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
		result = StepResult("failed", (_one_line(error),), 1)
	seconds = time.time() - started
	if result.status == "complete":
		mark_done(settings, name, result, seconds)
	log_progress(settings, name, result.status, seconds=round(seconds, 1), notes=list(result.notes))
	if result.status == "complete":
		return 0
	if result.status == "partial":
		return PARTIAL_EXIT
	return result.exit_code or 1


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(prog="python -m pokejax.lab", description=__doc__)
	commands = parser.add_subparsers(dest="command", required=True)
	step = commands.add_parser("step", help="run one step of the lab kit")
	step.add_argument("name", choices=STEPS)
	is_done_cmd = commands.add_parser("is-done", help="exit 0 when the step is done")
	is_done_cmd.add_argument("name", choices=STEPS)
	gpu = commands.add_parser("preflight-gpu", help=argparse.SUPPRESS)
	gpu.add_argument("--index", required=True)
	gpu.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	gpu.set_defaults(run=gpu_probe_main)
	jax_probe = commands.add_parser("preflight-jax", help=argparse.SUPPRESS)
	jax_probe.add_argument("--environment-out", type=Path, required=True)
	jax_probe.set_defaults(run=jax_probe_main)
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	if args.command == "step":
		return run_step(settings_from_env(), args.name)
	if args.command == "is-done":
		return 0 if is_done(settings_from_env(), args.name) else 1
	return args.run(args)


if __name__ == "__main__":
	raise SystemExit(main())
