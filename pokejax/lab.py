"""The lab kit's steps: what lab/run.sh runs, one step per `python -m pokejax.lab step NAME`.

Steps, in order: preflight (the GPUs and JAX, first minute), facts (the system), bench (the engine and
the environment per GPU), sweep (the training sizing grid per GPU), contention (one GPU alone and both
at once), train (a timed run of configs/rtx6000.toml per GPU) and pack (summary.md and the archive).
`python -m pokejax.lab is-done NAME` exits 0 when a step has its done marker.

Outputs go to LAB_DIR (default work/lab): results/*.jsonl (one JSON line per measurement), logs/,
runs/ (the training run directories), preflight/, facts.json, pip-freeze.txt, progress.log (one JSON
line per event) and done/NAME.json (the marker of a finished step). A step is complete when all its
measurements are in its results; rerunning skips complete steps and the measurements already
recorded. The time budget (LAB_HOURS) is set by make_plan: when the sweep starts it may run for
SWEEP_SHARE of the time left after the contention and pack reserves, and the training ends before the
pack reserve. The sweep and the training shrink to fit; the other steps have fixed sizes.

A contention run or a training probe that runs out of device memory is repeated with half the batch
(rollout.batch_size), down to BATCH_FLOOR (4,096); every attempt is recorded, and the batch a run used is
in its plan (see run_with_fallback).

Environment (set by lab/run.sh): LAB_HOURS (3), LAB_SMOKE (0, every step at minimal size),
LAB_DIR (work/lab), LAB_START (epoch seconds when this run started), LAB_TEAM_POOL
(work/teams/gen5-teams.npz).
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
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
from pokejax.records import (
	ENVIRONMENT_DEFAULTS,
	append_record,
	read_records,
	run_measurement,
	shard,
	xla_environment,
)
from pokejax.train.config import ConfigError, load_config

# The same path as pokejax.env.DEFAULT_POOL. Importing pokejax.env (or anything that imports the
# engine) initializes the JAX backend, and the orchestrator must run when no GPU works.
DEFAULT_POOL = Path("work/teams/gen5-teams.npz")
STEPS = ("preflight", "facts", "bench", "sweep", "contention", "train", "pack")
GPU_STEPS = ("bench", "sweep", "contention", "train")
PREFLIGHT_FAILED = 3  # exit status of a failed preflight (lab/run.sh stops the GPU steps)
PARTIAL_EXIT = 4  # a step that ran but is not complete yet (a rerun continues it)
SWEEP_SHARE = 0.55  # of the time left once the sweep starts, less the reserves below
CONTENTION_RESERVE_SECONDS = 600.0
PACK_RESERVE_SECONDS = 300.0
STARTUP_SECONDS = 60.0  # process start, tables and pool before a training run's first update
OVERHEAD = 1.25  # per update: evaluations and checkpoints on top of the rollout and the learner
SELFPLAY_CONFIG = Path("configs/selfplay.toml")
RTX_CONFIG = Path("configs/rtx6000.toml")
BENCH_BATCHES = (4096, 16384, 65536, 131072)
# The steps per scan and the repetitions of pokejax.bench's defaults, so that the lab's numbers compare
# with the README's (test_lab checks the equality). Each item compiles cold; the rollouts dominate
# at the large batches.
BENCH_STEPS, BENCH_REPETITIONS, BENCH_TIMEOUT = 256, 3, 1800
SMOKE_BENCH_BATCHES, SMOKE_BENCH_STEPS, SMOKE_BENCH_REPETITIONS = (4096,), 16, 1
SMOKE_BENCH_MODES = (("engine", None, False), ("env", None, True), ("env", "heuristics", False))
SWEEP_UPDATES, SMOKE_SWEEP_UPDATES = 3, 1
SMOKE_SWEEP_GRID = "batch=1024 rollout=64 width=1 core=gru"
CONTENTION_UPDATES, SMOKE_CONTENTION_UPDATES = 8, 2
PROBE_UPDATES, SMOKE_PROBE_UPDATES = 3, 1
SMOKE_TRAIN_UPDATES = 3
SMOKE_TRAIN_SETTINGS = (
	"evaluation.batch=512",
	"evaluation.battles=512",
	"evaluation.final_battles=512",
	'evaluation.opponents=["random", "max-damage"]',
	"evaluation.greedy=false",
)
TRAIN_TIMEOUT_SLACK = 900.0
EVALUATIONS_PER_RUN = 6
CHECKPOINTS_PER_RUN = 4
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
	train_config: Path

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

	The sweep may start cells until SWEEP_SHARE of the time left (less the contention and pack
	reserves) has passed. The training ends before the pack reserve, which is at most a tenth of
	the budget. Contention has a fixed size and runs between the two.
	"""
	end = start + hours * 3600.0
	pack = min(PACK_RESERVE_SECONDS, 0.1 * hours * 3600.0)
	left = max(0.0, end - now - CONTENTION_RESERVE_SECONDS - pack)
	return Plan(start, end, now + SWEEP_SHARE * left, end - pack)


def plan_updates(update_seconds: float, seconds: float, overhead: float = OVERHEAD) -> int:
	"""The updates that fit in `seconds` when each takes `update_seconds` plus the overhead."""
	if update_seconds <= 0 or seconds <= 0:
		return 1
	return max(1, int(seconds / (update_seconds * overhead)))


def settings_from_env(environ: Mapping[str, str] | None = None) -> Settings:
	environ = os.environ if environ is None else environ
	return Settings(
		lab_dir=Path(environ.get("LAB_DIR", "work/lab")),
		hours=float(environ.get("LAB_HOURS", "3")),
		smoke=environ.get("LAB_SMOKE", "0") == "1",
		start=float(environ.get("LAB_START", time.time())),
		team_pool=Path(environ.get("LAB_TEAM_POOL", str(DEFAULT_POOL))),
		train_config=Path(environ.get("LAB_TRAIN_CONFIG", str(RTX_CONFIG))),
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
		"train_config": str(settings.train_config),
		"train_config_sha256": _sha256(settings.train_config),
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


# Sweep.


def step_sweep(settings: Settings) -> StepResult:
	gpus = load_gpus(settings)
	from pokejax.sizing import DEFAULT_GRID, cell_config, grid_cells, parse_grid, run_sweep

	plan = make_plan(settings.hours, settings.start, time.time())
	grid = SMOKE_SWEEP_GRID if settings.smoke else DEFAULT_GRID
	updates = SMOKE_SWEEP_UPDATES if settings.smoke else SWEEP_UPDATES
	cells = grid_cells(parse_grid(grid), load_config(SELFPLAY_CONFIG))
	for cell in cells:
		cell_config(SELFPLAY_CONFIG, cell)
	log_progress(
		settings,
		"sweep",
		"deadline",
		seconds_from_now=round(plan.sweep_deadline - time.time()),
		cells=len(cells),
	)
	_run_parallel(
		[
			partial(
				run_sweep,
				shard(cells, position, len(gpus)),
				config_path=SELFPLAY_CONFIG,
				team_pool=settings.team_pool,
				updates=updates,
				out=settings.results / f"sweep-gpu{gpu['index']}.jsonl",
				gpu=int(gpu["index"]),
				deadline=plan.sweep_deadline,
				log_dir=settings.logs,
				log=lambda text: log_progress(settings, "sweep", "cell", text=text),
			)
			for position, gpu in enumerate(gpus)
		]
	)
	recorded = {
		record.get("key")
		for path in settings.results.glob("sweep-gpu*.jsonl")
		for record in read_records(path)
	}
	left = [cell.key for cell in cells if cell.key not in recorded]
	notes = [f"{len(cells) - len(left)} of {len(cells)} cells recorded (failed ones count)"]
	if left:
		notes.append(f"{len(left)} cells not measured yet (deadline or failure)")
		return StepResult("partial", tuple(notes))
	return StepResult("complete", tuple(notes))


# Contention.


# The batch fallback: a training probe or a contention run that runs out of device memory is repeated
# with half the batch (rollout.batch_size), down to BATCH_FLOOR. Every attempt is recorded.
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


def _latest_by_phase(records: Sequence[Mapping[str, Any]]) -> dict[tuple, Mapping[str, Any]]:
	"""The last record of each (phase, GPU): a failed attempt is replaced by a later one."""
	return {(record.get("phase"), record.get("gpu")): record for record in records}


def _phase_done(latest: Mapping[tuple, Mapping[str, Any]], phase: str, gpus: Sequence[int]) -> bool:
	return all(latest.get((phase, gpu), {}).get("status") == "ok" for gpu in gpus)


def _append_contention(out: Path, record: dict[str, Any], **fields: Any) -> dict[str, Any]:
	record.update(kind="contention", **fields)
	append_record(out, record)
	return record


def _contention_config(settings: Settings, batch: int) -> Path:
	return write_batch_config(
		settings.train_config, batch, settings.runs / "contention" / f"config-batch{batch}.toml"
	)


def _alone_attempt(settings: Settings, out: Path, gpu: int, updates: int, batch: int) -> dict:
	from pokejax.sizing import run_cell_process

	record = run_cell_process(
		key=f"contention-alone-batch{batch}",
		cell=None,
		config_path=_contention_config(settings, batch),
		team_pool=settings.team_pool,
		updates=updates,
		gpu=gpu,
		log_path=settings.logs / f"contention-alone-gpu{gpu}-batch{batch}.log",
	)
	return _append_contention(out, record, phase="alone", gpus=[gpu], batch_size=batch)


def _both_attempt(
	settings: Settings, out: Path, gpus: Sequence[int], updates: int, batch: int
) -> dict[str, Any]:
	"""Both GPUs at once at batch: the attempt fails when either run does, and is retried at half."""
	from pokejax.sizing import run_cell_process

	config = _contention_config(settings, batch)  # written once, before the two runs read it
	records = _run_parallel(
		[
			partial(
				run_cell_process,
				key=f"contention-both-gpu{gpu}-batch{batch}",
				cell=None,
				config_path=config,
				team_pool=settings.team_pool,
				updates=updates,
				gpu=gpu,
				log_path=settings.logs / f"contention-both-gpu{gpu}-batch{batch}.log",
			)
			for gpu in gpus
		]
	)
	for record in records:
		_append_contention(out, record, phase="both", gpus=list(gpus), batch_size=batch)
	return _combined(records)


def _combined(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
	"""One record for runs in parallel: ok when all are, else the first failure (out of memory first)."""
	failed = [record for record in records if record.get("status") != "ok"]
	if not failed:
		return {"status": "ok", "error_class": None, "error": None}
	first = next((record for record in failed if is_out_of_memory(record)), failed[0])
	return {
		"status": "failed",
		"error_class": first.get("error_class"),
		"error": first.get("error"),
	}


def _fallback_note(label: str, fallback: Fallback) -> str:
	status = "ok" if fallback.record.get("status") == "ok" else "failed"
	attempts = lab_report.attempts_text(fallback.attempts)
	return f"{label} {status} at batch {fallback.batch_size} ({attempts})"


def step_contention(settings: Settings) -> StepResult:
	gpus = [int(gpu["index"]) for gpu in load_gpus(settings)]
	updates = SMOKE_CONTENTION_UPDATES if settings.smoke else CONTENTION_UPDATES
	out = settings.results / "contention.jsonl"
	start = load_config(settings.train_config).rollout.batch_size
	latest = _latest_by_phase(read_records(out))
	notes: list[str] = []
	alone_ok = _phase_done(latest, "alone", gpus[:1])
	alone_batch = start
	if alone_ok:
		alone_batch = int(latest[("alone", gpus[0])].get("batch_size") or start)
	else:
		fallback = run_with_fallback(
			partial(_alone_attempt, settings, out, gpus[0], updates), start
		)
		alone_ok = fallback.record.get("status") == "ok"
		alone_batch = fallback.batch_size
		notes.append(_fallback_note("alone", fallback))
	if len(gpus) < 2:
		if ("both", None) not in latest:
			append_record(
				out,
				{
					"kind": "contention",
					"phase": "both",
					"status": "skipped",
					"error": "needs two GPUs; this host has one",
					"gpus": gpus,
				},
			)
	elif alone_ok and not _phase_done(latest, "both", gpus[:2]):
		both_attempt = partial(_both_attempt, settings, out, gpus[:2], updates)
		fallback = run_with_fallback(both_attempt, alone_batch)
		notes.append(_fallback_note("both", fallback))
	latest = _latest_by_phase(read_records(out))
	failed = [
		key for key, record in latest.items() if record.get("status") not in ("ok", "skipped")
	]
	if failed:
		return StepResult("partial", (f"contention failed for {len(failed)} run(s)", *notes))
	complete = alone_ok and (("both", None) in latest or _phase_done(latest, "both", gpus[:2]))
	return StepResult("complete" if complete else "partial", tuple(notes))


# Training.


def _train_probe(
	settings: Settings, gpu: int, run_dir: Path, deadline: float
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
	"""The plan of a training run from a probe of its configuration, or the failure when no batch fits.

	The probe runs at each batch of the ladder until one runs. The plan names that batch and the
	config file of the run directory that holds it.
	"""
	from pokejax.sizing import run_cell_process

	start = load_config(settings.train_config).rollout.batch_size
	probe_updates = SMOKE_PROBE_UPDATES if settings.smoke else PROBE_UPDATES

	def attempt(batch: int) -> dict[str, Any]:
		config = write_batch_config(
			settings.train_config, batch, run_dir / f"config-batch{batch}.toml"
		)
		record = run_cell_process(
			key=f"train-probe-gpu{gpu}-batch{batch}",
			cell=None,
			config_path=config,
			team_pool=settings.team_pool,
			updates=probe_updates,
			gpu=gpu,
			log_path=settings.logs / f"train-probe-gpu{gpu}-batch{batch}.log",
		)
		append_record(
			settings.results / "train-probe.jsonl", {"kind": "probe", "batch_size": batch, **record}
		)
		return record

	fallback = run_with_fallback(attempt, start)
	probe = fallback.record
	if probe.get("status") != "ok":
		failure = {
			"status": "failed",
			"gpu": gpu,
			"batch_size": fallback.batch_size,
			"attempts": fallback.attempts,
			"error": probe.get("error"),
			"probe": probe,
		}
		return None, failure
	if settings.smoke:
		updates = SMOKE_TRAIN_UPDATES
	else:
		available = deadline - time.time() - probe["cold_update_seconds"] - STARTUP_SECONDS
		updates = plan_updates(probe["update_seconds"], available)
	plan = {
		"gpu": gpu,
		"batch_size": fallback.batch_size,
		"config": str(run_dir / f"config-batch{fallback.batch_size}.toml"),
		"attempts": fallback.attempts,
		"updates": updates,
		"eval_every": 1 if settings.smoke else max(1, updates // EVALUATIONS_PER_RUN),
		"checkpoint_every": 1 if settings.smoke else max(1, updates // CHECKPOINTS_PER_RUN),
		"probe_update_seconds": probe["update_seconds"],
		"probe_cold_update_seconds": probe["cold_update_seconds"],
		"deadline": deadline,
	}
	(run_dir / "lab-plan.json").write_text(
		json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
	)
	return plan, None


def train_one_gpu(settings: Settings, gpu: int, deadline: float) -> dict[str, Any]:
	"""A timed training run of the training configuration on one GPU, resumed when interrupted.

	The batch is the one of the first probe in the ladder that runs (see _train_probe), and the run
	uses that batch's config file. A plan that exists is used as it is: a rerun resumes the run.
	"""
	run_dir = settings.runs / f"{settings.train_config.stem}-gpu{gpu}"
	run_dir.mkdir(parents=True, exist_ok=True)
	plan_path = run_dir / "lab-plan.json"
	result_path = run_dir / "result.json"
	plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
	if result_path.exists() and plan is not None:
		return {
			"status": "ok",
			"gpu": gpu,
			"run_dir": str(run_dir),
			"note": "already finished",
			"batch_size": plan.get("batch_size"),
			"attempts": plan.get("attempts", []),
		}
	if plan is None:
		plan, failure = _train_probe(settings, gpu, run_dir, deadline)
		if plan is None:
			return failure or {"status": "failed", "gpu": gpu}
	resume = (run_dir / "checkpoint.npz").exists()
	if not resume:  # an interrupted start left no checkpoint: begin the run again
		for name in ("config.json", "metrics.jsonl"):
			(run_dir / name).unlink(missing_ok=True)
	command = [
		sys.executable,
		"-m",
		"pokejax.train",
		"--config",
		plan["config"],
		"--team-pool",
		str(settings.team_pool),
		"--run-name",
		f"{settings.train_config.stem}-gpu{gpu}",
		"--run-dir",
		str(run_dir),
		"--updates",
		str(plan["updates"]),
		"--eval-every",
		str(plan["eval_every"]),
		"--checkpoint-every",
		str(plan["checkpoint_every"]),
	]
	if settings.smoke:
		for setting in SMOKE_TRAIN_SETTINGS:
			command += ["--set", setting]
	if resume:
		command.append("--resume")
	timeout = max(deadline - time.time(), 60.0) + TRAIN_TIMEOUT_SLACK
	measured = run_measurement(
		command, gpu=gpu, log_path=settings.logs / f"train-gpu{gpu}.log", timeout=timeout
	)
	ok = measured["returncode"] == 0 and result_path.exists()
	return {
		"status": "ok" if ok else "failed",
		"gpu": gpu,
		"run_dir": str(run_dir),
		"plan": plan,
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
	plan = make_plan(settings.hours, settings.start, time.time())
	out = settings.results / "train.jsonl"
	records = _run_parallel(
		[partial(train_one_gpu, settings, gpu, plan.training_deadline) for gpu in gpus]
	)
	for record in records:
		append_record(out, {"kind": "training", "step": "train", **record})
	notes = tuple(
		f"GPU {record['gpu']}: batch {record.get('batch_size')} "
		f"({lab_report.attempts_text(record.get('attempts', []))})"
		for record in records
	)
	failed = [record["gpu"] for record in records if record["status"] != "ok"]
	if failed:
		return StepResult("partial", (f"training failed on GPU {failed}", *notes))
	return StepResult("complete", (f"{len(gpus)} training runs finished", *notes))


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


def step_pack(settings: Settings) -> StepResult:
	summary = lab_report.summary_markdown(settings.lab_dir)
	(settings.lab_dir / "summary.md").write_text(summary, encoding="utf-8")
	archive = archive_name(settings, socket.gethostname())
	archive.parent.mkdir(parents=True, exist_ok=True)
	with tarfile.open(archive, "w:gz") as tar:
		tar.add(settings.lab_dir, arcname="lab", filter=_archive_filter)
		kit = Path("KIT_COMMIT")
		if kit.exists():
			tar.add(kit, arcname="KIT_COMMIT")
	return StepResult("complete", (f"archive {archive}",))


STEP_FUNCTIONS: dict[str, Callable[[Settings], StepResult]] = {
	"preflight": step_preflight,
	"facts": step_facts,
	"bench": step_bench,
	"sweep": step_sweep,
	"contention": step_contention,
	"train": step_train,
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
