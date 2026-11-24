"""The lab kit's logic without a GPU: the time budget, the bench matrix, done markers, the preflight
report and the summary tables, built from synthetic results."""

import json
from pathlib import Path

import pytest

from pokejax import lab, lab_report
from pokejax.lab import (
	Settings,
	StepResult,
	archive_name,
	bench_items,
	is_done,
	make_plan,
	mark_done,
	minimum_driver,
	parse_gpu_query,
	plan_updates,
	settings_from_env,
)
from pokejax.train.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
RTX_CONFIG = ROOT / "configs" / "rtx6000.toml"


def _settings(tmp_path: Path, **kwargs) -> Settings:
	values = {
		"lab_dir": tmp_path / "lab",
		"hours": 1.0,
		"smoke": False,
		"start": 1_000_000.0,
		"team_pool": tmp_path / "pool.npz",
		"arms": (Path("configs/p7-pairwise.toml"),),
		"bench": False,
	}
	values.update(kwargs)
	return Settings(**values)


def test_plan_gives_the_sweep_a_share_of_what_is_left_and_training_the_rest():
	plan = make_plan(1.0, 1000.0, 1000.0)
	left = 3600 - 300  # less the pack reserve
	assert plan.sweep_deadline == pytest.approx(1000.0 + lab.SWEEP_SHARE * left)
	assert plan.training_deadline == pytest.approx(1000.0 + 3600 - 300)
	later = make_plan(1.0, 1000.0, 2000.0)  # a sweep that starts later gets less
	assert later.sweep_deadline - 2000.0 < plan.sweep_deadline - 1000.0
	short = make_plan(0.05, 0.0, 0.0)
	assert short.training_deadline == pytest.approx(180 - 18)  # the reserve is a tenth at most
	assert make_plan(1.0, 0.0, 5000.0).sweep_deadline == pytest.approx(5000.0)  # nothing left


def test_plan_updates_counts_the_updates_that_fit():
	assert plan_updates(2.0, 100.0) == int(100 / (2.0 * lab.OVERHEAD))
	assert plan_updates(2.0, -5.0) == 1
	assert plan_updates(0.0, 100.0) == 1
	assert plan_updates(1e-9, 1e6) > 1


def test_bench_matrix_covers_both_modes_and_policies():
	items = bench_items((4096, 16384), steps=64, repetitions=3)
	assert len(items) == 2 * 6
	assert len(bench_items((4096,), steps=16, repetitions=1, smoke=True)) == 3
	assert len({item.key for item in items}) == len(items)
	heuristics = [item for item in items if item.policy == "heuristics"]
	assert {item.mode for item in heuristics} == {"env"}
	command = next(i for i in items if i.mode == "env" and i.observation).command(Path("p.npz"))
	assert "--env" in command and "--with-observation" in command
	engine = next(i for i in items if i.mode == "engine" and not i.observation)
	assert "--env" not in engine.command(Path("p.npz"))
	assert "--policy" not in engine.command(Path("p.npz"))


def test_settings_come_from_the_environment():
	settings = settings_from_env({"LAB_HOURS": "0.5", "LAB_SMOKE": "1", "LAB_START": "7"})
	assert settings.hours == 0.5 and settings.smoke and settings.start == 7.0
	assert settings.lab_dir == Path("work/lab")
	default = settings_from_env({})
	assert default.hours == 3.0 and not default.smoke


def test_done_marker_counts_only_complete_steps(tmp_path):
	settings = _settings(tmp_path)
	assert not is_done(settings, "sweep")
	mark_done(settings, "sweep", StepResult("partial", ("2 cells left",)), 5.0)
	assert not is_done(settings, "sweep")
	mark_done(settings, "sweep", StepResult("complete"), 5.0)
	assert is_done(settings, "sweep")
	marker = json.loads((settings.done_dir / "sweep.json").read_text())
	assert marker["status"] == "complete" and marker["seconds"] == 5.0


def test_archive_name_does_not_overwrite(tmp_path, monkeypatch):
	monkeypatch.chdir(tmp_path)
	settings = _settings(tmp_path)
	first = archive_name(settings, "my host")
	assert first.name.endswith("-my-host.tar.gz")
	first.parent.mkdir(parents=True)
	first.write_bytes(b"x")
	second = archive_name(settings, "my host")
	assert second != first and second.name.endswith("-my-host-2.tar.gz")


def test_minimum_driver_per_variant():
	assert minimum_driver("cuda13") == 580
	assert minimum_driver("cuda12") == 525
	assert minimum_driver("cuda12-driver550") == 525
	assert minimum_driver("unknown") is None


def test_gpu_query_parses_nvidia_smi_csv():
	text = "0, NVIDIA RTX 6000 Ada Generation, GPU-abc, 550.163.01, 49140, 8.9\n1, NVIDIA RTX 6000 Ada Generation, GPU-def, 550.163.01, 49140, 8.9\n"
	gpus = parse_gpu_query(text)
	assert [gpu["index"] for gpu in gpus] == ["0", "1"]
	assert gpus[0]["driver"] == "550.163.01" and gpus[1]["compute_capability"] == "8.9"
	assert parse_gpu_query("garbage") == []


def test_preflight_report_names_the_fallback_on_a_driver_failure():
	report = {
		"ok": False,
		"host": "h",
		"time": "t",
		"checks": {
			"image_variant": "cuda13",
			"driver": "550.163.01",
			"cuda_version_nvidia_smi": "12.4",
			"gpus_listed": 2,
			"nvidia_smi_error": None,
			"driver_ok_for_variant": False,
			"minimum_driver_for_variant": 580,
			"jax": {"ok": False, "error": "Unable to initialize backend 'cuda'", "devices": []},
		},
		"gpus": [],
		"per_gpu": [],
	}
	text = lab_report.preflight_text(report)
	assert text.startswith("Preflight report\nFAIL")
	assert "needs driver 580" in text and "cuda12" in text
	assert "Unable to initialize backend" in text


def test_preflight_report_passes_quietly():
	report = {
		"ok": True,
		"checks": {
			"image_variant": "cuda12",
			"driver": "550.163.01",
			"cuda_version_nvidia_smi": "12.4",
			"gpus_listed": 1,
			"nvidia_smi_error": None,
			"driver_ok_for_variant": True,
			"minimum_driver_for_variant": 525,
			"jax": {"ok": True, "devices": [{"platform": "gpu", "kind": "RTX", "id": 0}]},
		},
		"gpus": [{"index": "0", "name": "RTX", "memory_mib": "49140", "compute_capability": "8.9"}],
		"per_gpu": [
			{
				"index": "0",
				"jitted_product": {"ok": True},
				"engine_cold_compile": {"ok": True, "compile_seconds": 31.0},
			}
		],
	}
	text = lab_report.preflight_text(report)
	assert text.startswith("Preflight report\nPASS")
	assert "What to try" not in text


def test_run_step_exit_codes_and_markers(tmp_path, monkeypatch):
	settings = _settings(tmp_path)
	results = {
		"sweep": StepResult("partial", ("3 cells left",)),
		"panel": StepResult("complete"),
		"train": StepResult("failed", ("no run",), 1),
	}
	monkeypatch.setattr(
		lab, "STEP_FUNCTIONS", {name: (lambda s, r=result: r) for name, result in results.items()}
	)
	assert lab.run_step(settings, "sweep") == lab.PARTIAL_EXIT
	assert lab.run_step(settings, "panel") == 0
	assert lab.run_step(settings, "train") == 1
	assert not is_done(settings, "sweep") and is_done(settings, "panel")
	assert not is_done(settings, "train")
	events = [
		json.loads(line) for line in (settings.lab_dir / "progress.log").read_text().splitlines()
	]
	assert [event["event"] for event in events if event["step"] == "sweep"] == ["start", "partial"]


def test_bench_defaults_match_pokejax_bench():
	from pokejax import bench

	assert lab.BENCH_STEPS == bench.DEFAULT_STEPS
	assert lab.BENCH_REPETITIONS == bench.DEFAULT_REPETITIONS


def test_lab_processes_preallocate_unless_the_caller_set_it(monkeypatch):
	monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
	monkeypatch.delenv("XLA_PYTHON_CLIENT_MEM_FRACTION", raising=False)
	monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
	settings = lab.xla_settings()
	assert settings["CUDA_DEVICE_ORDER"] == {"value": "PCI_BUS_ID", "source": "lab default"}
	assert settings["XLA_PYTHON_CLIENT_PREALLOCATE"] == {"value": "true", "source": "lab default"}
	assert settings["XLA_PYTHON_CLIENT_MEM_FRACTION"]["value"] == "0.90"
	monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
	assert lab.xla_settings()["XLA_PYTHON_CLIENT_PREALLOCATE"] == {
		"value": "false",
		"source": "environment",
	}
	env = lab._jax_env("1")
	assert env["JAX_PLATFORMS"] == "cuda" and env["CUDA_VISIBLE_DEVICES"] == "1"
	assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_orchestrator_modules_do_not_load_jax():
	"""The lab's orchestrator and the sweep's parent must not initialize JAX: that would hold GPU memory
	that their child processes need (pokejax.sizing imports the engine only in the cell)."""
	import subprocess
	import sys

	code = (
		"import sys, pokejax.lab, pokejax.sizing, pokejax.lab_report, pokejax.train.config; "
		"print('jax' in sys.modules, any(m.startswith('pokejax.engine') for m in sys.modules))"
	)
	completed = subprocess.run(
		[sys.executable, "-c", code],
		capture_output=True,
		text=True,
		check=True,
		cwd=Path(__file__).resolve().parents[1],
	)
	assert completed.stdout.strip().splitlines()[-1] == "False False"


def _oom(batch):
	return {"status": "failed", "error_class": "oom", "error": "RESOURCE_EXHAUSTED", "batch": batch}


def test_batch_ladder_halves_down_to_the_floor():
	assert lab.batch_ladder(16384) == [16384, 8192, 4096]
	assert lab.batch_ladder(4096) == [4096]
	assert lab.batch_ladder(2048) == [2048]
	assert lab.batch_ladder(12288) == [12288, 6144, 4096]


def test_only_the_oom_class_falls_back():
	assert lab.is_out_of_memory({"status": "failed", "error_class": "oom"})
	assert not lab.is_out_of_memory({"status": "failed", "error_class": "crash"})
	assert not lab.is_out_of_memory({"status": "ok", "error_class": "oom"})


def test_fallback_stops_at_the_first_batch_that_runs():
	seen = []

	def attempt(batch):
		seen.append(batch)
		return (
			{"status": "ok", "error_class": None, "error": None} if batch <= 8192 else _oom(batch)
		)

	fallback = lab.run_with_fallback(attempt, 16384)
	assert seen == [16384, 8192]
	assert fallback.batch_size == 8192 and fallback.record["status"] == "ok"
	assert [(a["batch_size"], a["status"]) for a in fallback.attempts] == [
		(16384, "failed"),
		(8192, "ok"),
	]
	assert lab_report.attempts_text(fallback.attempts) == "16384 oom, 8192 ok"


def test_fallback_ends_at_the_floor_and_keeps_the_failure():
	seen = []

	def attempt(batch):
		seen.append(batch)
		return _oom(batch)

	fallback = lab.run_with_fallback(attempt, 16384)
	assert seen == [16384, 8192, 4096]
	assert fallback.batch_size == 4096 and fallback.record["status"] == "failed"
	assert len(fallback.attempts) == 3


def test_fallback_does_not_retry_other_failures():
	seen = []

	def attempt(batch):
		seen.append(batch)
		return {"status": "failed", "error_class": "timeout", "error": "timeout"}

	fallback = lab.run_with_fallback(attempt, 16384)
	assert seen == [16384] and fallback.batch_size == 16384


def test_write_batch_config_changes_only_the_batch(tmp_path):
	dest = tmp_path / "run" / "config-batch8192.toml"
	lab.write_batch_config(RTX_CONFIG, 8192, dest)
	base = RTX_CONFIG.read_text(encoding="utf-8")
	assert (
		dest.read_text(encoding="utf-8").replace("batch_size = 8192", "batch_size = 16384") == base
	)
	assert load_config(dest).rollout.batch_size == 8192


def test_write_batch_config_rejects_what_the_trainer_cannot_run(tmp_path):
	dest = tmp_path / "config-batch100.toml"
	with pytest.raises(ConfigError):
		lab.write_batch_config(RTX_CONFIG, 100, dest)
	assert not dest.exists()
	no_line = tmp_path / "no-batch.toml"
	no_line.write_text('[run]\nname = "x"\n', encoding="utf-8")
	with pytest.raises(ConfigError):
		lab.write_batch_config(no_line, 8192, tmp_path / "out.toml")


def _gpu_list(settings, count):
	settings.preflight_dir.mkdir(parents=True, exist_ok=True)
	gpus = [{"index": index, "name": f"GPU {index}", "memory_mib": 49152} for index in range(count)]
	(settings.preflight_dir / "preflight.json").write_text(json.dumps({"ok": True, "gpus": gpus}))


def test_summary_tables_from_synthetic_results(tmp_path):
	lab_dir = tmp_path / "lab"
	results = lab_dir / "results"
	results.mkdir(parents=True)
	bench = {
		"kind": "bench", "key": "engine-random-noobs-b4096", "gpu": 0, "status": "ok",
		"batch": 4096, "mode": "engine", "policy": "random", "observation": False,
		"result": {
			"steps_per_second": 3.0e6, "battles_per_second": 4.0e4,
			"compile_seconds": 29.0, "device_memory": {"peak_bytes_in_use": 2**30},
		},
	}  # fmt: skip
	failed = {
		"kind": "bench", "key": "env-random-obs-b131072", "gpu": 0, "status": "failed",
		"batch": 131072, "mode": "env", "policy": "random", "observation": True,
		"error": "RESOURCE_EXHAUSTED", "returncode": 1,
	}  # fmt: skip
	(results / "bench-gpu0.jsonl").write_text(json.dumps(bench) + "\n" + json.dumps(failed) + "\n")
	sweep = {
		"kind": "sweep", "gpu": 0, "status": "ok", "key": "k",
		"cell": {"batch": 1024, "rollout": 64, "width": 1, "core": "gru"},
		"parameters": 1000, "environment_steps_per_second": 2e6,
		"learner_steps_per_second": 5e4, "update_seconds": 1.0, "compile_seconds": 20.0,
		"device_memory": {"peak_bytes_in_use": 2**30},
	}  # fmt: skip
	(results / "sweep-gpu0.jsonl").write_text(json.dumps(sweep) + "\n")
	text = lab_report.summary_markdown(lab_dir)
	assert "## Benchmarks" in text and "3,000,000" in text and "RESOURCE_EXHAUSTED" in text
	assert "| gpu0 | 1024 | 2,000" not in text  # the parameters of the synthetic record are absent
	assert "| gpu0 | 1024 |" in text
	assert "## Training runs" in text
	assert "Benchmarks: not run" in text and "## Panel" in text and "## Cross-play" in text
