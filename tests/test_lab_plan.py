"""The lab kit's plan without a GPU: arms on GPUs, kept checkpoints at doubling step counts, the
training windows and the panel reserve, the sweep's jobs, the panel and cross-play commands, and the
report tables of a panel and cross-play run. The configurations are the committed ones; nothing trains."""

import itertools
import json
import time
from pathlib import Path

import pytest

from pokejax import lab, lab_report
from pokejax.lab import Run, assign_runs, doubling_checkpoints, panel_reserve_seconds
from pokejax.records import checkpoint_label, steps_label
from pokejax.train import crossplay
from pokejax.train.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE_ARM = ROOT / "configs" / "p7-pairwise.toml"
WIDE_ARM = ROOT / "configs" / "p7-pairwise-wide.toml"


def _settings(tmp_path: Path, **kwargs) -> lab.Settings:
	values = {
		"lab_dir": tmp_path / "lab",
		"hours": 12.0,
		"smoke": False,
		"start": 1_000_000.0,
		"team_pool": tmp_path / "pool.npz",
		"arms": (BASE_ARM, WIDE_ARM),
		"bench": False,
	}
	values.update(kwargs)
	return lab.Settings(**values)


def test_the_arm_configurations_load_and_differ_only_in_size():
	base, wide = load_config(BASE_ARM), load_config(WIDE_ARM)
	assert base.network.arch == wide.network.arch == "pairwise"
	assert base.rollout.batch_size == wide.rollout.batch_size == 4096
	assert (
		base.auxiliary == wide.auxiliary and base.ppo == wide.ppo and base.opponent == wide.opponent
	)
	assert (base.network.width, base.network.pair_width, base.network.core_width) == (96, 64, 128)
	assert (wide.network.width, wide.network.pair_width, wide.network.core_width) == (192, 96, 256)
	assert wide.network.context_width == wide.network.core_width == 256
	assert base.evaluation.every == 500 and base.checkpoint.every == 250
	p6 = load_config(ROOT / "configs" / "p6-3060ti.toml")
	assert p6.rollout.batch_size == 512 and p6.network.width == 96 and p6.run.updates == 14650


def test_one_arm_runs_on_every_gpu_and_several_arms_go_round_robin():
	replicas = assign_runs([BASE_ARM], [0, 1])
	assert [(run.arm, run.gpu) for run in replicas] == [(BASE_ARM, 0), (BASE_ARM, 1)]
	assert [run.name for run in replicas] == ["p7-pairwise-gpu0", "p7-pairwise-gpu1"]
	two = assign_runs([BASE_ARM, WIDE_ARM], [0, 1])
	assert [(run.arm.stem, run.gpu) for run in two] == [
		("p7-pairwise", 0),
		("p7-pairwise-wide", 1),
	]
	one = assign_runs([BASE_ARM, WIDE_ARM], [0])  # fewer GPUs than arms: they share GPU 0
	assert [run.gpu for run in one] == [0, 0]
	with pytest.raises(ValueError):
		assign_runs([BASE_ARM, ROOT / "other" / "p7-pairwise.toml"], [0, 1])
	with pytest.raises(ValueError):
		assign_runs([], [0])


def test_arms_come_from_the_environment_with_the_old_variable_as_a_fallback():
	default = lab.settings_from_env({})
	assert default.arms == lab.DEFAULT_ARMS and not default.bench
	listed = lab.settings_from_env({"LAB_ARMS": "configs/a.toml, configs/b.toml", "LAB_BENCH": "1"})
	assert listed.arms == (Path("configs/a.toml"), Path("configs/b.toml")) and listed.bench
	single = lab.settings_from_env({"LAB_TRAIN_CONFIG": "configs/rtx6000.toml"})
	assert single.arms == (Path("configs/rtx6000.toml"),)


def test_kept_checkpoints_double_from_thirty_million_steps_and_end_with_the_last_update():
	per_update = 4096 * 64  # 262,144 environment steps per update
	kept = doubling_checkpoints(600_000, per_update)
	steps = [update * per_update for update in kept]
	assert kept[0] == 115 and steps[0] >= 30_000_000 > steps[0] - per_update
	for a, b in itertools.pairwise(kept[:-1]):  # each kept count doubles (rounded up)
		assert abs(b - 2 * a) <= 1
	assert kept[-1] == 600_000
	assert doubling_checkpoints(100, per_update) == [100]  # shorter than the first doubling
	smoke = doubling_checkpoints(4, per_update, base_steps=per_update)
	assert smoke == [1, 2, 4]
	assert all(a < b for a, b in itertools.pairwise(kept))
	assert len(kept) == len(set(kept))


def test_labels_name_the_run_and_its_steps():
	assert steps_label(30_146_560) == "30.1M" and steps_label(262_144) == "262K"
	assert checkpoint_label("p7-pairwise-gpu0", 480_000_000) == "p7-pairwise-gpu0@480.0M"


def test_panel_reserve_counts_the_busiest_gpu_and_the_cross_play_on_the_first():
	one, two = Run(BASE_ARM, 0), Run(WIDE_ARM, 1)
	per_battle, job = lab.PANEL_SECONDS_PER_BATTLE, lab.PANEL_JOB_SECONDS
	assert lab.panel_battles(3) == 3 * 6 * 2 * lab.PANEL_BATTLES + 6 * 2 * lab.FINAL_PANEL_BATTLES
	assert lab.pairs(4) == 6
	cross = lab.crossplay_battles(4)  # the two finals and the two references: six pairs
	assert cross == 6 * 2 * 2 * lab.CROSSPLAY_DRAWS
	reserve = panel_reserve_seconds({one: 5, two: 3}, [0, 1])
	# GPU 0: a run of five checkpoints (two processes) and the cross-play (one process, six pairs)
	gpu0 = 2 * job + lab.panel_battles(5) * per_battle
	gpu0 += job + 6 * lab.CROSSPLAY_PAIR_SECONDS + cross * per_battle
	gpu1 = 2 * job + lab.panel_battles(3) * per_battle
	assert reserve == pytest.approx(lab.PANEL_SAFETY * max(gpu0, gpu1))
	assert gpu0 > gpu1
	only_refs = panel_reserve_seconds({}, [0])  # no finals: the cross-play of the references alone
	assert only_refs == pytest.approx(
		lab.PANEL_SAFETY
		* (job + lab.pairs(2) * lab.CROSSPLAY_PAIR_SECONDS + lab.crossplay_battles(2) * per_battle)
	)


def _probe(batch: int, steps_per_update: int = 262144, seconds: float = 2.0) -> dict:
	return {
		"run": "x",
		"gpu": 0,
		"status": "ok",
		"batch_size": batch,
		"attempts": [{"batch_size": batch, "status": "ok"}],
		"config": f"config-batch{batch}.toml",
		"steps_per_update": steps_per_update,
		"probe_update_seconds": seconds,
		"probe_cold_update_seconds": 20.0,
	}


def test_plan_fits_each_run_into_its_share_of_the_window_with_the_panel_reserve(tmp_path):
	settings = _settings(tmp_path, hours=12.0)
	runs = assign_runs([BASE_ARM, WIDE_ARM], [0, 1])
	now = 1_000_000.0
	deadline = now + 10 * 3600.0
	probes = {run.name: _probe(4096) for run in runs}
	plans = lab.plan_runs(settings, runs, probes, {}, deadline, now)
	base = plans["p7-pairwise-gpu0"]
	# updates fill the share: share = window less the reserve; each update costs 2.0 s x OVERHEAD
	reserve = base["panel_reserve_seconds"]
	share = deadline - reserve - now
	assert base["window_seconds"] == pytest.approx(share)
	assert base["updates"] == lab.plan_updates(2.0, share - 20.0 - lab.STARTUP_SECONDS)
	assert base["decay_updates"] == base["updates"]
	assert base["keep"] == doubling_checkpoints(base["updates"], 262144)
	assert base["eval_every"] == max(1, base["updates"] // lab.EVALUATIONS_PER_RUN)
	assert base["window_end"] == pytest.approx(now + share)
	kept_by_run = {run: len(plans[run.name]["keep"]) for run in runs}
	assert reserve == pytest.approx(panel_reserve_seconds(kept_by_run, [0, 1]))
	assert plans["p7-pairwise-wide-gpu1"]["window_end"] == pytest.approx(now + share)


def test_two_runs_on_one_gpu_share_its_window_and_end_one_after_the_other(tmp_path):
	settings = _settings(tmp_path, hours=12.0)
	runs = assign_runs([BASE_ARM, WIDE_ARM], [0])
	now = 0.0
	plans = lab.plan_runs(
		settings, runs, {run.name: _probe(4096) for run in runs}, {}, 36_000.0, now
	)
	first, second = plans["p7-pairwise-gpu0"], plans["p7-pairwise-wide-gpu0"]
	assert second["window_end"] == pytest.approx(2 * first["window_end"])
	assert first["window_seconds"] == pytest.approx(second["window_seconds"])


def test_smoke_plans_a_fixed_run_with_checkpoints_at_one_two_and_four_updates(tmp_path):
	settings = _settings(tmp_path, smoke=True)
	runs = assign_runs([BASE_ARM], [0])
	plans = lab.plan_runs(settings, runs, {runs[0].name: _probe(1024)}, {}, 1e6, 0.0)
	plan = plans[runs[0].name]
	assert plan["updates"] == lab.SMOKE_TRAIN_UPDATES == 4
	assert plan["keep"] == [1, 2, 4] and plan["eval_every"] == 1


def test_a_failed_probe_gets_no_plan_and_the_others_still_do(tmp_path):
	settings = _settings(tmp_path)
	runs = assign_runs([BASE_ARM, WIDE_ARM], [0, 1])
	probes = {runs[0].name: {"status": "failed"}, runs[1].name: _probe(1024)}
	plans = lab.plan_runs(settings, runs, probes, {}, 1e6, 0.0)
	assert set(plans) == {runs[1].name}


def test_sweep_jobs_give_the_base_arm_the_largest_size_only(tmp_path):
	jobs = lab.sweep_jobs([BASE_ARM, WIDE_ARM], smoke=False)
	assert jobs[0] == (BASE_ARM, (1024, 2048, 4096, 8192, 16384))
	assert jobs[1] == (WIDE_ARM, (1024, 2048, 4096, 8192))
	smoke = lab.sweep_jobs([BASE_ARM, WIDE_ARM], smoke=True)
	assert all(sizes == (512,) for _, sizes in smoke)


def test_sweep_cells_are_the_arm_at_each_batch_size():
	cells = lab.sweep_cells(BASE_ARM, (1024, 4096))
	assert [cell.batch for cell in cells] == [1024, 4096]
	assert {(cell.rollout, cell.width, cell.core) for cell in cells} == {(64, 1, "gru")}


def test_panel_and_crossplay_commands_name_the_checkpoints_and_references(tmp_path):
	settings = _settings(tmp_path)
	candidates = [("a@1.0M", Path("a1.npz")), ("a@2.0M", Path("a2.npz"))]
	command = lab.panel_command(
		settings,
		candidates,
		battles=2048,
		batch=1024,
		out=Path("p.json"),
		differences=[("a@2.0M", "a@1.0M")],
	)
	assert "a@1.0M=a1.npz" in command and "a@2.0M=a2.npz" in command
	assert command.count("--opponent") == 2
	assert command[command.index("--battles") + 1] == "2048"
	assert command[command.index("--difference") + 1 : command.index("--difference") + 3] == [
		"a@2.0M",
		"a@1.0M",
	]
	cross = lab.crossplay_command(settings, candidates, Path("c.json"))
	assert cross[1:3] == ["-m", "pokejax.train.crossplay"]
	assert "ref-P2-40M=" + str(lab.REFERENCE_CHECKPOINTS[0][1]) in cross
	assert cross[cross.index("--draws") + 1] == str(lab.CROSSPLAY_DRAWS)


def test_crossplay_pairs_are_the_unordered_pairs_in_order():
	assert crossplay.pairs(4) == [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
	assert crossplay.pairs(1) == []


def test_panel_step_fails_at_once_without_the_reference_checkpoints(tmp_path, monkeypatch):
	settings = _settings(tmp_path)
	_gpu_list(settings, 1)
	monkeypatch.setattr(lab, "REFERENCE_CHECKPOINTS", (("ref", tmp_path / "missing.npz"),))
	result = lab.step_panel(settings)
	assert result.status == "failed" and "missing reference checkpoint" in result.notes[0]


def _gpu_list(settings, count):
	settings.preflight_dir.mkdir(parents=True, exist_ok=True)
	gpus = [{"index": index, "name": f"GPU {index}", "memory_mib": 8192} for index in range(count)]
	(settings.preflight_dir / "preflight.json").write_text(json.dumps({"ok": True, "gpus": gpus}))


def test_panel_step_runs_each_run_then_the_cross_play_and_skips_what_is_done(tmp_path, monkeypatch):
	settings = _settings(tmp_path)
	_gpu_list(settings, 2)
	refs = []
	for label, path in lab.REFERENCE_CHECKPOINTS:
		refs.append((label, tmp_path / path.name))
		(tmp_path / path.name).write_bytes(b"x")
	monkeypatch.setattr(lab, "REFERENCE_CHECKPOINTS", tuple(refs))
	runs = assign_runs([BASE_ARM, WIDE_ARM], [0, 1])
	for run in runs:
		run_dir = settings.runs / run.name
		run_dir.mkdir(parents=True)
		(run_dir / "result.json").write_text("{}")
		plan = {"keep": [1, 2], "steps_per_update": 10, "updates": 2}
		lab._write_plan(settings, run, plan)
		for update in plan["keep"]:
			(run_dir / f"checkpoint-{update}.npz").write_bytes(b"x")
	commands = []

	def fake(command, *, gpu, log_path, timeout):
		commands.append((gpu, command))
		out = Path(command[command.index("--out") + 1]) if "--out" in command else None
		if out is not None:
			out.parent.mkdir(parents=True, exist_ok=True)
			out.write_text("{}")
		return {"returncode": 0, "timed_out": False, "wall_seconds": 1.0, "log": str(log_path)}

	monkeypatch.setattr(lab, "run_measurement", fake)
	result = lab.step_panel(settings)
	assert result.status == "complete", result.notes
	# each run: the panel of its two kept checkpoints and the final; the cross-play on GPU 0
	assert [gpu for gpu, _ in commands].count(0) == 3 + 0
	assert [gpu for gpu, _ in commands].count(1) == 2
	assert any("pokejax.train.crossplay" in command for _, command in commands)
	again = lab.step_panel(settings)  # every output exists: nothing runs
	assert again.status == "complete" and len(commands) == 5


def test_report_tables_of_a_panel_and_cross_play_run(tmp_path):
	lab_dir = tmp_path / "lab"
	run = "p7-pairwise-gpu0"
	(lab_dir / "runs" / run).mkdir(parents=True)
	plan = {"keep": [1, 2], "steps_per_update": 262144, "updates": 2, "batch_size": 4096,
		"attempts": [{"batch_size": 4096, "status": "ok"}], "eval_every": 1, "gpu": 0,
		"window_seconds": 100.0, "panel_reserve_seconds": 50.0}  # fmt: skip
	(lab_dir / "runs" / run / "lab-plan.json").write_text(json.dumps(plan))
	metrics = [
		{"kind": "update", "update": 1, "seconds": 60.0},
		{"kind": "update", "update": 2, "seconds": 120.0},
	]
	(lab_dir / "runs" / run / "metrics.jsonl").write_text(
		"".join(json.dumps(m) + "\n" for m in metrics)
	)
	opponents = ["random", "random-move", "max-damage", "heuristics", "ref-P2-40M", "ref-P6-480M"]

	def cell(rate):
		return {
			"opponents": {name: {"win_rate": rate, "ci95": [rate, rate]} for name in opponents},
			"panel_mean": rate,
			"panel_ci95": {"estimate": rate, "ci95": [rate - 0.01, rate + 0.01]},
		}

	labels = [f"{run}@{steps_label(262144)}", f"{run}@{steps_label(2 * 262144)}"]
	results = {label: {"sampled": cell(0.6), "greedy": cell(0.62)} for label in labels}
	panel = {"battles": 512, "results": results, "differences": [{
		"a": labels[1], "b": labels[0], "mode": "sampled",
		"panel": {"estimate": 0.02, "ci95": [0.01, 0.03]},
		"opponents": {name: {"estimate": 0.02, "ci95": [0.0, 0.04]} for name in opponents},
	}]}  # fmt: skip
	(lab_dir / "panel").mkdir(parents=True)
	(lab_dir / "panel" / f"{run}.json").write_text(json.dumps(panel))
	(lab_dir / "panel" / f"{run}-final.json").write_text(json.dumps(
		{"battles": 1024, "results": {labels[1]: {"sampled": cell(0.7), "greedy": cell(0.7)}}}))  # fmt: skip
	pair = {"first": labels[0], "second": labels[1], "draws": 256, "games": 512, "modes": {
		mode: {"win_rate": 0.5, "ci95": [0.45, 0.55], "side0_win_rate": 0.5, "side1_win_rate": 0.5,
			"paired_draws": {"estimate": 0.5, "ci95": [0.4, 0.6]}, "truncated": 1,
			"mean_battle_length": 40.0, "battles": 512} for mode in ("sampled", "greedy")}}  # fmt: skip
	(lab_dir / "panel" / "crossplay.json").write_text(json.dumps(
		{"labels": labels, "pairs": [pair]}))  # fmt: skip
	text = lab_report.summary_markdown(lab_dir)
	assert "## Panel" in text and "Panel mean, sampled [95%]" in text
	assert f"{run}@{steps_label(262144)}" in text and "60.00" in text and "70.00" in text
	assert "paired differences between successive kept checkpoints" in text
	assert "+2.00 [+1.00, +3.00]" in text
	assert "## Cross-play" in text and "50.00 [45.00, 55.00]" in text
	assert "| 1 |" not in text.split("## Cross-play")[0].split("## Panel")[1][:0]
	plan_table = lab_report._plans_table(lab_dir)
	assert "1, 2" in plan_table and "p7-pairwise-gpu0" in plan_table


def test_probe_falls_back_to_half_the_rows_and_caches_the_plan_inputs(tmp_path, monkeypatch):
	from pokejax import sizing

	settings = _settings(tmp_path)
	run = Run(BASE_ARM, 0)
	seen = []

	def fake_cell(**kwargs):
		config = load_config(kwargs["config_path"])
		seen.append(config.rollout.batch_size)
		if config.rollout.batch_size > 1024:
			return {"status": "failed", "error_class": "oom", "error": "RESOURCE_EXHAUSTED"}
		return {"status": "ok", "error_class": None, "error": None,
			"update_seconds": 2.0, "cold_update_seconds": 20.0}  # fmt: skip

	monkeypatch.setattr(sizing, "run_cell_process", fake_cell)
	probe = lab.probe_run(settings, run)
	assert seen == [4096, 2048, 1024]
	assert probe["status"] == "ok" and probe["batch_size"] == 1024
	assert probe["steps_per_update"] == 1024 * 64
	assert [a["batch_size"] for a in probe["attempts"]] == [4096, 2048, 1024]
	assert (settings.runs / run.name / "probe.json").exists()
	seen.clear()
	assert lab.probe_run(settings, run) == probe and seen == []  # cached: no second probe


def test_probe_that_never_fits_is_a_failure_at_the_floor(tmp_path, monkeypatch):
	from pokejax import sizing

	settings = _settings(tmp_path)
	monkeypatch.setattr(
		sizing,
		"run_cell_process",
		lambda **kwargs: {"status": "failed", "error_class": "oom", "error": "RESOURCE_EXHAUSTED"},
	)
	probe = lab.probe_run(settings, Run(WIDE_ARM, 1))
	assert probe["status"] == "failed" and probe["batch_size"] == lab.TRAIN_FLOOR
	assert [a["batch_size"] for a in probe["attempts"]] == [4096, 2048, 1024]
	assert not (settings.runs / "p7-pairwise-wide-gpu1" / "probe.json").exists()


def test_train_run_command_sets_the_decay_horizon_and_the_kept_checkpoints(tmp_path, monkeypatch):
	settings = _settings(tmp_path, team_pool=tmp_path / "pool.npz")
	run = Run(BASE_ARM, 0)
	plan = {
		"config": str(tmp_path / "config-batch4096.toml"),
		"updates": 600,
		"decay_updates": 600,
		"keep": [115, 229, 600],
		"eval_every": 50,
		"batch_size": 4096,
		"attempts": [],
		"window_end": time.time() + 3600.0,
	}
	commands = []

	def fake(command, *, gpu, log_path, timeout):
		commands.append(command)
		(settings.runs / run.name / "result.json").write_text("{}")
		return {"returncode": 0, "timed_out": False, "wall_seconds": 1.0, "log": str(log_path)}

	monkeypatch.setattr(lab, "run_measurement", fake)
	record = lab.train_run(settings, run, plan)
	assert record["status"] == "ok"
	command = commands[0]
	assert command[command.index("--updates") + 1] == "600"
	assert command[command.index("--eval-every") + 1] == "50"
	sets = [command[i + 1] for i, part in enumerate(command) if part == "--set"]
	assert "ppo.decay_updates=600" in sets and "checkpoint.keep=[115,229,600]" in sets
	assert "--checkpoint-every" not in command and "--resume" not in command
	again = lab.train_run(settings, run, plan)  # a finished run is not trained again
	assert again["note"] == "already finished" and len(commands) == 1


def test_a_resumed_run_keeps_its_checkpoint_files_and_adds_the_resume_flag(tmp_path, monkeypatch):
	settings = _settings(tmp_path)
	run = Run(BASE_ARM, 0)
	run_dir = settings.runs / run.name
	run_dir.mkdir(parents=True)
	(run_dir / "checkpoint.npz").write_bytes(b"x")
	(run_dir / "checkpoint-115.npz").write_bytes(b"x")
	plan = {"config": "c.toml", "updates": 600, "decay_updates": 600, "keep": [115, 600],
		"eval_every": 50, "window_end": time.time() + 60.0}  # fmt: skip
	commands = []

	def fake(command, *, gpu, log_path, timeout):
		commands.append(command)
		return {"returncode": 1, "timed_out": True, "wall_seconds": 1.0, "log": str(log_path)}

	monkeypatch.setattr(lab, "run_measurement", fake)
	record = lab.train_run(settings, run, plan)
	assert record["status"] == "failed" and record["resumed"] is True
	assert "--resume" in commands[0]
	assert (run_dir / "checkpoint-115.npz").exists()


def test_a_fresh_start_clears_the_kept_checkpoints_of_an_aborted_run(tmp_path, monkeypatch):
	settings = _settings(tmp_path)
	run = Run(BASE_ARM, 0)
	run_dir = settings.runs / run.name
	run_dir.mkdir(parents=True)
	(run_dir / "checkpoint-115.npz").write_bytes(b"stale")
	plan = {"config": "c.toml", "updates": 600, "decay_updates": 600, "keep": [600],
		"eval_every": 50, "window_end": time.time() + 60.0}  # fmt: skip
	monkeypatch.setattr(
		lab,
		"run_measurement",
		lambda command, *, gpu, log_path, timeout: {
			"returncode": 1,
			"timed_out": False,
			"wall_seconds": 1.0,
			"log": str(log_path),
		},
	)
	lab.train_run(settings, run, plan)
	assert not (run_dir / "checkpoint-115.npz").exists()
