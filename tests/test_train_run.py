"""The training loop on the CPU: metric lines, evaluations, checkpoints and a resume."""

import json

import jax
import pytest
from train_support import on_cpu, tiny_config, write_pool

from pokejax.train.run import RunError, run_training


@pytest.fixture(scope="module", autouse=True)
def cpu():
	with on_cpu():
		yield
	# Compiled executables of a module are not needed after it; a worker keeps them otherwise.
	jax.clear_caches()


def records(run_dir):
	lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
	return [json.loads(line) for line in lines]


def test_a_tiny_run_finishes_and_resumes(tables, tmp_path):
	run_dir = tmp_path / "run"
	pool = write_pool(tmp_path, tables)
	options = [f"run.run_dir={run_dir}", "logging.every=1", "evaluation.every=2"]
	lines = []
	result = run_training(
		tiny_config("run.updates=2", "checkpoint.every=1", *options),
		team_pool=pool,
		log=lines.append,
	)
	assert result["environment_steps"] == 2 * 4 * 8
	assert (run_dir / "config.json").exists() and (run_dir / "checkpoint.npz").exists()
	assert json.loads((run_dir / "result.json").read_text())["updates"] == 2
	assert [record["kind"] for record in records(run_dir)] == [
		"update",
		"update",
		"evaluation",
		"final_evaluation",
	]
	update = records(run_dir)[1]
	for name in ("win_rate", "approx_kl", "clip_fraction", "explained_variance", "entropy"):
		assert name in update and update[name] == update[name]
	assert update["steps_per_second"] > 0 and update["environment_steps_per_second"] > 0
	evaluation = records(run_dir)[2]["results"]
	assert [r["opponent"] for r in evaluation] == ["random", "max-damage"]
	assert all(0 <= r["win_rate"] <= 1 and len(r["ci95"]) == 2 for r in evaluation)
	assert any("final evaluation" in line for line in lines)

	with pytest.raises(RunError, match="already holds a run"):
		run_training(tiny_config("run.updates=2", *options), team_pool=pool, log=lines.append)

	resumed = run_training(
		tiny_config("run.updates=3", "checkpoint.every=1", *options),
		team_pool=pool,
		resume=True,
		log=lines.append,
	)
	assert resumed["resumed_from_update"] == 2
	kinds = [(record["kind"], record["update"]) for record in records(run_dir)]
	assert kinds == [
		("update", 1),
		("update", 2),
		("evaluation", 2),
		("update", 3),
		("final_evaluation", 3),
	]


def test_resume_needs_a_checkpoint(tables, tmp_path):
	with pytest.raises(RunError, match="nothing to resume"):
		run_training(
			tiny_config(f"run.run_dir={tmp_path / 'missing'}"),
			team_pool=write_pool(tmp_path, tables),
			resume=True,
		)
