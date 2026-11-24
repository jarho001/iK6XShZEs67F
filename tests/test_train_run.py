"""The training loop: the run directory's files, a resume, and the refusals that need no training.

The runs that train (a few updates each) are marked large: each builds its own trainer and compiles
the update again, which is too much for the default suite.
"""

import json

import numpy as np
import pytest
from train_support import league_config, tiny_config, write_pool

from pokejax.train.run import RunError, _keep_checkpoint, run_training

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


def records(run_dir):
	lines = (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
	return [json.loads(line) for line in lines]


def test_a_run_directory_that_holds_a_run_is_not_overwritten(tmp_path):
	(tmp_path / "config.json").write_text("{}")
	with pytest.raises(RunError, match="already holds a run"):
		run_training(tiny_config(f"run.run_dir={tmp_path}"), team_pool=tmp_path / "none.npz")


def test_resume_needs_a_checkpoint(tmp_path):
	with pytest.raises(RunError, match="nothing to resume"):
		run_training(
			tiny_config(f"run.run_dir={tmp_path / 'missing'}"),
			team_pool=tmp_path / "none.npz",
			resume=True,
		)


@pytest.mark.large
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


@pytest.mark.large
def test_a_league_run_reports_every_group_and_each_baseline_and_action_policy(tables, tmp_path):
	run_dir = tmp_path / "league"
	lines = []
	result = run_training(
		league_config(
			"run.updates=2",
			"logging.every=1",
			"evaluation.every=2",
			"checkpoint.every=1",
			f"run.run_dir={run_dir}",
		),
		team_pool=write_pool(tmp_path, tables),
		log=lines.append,
	)
	assert result["groups"] == {"selfplay": 4, "pool": 2, "random": 1, "max-damage": 1}
	assert result["lanes"] == 12
	assert [record["kind"] for record in records(run_dir)] == [
		"update",
		"update",
		"evaluation",
		"final_evaluation",
	]
	update = records(run_dir)[1]
	for name in ("win_rate_selfplay", "win_rate_pool", "win_rate_max-damage"):
		assert name in update
	assert update["lane_steps_per_second"] > 0
	evaluation = records(run_dir)[2]["results"]
	assert [(r["opponent"], r["policy"]) for r in evaluation] == [
		("max-damage", "sampled"),
		("max-damage", "greedy"),
	]
	assert any(line.startswith("evaluation at 2 (4 battles each)") for line in lines)
	assert any(line.startswith("final evaluation at 2") for line in lines)


@pytest.mark.large
def test_kept_checkpoints_are_written_at_their_updates(tables, tmp_path):
	run_dir = tmp_path / "kept"
	lines = []
	run_training(
		tiny_config(
			"run.updates=3",
			"checkpoint.every=0",
			"checkpoint.keep=[2]",
			f"run.run_dir={run_dir}",
			"logging.every=1",
			"evaluation.every=0",
		),
		team_pool=write_pool(tmp_path, tables),
		log=lines.append,
	)
	kept = run_dir / "checkpoint-2.npz"
	assert kept.exists() and (run_dir / "checkpoint.npz").exists()
	assert not (run_dir / "checkpoint-3.npz").exists()
	with np.load(kept, allow_pickle=False) as data:
		assert int(data[".update"]) == 2
	with np.load(run_dir / "checkpoint.npz", allow_pickle=False) as data:
		assert int(data[".update"]) == 3
	assert "kept checkpoint-2.npz at update 2" in lines


def test_a_kept_checkpoint_is_never_overwritten(tmp_path):
	kept = tmp_path / "checkpoint-2.npz"
	kept.write_bytes(b"earlier")
	messages = []
	_keep_checkpoint(tmp_path, 2, None, None, None, messages.append)
	assert kept.read_bytes() == b"earlier" and messages == []
