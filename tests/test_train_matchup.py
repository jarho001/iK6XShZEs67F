"""The engine matchups: paired battle setups, the paired bootstrap, and the head-to-head and panel
reports run on a small checkpoint (CPU, tiny networks, a short battle limit)."""

import json
from pathlib import Path

import jax
import numpy as np
import pytest
from train_support import tiny_config, write_pool

from pokejax.env import Env, EnvConfig
from pokejax.train import headtohead, panel
from pokejax.train.checkpoint import save_checkpoint
from pokejax.train.headtohead import battle_setups as head_to_head_setups
from pokejax.train.headtohead import parse_checkpoint
from pokejax.train.matchup import paired_mean_interval, summarize
from pokejax.train.panel import battle_setups as panel_setups
from pokejax.train.ppo import Trainer

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


@pytest.fixture(scope="module")
def pool(tables, tmp_path_factory):
	return write_pool(tmp_path_factory.mktemp("matchup-pool"), tables)


def engine(tables, pool, batch: int, max_steps: int = 3) -> Env:
	return Env(
		EnvConfig(batch_size=batch, max_steps=max_steps, observe="none"), pool, tables=tables
	)


def test_parse_checkpoint_takes_a_name_or_the_run_directory_name():
	assert parse_checkpoint("p6-p2=work/runs/p6-p2/checkpoint.npz") == (
		"p6-p2",
		Path("work/runs/p6-p2/checkpoint.npz"),
	)
	assert parse_checkpoint("work/runs/p6-p2") == ("p6-p2", Path("work/runs/p6-p2"))
	assert parse_checkpoint("work/runs/p6-long/checkpoint-480M.npz")[0] == "checkpoint-480M"


def test_paired_bootstrap_gives_the_mean_and_a_zero_width_interval_for_constant_values():
	result = paired_mean_interval(np.array([1.0, 0.0, 1.0, 1.0]), resamples=4000, seed=0)
	assert result["estimate"] == pytest.approx(0.75)
	low, high = result["ci95"]
	assert 0.0 <= low < 0.75 < high <= 1.0
	assert paired_mean_interval(np.full(10, 0.5), resamples=100, seed=1) == {
		"estimate": 0.5,
		"ci95": [0.5, 0.5],
	}


def test_summary_counts_wins_truncations_and_unfinished_battles():
	won = np.array([True, False, True, True])
	result = summarize(
		won, np.array([False, False, True, False]), np.array([3, 4, 5, 6]), np.ones(4, bool)
	)
	assert result["wins"] == 3 and result["battles"] == 4 and result["win_rate"] == 0.75
	assert result["truncated"] == 1 and result["unfinished"] == 0
	assert result["mean_battle_length"] == pytest.approx(4.5)
	assert len(result["ci95"]) == 2 and result["ci95"][0] < 0.75 < result["ci95"][1]


def test_head_to_head_draws_put_each_draw_on_both_seats(tables, pool):
	setups = head_to_head_setups(engine(tables, pool, 4), draws=8, batch=4, seed=0)
	assert len(setups) == 4
	for states, seats, _ in setups:
		assert np.asarray(seats).tolist() == [0, 0, 1, 1]
		# Rows 0 and 2 (and 1 and 3) are the same battle: the teams and every battle field agree.
		same = jax.tree.map(lambda x: np.array_equal(np.asarray(x)[:2], np.asarray(x)[2:]), states)
		assert all(jax.tree.leaves(same))


def test_panel_setups_are_a_function_of_the_seed_alone(tables, pool):
	env = engine(tables, pool, 4)
	first = panel_setups(env, battles=8, batch=4, seed=5)
	again = panel_setups(env, battles=8, batch=4, seed=5)
	for (a_states, a_sides, _), (b_states, b_sides, _) in zip(first, again, strict=True):
		assert np.array_equal(np.asarray(a_sides), np.asarray(b_sides))
		assert all(
			jax.tree.leaves(
				jax.tree.map(
					lambda x, y: np.array_equal(np.asarray(x), np.asarray(y)), a_states, b_states
				)
			)
		)


@pytest.fixture(scope="module")
def small_checkpoint(tables, pool, tmp_path_factory):
	config = tiny_config("checkpoint.every=0", "env.max_steps=60")
	trainer = Trainer(config, pool)
	learner = trainer.init()
	path = tmp_path_factory.mktemp("small") / "checkpoint.npz"
	save_checkpoint(path, learner, config, trainer.team_digest)
	return path


@pytest.mark.large
def test_the_panel_pairs_identical_checkpoints_exactly(small_checkpoint, pool, tmp_path):
	out = tmp_path / "panel.json"
	status = panel.main(
		[
			f"a={small_checkpoint}",
			f"b={small_checkpoint}",
			"--baselines",
			"random",
			"max-damage",
			"--opponent",
			f"fixed={small_checkpoint}",
			"--battles",
			"8",
			"--batch",
			"4",
			"--difference",
			"a",
			"b",
			"--team-pool",
			str(pool),
			"--out",
			str(out),
		]
	)
	assert status == 0
	report = json.loads(out.read_text())
	assert report["opponents"] == ["random", "max-damage", "fixed"]
	assert set(report["results"]["a"]) == {"sampled", "greedy"}
	for entry in report["differences"]:
		assert entry["panel"]["estimate"] == 0.0 and entry["panel"]["ci95"] == [0.0, 0.0]
		assert all(cell["estimate"] == 0.0 for cell in entry["opponents"].values())
	for mode in ("sampled", "greedy"):
		cells = report["results"]["a"][mode]["opponents"]
		assert all(cell["battles"] == 8 for cell in cells.values())


@pytest.mark.large
def test_head_to_head_plays_both_seats_and_both_action_modes(small_checkpoint, pool, tmp_path):
	out = tmp_path / "h2h.json"
	status = headtohead.main(
		[
			f"first={small_checkpoint}",
			f"second={small_checkpoint}",
			"--draws",
			"8",
			"--batch",
			"4",
			"--team-pool",
			str(pool),
			"--out",
			str(out),
		]
	)
	assert status == 0
	report = json.loads(out.read_text())
	assert report["games"] == 16
	for mode in ("sampled", "greedy"):
		row = report["modes"][mode]
		assert row["battles"] == 16
		assert 0 <= row["wins"] <= 16
		assert row["win_rate"] == pytest.approx(row["wins"] / 16)
		assert 0.0 <= row["side0_win_rate"] <= 1.0 and 0.0 <= row["side1_win_rate"] <= 1.0
		assert len(row["paired_draws"]["ci95"]) == 2


def test_head_to_head_needs_an_even_batch(small_checkpoint):
	with pytest.raises(SystemExit):
		headtohead.main([str(small_checkpoint), str(small_checkpoint), "--batch", "3"])
