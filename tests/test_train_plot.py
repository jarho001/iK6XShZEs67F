"""The plot tool on synthetic metrics files: the lines, markers and exit line of the SVG, the wall
time of a resumed run, the smoothing of training rates, and the errors it reports."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pokejax.train.plot import PlotError, _clock, _pooled, main, run_series

NS = "{http://www.w3.org/2000/svg}"


def evaluation(update, rate):
	"""An evaluation record: the learner against two baselines, sampled and greedy."""
	results = []
	for opponent, value in (("max-damage", rate), ("random", 0.99)):
		for policy in ("sampled", "greedy"):
			results.append(
				{
					"opponent": opponent,
					"policy": policy,
					"battles": 1024,
					"wins": 0,
					"win_rate": value,
					"ci95": [value - 0.02, value + 0.02],
					"truncated": 0,
					"mean_battle_length": 30.0,
				}
			)
	return {"kind": "evaluation", "update": update, "results": results}


def write_run(directory, name, rate, updates=range(10, 201, 10), env_steps=True):
	"""A run directory with update records (six seconds per update) and evaluations at 100, 200.

	Each update is 32,768 environment steps, as in the training runs (env_steps), unless env_steps is
	false.
	"""
	directory.mkdir(parents=True)
	(directory / "config.json").write_text(json.dumps({"run": {"name": name}}), encoding="utf-8")
	rows = [
		{
			"kind": "update",
			"update": update,
			"seconds": 6.0 * update,
			"win_rate_max-damage": rate,
			"completed_max-damage": 10.0,
		}
		for update in updates
	]
	if env_steps:
		for row in rows:
			row["env_steps"] = 32768 * row["update"]
	rows += [evaluation(100, rate), evaluation(200, rate + 0.05)]
	text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
	(directory / "metrics.jsonl").write_text(text, encoding="utf-8")
	return directory


def classes(element):
	return element.get("class", "").split()


def test_clock_continues_a_resumed_run():
	rows = [
		{"update": 1, "seconds": 60.0},
		{"update": 2, "seconds": 120.0},
		{"update": 3, "seconds": 30.0},  # resumed: the clock starts again from zero
		{"update": 4, "seconds": 90.0},
	]
	updates, minutes = _clock(rows)
	assert list(updates) == [1, 2, 3, 4]
	np.testing.assert_allclose(minutes, [1.0, 2.0, 2.5, 3.5])


def test_pooled_rates_use_the_updates_within_half_the_window():
	updates = np.array([10.0, 20.0, 30.0, 40.0])
	ended = np.array([10.0, 10.0, 10.0, 10.0])
	won = np.array([10.0, 0.0, 0.0, 0.0])
	rates = _pooled(updates, won, ended, window=20)
	np.testing.assert_allclose(rates, [0.5, 1 / 3, 0.0, 0.0])
	empty = _pooled(updates, won, np.zeros(4), window=20)
	assert np.all(np.isnan(empty))


def test_svg_draws_each_run_its_training_line_and_the_exit_line(tmp_path):
	first = write_run(tmp_path / "abl-a", "abl-a", 0.7)
	second = write_run(tmp_path / "abl-b", "abl-b", 0.8)
	out = tmp_path / "figure.svg"
	assert main([str(first), str(second), "--training", "--out", str(out)]) == 0
	root = ET.parse(out).getroot()
	assert root.tag == NS + "svg"
	polylines = list(root.iter(NS + "polyline"))
	evaluation_lines = [p for p in polylines if "line" in classes(p)]
	training_lines = [p for p in polylines if "train" in classes(p)]
	assert len(evaluation_lines) == 2 and len(training_lines) == 2
	assert [c for c in classes(evaluation_lines[0]) if c.startswith("k")] == ["k0"]
	assert [c for c in classes(evaluation_lines[1]) if c.startswith("k")] == ["k1"]
	assert len(training_lines[0].get("points").split()) == 20  # one per logged update
	dashed = [line for line in root.iter(NS + "line") if "ref" in classes(line)]
	assert len(dashed) == 1 and dashed[0].get("y1") == dashed[0].get("y2")
	texts = [element.text for element in root.iter(NS + "text")]
	assert "abl-a" in texts and "abl-b" in texts  # the legend names the runs
	markers = [c for c in root.iter(NS + "circle") if len(c) and c[0].tag == NS + "title"]
	assert len(markers) == 4  # two evaluation points per run
	tooltips = [marker[0].text for marker in markers]
	assert any("abl-a, update 100: 70.00%" in text for text in tooltips)
	assert any("abl-b, update 200: 85.00%" in text for text in tooltips)


def test_minutes_axis_and_the_tooltips_of_the_greedy_policy(tmp_path):
	run = write_run(tmp_path / "exp", "exp", 0.7)
	out = tmp_path / "minutes.svg"
	args = [str(run), "--x", "minutes", "--policy", "greedy", "--out", str(out)]
	assert main(args) == 0
	series = run_series(
		run, opponent="max-damage", policy="greedy", x_mode="minutes", training=False
	)
	assert [round(x, 6) for x, _, _ in series.evaluation] == [10.0, 20.0]
	assert "10.0 min" in series.evaluation[0][2] and "greedy" in series.evaluation[0][2]


def test_steps_axis_places_the_evaluations_at_their_environment_steps(tmp_path):
	run = write_run(tmp_path / "exp", "exp", 0.7)
	series = run_series(
		run, opponent="max-damage", policy="sampled", x_mode="steps", training=False
	)
	assert [round(x) for x, _, _ in series.evaluation] == [100 * 32768, 200 * 32768]
	assert "3.3M steps" in series.evaluation[0][2]
	out = tmp_path / "steps.svg"
	assert main([str(run), "--x", "steps", "--out", str(out)]) == 0
	texts = [element.text for element in ET.parse(out).getroot().iter(NS + "text")]
	assert "environment steps" in texts and "2M" in texts and "8M" in texts
	without = write_run(tmp_path / "old", "old", 0.7, env_steps=False)
	with pytest.raises(PlotError, match="without env_steps"):
		run_series(without, opponent="max-damage", policy="sampled", x_mode="steps", training=False)


def test_html_output_wraps_the_same_figure(tmp_path):
	run = write_run(tmp_path / "exp", "exp", 0.7)
	out = tmp_path / "figure.html"
	assert main([str(run), "--out", str(out)]) == 0
	text = out.read_text(encoding="utf-8")
	assert text.startswith("<!doctype html>")
	assert text.count("<svg") == 1 and "</svg>" in text


def test_the_errors_name_the_missing_data(tmp_path):
	run = write_run(tmp_path / "exp", "exp", 0.7)
	with pytest.raises(PlotError, match="no sampled evaluation against heuristics"):
		run_series(run, opponent="heuristics", policy="sampled", x_mode="updates", training=False)
	with pytest.raises(PlotError, match="no training win rates against random"):
		run_series(run, opponent="random", policy="sampled", x_mode="updates", training=True)
	with pytest.raises(PlotError, match="does not exist"):
		run_series(
			tmp_path / "missing",
			opponent="max-damage",
			policy="sampled",
			x_mode="updates",
			training=False,
		)
	runs = [write_run(tmp_path / f"run{index}", f"run{index}", 0.7) for index in range(9)]
	with pytest.raises(SystemExit):
		main([*map(str, runs), "--out", str(tmp_path / "nine.svg")])
	with pytest.raises(SystemExit):
		main([str(run), "--out", str(tmp_path / "figure.png")])
