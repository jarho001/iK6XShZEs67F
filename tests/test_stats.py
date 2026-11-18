import json

import jax.numpy as jnp

from pokejax import stats
from pokejax.stats import (
	DEFAULT_POOL,
	count_battle_summary,
	holm_adjust,
	ks_distribution_test,
	main,
	two_proportion_test,
	welch_mean_test,
)
from pokejax.tables import load_tables


def event(
	*,
	moves=(None, None),
	switches=(False, False),
	statuses=None,
	fainted=None,
	confusion=(False, False),
):
	return {
		"moves": list(moves),
		"switches": list(switches),
		"statuses": list(statuses or [""] * 12),
		"fainted": list(fainted or [False] * 12),
		"confusion": list(confusion),
	}


def test_decision_trace_counting_rules():
	first_status = [""] * 12
	first_status[0] = "brn"
	second_status = first_status.copy()
	second_status[0] = "tox"
	fainted = [False] * 12
	fainted[7] = True
	battle = {
		"winner": 0,
		"turns": 3,
		"decisions": [
			event(
				moves=("tackle", None),
				switches=(False, True),
				statuses=first_status,
				confusion=(True, False),
			),
			event(moves=("struggle", "toxic"), statuses=second_status, fainted=fainted),
			event(
				moves=(None, None), statuses=second_status, fainted=fainted, confusion=(True, False)
			),
		],
	}
	assert count_battle_summary(battle) == {
		"winner": 0,
		"turns": 3,
		"moves": {"tackle": 1, "struggle": 1, "toxic": 1},
		"statuses": {"brn": 1, "psn": 0, "tox": 1, "par": 0, "slp": 0, "frz": 0},
		"faints": 1,
		"switches": [0, 1],
		"struggle": 1,
		"confusion": 2,
		"itemsConsumed": {},
	}


def test_engine_item_counts_keep_team_pair_offsets_across_batches(monkeypatch):
	team_pairs = [
		[[{"item": "whiteherb"}], [{"item": ""}]],
		[[{"item": "sitrusberry"}], [{"item": ""}]],
	]
	monkeypatch.setattr(stats, "BATTLE_BATCH", 1)
	monkeypatch.setattr(
		stats,
		"_teams_for_pairs",
		lambda pairs, _tables: {"species": jnp.zeros((len(pairs), 2, 6), jnp.int16)},
	)

	def rollout(chunk, _keys, _engine_tables):
		batch = chunk["species"].shape[0]
		consumed = jnp.zeros((1, batch, 2, 6), jnp.bool_).at[0, :, 0, 0].set(True)
		return (
			jnp.zeros((batch,), jnp.int8),
			jnp.ones((batch,), jnp.int32),
			jnp.ones((batch,), jnp.int32),
			jnp.full((1, batch, 2), -1, jnp.int16),
			jnp.zeros((1, batch, 2), jnp.bool_),
			jnp.zeros((1, batch, 2, 6), jnp.int8),
			jnp.zeros((1, batch, 2, 6), jnp.bool_),
			jnp.zeros((1, batch, 2), jnp.bool_),
			consumed,
		)

	monkeypatch.setattr(stats, "_rollout_batch", rollout)
	result = stats.engine_summaries(
		{"teamPairs": team_pairs, "count": 2}, seed=1, tables=load_tables()
	)
	assert result["battles"][0]["itemsConsumed"] == {"whiteherb": 1}
	assert result["battles"][1]["itemsConsumed"] == {"sitrusberry": 1}


def test_statistical_functions_on_known_samples():
	difference, standard_error, p_value = two_proportion_test([1] * 10, [0] * 10)
	assert difference == 1
	assert standard_error > 0
	assert p_value < 0.001

	difference, standard_error, p_value = welch_mean_test([0, 1, 2, 10], [0, 0, 0, 0])
	assert difference == 3.25
	assert standard_error > 0
	assert p_value > 0.05  # A single high-count battle is not four independent events.
	assert welch_mean_test([0, 0], [0, 0]) == (0.0, 0.0, 1.0)

	difference, standard_error, p_value = welch_mean_test([1, 2, 3, 4], [1, 2, 3, 4])
	assert difference == 0
	assert standard_error > 0
	assert p_value == 1

	difference, standard_error, p_value = ks_distribution_test([1] * 20, [2] * 20)
	assert difference == 1
	assert standard_error > 0
	assert p_value < 0.001
	assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_whole_battle_smoke_runs_both_simulators_and_writes_report(tmp_path, capsys):
	report_path = tmp_path / "report.json"
	stats_dir = tmp_path / "stats"
	assert (
		main(
			[
				"--battles",
				"24",
				"--seed",
				"9",
				"--team-pool",
				str(DEFAULT_POOL),
				"--stats-dir",
				str(stats_dir),
				"--out",
				str(report_path),
			]
		)
		== 0
	)
	output = capsys.readouterr().out
	assert "Whole-battle comparison: 24 battles per simulator" in output
	assert "Showdown" in output and "Engine" in output
	assert report_path.exists()
	report = json.loads(report_path.read_text())
	assert report["battles"] == 24
	assert report["results"]
	assert (stats_dir / "showdown.json").exists()
	assert (stats_dir / "engine.json").exists()
	assert all("p_adjusted" in row and "standard_error" in row for row in report["results"])
