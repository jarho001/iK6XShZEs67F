"""Wilson intervals and the summary of a learner's battles against one opponent."""

import pytest

from pokejax.train.evaluation import summarize_battles, wilson_interval


def test_wilson_interval_matches_the_formula_at_half():
	low, high = wilson_interval(50, 100)
	assert low == pytest.approx(0.40386, abs=1e-4)
	assert high == pytest.approx(0.59614, abs=1e-4)


def test_wilson_interval_stays_inside_the_unit_interval():
	low, high = wilson_interval(0, 40)
	assert low == pytest.approx(0.0, abs=1e-12) and 0 < high < 0.1
	low, high = wilson_interval(40, 40)
	assert 0.9 < low < 1 and high == pytest.approx(1.0, abs=1e-12)


def test_wilson_interval_rejects_impossible_counts():
	with pytest.raises(ValueError):
		wilson_interval(3, 2)
	with pytest.raises(ValueError):
		wilson_interval(0, 0)


def test_summary_counts_truncated_battles_as_not_won():
	result = summarize_battles("random", 100, 60, 5, 42.5)
	assert result["win_rate"] == pytest.approx(0.6)
	assert result["battles"] == 100 and result["truncated"] == 5
	low, high = result["ci95"]
	assert low < 0.6 < high
