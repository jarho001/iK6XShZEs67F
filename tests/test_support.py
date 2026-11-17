"""Support filtering uses all mechanics and permits empty move/item slots."""

import json
from pathlib import Path

import numpy as np
import pytest

from pokejax import support


def test_initial_support(synthetic_pool, tables):
	assert {"tackle", "swordsdance", "closecombat", "bellydrum"} <= support.SUPPORTED_MOVES
	assert support.SUPPORTED_ABILITIES == frozenset()
	assert support.SUPPORTED_ITEMS == frozenset()
	result = support.team_supported(synthetic_pool, tables)
	assert result.shape == (3,)
	assert result.dtype == np.bool_
	np.testing.assert_array_equal(result, [False, False, False])


def test_secondary_group_matches_allowed_file():
	allowed = json.loads(Path("tests/fixtures/diff-secondaries.allowed.json").read_text())
	assert allowed["moves"] == sorted(support.SECONDARY_MOVES | support.CONFUSION_MOVES)
	assert set(allowed["moves"]) <= support.SUPPORTED_MOVES


def test_effects2_moves_match_allowed_file():
	allowed = json.loads(Path("tests/fixtures/diff-effects2.allowed.json").read_text())
	assert allowed["moves"] == sorted(allowed["moves"])
	assert set(allowed["moves"]) <= support.SUPPORTED_MOVES


@pytest.mark.parametrize(
	"moves,abilities,items,expected",
	[
		({"tackle"}, {"overgrow"}, {"leftovers"}, [True, False, False]),
		({"surf"}, {"torrent"}, set(), [False, True, False]),
		(set(), {"overgrow"}, {"leftovers"}, [False, False, False]),
		({"tackle"}, set(), {"leftovers"}, [False, False, False]),
		({"tackle"}, {"overgrow"}, set(), [False, False, False]),
	],
)
def test_selected_support(synthetic_pool, tables, monkeypatch, moves, abilities, items, expected):
	monkeypatch.setattr(support, "SUPPORTED_MOVES", frozenset(moves))
	monkeypatch.setattr(support, "SUPPORTED_ABILITIES", frozenset(abilities))
	monkeypatch.setattr(support, "SUPPORTED_ITEMS", frozenset(items))
	np.testing.assert_array_equal(support.team_supported(synthetic_pool, tables), expected)


def test_none_ability_needs_support(synthetic_pool, tables, monkeypatch):
	monkeypatch.setattr(support, "SUPPORTED_MOVES", frozenset({"tackle"}))
	monkeypatch.setattr(support, "SUPPORTED_ITEMS", frozenset({"leftovers"}))
	synthetic_pool.ability[0] = 0
	assert not support.team_supported(synthetic_pool, tables).any()
