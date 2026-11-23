"""The engine observation against the one that Node builds from each player's own stream.

reference/player-view.cjs builds the observation from one player's requests and protocol lines
only; these tests replay the recorded battles through the engine with forced outcomes and compare
every leaf at every decision point, for both sides (pokejax/obs_diff.py).
"""

import copy
from pathlib import Path

import pytest

from pokejax.diff import load_recording
from pokejax.engine.observation import EventObservation, Observation
from pokejax.obs_diff import ambiguous_from, compare_file, flatten, node_history

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/obs-pool.jsonl.gz"


def test_fixture_matches():
	result = compare_file(FIXTURE)
	assert result.battles == 40
	assert not result.mismatches, result.report()


# The 300-battle recording in work/diff/ (git-ignored) runs only with --large.
@pytest.mark.large
def test_large_recording():
	path = ROOT / "work/diff/obs-pool.jsonl.gz"
	if not path.exists():
		pytest.skip(f"Regenerate {path} with reference/record-battles.cjs --views")
	result = compare_file(path)
	assert result.battles == 300
	assert not result.mismatches, result.report()


def one_battle(index=0):
	header, battles = load_recording(FIXTURE)
	header = dict(header, count=1)
	return header, [copy.deepcopy(battles[index])]


def test_player_view_has_the_observation_fields():
	"""The Node observation has the engine observation's names, nesting and shapes."""
	_, battles = load_recording(FIXTURE)
	node = node_history(battles[0])
	fields = Observation._fields
	assert node["turn"].shape[1:] == (2,)
	assert {path.split(".")[0] for path in node} == set(fields)
	assert {path for path in node if path.startswith("events.")} == {
		f"events.{name}" for name in EventObservation._fields
	}
	assert node["own.species"].shape[1:] == (2, 6)
	assert node["opponent.moves"].shape[1:] == (2, 6, 4)
	assert node["hazards"].shape[1:] == (2, 2, 2)
	assert len(flatten(battles[0]["initialViews"][0])) == len(node)


@pytest.mark.parametrize(
	"path, mutate",
	[
		("own.hp", lambda view: view["own"]["hp"].__setitem__(0, 0.123)),
		("opponent.species", lambda view: view["opponent"]["species"].__setitem__(0, 3)),
		("legal_actions", lambda view: view["legal_actions"].__setitem__(0, False)),
		("events.order", lambda view: view["events"]["order"].__setitem__(1, 2)),
		("weather", lambda view: view.__setitem__("weather", 3)),
	],
)
def test_corrupt_view_is_reported(path, mutate):
	"""A changed leaf is found, at the right decision point, side and field."""
	header, battles = one_battle()
	mutate(battles[0]["steps"][4]["views"][1])
	result = compare_file(FIXTURE, recording=(header, battles))
	assert result.mismatches, path
	first = result.mismatches[0]
	assert (first.battle, first.step, first.side) == (battles[0]["index"], 4, 1)
	assert first.path.startswith(path)


def test_float_leaves_are_compared_to_a_tolerance():
	header, battles = one_battle()
	hp = battles[0]["steps"][2]["views"][0]["own"]["hp"]
	hp[0] += 5e-7
	assert not compare_file(FIXTURE, recording=(header, battles)).mismatches
	hp[0] += 5e-6
	assert compare_file(FIXTURE, recording=(header, battles)).mismatches


def test_a_nan_in_the_player_view_is_a_mismatch():
	"""NaN compares false to every number, so a tolerance test alone would pass it."""
	header, battles = one_battle()
	battles[0]["steps"][2]["views"][0]["own"]["hp"][0] = float("nan")
	result = compare_file(FIXTURE, recording=(header, battles))
	assert any(
		m.path.startswith("own.hp") and m.step == 2 and m.side == 0 for m in result.mismatches
	)


def test_illusion_makes_the_opponent_roster_ambiguous():
	"""From the first decision point at which the foe's Illusion holder is active, the roster
	arrays are left out (the protocol cannot say which entry a switch-in belongs to); the
	holder's own player and everything else keep being compared."""
	_, battles = load_recording(FIXTURE)
	holders = [
		battle
		for battle in battles
		if any(member["ability"] == "illusion" for team in battle["teams"] for member in team)
	]
	assert holders
	battle = holders[0]
	side = next(
		s for s, team in enumerate(battle["teams"]) if any(m["ability"] == "illusion" for m in team)
	)
	assert ambiguous_from(battle, side) is None
	foe_start = ambiguous_from(battle, 1 - side)
	assert foe_start is not None and foe_start >= 0
	plain = next(battle for battle in battles if battle not in holders)
	assert ambiguous_from(plain, 0) is None and ambiguous_from(plain, 1) is None
	result = compare_file(FIXTURE)
	assert result.skipped
	assert all(name.startswith("opponent.") for name in result.skipped)
