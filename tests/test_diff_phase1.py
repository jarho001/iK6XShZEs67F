"""Whole-file forced-outcome replay of the pinned Phase 1 Showdown oracle."""

import copy
from pathlib import Path

import pytest

from pokejax.diff import load_recording, recorded_state, replay_file
from pokejax.engine.state import STATUS_NAMES
from pokejax.tables import load_tables

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/diff-phase1.jsonl.gz"


def test_snapshot_status_mapping():
	_, battles = load_recording(FIXTURE)
	snapshot = copy.deepcopy(battles[0]["initialState"])
	for member, name in zip(snapshot["sides"][0]["members"], STATUS_NAMES[1:], strict=True):
		member["status"] = name
	assert recorded_state(snapshot, load_tables())["status"][0].tolist() == list(range(1, 7))


@pytest.mark.parametrize("fixture", sorted((ROOT / "tests/fixtures").glob("diff-*.jsonl.gz")))
def test_fixtures(fixture):
	result = replay_file(fixture)
	assert not result.mismatches, result.report()


# The 300-battle recordings in work/diff/ (git-ignored) run only with --large.
@pytest.mark.large
@pytest.mark.parametrize("name", ["phase1", "status", "boosts", "secondaries"])
def test_large_recording(name):
	path = ROOT / f"work/diff/{name}.jsonl.gz"
	if not path.exists():
		pytest.skip(f"Regenerate {path} with reference/record-battles.cjs")
	result = replay_file(path)
	assert result.battles == 300
	assert not result.mismatches, result.report()


@pytest.mark.parametrize("field", ["status", "volatile", "legal", "weather"])
def test_corrupt_snapshot(field):
	header, battles = load_recording(FIXTURE)
	header, battles = copy.deepcopy((header, battles[:1]))
	header["count"] = 1
	snapshot = battles[0]["initialState"]
	if field == "status":
		snapshot["sides"][0]["status"] = "brn"
		snapshot["sides"][0]["members"][snapshot["sides"][0]["active"]]["status"] = "brn"
	elif field == "volatile":
		snapshot["sides"][0]["volatiles"] = ["confusion"]
	elif field == "legal":
		snapshot["sides"][0]["legalActions"][0] ^= 1
	else:
		snapshot["weather"] = "raindance"
	result = replay_file(FIXTURE, recording=(header, battles))
	assert result.mismatches, result.report()
	assert result.mismatches[0].step == -1


@pytest.mark.parametrize("field", ["counter", "unknown_volatile", "unknown_counter"])
def test_corrupt_secondary_snapshot(field):
	path = ROOT / "tests/fixtures/diff-secondaries.jsonl.gz"
	header, battles = load_recording(path)
	battle = copy.deepcopy(
		next(
			battle
			for battle in battles
			if any(
				side.get("volatileCounters", {}).get("confusion", 0)
				for record in battle["steps"]
				for side in record["state"]["sides"]
			)
		)
	)
	header = dict(header, count=1)
	if field == "counter":
		side = next(
			side
			for record in battle["steps"]
			for side in record["state"]["sides"]
			if side.get("volatileCounters", {}).get("confusion", 0)
		)
		side["volatileCounters"]["confusion"] += 1
	elif field == "unknown_volatile":
		battle["initialState"]["sides"][0]["volatiles"].append("substitute")
	else:
		battle["initialState"]["sides"][0]["volatileCounters"] = {"encore": 3}
	result = replay_file(path, recording=(header, [battle]))
	assert result.mismatches, result.report()


def test_mismatch_after_chunk_boundaries():
	header, battles = load_recording(FIXTURE)
	header, battles = copy.deepcopy((header, battles))
	battle = battles[32]
	assert len(battle["steps"]) > 32
	snapshot = battle["steps"][32]["state"]
	snapshot["sides"][0]["members"][0]["hp"] += 1
	result = replay_file(FIXTURE, recording=(header, battles))
	assert len(result.mismatches) == 1
	mismatch = result.mismatches[0]
	assert (mismatch.battle, mismatch.step, mismatch.field) == (battle["index"], 32, "hp[0][0]")
	assert mismatch.showdown == mismatch.engine + 1
	assert mismatch.actions == battle["steps"][32]["actions"]
	assert mismatch.outcomes == battle["steps"][32]["outcomes"]


def test_fainted_active_nullable_fields_and_other_fields():
	header, battles = load_recording(FIXTURE)
	for original in battles:
		for record in original["steps"]:
			for side in record["state"]["sides"]:
				if side["types"] is None:
					battle = copy.deepcopy(original)
					break
			else:
				continue
			break
		else:
			continue
		break
	else:
		pytest.fail("Fixture has no fainted active Pokemon")
	header = dict(header, count=1)
	side = next(
		side
		for record in battle["steps"]
		for side in record["state"]["sides"]
		if side["types"] is None
	)
	assert side["members"][side["active"]]["fainted"]
	assert side["volatiles"] is None
	result = replay_file(FIXTURE, recording=(header, [battle]))
	assert not result.mismatches, result.report()
	side["members"][side["active"]]["hp"] = 1
	result = replay_file(FIXTURE, recording=(header, [battle]))
	assert result.mismatches and result.mismatches[0].field.startswith("hp")
