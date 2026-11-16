"""Whole-file forced-outcome replay of the pinned Phase 1 Showdown oracle."""

from pathlib import Path

import pytest

from pokejax.diff import replay_file

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/diff-phase1.jsonl.gz"
LARGE = ROOT / "work/diff/phase1.jsonl.gz"


def test_committed_phase1():
	result = replay_file(FIXTURE)
	assert result.battles == 40
	assert not result.mismatches, result.report()


@pytest.mark.skipif(not LARGE.exists(), reason="Regenerate with reference/record-battles.cjs")
def test_large_phase1():
	result = replay_file(LARGE)
	assert result.battles == 300
	assert not result.mismatches, result.report()
