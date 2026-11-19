"""Exact coverage, ordering and deterministic CLI output without Showdown."""

import subprocess
import sys
from dataclasses import replace

from pokejax import support
from pokejax.coverage import REGENERATE_COMMAND, render_coverage
from pokejax.teams import save_npz


def section_rows(markdown, title):
	section = markdown.split(f"## {title}\n", 1)[1].split("\n## ", 1)[0]
	return [line for line in section.splitlines() if line.startswith("| ")][2:]


def test_counts_and_sorting(synthetic_pool, tables, monkeypatch):
	monkeypatch.setattr(support, "SUPPORTED_MOVES", frozenset())
	monkeypatch.setattr(support, "SUPPORTED_ABILITIES", frozenset())
	monkeypatch.setattr(support, "SUPPORTED_ITEMS", frozenset())
	output = render_coverage(synthetic_pool, tables)
	assert output.splitlines()[0] == (
		f"Generated file; do not edit by hand. Regenerate with `{REGENERATE_COMMAND}`."
	)
	assert "Teams: 3\nSeed: 7\nShowdown commit: `synthetic-commit`" in output
	assert section_rows(output, "Summary") == [
		"| Moves | 3 | 0 | 0.0% |",
		"| Abilities | 2 | 0 | 0.0% |",
		"| Items | 2 | 0 | 0.0% |",
	]
	assert "Fully supported teams: 0.0%" in output
	assert section_rows(output, "Moves") == [
		"| tackle | Tackle | 12 | 66.7% | no |",
		"| protect | Protect | 1 | 33.3% | no |",
		"| surf | Surf | 6 | 33.3% | no |",
	]
	assert section_rows(output, "Abilities") == [
		"| overgrow | Overgrow | 12 | 66.7% | no |",
		"| torrent | Torrent | 6 | 33.3% | no |",
	]
	assert section_rows(output, "Items") == [
		"| leftovers | Leftovers | 6 | 33.3% | no |",
		"| lifeorb | Life Orb | 6 | 33.3% | no |",
	]


def test_supported_summary(synthetic_pool, tables, monkeypatch):
	monkeypatch.setattr(support, "SUPPORTED_MOVES", frozenset({"tackle"}))
	monkeypatch.setattr(support, "SUPPORTED_ABILITIES", frozenset({"overgrow"}))
	monkeypatch.setattr(support, "SUPPORTED_ITEMS", frozenset({"leftovers"}))
	output = render_coverage(synthetic_pool, tables)
	assert section_rows(output, "Summary") == [
		"| Moves | 3 | 1 | 63.2% |",
		"| Abilities | 2 | 1 | 66.7% |",
		"| Items | 2 | 1 | 50.0% |",
	]
	assert "Fully supported teams: 33.3%" in output
	assert "| tackle | Tackle | 12 | 66.7% | yes |" in output
	assert "| leftovers | Leftovers | 6 | 33.3% | yes |" in output


def test_cli_is_deterministic(synthetic_pool, tables, tmp_path):
	pool_path = tmp_path / "pool.npz"
	output_path = tmp_path / "COVERAGE.md"
	save_npz(synthetic_pool, pool_path)
	command = [sys.executable, "-m", "pokejax.coverage", str(pool_path), str(output_path)]
	subprocess.run(command, check=True, capture_output=True)
	first = output_path.read_bytes()
	subprocess.run(command, check=True, capture_output=True)
	assert output_path.read_bytes() == first
	assert first.decode("utf-8") == render_coverage(synthetic_pool, tables)


def test_display_names_use_table(synthetic_pool, tables):
	names = list(tables.moves.names)
	names[tables.moves.index["tackle"]] = "Custom Tackle"
	custom = replace(tables, moves=replace(tables.moves, names=tuple(names)))
	assert "| tackle | Custom Tackle |" in render_coverage(synthetic_pool, custom)
