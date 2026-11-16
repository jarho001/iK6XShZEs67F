"""Generate deterministic engine coverage from a compact team pool."""

import argparse
from pathlib import Path

import numpy as np

from pokejax.support import support_masks, team_supported
from pokejax.tables import NONE_ROW, Tables, load_tables
from pokejax.teams import TeamPool, load_npz

CATEGORIES = (
	("Moves", "moves", "moves"),
	("Abilities", "abilities", "ability"),
	("Items", "items", "item"),
)
REGENERATE_COMMAND = "python -m pokejax.coverage work/teams/gen5-teams.npz COVERAGE.md"
PERCENT_SCALE = 100


def _percentage(numerator, denominator):
	return f"{PERCENT_SCALE * numerator / denominator if denominator else 0:.1f}%"


def _counts(values, row_count):
	"""Count occurrences and distinct (team, row) pairs using NumPy."""
	occurrences = np.bincount(values.ravel(), minlength=row_count)
	team_rows = np.broadcast_to(
		np.arange(values.shape[0]).reshape((-1,) + (1,) * (values.ndim - 1)), values.shape
	)
	pairs = np.unique(np.stack((team_rows.ravel(), values.ravel()), axis=1), axis=0)
	teams = np.bincount(pairs[:, 1], minlength=row_count)
	return occurrences, teams


def render_coverage(pool: TeamPool, tables: Tables) -> str:
	"""Render only nonempty ids that occur in the pool; ties sort by id."""
	team_count = len(pool.team_size)
	lines = [
		f"Generated file; do not edit by hand. Regenerate with `{REGENERATE_COMMAND}`.",
		"",
		f"Teams: {team_count}",
		f"Seed: {pool.seed}",
		f"Showdown commit: `{pool.showdown_commit}`",
		"",
		"## Summary",
		"",
		"| Kind | Distinct | Supported | Supported occurrences |",
		"| --- | ---: | ---: | ---: |",
	]
	sections = []
	for (title, table_name, column), supported in zip(CATEGORIES, support_masks(tables)):
		table = getattr(tables, table_name)
		occurrences, teams = _counts(getattr(pool, column), len(table.ids))
		rows = np.flatnonzero(occurrences)
		rows = rows[rows != NONE_ROW]
		supported_count = np.count_nonzero(supported[rows])
		share = _percentage(occurrences[rows][supported[rows]].sum(), occurrences[rows].sum())
		lines.append(f"| {title} | {len(rows)} | {supported_count} | {share} |")
		rows = rows[np.lexsort((np.asarray(table.ids)[rows], -teams[rows]))]
		sections.extend(
			[
				"",
				f"## {title}",
				"",
				"| Id | Display name | Occurrences | Teams | Supported |",
				"| --- | --- | ---: | ---: | --- |",
			]
		)
		for row in rows:
			name = table.names[row].replace("|", r"\|")
			sections.append(
				f"| {table.ids[row]} | {name} | {occurrences[row]} | "
				f"{_percentage(teams[row], team_count)} | {'yes' if supported[row] else 'no'} |"
			)
	lines.extend(
		[
			"",
			f"Fully supported teams: {_percentage(np.count_nonzero(team_supported(pool, tables)), team_count)}",
		]
	)
	return "\n".join(lines + sections) + "\n"


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("pool_path", type=Path)
	parser.add_argument("output_path", type=Path)
	args = parser.parse_args()
	args.output_path.write_text(
		render_coverage(load_npz(args.pool_path), load_tables()), encoding="utf-8"
	)


if __name__ == "__main__":
	main()
