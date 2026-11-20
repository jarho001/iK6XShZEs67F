"""Draw teams for battles only from fully supported pool teams.

The supported team indices are computed once on the host from the support registry.
Each draw indexes that array with a JAX PRNG key, so one key always gives the same
team pairs and the host does no work per step. `python -m pokejax.sampler POOL`
prints the indices as JSON for the Node scripts.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from numpy.typing import NDArray

from pokejax.support import support_masks, team_supported
from pokejax.tables import RowTable, Tables, load_tables
from pokejax.teams import TeamPool, load_npz, load_team_pool

NO_ID = "(none)"


@dataclass(frozen=True)
class ExcludedTeam:
	"""One pool team that the sampler does not draw, with the ids that exclude it."""

	index: int
	moves: tuple[str, ...]
	abilities: tuple[str, ...]
	items: tuple[str, ...]


def supported_team_indices(pool: TeamPool, tables: Tables) -> NDArray[np.int32]:
	"""Ascending pool indices of the teams whose moves, abilities and items are all supported."""
	return np.flatnonzero(team_supported(pool, tables)).astype(np.int32)


def excluded_teams(pool: TeamPool, tables: Tables) -> tuple[ExcludedTeam, ...]:
	"""Every team the sampler leaves out, in pool order, with its unsupported ids."""
	supported = team_supported(pool, tables)
	move_mask, ability_mask, item_mask = support_masks(tables)

	def unsupported(values, mask, table: RowTable) -> tuple[str, ...]:
		rows = np.unique(values[~mask[values]])
		return tuple(sorted(str(table.ids[row]) or NO_ID for row in rows))

	excluded = []
	for index in np.flatnonzero(~supported):
		excluded.append(
			ExcludedTeam(
				index=int(index),
				moves=unsupported(pool.moves[index], move_mask, tables.moves),
				abilities=unsupported(pool.ability[index], ability_mask, tables.abilities),
				items=unsupported(pool.item[index], item_mask, tables.items),
			)
		)
	return tuple(excluded)


def sample_team_indices(key, team_indices: jax.Array, batch: int) -> jax.Array:
	"""Pool indices [batch, 2] for the two sides, drawn uniformly from team_indices with replacement."""
	if team_indices.ndim != 1 or team_indices.shape[0] == 0:
		raise ValueError("team_indices must be a nonempty one-dimensional array")
	picks = jax.random.randint(key, (batch, 2), 0, team_indices.shape[0])
	return team_indices[picks]


def load_pool_file(path, tables: Tables) -> TeamPool:
	"""Read a pool saved as .npz, or the JSON export from the Node team generator."""
	path = Path(path)
	if path.suffix == ".npz":
		return load_npz(path)
	return load_team_pool(path, tables)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("pool_path", type=Path)
	args = parser.parse_args()
	tables = load_tables()
	indices = supported_team_indices(load_pool_file(args.pool_path, tables), tables)
	print(json.dumps(indices.tolist()))


if __name__ == "__main__":
	main()
