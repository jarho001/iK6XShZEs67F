"""Engine support: later phases add ids in the same commit that implements them."""

import numpy as np
from numpy.typing import NDArray

from pokejax.tables import NONE_ROW, Tables
from pokejax.teams import TeamPool

SUPPORTED_MOVES: frozenset[str] = frozenset(
	[
		"aircutter",
		"aquajet",
		"aquatail",
		"attackorder",
		"bulletpunch",
		"crabhammer",
		"crosschop",
		"cut",
		"dragonclaw",
		"drillrun",
		"earthquake",
		"eggbomb",
		"extremespeed",
		"feintattack",
		"hornattack",
		"hydropump",
		"hypervoice",
		"iceshard",
		"karatechop",
		"leafblade",
		"machpunch",
		"magicalleaf",
		"magnetbomb",
		"megahorn",
		"megakick",
		"megapunch",
		"nightslash",
		"payday",
		"pound",
		"powergem",
		"powerwhip",
		"psychocut",
		"quickattack",
		"razorleaf",
		"rockthrow",
		"scratch",
		"seedbomb",
		"shadowclaw",
		"shadowpunch",
		"shadowsneak",
		"shockwave",
		"skyuppercut",
		"slam",
		"slash",
		"spacialrend",
		"stoneedge",
		"strength",
		"surf",
		"swift",
		"tackle",
		"vacuumwave",
		"vinewhip",
		"visegrip",
		"vitalthrow",
		"watergun",
		"xscissor",
	]
)
SUPPORTED_ABILITIES: frozenset[str] = frozenset()
SUPPORTED_ITEMS: frozenset[str] = frozenset()


def support_masks(tables: Tables) -> tuple[NDArray[np.bool_], ...]:
	"""Return move, ability and item support lookups indexed by table row."""
	masks = tuple(
		np.isin(table.ids, tuple(supported))
		for table, supported in (
			(tables.moves, SUPPORTED_MOVES),
			(tables.abilities, SUPPORTED_ABILITIES),
			(tables.items, SUPPORTED_ITEMS),
		)
	)
	masks[0][NONE_ROW] = True
	masks[2][NONE_ROW] = True
	return masks


def team_supported(pool: TeamPool, tables: Tables) -> NDArray[np.bool_]:
	"""Check every move, ability and item without looping over teams."""
	moves, abilities, items = support_masks(tables)
	return (
		np.all(moves[pool.moves], axis=(1, 2))
		& np.all(abilities[pool.ability], axis=1)
		& np.all(items[pool.item], axis=1)
	)


if __name__ == "__main__":
	import json

	print(
		json.dumps(
			{
				"moves": sorted(SUPPORTED_MOVES),
				"abilities": sorted(SUPPORTED_ABILITIES),
				"items": sorted(SUPPORTED_ITEMS),
			}
		)
	)
