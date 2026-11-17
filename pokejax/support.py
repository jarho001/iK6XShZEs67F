"""Engine support: later phases add ids in the same commit that implements them."""

import numpy as np
from numpy.typing import NDArray

from pokejax.tables import NONE_ROW, Tables
from pokejax.teams import TeamPool

SUPPORTED_MOVES: frozenset[str] = frozenset(
	[
		"agility",
		"aircutter",
		"aquajet",
		"aquatail",
		"aromatherapy",
		"attackorder",
		"bellydrum",
		"bulkup",
		"bulletpunch",
		"calmmind",
		"clearsmog",
		"closecombat",
		"coil",
		"cosmicpower",
		"crabhammer",
		"crosschop",
		"cut",
		"darkvoid",
		"dracometeor",
		"dragonclaw",
		"dragondance",
		"drillrun",
		"earthquake",
		"eggbomb",
		"extremespeed",
		"feintattack",
		"glare",
		"haze",
		"healbell",
		"honeclaws",
		"hornattack",
		"hydropump",
		"hypervoice",
		"hypnosis",
		"iceshard",
		"karatechop",
		"leafblade",
		"leafstorm",
		"lovelykiss",
		"machpunch",
		"magicalleaf",
		"magnetbomb",
		"megahorn",
		"megakick",
		"megapunch",
		"nastyplot",
		"nightslash",
		"overheat",
		"payday",
		"pound",
		"powergem",
		"powerwhip",
		"psychoboost",
		"psychocut",
		"psychoshift",
		"quickattack",
		"quiverdance",
		"razorleaf",
		"refresh",
		"rest",
		"rockpolish",
		"rockthrow",
		"scratch",
		"seedbomb",
		"shadowclaw",
		"shadowpunch",
		"shadowsneak",
		"shellsmash",
		"shiftgear",
		"shockwave",
		"skyuppercut",
		"slam",
		"slash",
		"sleeppowder",
		"spacialrend",
		"spore",
		"stoneedge",
		"strength",
		"stunspore",
		"superpower",
		"surf",
		"swift",
		"swordsdance",
		"tackle",
		"tailglow",
		"thunderwave",
		"toxic",
		"vacuumwave",
		"vcreate",
		"vinewhip",
		"visegrip",
		"vitalthrow",
		"watergun",
		"willowisp",
		"workup",
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
