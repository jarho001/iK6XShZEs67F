"""Engine support: later phases add ids in the same commit that implements them."""

import numpy as np
from numpy.typing import NDArray

from pokejax.tables import NONE_ROW, Tables
from pokejax.teams import TeamPool

# A group of rows, not per-move execution branches. Fake Out and form changes
# remain in their roadmap groups.
SECONDARY_MOVES = frozenset(
	[
		"airslash",
		"ancientpower",
		"bite",
		"blazekick",
		"blizzard",
		"blueflare",
		"bodyslam",
		"boltstrike",
		"bugbuzz",
		"bulldoze",
		"chargebeam",
		"chatter",
		"crunch",
		"darkpulse",
		"discharge",
		"dynamicpunch",
		"earthpower",
		"energyball",
		"fierydance",
		"fireblast",
		"firefang",
		"firepunch",
		"flamecharge",
		"flamethrower",
		"flamewheel",
		"flashcannon",
		"focusblast",
		"gunkshot",
		"headbutt",
		"heatwave",
		"hurricane",
		"icebeam",
		"icefang",
		"icepunch",
		"iciclecrash",
		"ironhead",
		"irontail",
		"lavaplume",
		"meteormash",
		"ominouswind",
		"poisonjab",
		"psychic",
		"rockslide",
		"sacredfire",
		"scald",
		"seedflare",
		"shadowball",
		"signalbeam",
		"silverwind",
		"sludgebomb",
		"sludgewave",
		"thunder",
		"thunderbolt",
		"thunderfang",
		"thunderpunch",
		"triattack",
		"waterfall",
		"zenheadbutt",
	]
)
CONFUSION_MOVES = frozenset(
	["confuseray", "flatter", "supersonic", "swagger", "sweetkiss", "teeterdance"]
)


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
		"hail",
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
		"raindance",
		"rapidspin",
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
		"spikes",
		"strength",
		"stunspore",
		"superpower",
		"surf",
		"swift",
		"swordsdance",
		"tackle",
		"tailglow",
		"thunderwave",
		"toxicspikes",
		"toxic",
		"vacuumwave",
		"vcreate",
		"vinewhip",
		"visegrip",
		"vitalthrow",
		"watergun",
		"weatherball",
		"willowisp",
		"workup",
		"sunnyday",
		"sandstorm",
		"xscissor",
	]
)
SUPPORTED_MOVES |= SECONDARY_MOVES | CONFUSION_MOVES
MOVE_EFFECT_MOVES = frozenset(
	[
		"absorb",
		"aeroblast",
		"bonemerang",
		"bravebird",
		"bulletseed",
		"doubleedge",
		"doublekick",
		"drainpunch",
		"dualchop",
		"explosion",
		"flareblitz",
		"frostbreath",
		"geargrind",
		"gigadrain",
		"headcharge",
		"headsmash",
		"highjumpkick",
		"hornleech",
		"iciclespear",
		"jumpkick",
		"karatechop",
		"leechlife",
		"megadrain",
		"milkdrink",
		"moonlight",
		"morningsun",
		"nightshade",
		"painsplit",
		"pinmissile",
		"recover",
		"rockblast",
		"roost",
		"seismictoss",
		"selfdestruct",
		"slackoff",
		"softboiled",
		"sonicboom",
		"spacialrend",
		"stormthrow",
		"submission",
		"superfang",
		"synthesis",
		"tailslap",
		"takedown",
		"twineedle",
		"volttackle",
		"wildcharge",
		"woodhammer",
	]
)
SUPPORTED_MOVES |= MOVE_EFFECT_MOVES

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
