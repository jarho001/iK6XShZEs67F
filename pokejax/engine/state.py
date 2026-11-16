"""Fixed-shape state for one battle; batch externally with vmap."""

from typing import NamedTuple

import jax

from pokejax import teams

MOVE_SLOTS = teams.MOVE_SLOTS
TEAM_SLOTS = teams.TEAM_SLOTS

NUM_SIDES = 2
NUM_TYPES = 2
NUM_STATS = 6
NUM_BOOSTS = 7
HP, ATK, DEF, SPA, SPD, SPE = range(NUM_STATS)
BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE, ACCURACY, EVASION = range(NUM_BOOSTS)
EMPTY = 0
NO_TYPE = -1
NONE, MOVE, SWITCH = range(3)
ONGOING, SIDE0, SIDE1, TIE = -1, 0, 1, 2
FIRST_TURN = 1


class BattleState(NamedTuple):
	"""Member axes [side, slot]; moves/stats/types add their final axis.

	Species, moves, ability, item and stored stats/HP are int16. Level, PP,
	EVs, IVs and happiness are uint8. Other member/side fields are int8;
	turn is int32. Boost order is atk/def/spa/spd/spe/accuracy/evasion.
	seen_order is int8 [2, 6]: 0 for unseen, otherwise 1-based first appearance.
	revealed_moves is uint8 [2, 6, 4], indexed by the owner's fixed move slots.
	"""

	species: jax.Array
	level: jax.Array
	types: jax.Array
	stats: jax.Array
	hp: jax.Array
	moves: jax.Array
	pp: jax.Array
	max_pp: jax.Array
	ability: jax.Array
	item: jax.Array
	nature: jax.Array
	gender: jax.Array
	evs: jax.Array
	ivs: jax.Array
	happiness: jax.Array
	status: jax.Array
	status_counter: jax.Array
	seen_order: jax.Array
	revealed_moves: jax.Array
	active: jax.Array
	boosts: jax.Array
	request: jax.Array
	turn: jax.Array
	winner: jax.Array
