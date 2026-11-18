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
(
	CONFUSION,
	FLINCH,
	ROOST,
	LEECH_SEED,
	PARTIALLY_TRAPPED,
	PROTECT,
	ENDURE,
	STALL,
	TWO_TURN_MOVE,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	MUST_RECHARGE,
	QUICK_GUARD,
	WIDE_GUARD,
	FAKE_OUT_USED,
	PARTIAL_TRAP_SOURCE,
	STALL_USED,
	LOCKED_TRUE_DURATION,
) = range(18)
SMACKDOWN = 18
VOLATILE_NAMES = (
	"confusion",
	"flinch",
	"roost",
	"leechseed",
	"partiallytrapped",
	"protect",
	"endure",
	"stall",
	"twoturnmove",
	"lockedmove",
	"mustrecharge",
	"fly",
	"dig",
	"bounce",
	"dive",
	"shadowforce",
	"skullbash",
	"skyattack",
	"razorwind",
	"solarbeam",
	"freezeshock",
	"iceburn",
	"smackdown",
)
NUM_VOLATILES = 19
HP, ATK, DEF, SPA, SPD, SPE = range(NUM_STATS)
BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, BOOST_SPE, ACCURACY, EVASION = range(NUM_BOOSTS)
EMPTY = 0
HEALTHY, BURN, POISON, TOXIC, PARALYSIS, SLEEP, FREEZE = range(7)
STATUS_NAMES = ("", "brn", "psn", "tox", "par", "slp", "frz")
NO_TYPE = -1
NONE, MOVE, SWITCH = range(3)
ONGOING, SIDE0, SIDE1, TIE = -1, 0, 1, 2
FIRST_TURN = 1
NO_WEATHER, RAIN, SUN, SAND, HAIL = range(5)
WEATHER_NAMES = ("", "raindance", "sunnyday", "sandstorm", "hail")
NUM_WEATHERS = len(WEATHER_NAMES)
SPIKES, TOXIC_SPIKES = range(2)

# Resumable turn stages. The stored values describe hidden queue work; only the
# public request and the boolean ``turn_stage > 0`` are exposed to a policy.
TURN_IDLE, TURN_MOVES, TURN_RESIDUAL, TURN_REPLACEMENT = range(4)


class BattleState(NamedTuple):
	"""Member axes [side, slot]; moves/stats/types add their final axis.

	Species, moves, ability, item and stored stats/HP are int16. Level, PP,
	EVs, IVs and happiness are uint8. Other member/side fields are int8;
	turn is int32. Boost order is atk/def/spa/spd/spe/accuracy/evasion.
	seen_order is int8 [2, 6]: 0 for unseen, otherwise 1-based first appearance.
	revealed_moves is uint8 [2, 6, 4], indexed by the owner's fixed move slots.
	sleep_start and sleep_from_foe are [2, 6] bytes for the Gen 5 switch reset
	and Sleep Clause, respectively; status and status_counter are [2, 6] int8.
	volatiles is int16 [2, NUM_VOLATILES], active only: zero is absent;
	confusion stores the private BeforeMove timer (2..5 initially), flinch and
	roost are 1. Charge/lock state stores move IDs and private turn counters; trap
	state stores a private duration and source slot. These counters are never
	observed. The wider dtype also fits future volatile HP (Substitute) in another column. Weather is
	an int8 enum (0 clear, 1 rain, 2 sun, 3 sand, 4 hail); zero turns means
	permanent weather or no weather. Hazards and their private switch-in creation
	order are int8 [side, spikes/toxic spikes].
	"""

	species: jax.Array
	level: jax.Array
	types: jax.Array
	roost_types: jax.Array
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
	sleep_start: jax.Array
	sleep_from_foe: jax.Array
	seen_order: jax.Array
	revealed_moves: jax.Array
	active: jax.Array
	active_turns: jax.Array  # uint8 [2, 6], increments at Showdown turn advance
	boosts: jax.Array
	volatiles: jax.Array  # int16 [2, NUM_VOLATILES], active Pokemon only
	weather: jax.Array  # int8 scalar; NO_WEATHER, RAIN, SUN, SAND, HAIL
	weather_turns: jax.Array  # int8 scalar; zero means permanent or no weather
	hazards: jax.Array  # int8 [2, 2]: Spikes, Toxic Spikes layers on each side
	hazard_order: jax.Array  # int8 [2, 2]: relative creation order for switch-in handlers
	request: jax.Array
	turn_stage: jax.Array  # int8 scalar; idle, saved moves, residual, replacement
	turn_actions: jax.Array  # int8 [2], original move slots for the saved queue
	turn_moves: jax.Array  # int16 [2], resolved move IDs for the saved queue
	turn_pending: jax.Array  # bool [2], queued actions that have not run
	turn_first: jax.Array  # int8 scalar, first side in the fixed move order
	turn: jax.Array
	winner: jax.Array
