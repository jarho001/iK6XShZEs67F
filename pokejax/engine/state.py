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
CHOICE_MOVE = 19
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
	"choicelock",
	"flashfire",
	"unburden",
	"truant",
	"disable",
	"curse",
	"autotomize",
)
FLASH_FIRE = 20
UNBURDEN, TRUANT, DISABLE_MOVE, DISABLE_REMAINING = range(21, 25)
SLOW_START = 25
CURSE = 26
AUTOTOMIZE = 27
# Task 13 volatiles: Substitute holds its HP here; Encore keeps the move and its
# duration; Taunt, Yawn and Destiny Bond keep a duration or flag.
SUBSTITUTE = 28
ENCORE_MOVE = 29
ENCORE_REMAINING = 30
TAUNT = 31
YAWN = 32
DESTINY_BOND = 33
# Task 14 volatiles: a Transformed active (Transform or Imposter); the Illusion disguise as
# the disguised member's slot plus one (zero means none); Trace's seek flag.
TRANSFORMED = 34
ILLUSION_SLOT = 35
TRACE_SEEK = 36
NUM_VOLATILES = 37
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
	The event record (events.py) is cleared at the start of each step and holds only that
	step's visible events. weather_age, trick_room_age and wish_age count turns since each
	effect started, from the protocol; the observation exposes those, never a remaining count.
	"""

	species: jax.Array
	level: jax.Array
	types: jax.Array
	base_types: jax.Array
	roost_types: jax.Array
	stats: jax.Array
	hp: jax.Array
	moves: jax.Array
	pp: jax.Array
	max_pp: jax.Array
	ability: jax.Array  # int16 [2, 6], current ability
	ability_base: jax.Array  # int16 [2, 6], restored when volatile state clears
	ability_public: jax.Array  # int16 [2, 6], last protocol reveal
	item: jax.Array  # int16 [2, 6], current held item ID; zero means none
	item_consumed: jax.Array  # bool [2, 6], removed by a consumable effect
	item_knocked_off: jax.Array  # bool [2, 6], removed by Knock Off or another takeItem
	item_revealed: jax.Array  # bool [2, 6], protocol has publicly named the item
	item_public: jax.Array  # int16 [2, 6], last item ID named in the public protocol
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
	turn_damage: jax.Array  # int16 [2, 3]: last any, physical, special damage taken this turn
	turn_hit: jax.Array  # bool [2]: took a damaging move hit this turn (Metal Burst)
	focus_lost: jax.Array  # bool [2]: hit by a damaging move this turn (Focus Punch)
	last_success: jax.Array  # int16 scalar: last successful move ID this turn (Fusion Bolt/Flare)
	fainted_last: jax.Array  # bool [2]: a member fainted in the previous turn (Retaliate)
	alive_mark: jax.Array  # bool [2, 6]: members alive at the last turn end
	last_move: jax.Array  # int16 [2, 6]: last move used by each member; cleared on switch-out
	wish_turns: jax.Array  # int8 [2]: Wish turns left per slot; zero means no Wish
	wish_hp: jax.Array  # int16 [2]: HP the pending Wish restores
	healing_wish: jax.Array  # bool [2]: Healing Wish waits for the next switch-in to that side
	trick_room: jax.Array  # int8 scalar: Trick Room turns left; zero means inactive
	# Task 14 forms. species_base is each member's own species; a member leaving the field
	# restores its own species, and saved_stats / saved_* hold the active side's own stats and
	# moveset while it is Transformed or a Meloetta has changed forme. shown_species is
	# the species the opponent was last told (Illusion keeps the disguise until it breaks).
	species_base: jax.Array  # int16 [2, 6]
	saved_stats: jax.Array  # int16 [2, 5]: the active member's own Atk..Spe before a stat change
	saved_moves: jax.Array  # int16 [2, 4]
	saved_pp: jax.Array  # uint8 [2, 4]
	saved_max_pp: jax.Array  # uint8 [2, 4]
	saved_revealed: jax.Array  # uint8 [2, 4]
	# Atk..Spe a Meloetta member takes in Pirouette forme, derived at reset from its immutable
	# level, IVs, EVs and nature (other members hold unused values).
	pirouette_stats: jax.Array  # int16 [2, 6, 5]
	shown_species: jax.Array  # int16 [2, 6]
	# Showdown's side.pokemon order (array index to team slot). Index 0 is the active member;
	# a switch-in swaps the entering and leaving positions. Illusion's disguise reads it.
	party: jax.Array  # int8 [2, 6]
	event: jax.Array  # int8 [2, EVENT_FIELDS]: the step's visible events (events.py columns)
	event_move: jax.Array  # int16 [2]: the move each side last used in the step
	weather_age: jax.Array  # int8 scalar: turns since the current weather started (0 if none)
	trick_room_age: jax.Array  # int8 scalar: turns since Trick Room started (0 if inactive)
	wish_age: jax.Array  # int8 [2]: turns since each side's pending Wish was used
