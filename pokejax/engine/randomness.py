"""Sample or force addressed outcomes, independently of evaluation order.

The harness supplies an int8 Forced array [NUM_KINDS, 2, MAX_OCCURRENCES].
-1 means sample; other values specify the outcome, not a PRNG draw. Records
use exactly kind/side/occurrence/value keys. Kind names and row order are in
KIND_NAMES; append a name to add a kind without changing existing addresses.

order: side must be 0; value is the first side (0/1). Occurrences are tie
contexts ACTION_ORDER=0, SWITCH_IN_ORDER=1, RESIDUAL_ORDER=2 (others reserved).
hit: side is attacker, occurrence is hit index, value 1=hit, 0=miss. Consult
only if the move checks accuracy. crit: same address axes, value 1=crit, 0=no.
damage_roll: same axes, value 0..15 is Showdown random(16), multiplying damage
by (100-value)/100. Occurrences are zero-based, at most 7.
full_paralysis and thaw: side is the affected Pokemon, value 0/1. sleep_duration:
side is the recipient, value 0..2, added to 2 for Showdown's initial timer.
secondary: attacker side, occurrence is the secondary row (fangs use 0 and 1),
value 0/1 is whether its chance passed. secondary_status: same axes, value 0..2
selects Tri Attack's burn/paralysis/freeze. confusion_duration: recipient side,
occurrence 0, value 0..3 added to 2. confusion_hit: affected side, occurrence 0,
value 1=self-hit, 0=acts. confusion_damage_roll: same axes, value 0..15.

The step draws a table of random bits once, addressed by kind, side and occurrence. integer
requires n>0; chance requires 0<=numerator<=denominator and denominator>0.
These are caller preconditions. A forced integer must be less than n at its
call site. Samples still evaluate when forced; selection is branch-free.
"""

import jax
import jax.numpy as jnp
import numpy as np

OUTCOME_KINDS = (
	("order", 1),
	("hit", 1),
	("crit", 1),
	("damage_roll", 15),
	("full_paralysis", 1),
	("thaw", 1),
	("sleep_duration", 2),
	("secondary", 1),
	("secondary_status", 2),
	("confusion_duration", 3),
	("confusion_hit", 1),
	("confusion_damage_roll", 15),
)
KIND_NAMES = tuple(name for name, _ in OUTCOME_KINDS)
NUM_KINDS = len(KIND_NAMES)
ORDER, HIT, CRIT, DAMAGE_ROLL = map(KIND_NAMES.index, ("order", "hit", "crit", "damage_roll"))
FULL_PARALYSIS, THAW, SLEEP_DURATION = map(KIND_NAMES.index, KIND_NAMES[4:7])
SECONDARY, SECONDARY_STATUS, CONFUSION_DURATION, CONFUSION_HIT, CONFUSION_DAMAGE_ROLL = map(
	KIND_NAMES.index, KIND_NAMES[7:]
)
OUTCOME_MAXIMUM = tuple(maximum for _, maximum in OUTCOME_KINDS)
ACTION_ORDER, SWITCH_IN_ORDER, RESIDUAL_ORDER = range(3)
MAX_OCCURRENCES = 8
NUM_SIDES = 2
NOT_FORCED = -1
type Forced = jax.Array
UNFORCED = jnp.full((NUM_KINDS, NUM_SIDES, MAX_OCCURRENCES), NOT_FORCED, dtype=jnp.int8)


def forced_from_records(records, *, host=False) -> Forced:
	"""Validate host records; reject duplicates rather than silently overwriting."""
	values = np.full(UNFORCED.shape, NOT_FORCED, dtype=np.int8)
	for record in records:
		if not isinstance(record, dict) or set(record) != {"kind", "side", "occurrence", "value"}:
			raise ValueError("Expected kind, side, occurrence and value")
		name = record["kind"]
		if name not in KIND_NAMES:
			raise ValueError(f"Unknown outcome kind: {name!r}")
		kind = KIND_NAMES.index(name)
		side, occ, value = (record[key] for key in ("side", "occurrence", "value"))
		if any(type(v) is not int for v in (side, occ, value)):
			raise ValueError("Outcome axes and value must be integers")
		if not 0 <= side < NUM_SIDES or not 0 <= occ < MAX_OCCURRENCES:
			raise ValueError("Outcome address out of range")
		if kind == ORDER and (side != 0 or occ > RESIDUAL_ORDER):
			raise ValueError("Invalid order context")
		maximum = OUTCOME_MAXIMUM[kind]
		if not 0 <= value <= maximum:
			raise ValueError("Outcome value out of range")
		if values[kind, side, occ] != NOT_FORCED:
			raise ValueError("Duplicate outcome address")
		values[kind, side, occ] = value
	return values if host else jnp.asarray(values)


def random_bits(key):
	"""Draw independent bits for every address used by a step."""
	return jax.random.bits(key, (NUM_KINDS, NUM_SIDES, MAX_OCCURRENCES), dtype=jnp.uint32)


def _sample_or_force(key, forced, kind, side, occ, n):
	# Direct helper calls accept a PRNG key too; step passes the shared table.
	bits = random_bits(key) if key.ndim < 3 else key
	sample = (bits[kind, side, occ] % jnp.asarray(n, jnp.uint32)).astype(jnp.int32)
	value = forced[kind, side, occ].astype(jnp.int32)
	return jnp.where(value >= 0, value, sample)


def integer(key, forced, kind, side, occ, n):
	"""Return a forced outcome or a uniform integer in [0, n)."""
	return _sample_or_force(key, forced, kind, side, occ, n)


def chance(key, forced, kind, side, occ, numerator, denominator):
	"""Return a forced boolean or randomChance(numerator, denominator)."""
	# Convert the sample to an outcome before forcing: forced 0/1 is not a draw.
	sample = integer(key, UNFORCED, kind, side, occ, denominator) < numerator
	value = forced[kind, side, occ]
	return jnp.where(value >= 0, value != 0, sample)


def first_side(key, forced, context=ACTION_ORDER):
	"""Choose the side going first for an order tie context."""
	return integer(key, forced, ORDER, 0, context, NUM_SIDES)
