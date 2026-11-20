"""Sample or force addressed outcomes, independently of evaluation order.

The harness supplies an int8 Forced array [NUM_KINDS, 2, MAX_OCCURRENCES].
-1 means sample; other values specify the outcome, not a PRNG draw. Records
use exactly kind/side/occurrence/value keys. Kind names and row order are in
KIND_NAMES; append a name to add a kind without changing existing addresses.

order: side must be 0; value is the first side (0/1). Occurrences are tie
contexts ACTION_ORDER=0, SWITCH_IN_ORDER=1, locked-move residual=2,
WEATHER_ORDER=3, WEATHER_CHANGE_ORDER=4, SWITCH_HANDLER_ORDER=5,
poison/toxic residual=6, burn residual=7, partial-trap residual=8, and ability
residual=9. Ability and held-item residual handler shuffles have their own
one-slot outcome kinds; Roost's type-reset shuffle has its own one-slot kind.
sleep_talk: side is the Sleep Talk user, occurrence 0, value 0..2 indexes the
callable moveset slots in slot order (Showdown samples that list uniformly).
hit: side is attacker, occurrence is hit index, value 1=hit, 0=miss. Consult
only if the move checks accuracy. crit: same address axes, value 1=crit, 0=no.
damage_roll: same axes, value 0..15 is Showdown random(16), multiplying damage
by (100-value)/100. Occurrences are zero-based.
full_paralysis and thaw: side is the affected Pokemon, value 0/1. sleep_duration:
side is the recipient, value 0..2, added to 2 for Showdown's initial timer.
secondary: attacker side, occurrence is the secondary row (fangs use 0 and 1),
value 0/1 is whether its chance passed. secondary_status: same axes, value 0..2
selects Tri Attack's burn/paralysis/freeze. confusion_duration: recipient side,
occurrence 0, value 0..3 added to 2. confusion_hit: affected side, occurrence 0,
value 1=self-hit, 0=acts. confusion_damage_roll: same axes, value 0..15.
hit_count: attacker side, occurrence 0, value 2..5. Gen 5 uses 35% each for 2 and 3 hits,
then 15% each for 4 and 5.
drag_in: side being dragged, occurrence 0, value is the selected fixed team slot
0..5. This forces the chosen member; unforced sampling is uniform over its
currently usable bench.
bounce_hit: Magic Bounce's reflected move accuracy check: side is the reflecting holder, value
1=hit, 0=miss. ability_chance: on-hit ability randomChance outcomes, occurrence 2 * hit index
for the defending Pokemon and 2 * hit index + 1 for the attacker. effect_spore
is Showdown random(100). Harvest and residual_chance address their end-of-turn
ability chances by holder side.

forced=None is an unforced run: every outcome samples, and the rollout and bench pass it.
A step keeps the key's two words and hashes each address (kind, side, occurrence) when
it is used. integer requires n>0; chance requires 0<=numerator<=denominator and denominator>0.
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
	("hit_count", 5),
	("stall_success", 1),
	("trap_duration", 1),
	("lock_duration", 1),
	("magnitude", 99),
	("drag_in", 5),
	("ability_chance", 1),
	("effect_spore", 99),
	("harvest", 1),
	("residual_chance", 1),
	("ability_residual_order", 1),
	("item_residual_order", 1),
	("roost_residual_order", 1),
	("sleep_talk", 3),
	("bounce_hit", 1),
)
KIND_NAMES = tuple(name for name, _ in OUTCOME_KINDS)
NUM_KINDS = len(KIND_NAMES)
ORDER, HIT, CRIT, DAMAGE_ROLL = map(KIND_NAMES.index, ("order", "hit", "crit", "damage_roll"))
FULL_PARALYSIS, THAW, SLEEP_DURATION = map(KIND_NAMES.index, KIND_NAMES[4:7])
SECONDARY, SECONDARY_STATUS, CONFUSION_DURATION, CONFUSION_HIT, CONFUSION_DAMAGE_ROLL = map(
	KIND_NAMES.index, KIND_NAMES[7:12]
)
HIT_COUNT = KIND_NAMES.index("hit_count")
STALL_SUCCESS, TRAP_DURATION, LOCK_DURATION = map(
	KIND_NAMES.index, ("stall_success", "trap_duration", "lock_duration")
)
MAGNITUDE = KIND_NAMES.index("magnitude")
DRAG_IN = KIND_NAMES.index("drag_in")
ABILITY_CHANCE, EFFECT_SPORE, HARVEST = map(
	KIND_NAMES.index, ("ability_chance", "effect_spore", "harvest")
)
RESIDUAL_CHANCE = KIND_NAMES.index("residual_chance")
ABILITY_RESIDUAL_ORDER = KIND_NAMES.index("ability_residual_order")
ITEM_RESIDUAL_ORDER = KIND_NAMES.index("item_residual_order")
ROOST_RESIDUAL_ORDER = KIND_NAMES.index("roost_residual_order")
SLEEP_TALK = KIND_NAMES.index("sleep_talk")
# Magic Bounce's reflected move: its own accuracy check (side is the reflecting holder).
BOUNCE_HIT = KIND_NAMES.index("bounce_hit")
OUTCOME_MAXIMUM = tuple(maximum for _, maximum in OUTCOME_KINDS)
OUTCOME_MINIMUM = tuple(2 if name == "hit_count" else 0 for name, _ in OUTCOME_KINDS)
(
	ACTION_ORDER,
	SWITCH_IN_ORDER,
	RESIDUAL_ORDER,
	WEATHER_ORDER,
	WEATHER_CHANGE_ORDER,
	SWITCH_HANDLER_ORDER,
) = range(6)
# Secondary outcomes address hit_index * SECONDARY_SLOTS + row; fifth-hit row
# two is occurrence 9, so ten slots cover the largest table-defined sequence.
MAX_OCCURRENCES = 10
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
		order_kinds = (ORDER, ABILITY_RESIDUAL_ORDER, ITEM_RESIDUAL_ORDER, ROOST_RESIDUAL_ORDER)
		if kind in order_kinds and (
			side != 0 or (kind != ORDER and occ != 0) or occ not in tuple(range(MAX_OCCURRENCES))
		):
			raise ValueError("Invalid order context")
		minimum, maximum = OUTCOME_MINIMUM[kind], OUTCOME_MAXIMUM[kind]
		if not minimum <= value <= maximum:
			raise ValueError("Outcome value out of range")
		if values[kind, side, occ] != NOT_FORCED:
			raise ValueError("Duplicate outcome address")
		values[kind, side, occ] = value
	return values if host else jnp.asarray(values)


_FMIX_MUL_1, _FMIX_MUL_2, _ADDRESS_MUL = 0x85EBCA6B, 0xC2B2AE35, 0x9E3779B9


def _key_words(key):
	"""The two uint32 words of a step key; a words array passes through unchanged."""
	if jnp.issubdtype(key.dtype, jax.dtypes.prng_key):
		key = jax.random.key_data(key)
	return key.astype(jnp.uint32).reshape(-1)[:2]


def random_bits(key):
	"""Per-step randomness: the key's two words. Each address hashes them on use.

	Hashing an address costs a few fused integer ops, so a step never materializes
	the whole kinds x sides x occurrences table.
	"""
	return _key_words(key)


def address_bits(key, kind, side, occ):
	"""Addressed 32-bit draw: a murmur3 finalizer over the key words and the address."""
	words = _key_words(key)
	address = jnp.asarray((kind * NUM_SIDES + side) * MAX_OCCURRENCES + occ, jnp.uint32)
	mixed = _fmix32(words[0] ^ (address * jnp.uint32(_ADDRESS_MUL)))
	return _fmix32(mixed ^ words[1])


def _fmix32(value):
	value = value ^ (value >> 16)
	value = value * jnp.uint32(_FMIX_MUL_1)
	value = value ^ (value >> 13)
	value = value * jnp.uint32(_FMIX_MUL_2)
	return value ^ (value >> 16)


def forced_at(forced, kind, side, occ):
	"""The forced value at one address; None (an unforced run) means every outcome samples."""
	if forced is None:
		return jnp.asarray(NOT_FORCED, jnp.int8)
	return forced[kind, side, occ]


def _sample_or_force(key, forced, kind, side, occ, n):
	sample = (address_bits(key, kind, side, occ) % jnp.asarray(n, jnp.uint32)).astype(jnp.int32)
	if forced is None:
		return sample
	value = forced[kind, side, occ].astype(jnp.int32)
	return jnp.where(value >= 0, value, sample)


def integer(key, forced, kind, side, occ, n):
	"""Return a forced outcome or a uniform integer in [0, n)."""
	return _sample_or_force(key, forced, kind, side, occ, n)


def hit_count(key, forced, side, hit_min, hit_max):
	"""Sample Gen 5's 2-5 distribution or force the resulting hit count."""
	bits = address_bits(key, HIT_COUNT, side, 0)
	selector = bits % jnp.asarray(20, jnp.uint32)
	gen5_count = jnp.where(
		selector < 7, 2, jnp.where(selector < 14, 3, jnp.where(selector < 17, 4, 5))
	)
	low = hit_min.astype(jnp.uint32)
	span = (hit_max - hit_min + 1).astype(jnp.uint32)
	uniform = low + bits % span
	sample = jnp.where((hit_min == 2) & (hit_max == 5), gen5_count, uniform)
	value = forced_at(forced, HIT_COUNT, side, 0).astype(jnp.int32)
	return jnp.where(value >= 0, value, sample.astype(jnp.int32))


def chance(key, forced, kind, side, occ, numerator, denominator):
	"""Return a forced boolean or randomChance(numerator, denominator)."""
	# Convert the sample to an outcome before forcing: forced 0/1 is not a draw.
	sample = integer(key, None, kind, side, occ, denominator) < numerator
	value = forced_at(forced, kind, side, occ)
	return jnp.where(value >= 0, value != 0, sample)


def first_side(key, forced, context=ACTION_ORDER):
	"""Choose the side going first for an order tie context."""
	return integer(key, forced, ORDER, 0, context, NUM_SIDES)
