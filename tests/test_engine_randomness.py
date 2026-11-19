"""Addressed random outcomes under jit/vmap, forced replay and host validation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.randomness import (
	ACTION_ORDER,
	CRIT,
	DAMAGE_ROLL,
	HIT,
	KIND_NAMES,
	MAX_OCCURRENCES,
	NUM_KINDS,
	ORDER,
	RESIDUAL_ORDER,
	SWITCH_IN_ORDER,
	UNFORCED,
	chance,
	first_side,
	forced_from_records,
	hit_count,
	integer,
	random_bits,
)


def test_forced_outcomes_exact():
	records = []
	for name in KIND_NAMES:
		order_kind = name in {
			"order",
			"ability_residual_order",
			"item_residual_order",
			"roost_residual_order",
		}
		for side in range(1 if order_kind else 2):
			for occ in range(3 if name == "order" else 1 if order_kind else MAX_OCCURRENCES):
				records.append(
					{
						"kind": name,
						"side": side,
						"occurrence": occ,
						"value": (
							2 + (occ + side) % 4
							if name == "hit_count"
							else (occ + side) % (16 if name == "damage_roll" else 2)
						),
					}
				)
	forced = forced_from_records(records)
	assert forced.dtype == jnp.int8
	assert forced.shape == (NUM_KINDS, 2, MAX_OCCURRENCES)
	key = jax.random.key(42)
	jit_int = jax.jit(integer)
	jit_chance = jax.jit(chance)
	for record in records:
		kind, side, occ, value = (
			KIND_NAMES.index(record["kind"]),
			record["side"],
			record["occurrence"],
			record["value"],
		)
		assert int(jit_int(key, forced, kind, side, occ, 16)) == value
		if kind in (HIT, CRIT):
			# Opposite guaranteed probabilities show forcing chooses outcomes, not draws.
			assert bool(jit_chance(key, forced, kind, side, occ, 1 - value, 1)) == bool(value)
	for context in (ACTION_ORDER, SWITCH_IN_ORDER, RESIDUAL_ORDER):
		assert int(jax.jit(first_side)(key, forced, context)) == context % 2


def test_frequencies_and_address_independence():
	keys = jax.random.split(jax.random.key(12), 32768)

	@jax.jit
	@jax.vmap
	def outcomes(key):
		rolls = jnp.stack(
			[
				integer(key, UNFORCED, DAMAGE_ROLL, 0, 0, 16),
				integer(key, UNFORCED, DAMAGE_ROLL, 1, 0, 16),
				integer(key, UNFORCED, DAMAGE_ROLL, 0, 1, 16),
				integer(key, UNFORCED, CRIT, 0, 0, 16),
			]
		)
		return (
			rolls,
			chance(key, UNFORCED, HIT, 0, 0, 3, 10),
			chance(key, UNFORCED, CRIT, 0, 0, 1, 16),
			first_side(key, UNFORCED),
		)

	rolls, hits, crits, order = map(np.asarray, outcomes(keys))
	assert rolls.dtype == np.int32
	assert hits.dtype == np.bool_
	for column in rolls.T:
		counts = np.bincount(column, minlength=16)
		assert np.all((counts > 1750) & (counts < 2350))
	assert 0.28 < hits.mean() < 0.32
	assert 0.05 < crits.mean() < 0.075
	assert 0.48 < order.mean() < 0.52
	for i in range(4):
		for j in range(i):
			assert 0.05 < np.mean(rolls[:, i] == rolls[:, j]) < 0.075
			assert abs(np.corrcoef(rolls[:, i], rolls[:, j])[0, 1]) < 0.03


def test_gen5_multihit_count_distribution():
	keys = jax.random.split(jax.random.key(902), 32768)

	@jax.jit
	@jax.vmap
	def outcomes(key):
		return hit_count(key, UNFORCED, 0, jnp.int32(2), jnp.int32(5))

	counts = np.bincount(np.asarray(outcomes(keys)), minlength=6)[2:]
	assert np.all(np.abs(counts / counts.sum() - [0.35, 0.35, 0.15, 0.15]) < 0.02)


def test_address_is_stable_and_forcing_is_local():
	key = jax.random.key(7)
	a = integer(key, UNFORCED, DAMAGE_ROLL, 0, 0, 16)
	_ = first_side(key, UNFORCED)
	b = integer(key, UNFORCED, DAMAGE_ROLL, 0, 0, 16)
	assert int(a) == int(b)
	forced = forced_from_records([{"kind": "hit", "side": 0, "occurrence": 0, "value": 0}])
	assert int(integer(key, forced, DAMAGE_ROLL, 0, 0, 16)) == int(a)
	assert not bool(chance(key, UNFORCED, HIT, 0, 0, 0, 100))
	assert bool(chance(key, UNFORCED, HIT, 0, 0, 100, 100))
	assert int(integer(key, UNFORCED, ORDER, 0, 0, 1)) == 0
	assert np.all(np.asarray(UNFORCED) == -1)


@pytest.mark.parametrize("key", [jax.random.key(7), jax.random.PRNGKey(7)])
def test_shared_bits_preserve_addressed_results(key):
	bits = random_bits(key)
	for kind, side, occurrence in ((ORDER, 0, 2), (HIT, 1, 0), (DAMAGE_ROLL, 1, 7)):
		assert int(integer(key, UNFORCED, kind, side, occurrence, 16)) == int(
			integer(bits, UNFORCED, kind, side, occurrence, 16)
		)
	forced = forced_from_records([{"kind": "hit", "side": 1, "occurrence": 0, "value": 1}])
	assert bool(chance(bits, forced, HIT, 1, 0, 0, 100))


def test_vmap_batched_forced_arrays():
	keys = jax.random.split(jax.random.key(0), 16)
	forced = jnp.stack(
		[
			forced_from_records([{"kind": "damage_roll", "side": 1, "occurrence": 7, "value": i}])
			for i in range(16)
		]
	)
	actual = jax.jit(jax.vmap(lambda k, f: integer(k, f, DAMAGE_ROLL, 1, 7, 16)))(keys, forced)
	np.testing.assert_array_equal(actual, np.arange(16))


@pytest.mark.parametrize(
	"update",
	[
		{"kind": "unknown"},
		{"kind": []},
		{"side": -1},
		{"side": 2},
		{"side": True},
		{"occurrence": -1},
		{"occurrence": 16},
		{"occurrence": 0.0},
		{"value": -1},
		{"value": 16},
		{"value": True},
		{"extra": 1},
		{"kind": "hit", "value": 2},
		{"kind": "crit", "value": 2},
		{"kind": "order", "side": 1},
		{"kind": "order", "occurrence": 3},
	],
)
def test_invalid_records(update):
	record = {"kind": "damage_roll", "side": 0, "occurrence": 0, "value": 7}
	with pytest.raises(ValueError):
		forced_from_records([record | update])


def test_missing_keys_nonrecords_and_duplicates():
	for records in ([{}], [None], ["hit"]):
		with pytest.raises(ValueError):
			forced_from_records(records)
	record = {"kind": "hit", "side": 0, "occurrence": 0, "value": 1}
	with pytest.raises(ValueError, match="Duplicate"):
		forced_from_records([record, record])
	np.testing.assert_array_equal(forced_from_records([]), UNFORCED)
