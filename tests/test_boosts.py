"""Stage caps, attack drops, and Belly Drum at decision boundaries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_status
from test_status import KEY

from pokejax.engine.boosts import apply_boosts
from pokejax.engine.randomness import HIT, ORDER, UNFORCED
from pokejax.engine.step import step

boost_setup = pytest.fixture(scope="module", name="boost_setup")(test_status.setup.__wrapped__)


def test_stage_caps_and_multiple_changes(boost_setup):
	tables, t, s = boost_setup
	s = s._replace(boosts=s.boosts.at[0].set(jnp.array([6, 6, 0, 0, 0, 0, 0], dtype=jnp.int8)))
	out = apply_boosts(s, 0, tables.moves.index["bulkup"], True, True, False, t)
	np.testing.assert_array_equal(out.boosts, s.boosts)
	out = apply_boosts(s, 0, tables.moves.index["shellsmash"], True, True, False, t)
	np.testing.assert_array_equal(out.boosts[0], [6, 5, 2, -1, 2, 0, 0])


def test_belly_drum_cost_and_failure(boost_setup):
	tables, t, s = boost_setup
	move = tables.moves.index["bellydrum"]
	for hp in (79, 80):
		out = apply_boosts(s._replace(hp=s.hp.at[0, 0].set(hp)), 0, move, True, True, False, t)
		assert int(out.hp[0, 0]) == hp
		assert int(out.boosts[0, 0]) == 0
	out = apply_boosts(s._replace(hp=s.hp.at[0, 0].set(81)), 0, move, True, True, False, t)
	assert int(out.hp[0, 0]) == 1 and int(out.boosts[0, 0]) == 6
	out = apply_boosts(out, 0, move, True, True, False, t)
	assert int(out.hp[0, 0]) == 1


def test_attack_drops_require_hit_and_ignore_knockout(boost_setup):
	tables, t, s = boost_setup
	s = s._replace(
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["closecombat"]),
		stats=s.stats.at[0, 0, 5].set(200),
	)
	forced = UNFORCED.at[ORDER, 0, 0].set(0).at[HIT, :, 0].set(1)
	miss = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced.at[HIT, 0, 0].set(0), t)
	assert int(miss.boosts[0, 1]) == 0
	ghost = s._replace(types=s.types.at[1, 0, 0].set(tables.types.index["ghost"]))
	immune = jax.jit(step)(ghost, jnp.array([0, 0]), KEY, forced, t)
	assert int(immune.boosts[0, 1]) == 0
	ko = jax.jit(step)(s._replace(hp=s.hp.at[1, 0].set(1)), jnp.array([0, 0]), KEY, forced, t)
	np.testing.assert_array_equal(ko.boosts[0, 1:4:2], [-1, -1])
