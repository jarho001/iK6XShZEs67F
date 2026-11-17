"""Gen 5 secondary effects and rare volatile interactions from pinned Showdown."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_status

from pokejax.engine.observation import observe
from pokejax.engine.randomness import (
	CONFUSION_DAMAGE_ROLL,
	CONFUSION_DURATION,
	CONFUSION_HIT,
	CRIT,
	DAMAGE_ROLL,
	FULL_PARALYSIS,
	HIT,
	ORDER,
	SECONDARY,
	SECONDARY_STATUS,
	THAW,
	UNFORCED,
	forced_from_records,
)
from pokejax.engine.state import BURN, CONFUSION, FLINCH, FREEZE, PARALYSIS, SLEEP
from pokejax.engine.step import step
from pokejax.engine.volatiles import confuse, confusion_damage

KEY = jax.random.key(31)
RUN = jax.jit(step)
FORCED = (
	UNFORCED.at[HIT]
	.set(1)
	.at[CRIT]
	.set(0)
	.at[DAMAGE_ROLL]
	.set(0)
	.at[SECONDARY]
	.set(0)
	.at[CONFUSION_HIT]
	.set(0)
	.at[CONFUSION_DAMAGE_ROLL]
	.set(0)
	.at[CONFUSION_DURATION]
	.set(0)
	.at[FULL_PARALYSIS]
	.set(0)
	.at[THAW]
	.set(0)
	.at[ORDER, 0, 0]
	.set(0)
)


@pytest.fixture(scope="module")
def setup():
	tables, t, s = test_status.setup.__wrapped__()
	s = s._replace(
		level=jnp.full_like(s.level, 100),
		hp=jnp.full_like(s.hp, 1000),
		stats=s.stats.at[:, :, 0].set(1000).at[0, 0, 5].set(200),
		moves=jnp.full_like(s.moves, tables.moves.index["splash"]),
	)
	return tables, t, s


def play(setup, move, *, forced=FORCED, state=None, actions=(0, 0)):
	tables, t, s = setup
	s = s if state is None else state
	s = s._replace(moves=s.moves.at[0, 0, 0].set(tables.moves.index[move]))
	return RUN(s, jnp.asarray(actions), KEY, forced, t)


def test_flinch_only_prevents_a_pending_move_and_expires(setup):
	tables, _, s = setup
	s = s._replace(moves=s.moves.at[1, 0, 0].set(tables.moves.index["tackle"]))
	f = FORCED.at[SECONDARY, 0, 0].set(1)
	fast = play(setup, "ironhead", state=s, forced=f)
	assert int(fast.pp[1, 0, 0]) == 20
	assert int(fast.hp[0, 0]) == 1000
	assert np.all(np.asarray(fast.volatiles[:, FLINCH]) == 0)
	slow = play(setup, "ironhead", state=s._replace(stats=s.stats.at[0, 0, 5].set(1)), forced=f)
	assert int(slow.pp[1, 0, 0]) == 19
	assert int(slow.hp[0, 0]) < 1000
	assert np.all(np.asarray(slow.volatiles[:, FLINCH]) == 0)
	assert int(play(setup, "ironhead", state=s, forced=FORCED).pp[1, 0, 0]) == 19


@pytest.mark.parametrize(
	"move,status,type_,existing,expected",
	[
		("flamethrower", BURN, "fire", 0, 0),
		("icebeam", FREEZE, "ice", 0, 0),
		("sludgebomb", 2, "steel", 0, 0),
		("sludgebomb", 2, "poison", 0, 0),
		("thunderbolt", PARALYSIS, "electric", 0, PARALYSIS),
		("scald", BURN, "normal", PARALYSIS, PARALYSIS),
		("scald", BURN, "normal", 0, BURN),
	],
)
def test_secondary_status_immunity_and_existing_status(
	setup, move, status, type_, existing, expected
):
	tables, _, s = setup
	s = s._replace(
		types=s.types.at[1, 0, 0].set(tables.types.index[type_]),
		status=s.status.at[1, 0].set(existing),
	)
	out = play(setup, move, state=s, forced=FORCED.at[SECONDARY, 0, 0].set(1))
	assert int(out.status[1, 0]) == expected
	assert int(play(setup, move, state=s).status[1, 0]) == existing


def test_fire_hit_thaws_after_attempting_the_secondary(setup):
	_, _, s = setup
	s = s._replace(status=s.status.at[1, 0].set(FREEZE))
	# DamagingHit (thaw) runs after the burn attempt, so the frozen target
	# finishes healthy even when the secondary chance succeeds.
	out = play(setup, "flamethrower", state=s, forced=FORCED.at[SECONDARY, 0, 0].set(1))
	assert int(out.status[1, 0]) == 0


def test_secondaries_require_a_hit_but_self_boosts_survive_ko(setup):
	tables, _, s = setup
	f = FORCED.at[SECONDARY, 0, 0].set(1)
	assert int(play(setup, "chargebeam", forced=f).boosts[0, 2]) == 1
	assert int(play(setup, "chargebeam", forced=f.at[HIT, 0, 0].set(0)).boosts[0, 2]) == 0
	ground = s._replace(types=s.types.at[1, 0, 0].set(tables.types.index["ground"]))
	assert int(play(setup, "chargebeam", state=ground, forced=f).boosts[0, 2]) == 0
	ko = s._replace(hp=s.hp.at[1, 0].set(1))
	assert int(play(setup, "flamecharge", state=ko, forced=f).boosts[0, 4]) == 1
	assert int(play(setup, "scald", state=ko, forced=f).status[1, 0]) == 0
	assert int(play(setup, "dynamicpunch", state=ko, forced=f).volatiles[1, CONFUSION]) == 0
	assert int(play(setup, "psychic", forced=f).boosts[1, 3]) == -1
	np.testing.assert_array_equal(play(setup, "ancientpower", forced=f).boosts[0, :5], [1] * 5)


@pytest.mark.parametrize("burn,flinch", [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_fang_secondaries_are_independent(setup, burn, flinch):
	tables, _, s = setup
	s = s._replace(moves=s.moves.at[1, 0, 0].set(tables.moves.index["tackle"]))
	f = FORCED.at[SECONDARY, 0, 0].set(burn).at[SECONDARY, 0, 1].set(flinch)
	out = play(setup, "firefang", state=s, forced=f)
	assert int(out.status[1, 0]) == (BURN if burn else 0)
	assert int(out.pp[1, 0, 0]) == (20 if flinch else 19)


@pytest.mark.parametrize("choice,status", [(0, BURN), (1, PARALYSIS), (2, FREEZE)])
def test_tri_attack_status_selection(setup, choice, status):
	f = FORCED.at[SECONDARY, 0, 0].set(1).at[SECONDARY_STATUS, 0, 0].set(choice)
	assert int(play(setup, "triattack", forced=f).status[1, 0]) == status


def test_chatter_requires_chatot(setup):
	tables, _, s = setup
	f = FORCED.at[SECONDARY, 0, 0].set(1)
	assert int(play(setup, "chatter", forced=f).volatiles[1, CONFUSION]) == 0
	s = s._replace(species=s.species.at[0, 0].set(tables.species.index["chatot"]))
	assert int(play(setup, "chatter", state=s, forced=f).volatiles[1, CONFUSION]) == 1


@pytest.mark.parametrize(
	"move,boost,amount",
	[
		("confuseray", 0, 0),
		("supersonic", 0, 0),
		("sweetkiss", 0, 0),
		("teeterdance", 0, 0),
		("swagger", 0, 2),
		("flatter", 2, 1),
	],
)
def test_confusion_status_moves_ignore_damage_type_immunity(setup, move, boost, amount):
	tables, _, s = setup
	s = s._replace(types=s.types.at[1, 0, 0].set(tables.types.index["ghost"]))
	out = play(setup, move, state=s)
	assert int(out.volatiles[1, CONFUSION]) == 1  # initial 2, decremented before moving
	assert int(out.boosts[1, boost]) == amount
	miss = play(setup, move, state=s, forced=FORCED.at[HIT, 0, 0].set(0))
	assert int(miss.volatiles[1, CONFUSION]) == 0
	assert int(miss.boosts[1, boost]) == 0


def test_confusion_duration_does_not_restart_and_ends_before_self_hit(setup):
	_, t, s = setup
	for duration in range(4):
		f = FORCED.at[CONFUSION_DURATION, 1, 0].set(duration)
		out = confuse(s, 1, True, KEY, f)
		assert int(out.volatiles[1, CONFUSION]) == duration + 2
		assert int(confuse(out, 1, True, KEY, FORCED).volatiles[1, CONFUSION]) == duration + 2
	s = s._replace(volatiles=s.volatiles.at[0, CONFUSION].set(1))
	out = RUN(s, jnp.array([0, 0]), KEY, FORCED.at[CONFUSION_HIT].set(1), t)
	assert int(out.volatiles[0, CONFUSION]) == 0
	assert int(out.hp[0, 0]) == 1000
	assert int(out.pp[0, 0, 0]) == 19


def test_confusion_self_hit_skips_modifiers_and_pp_and_can_faint(setup):
	_, t, s = setup
	f = FORCED.at[CONFUSION_HIT, 0, 0].set(1)
	s = s._replace(volatiles=s.volatiles.at[0, CONFUSION].set(3))
	assert int(confusion_damage(s, 0, KEY, f)) == 35
	assert int(confusion_damage(s, 0, KEY, f.at[CONFUSION_DAMAGE_ROLL, 0, 0].set(15))) == 29
	out = RUN(s, jnp.array([0, 0]), KEY, f, t)
	assert int(out.hp[0, 0]) == 965
	assert int(out.pp[0, 0, 0]) == 20
	assert int(out.volatiles[0, CONFUSION]) == 2
	# Burn affects neither Attack nor the self-hit damage (residual is separate).
	burned = s._replace(status=s.status.at[0, 0].set(BURN))
	assert int(confusion_damage(burned, 0, KEY, f)) == 35
	boosted = s._replace(boosts=s.boosts.at[0, 0].set(2).at[0, 1].set(-2))
	assert int(confusion_damage(boosted, 0, KEY, f)) == 136
	extreme = s._replace(stats=s.stats.at[0, 0, 1].set(30000).at[0, 0, 2].set(1))
	# 1,008,002 wraps to 24,962 before the 85% roll, then floors to 21,217.
	assert (
		int(confusion_damage(extreme, 0, KEY, f.at[CONFUSION_DAMAGE_ROLL, 0, 0].set(15))) == 21217
	)
	last = s._replace(hp=s.hp.at[0].set(0).at[0, 0].set(1))
	out = RUN(last, jnp.array([0, 0]), KEY, f, t)
	assert int(out.winner) == 1
	assert int(out.volatiles[0, CONFUSION]) == 0
	assert int(out.pp[1, 0, 0]) == 20


def test_prevention_order_and_switch_clearing(setup):
	_, t, s = setup
	s = s._replace(volatiles=s.volatiles.at[0, CONFUSION].set(3))
	for status in (SLEEP, FREEZE):
		blocked = s._replace(
			status=s.status.at[0, 0].set(status), status_counter=s.status_counter.at[0, 0].set(3)
		)
		out = RUN(blocked, jnp.array([0, 0]), KEY, FORCED, t)
		assert int(out.volatiles[0, CONFUSION]) == 3
	flinched = s._replace(volatiles=s.volatiles.at[0, FLINCH].set(1))
	out = RUN(flinched, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.volatiles[0, CONFUSION]) == 3
	paralyzed = s._replace(status=s.status.at[0, 0].set(PARALYSIS))
	out = RUN(paralyzed, jnp.array([0, 0]), KEY, FORCED.at[FULL_PARALYSIS].set(1), t)
	assert int(out.volatiles[0, CONFUSION]) == 2
	out = RUN(s, jnp.array([4, 0]), KEY, FORCED, t)
	assert np.all(np.asarray(out.volatiles[0]) == 0)
	out = RUN(out, jnp.array([4, 0]), KEY, FORCED, t)
	assert int(out.active[0]) == 0
	assert np.all(np.asarray(out.volatiles[0]) == 0)


def test_observation_exposes_confusion_presence_only(setup):
	_, t, s = setup
	observe_ = jax.jit(observe)
	s = s._replace(volatiles=s.volatiles.at[:, CONFUSION].set(2))
	a = observe_(s, 0, t)
	assert float(a.own.confused) == float(a.opponent.confused) == 1
	private = s._replace(volatiles=s.volatiles.at[:, CONFUSION].set(5).at[:, FLINCH].set(1))
	b = observe_(private, 0, t)
	for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
		np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize(
	"kind,maximum",
	[
		("secondary", 1),
		("secondary_status", 2),
		("confusion_duration", 3),
		("confusion_hit", 1),
		("confusion_damage_roll", 15),
	],
)
def test_new_forced_outcome_ranges(kind, maximum):
	for value in (0, maximum):
		forced_from_records([{"kind": kind, "side": 1, "occurrence": 0, "value": value}])
	with pytest.raises(ValueError, match="value out of range"):
		forced_from_records([{"kind": kind, "side": 1, "occurrence": 0, "value": maximum + 1}])
