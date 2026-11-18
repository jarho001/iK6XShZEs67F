"""Focused Gen 5 held-item rules that random battles do not reliably exercise."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.items import (
	apply_stat_modifier,
	gather_active_items,
	item_base_power_modifier,
	update_active_items,
)
from pokejax.engine.observation import observe
from pokejax.engine.order import action_order
from pokejax.engine.randomness import (
	ACTION_ORDER,
	CONFUSION_HIT,
	CRIT,
	DAMAGE_ROLL,
	HIT,
	HIT_COUNT,
	ORDER,
	UNFORCED,
)
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import (
	ATK,
	CHOICE_MOVE,
	CONFUSION,
	MOVE,
	MUST_RECHARGE,
	NONE,
	SPE,
	SWITCH,
)
from pokejax.engine.step import step

KEY = jax.random.key(19)
RUN = jax.jit(step)


@pytest.fixture
def item_battle(synthetic_pool, tables):
	engine = build_engine_tables(tables)
	state = reset(KEY, teams_from_pool(synthetic_pool, 0, 1), engine)
	stats = jnp.full_like(state.stats, 100).at[:, :, 0].set(1000)
	return (
		tables,
		engine,
		state._replace(
			level=jnp.full_like(state.level, 100),
			stats=stats,
			hp=jnp.full_like(state.hp, 1000),
			moves=jnp.zeros_like(state.moves).at[0, 0, 0].set(tables.moves.index["tackle"]),
			pp=jnp.zeros_like(state.pp).at[0, 0, 0].set(10),
			max_pp=jnp.zeros_like(state.max_pp).at[0, 0, 0].set(10),
			types=jnp.full_like(state.types, -1),
			item=jnp.zeros_like(state.item),
			item_consumed=jnp.zeros_like(state.item_consumed),
			item_knocked_off=jnp.zeros_like(state.item_knocked_off),
			item_revealed=jnp.zeros_like(state.item_revealed),
			item_public=jnp.zeros_like(state.item_public),
			request=jnp.array([MOVE, NONE], dtype=state.request.dtype),
		),
	)


def forced_rolls(*, hits=1, count=None):
	forced = UNFORCED.at[ORDER, 0, ACTION_ORDER].set(0)
	forced = forced.at[HIT, 0, 0].set(hits)
	for occurrence in range(5):
		forced = forced.at[CRIT, 0, occurrence].set(0)
		forced = forced.at[DAMAGE_ROLL, 0, occurrence].set(0)
	if count is not None:
		forced = forced.at[HIT_COUNT, 0, 0].set(count)
	return forced


def held(tables, state, side, item):
	return state._replace(
		item=state.item.at[side, state.active[side]].set(tables.items.index[item])
	)


def use_move(tables, engine, state, move, *, forced=None):
	state = state._replace(moves=state.moves.at[0, 0, 0].set(tables.moves.index[move]))
	return RUN(
		state,
		jnp.array([0, 0], dtype=jnp.int32),
		KEY,
		forced_rolls() if forced is None else forced,
		engine,
	)


def test_focus_sash_saves_only_the_first_hit(item_battle):
	tables, engine, state = item_battle
	state = state._replace(
		stats=state.stats.at[0, 0, ATK].set(1000).at[1, 0, 0].set(100).at[1, 0, 2].set(1),
		hp=state.hp.at[1, 0].set(100),
		moves=state.moves.at[0, 0, 0].set(tables.moves.index["tailslap"]),
		item=state.item.at[1, 0].set(tables.items.index["focussash"]),
	)
	out = use_move(tables, engine, state, "tailslap", forced=forced_rolls(hits=1, count=2))
	assert int(out.hp[1, 0]) == 0
	assert bool(out.item_consumed[1, 0])
	assert int(out.item[1, 0]) == 0
	observation = observe(out, 0, engine)
	assert int(observation.opponent.item[0]) == tables.items.index["focussash"]
	assert observation.opponent.item_revealed[0] == 1
	assert observation.opponent.item_consumed[0] == 1


def test_gen5_gem_boosts_every_hit_of_a_multihit_move(item_battle):
	tables, engine, state = item_battle
	state = state._replace(
		moves=state.moves.at[0, 0, 0].set(tables.moves.index["tailslap"]),
		hp=state.hp.at[1, 0].set(1000),
		stats=state.stats.at[0, 0, ATK].set(300),
	)
	forced = forced_rolls(hits=1, count=3)
	plain = use_move(tables, engine, state, "tailslap", forced=forced)
	boosted = use_move(
		tables, engine, held(tables, state, 0, "normalgem"), "tailslap", forced=forced
	)
	assert int(boosted.hp[1, 0]) < int(plain.hp[1, 0])
	assert bool(boosted.item_consumed[0, 0])
	assert int(boosted.item[0, 0]) == 0


def test_choice_lock_pp_exhaustion_broken_lock_and_switch_out(item_battle):
	tables, engine, state = item_battle
	state = state._replace(
		moves=state.moves.at[0, 0].set(
			jnp.array(
				[
					tables.moves.index["tackle"],
					tables.moves.index["protect"],
					tables.moves.index["thunderbolt"],
					0,
				],
				dtype=state.moves.dtype,
			)
		),
		pp=state.pp.at[0, 0].set(jnp.array([0, 4, 4, 0], dtype=state.pp.dtype)),
		item=state.item.at[0, 0].set(tables.items.index["choiceband"]),
		volatiles=state.volatiles.at[0, CHOICE_MOVE].set(tables.moves.index["tackle"]),
	)
	locked = legal_actions(state, engine)
	np.testing.assert_array_equal(locked[0, :4], [True, False, False, False])

	# Without the held Choice item, Showdown removes the soft lock and exposes
	# every move with PP again.
	broken = state._replace(item=state.item.at[0, 0].set(0))
	np.testing.assert_array_equal(legal_actions(broken, engine)[0, :4], [False, True, True, False])

	# The first legal action represents Struggle when the locked slot has no PP.
	state = state._replace(request=jnp.array([MOVE, NONE], dtype=state.request.dtype))
	out = RUN(state, jnp.array([0, 0]), KEY, forced_rolls(), engine)
	assert int(out.pp[0, 0, 0]) == 0
	assert int(out.volatiles[0, CHOICE_MOVE]) == tables.moves.index["tackle"]

	# Switching clears the active soft lock and a new active Pokemon can choose freely.
	out = RUN(state, jnp.array([4, 0]), KEY, forced_rolls(), engine)
	assert int(out.active[0]) == 1
	assert int(out.volatiles[0, CHOICE_MOVE]) == 0


@pytest.mark.parametrize(
	"item,stat,species,expected",
	[
		("choiceband", ATK, "bulbasaur", 150),
		("choicespecs", 3, "bulbasaur", 150),
		("choicescarf", SPE, "bulbasaur", 150),
		("souldew", 3, "latias", 150),
		("souldew", 4, "latias", 150),
		("souldew", 3, "bulbasaur", 100),
		("eviolite", 2, "bulbasaur", 150),
		("eviolite", 2, "venusaur", 100),
		("lightball", ATK, "pikachu", 200),
		("thickclub", ATK, "marowak", 200),
	],
)
def test_item_stat_modifier_rows(item_battle, item, stat, species, expected):
	tables, engine, state = item_battle
	state = held(tables, state, 0, item)
	state = state._replace(species=state.species.at[0, 0].set(tables.species.index[species]))
	rows = gather_active_items(state, engine)
	assert int(apply_stat_modifier(jnp.int16(100), 0, stat, rows)) == expected
	if item == "souldew":
		assert (
			int(item_base_power_modifier(0, tables.types.index["psychic"], False, 0, rows)) == 4096
		)


def test_custap_berry_priority_and_recharge_exclusion(item_battle):
	tables, engine, state = item_battle
	state = state._replace(
		request=jnp.array([MOVE, MOVE], dtype=state.request.dtype),
		stats=state.stats.at[0, 0, SPE].set(100).at[1, 0, SPE].set(200),
		hp=state.hp.at[0, 0].set(250),
		item=state.item.at[0, 0].set(tables.items.index["custapberry"]),
	)
	state, first = action_order(
		state,
		jnp.array([0, 0]),
		jnp.array([tables.moves.index["tackle"], tables.moves.index["tackle"]]),
		False,
		KEY,
		UNFORCED,
		gather_active_items(state, engine),
		engine,
	)
	assert int(first) == 0
	assert bool(state.item_consumed[0, 0])

	state = state._replace(
		item=state.item.at[0, 0].set(tables.items.index["custapberry"]),
		item_consumed=state.item_consumed.at[0, 0].set(False),
		volatiles=state.volatiles.at[0, MUST_RECHARGE].set(1),
	)
	state, first = action_order(
		state,
		jnp.array([0, 0]),
		jnp.array([tables.moves.index["tackle"], tables.moves.index["tackle"]]),
		False,
		KEY,
		UNFORCED,
		gather_active_items(state, engine),
		engine,
	)
	assert int(first) == 1
	assert not bool(state.item_consumed[0, 0])


@pytest.mark.parametrize("move,recoil", [("protect", 0), ("seismictoss", 100)])
def test_life_orb_status_and_fixed_damage_rules(item_battle, move, recoil):
	tables, engine, state = item_battle
	state = held(tables, state, 0, "lifeorb")
	out = use_move(tables, engine, state, move)
	assert int(out.hp[0, 0]) == 1000 - recoil
	if move == "seismictoss":
		assert int(out.hp[1, 0]) == 900
	assert not bool(out.item_consumed[0, 0])


def test_lum_cures_confusion_even_when_it_prevents_the_move(item_battle):
	tables, engine, state = item_battle
	state = held(tables, state, 0, "lumberry")
	state = state._replace(volatiles=state.volatiles.at[0, CONFUSION].set(2))
	forced = forced_rolls().at[CONFUSION_HIT, 0, 0].set(1)
	out = use_move(tables, engine, state, "tackle", forced=forced)
	assert int(out.volatiles[0, CONFUSION]) == 0
	assert bool(out.item_consumed[0, 0])
	assert int(out.item[0, 0]) == 0


@pytest.mark.parametrize(
	"move,expected",
	[
		("closecombat", [0, 0, 0, 0, 0]),
		("superpower", [0, 0, 0, 0, 0]),
		("shellsmash", [2, 0, 2, 0, 2]),
	],
)
def test_white_herb_clears_negative_stages_after_stat_drops(item_battle, move, expected):
	tables, engine, state = item_battle
	state = held(tables, state, 0, "whiteherb")
	out = use_move(tables, engine, state, move)
	boosts = np.asarray(out.boosts[0, :5])
	assert boosts.tolist() == expected
	assert bool(out.item_consumed[0, 0])
	assert int(out.item[0, 0]) == 0


def test_white_herb_updates_after_an_opponent_switch_in(item_battle):
	tables, engine, state = item_battle
	state = state._replace(
		item=state.item.at[0, 0].set(tables.items.index["whiteherb"]),
		boosts=state.boosts.at[0, ATK].set(-1),
		request=jnp.array([NONE, SWITCH], dtype=state.request.dtype),
	)
	out = RUN(state, jnp.array([0, 4]), KEY, forced_rolls(), engine)
	assert int(out.boosts[0, ATK]) == 0
	assert bool(out.item_consumed[0, 0])


def test_unrevealed_opponent_item_never_leaks(item_battle):
	tables, engine, state = item_battle
	state = state._replace(item=state.item.at[1, 0].set(tables.items.index["choicescarf"]))
	observation = observe(state, 0, engine)
	assert int(observation.opponent.item[0]) == 0
	assert observation.opponent.item_revealed[0] == 0

	rows = gather_active_items(state, engine)
	state = update_active_items(
		state._replace(status=state.status.at[1, 0].set(1)),
		True,
		rows,
	)
	# A passive Choice modifier is not a protocol reveal either.
	assert int(observe(state, 0, engine).opponent.item[0]) == 0
