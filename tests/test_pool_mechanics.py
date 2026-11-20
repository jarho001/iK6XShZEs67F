"""Focused checks for rules the unchanged pool recording reached (task 15).

Each scenario is a two-lead battle from explicit members, played through compiled steps
with forced outcomes, as in the recorder. Showdown's behaviour is cited per test.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine import state as S
from pokejax.engine.data import build_engine_tables
from pokejax.engine.items import showdown_item_consumed
from pokejax.engine.randomness import forced_from_records
from pokejax.engine.reset import Teams, reset
from pokejax.engine.step import step
from pokejax.tables import load_tables

KEY = jax.random.key(15)
STEP = jax.jit(step)
FILLER = {"species": "mew", "moves": ["tackle"]}


@pytest.fixture(scope="module")
def world():
	tables = load_tables()
	return tables, build_engine_tables(tables)


def make_teams(tables, side0, side1):
	species = np.zeros((2, 6), np.int16)
	ability = np.zeros((2, 6), np.int16)
	item = np.zeros((2, 6), np.int16)
	moves = np.zeros((2, 6, 4), np.int16)
	for s, side in enumerate((side0, side1)):
		members = list(side) + [FILLER] * (6 - len(side))
		for slot, member in enumerate(members):
			species[s, slot] = tables.species.index[member["species"]]
			ability[s, slot] = tables.abilities.index[member.get("ability", "noability")]
			item[s, slot] = tables.items.index.get(member.get("item", ""), 0)
			for m, move in enumerate(member["moves"]):
				moves[s, slot, m] = tables.moves.index[move]
	full = (2, 6)
	return Teams(
		species=species,
		level=np.full(full, 80, np.uint8),
		moves=moves,
		ability=ability,
		item=item,
		nature=np.full(full, tables.natures.index["serious"], np.int8),
		gender=np.full(full, 1, np.int8),
		evs=np.full((*full, 6), 85, np.uint8),
		ivs=np.full((*full, 6), 31, np.uint8),
		happiness=np.full(full, 255, np.uint8),
	)


def start(world, side0, side1):
	tables, engine = world
	return reset(KEY, make_teams(tables, side0, side1), engine)


def turn(world, state, actions, records=()):
	_, engine = world
	forced = forced_from_records(list(records))
	return STEP(state, jnp.asarray(actions, jnp.int32), KEY, forced, engine)


def record(kind, side, value, occurrence=0):
	return {"kind": kind, "side": side, "occurrence": occurrence, "value": value}


def test_reflected_encore_fails_on_its_users_own_last_move(world):
	"""Showdown's moveUsed runs before the hit, so Magic Bounce's Encore reads Encore as the
	user's last move (failencore) and fails, instead of encoring the previous move."""
	state = start(
		world,
		[{"species": "jumpluff", "moves": ["tackle", "encore"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 0])
	state = turn(world, state, [1, 0], [record("bounce_hit", 1, 1)])
	assert int(state.volatiles[0, S.ENCORE_REMAINING]) == 0
	assert int(state.volatiles[0, S.ENCORE_MOVE]) == 0


def test_toxic_spikes_poison_before_an_imposter_copies_the_foe(world):
	"""SwitchIn hazards run before the Imposter copy: Ditto is poisoned while still Normal,
	and only then copies a Steel foe, which would have been immune to the poison."""
	tables, _ = world
	state = start(
		world,
		[
			{"species": "mew", "moves": ["tackle"]},
			{"species": "ditto", "ability": "imposter", "moves": ["transform"]},
		],
		[{"species": "forretress", "ability": "overcoat", "moves": ["tackle"]}],
	)
	state = state._replace(hazards=state.hazards.at[0, S.TOXIC_SPIKES].set(2))
	state = turn(world, state, [4, 0])
	assert int(state.status[0, 1]) == S.TOXIC
	assert int(state.ability[0, 1]) == tables.abilities.index["overcoat"]


def test_moxie_reveals_a_boost_that_reaches_the_cap(world):
	"""Showdown names Moxie when its boost is effective: from +5 to +6 is effective."""
	tables, _ = world
	state = start(
		world,
		[{"species": "gyarados", "ability": "moxie", "moves": ["tackle"]}],
		[{"species": "mew", "moves": ["tackle"]}],
	)
	state = state._replace(
		boosts=state.boosts.at[0, S.BOOST_ATK].set(5),
		hp=state.hp.at[1, state.active[1]].set(1),
	)
	state = turn(world, state, [0, 0])
	assert int(state.boosts[0, S.BOOST_ATK]) == 6
	assert int(state.ability_public[0, state.active[0]]) == tables.abilities.index["moxie"]


def test_synchronize_reveals_the_source_when_the_status_comes_back(world):
	"""Showdown's Synchronize on the source runs onAfterSetStatus when the copy lands, so
	both holders are named."""
	tables, _ = world
	state = start(
		world,
		[{"species": "musharna", "ability": "synchronize", "moves": ["toxic"]}],
		[{"species": "gardevoir", "ability": "synchronize", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 0], [record("hit", 0, 1)])
	sync = tables.abilities.index["synchronize"]
	assert int(state.status[1, state.active[1]]) == S.TOXIC
	assert int(state.status[0, state.active[0]]) == S.TOXIC
	assert int(state.ability_public[0, state.active[0]]) == sync
	assert int(state.ability_public[1, state.active[1]]) == sync


def test_sheer_force_removes_color_change(world):
	"""Showdown suppresses AfterMoveSecondary (Color Change) when Sheer Force removed the
	move's secondary effects, so the target keeps its type and its ability stays hidden."""
	tables, _ = world
	state = start(
		world,
		[{"species": "tauros", "ability": "sheerforce", "moves": ["fireblast"]}],
		[{"species": "kecleon", "ability": "colorchange", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 0], [record("hit", 0, 1), record("crit", 0, 0)])
	active = state.active[1]
	assert np.asarray(state.types[1, active]).tolist() == [tables.types.index["normal"], -1]
	assert int(state.ability_public[1, active]) == 0


def test_trick_keeps_the_consumed_item_flag(world):
	"""Showdown's lastItem is set only when an item is eaten or used. Trick moves items and
	leaves it alone, so a holder that ate its berry still reads as consumed when it holds
	nothing, even after receiving an item."""
	tables, _ = world
	state = start(
		world,
		[{"species": "gengar", "item": "leftovers", "moves": ["trick"]}],
		[{"species": "mew", "moves": ["tackle"]}],
	)
	state = state._replace(item_consumed=state.item_consumed.at[1, 0].set(True))
	state = turn(world, state, [0, 0])
	assert int(state.item[1, 0]) == tables.items.index["leftovers"]
	assert bool(state.item_consumed[1, 0])
	assert not bool(showdown_item_consumed(state)[1, 0])
	state = turn(world, state, [0, 0])
	assert int(state.item[1, 0]) == 0
	assert bool(showdown_item_consumed(state)[1, 0])


def test_trace_copies_intimidate_onto_the_entrant_before_it_switches_in(world):
	"""Showdown's Update copies a foe into a Trace holder that is leaving, before the entrant's
	own runSwitch, so the copied Intimidate lowers the entrant's Attack."""
	state = start(
		world,
		[
			{"species": "arceus", "ability": "multitype", "moves": ["tackle"]},
			{"species": "gyarados", "ability": "intimidate", "moves": ["tackle"]},
		],
		[
			{"species": "gardevoir", "ability": "trace", "moves": ["tackle"]},
			{"species": "cofagrigus", "ability": "mummy", "moves": ["tackle"]},
		],
	)
	state = turn(world, state, [4, 4])
	assert int(state.active[0]) == 1 and int(state.active[1]) == 1
	assert int(state.boosts[0, S.BOOST_ATK]) == -1
	assert int(state.boosts[1, S.BOOST_ATK]) == 0


def test_choice_item_locks_a_two_turn_move_it_continues(world):
	"""A Choice item gained during a charge locks the move it continues: Showdown's ModifyMove
	adds the choicelock on every use, including a continuation. The Trick user is paralysed so
	that the charging Scyther moves first (Showdown: Razor Wind, Trick, Razor Wind)."""
	tables, _ = world
	state = start(
		world,
		[{"species": "scyther", "ability": "technician", "moves": ["razorwind", "tackle"]}],
		[
			{
				"species": "girafarig",
				"ability": "innerfocus",
				"item": "choicescarf",
				"moves": ["trick", "tackle"],
			}
		],
	)
	state = state._replace(status=state.status.at[1, 0].set(S.PARALYSIS))
	state = turn(world, state, [0, 0], [record("full_paralysis", 1, 0)])
	assert int(state.item[0, 0]) == tables.items.index["choicescarf"]
	state = turn(world, state, [0, 1])
	assert int(state.volatiles[0, S.CHOICE_MOVE]) == tables.moves.index["razorwind"]
