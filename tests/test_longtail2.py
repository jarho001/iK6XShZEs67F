"""Focused checks for task 14 rules that random battles rarely reach.

Each scenario is a two-lead battle built from explicit members and played through one
compiled step with forced outcomes (forced_from_records addresses, as in the recorder).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine import state as S
from pokejax.engine.abilities import active_abilities
from pokejax.engine.data import build_engine_tables
from pokejax.engine.forms import sync_weather_formes
from pokejax.engine.observation import observe
from pokejax.engine.randomness import forced_from_records
from pokejax.engine.reset import Teams, reset
from pokejax.engine.step import step
from pokejax.tables import load_tables

KEY = jax.random.key(14)
STEP = jax.jit(step)
FILLER = {"species": "mew", "moves": ["tackle"]}


@pytest.fixture(scope="module")
def world():
	tables = load_tables()
	return tables, build_engine_tables(tables)


def make_teams(tables, side0, side1):
	"""Six members per side; unlisted slots are plain Mew. Members: species, ability, item, moves."""
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


def hit(side, value=1, occurrence=0):
	return {"kind": "hit", "side": side, "occurrence": occurrence, "value": value}


def test_multitype_takes_plate_type_and_normal_without_plate(world):
	tables, _engine = world
	state = start(
		world,
		[
			{
				"species": "arceusfire",
				"ability": "multitype",
				"item": "flameplate",
				"moves": ["tackle"],
			}
		],
		[{"species": "arceus", "ability": "multitype", "item": "lifeorb", "moves": ["tackle"]}],
	)
	assert int(state.types[0, 0, 0]) == tables.types.index["fire"]
	assert int(state.types[1, 0, 0]) == tables.types.index["normal"]


def test_magic_bounce_reflects_toxic(world):
	_tables, _ = world
	state = start(
		world,
		[{"species": "mew", "moves": ["toxic"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	state = turn(
		world, state, [0, 0], [{"kind": "bounce_hit", "side": 1, "occurrence": 0, "value": 1}]
	)
	assert int(state.status[0, 0]) == S.TOXIC
	assert int(state.status[1, 0]) == S.HEALTHY


def test_magic_bounce_misses_reflected_toxic_when_its_accuracy_check_fails(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["toxic"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	state = turn(
		world, state, [0, 0], [{"kind": "bounce_hit", "side": 1, "occurrence": 0, "value": 0}]
	)
	assert int(state.status[0, 0]) == S.HEALTHY
	assert int(state.status[1, 0]) == S.HEALTHY


def test_mold_breaker_ignores_magic_bounce(world):
	state = start(
		world,
		[{"species": "mew", "ability": "moldbreaker", "moves": ["toxic"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 0], [hit(0)])
	assert int(state.status[1, 0]) == S.TOXIC
	assert int(state.status[0, 0]) == S.HEALTHY


def test_magic_bounce_reflects_spikes_onto_the_user_side(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["spikes"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 0])
	assert int(state.hazards[0, S.SPIKES]) == 1
	assert int(state.hazards[1, S.SPIKES]) == 0


def test_magic_bounce_reflects_roar_and_drags_the_user(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["roar"]}, {"species": "ditto", "moves": ["transform"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	records = [
		{"kind": "bounce_hit", "side": 1, "occurrence": 0, "value": 1},
		{"kind": "drag_in", "side": 0, "occurrence": 0, "value": 3},
	]
	state = turn(world, state, [0, 0], records)
	assert int(state.active[0]) == 3


def test_uncopyable_trace_keeps_seeking(world):
	tables, _ = world
	state = start(
		world,
		[
			{
				"species": "arceussteel",
				"ability": "multitype",
				"item": "ironplate",
				"moves": ["tackle"],
			}
		],
		[{"species": "gardevoir", "ability": "trace", "moves": ["tackle"]}],
	)
	assert int(state.ability[1, 0]) == tables.abilities.index["trace"]
	assert int(state.volatiles[1, S.TRACE_SEEK]) == 1
	state = turn(world, state, [0, 0], [hit(0), hit(1)])
	assert int(state.ability[1, 0]) == tables.abilities.index["trace"]
	assert int(state.volatiles[1, S.TRACE_SEEK]) == 1


def test_trace_copies_intimidate_on_switch_in(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "gyarados", "ability": "intimidate", "moves": ["tackle"]}],
		[
			{"species": "mew", "moves": ["tackle"]},
			{"species": "gardevoir", "ability": "trace", "moves": ["tackle"]},
		],
	)
	state = turn(world, state, [0, 4], [hit(0)])
	assert int(state.active[1]) == 1
	assert int(state.ability[1, 1]) == tables.abilities.index["intimidate"]
	assert int(state.boosts[0, S.BOOST_ATK]) == -1


def test_transform_copies_the_foe_and_switch_out_restores_the_user(world):
	tables, _ = world
	state = start(
		world,
		[
			{"species": "ditto", "ability": "limber", "moves": ["transform"]},
			{"species": "mew", "moves": ["tackle"]},
		],
		[{"species": "mew", "moves": ["tackle", "toxic"]}],
	)
	before = state
	state = turn(world, state, [0, 0], [hit(1)])
	assert int(state.species[0, 0]) == tables.species.index["mew"]
	assert int(state.volatiles[0, S.TRANSFORMED]) == 1
	assert int(state.moves[0, 0, 1]) == tables.moves.index["toxic"]
	assert int(state.pp[0, 0, 0]) == 5
	assert int(state.max_pp[0, 0, 0]) == 5
	state = turn(world, state, [4, 0], [hit(1)])
	assert int(state.active[0]) == 1
	assert int(state.species[0, 0]) == tables.species.index["ditto"]
	assert int(state.volatiles[0, S.TRANSFORMED]) == 0
	assert int(state.moves[0, 0, 0]) == tables.moves.index["transform"]
	assert int(state.pp[0, 0, 0]) == int(before.pp[0, 0, 0]) - 1
	np.testing.assert_array_equal(state.stats[0, 0], before.stats[0, 0])


def test_imposter_fails_against_a_substitute(world):
	tables, _ = world
	state = start(
		world,
		[
			{"species": "mew", "moves": ["tackle"]},
			{"species": "ditto", "ability": "imposter", "moves": ["transform"]},
		],
		[{"species": "mew", "moves": ["tackle"]}],
	)
	state = state._replace(volatiles=state.volatiles.at[1, S.SUBSTITUTE].set(5))
	state = turn(world, state, [4, 0], [hit(1)])
	assert int(state.active[0]) == 1
	assert int(state.species[0, 1]) == tables.species.index["ditto"]
	assert int(state.ability[0, 1]) == tables.abilities.index["imposter"]
	assert int(state.volatiles[0, S.TRANSFORMED]) == 0


def test_imposter_copies_the_foe_on_switch_in(world):
	tables, _ = world
	state = start(
		world,
		[
			{"species": "mew", "moves": ["tackle"]},
			{"species": "ditto", "ability": "imposter", "moves": ["transform"]},
		],
		[{"species": "gengar", "ability": "levitate", "moves": ["tackle"]}],
	)
	state = turn(world, state, [4, 0], [hit(1)])
	assert int(state.species[0, 1]) == tables.species.index["gengar"]
	assert int(state.volatiles[0, S.TRANSFORMED]) == 1
	assert int(state.ability[0, 1]) == tables.abilities.index["levitate"]
	assert int(state.types[0, 1, 0]) == tables.types.index["ghost"]


def test_illusion_breaks_on_a_multihit_move_and_the_opponent_sees_the_disguise(world):
	tables, engine = world
	state = start(
		world,
		[{"species": "mew", "moves": ["bulletseed"]}],
		[{"species": "zoroark", "ability": "illusion", "moves": ["tackle"]}],
	)
	assert int(state.volatiles[1, S.ILLUSION_SLOT]) == 6
	shown = observe(state, 0, engine).opponent
	assert int(shown.species[0]) == tables.species.index["mew"]
	assert int(shown.ability[0]) == 0
	records = [hit(0), {"kind": "hit_count", "side": 0, "occurrence": 0, "value": 2}]
	state = turn(world, state, [0, 0], records)
	assert int(state.volatiles[1, S.ILLUSION_SLOT]) == 0
	shown = observe(state, 0, engine).opponent
	assert int(shown.species[0]) == tables.species.index["zoroark"]
	assert int(shown.ability[0]) == tables.abilities.index["illusion"]


def test_relic_song_changes_forme_and_faint_reverts_it(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "mew", "moves": ["tackle"]}],
		[{"species": "meloetta", "ability": "serenegrace", "moves": ["relicsong", "tackle"]}],
	)
	base_stats = np.asarray(state.stats[1, 0])
	state = turn(
		world,
		state,
		[0, 0],
		[hit(1), {"kind": "secondary", "side": 1, "occurrence": 0, "value": 0}, hit(0)],
	)
	assert int(state.species[1, 0]) == tables.species.index["meloettapirouette"]
	assert int(state.saved_stats[1, 0]) == int(base_stats[1])
	assert not np.array_equal(np.asarray(state.stats[1, 0]), base_stats)
	state = state._replace(hp=state.hp.at[1, 0].set(1))
	state = turn(world, state, [0, 1], [hit(0)])
	assert int(state.hp[1, 0]) == 0
	assert int(state.species[1, 0]) == tables.species.index["meloetta"]
	np.testing.assert_array_equal(np.asarray(state.stats[1, 0]), base_stats)


def test_forecast_follows_the_weather_unless_air_lock_suppresses_it(world):
	tables, engine = world
	for ability, expected in (("airlock", "castform"), ("noability", "castformrainy")):
		state = start(
			world,
			[{"species": "mew", "ability": ability, "moves": ["tackle"]}],
			[{"species": "castform", "ability": "forecast", "moves": ["tackle"]}],
		)
		state = state._replace(
			weather=jnp.asarray(S.RAIN, jnp.int8), weather_turns=jnp.asarray(5, jnp.int8)
		)
		state = sync_weather_formes(state, engine, active_abilities(state, engine))
		assert int(state.species[1, 0]) == tables.species.index[expected], ability


def test_flower_gift_takes_sun_forme(world):
	tables, engine = world
	state = start(
		world,
		[{"species": "mew", "moves": ["tackle"]}],
		[{"species": "cherrim", "ability": "flowergift", "moves": ["tackle"]}],
	)
	state = state._replace(
		weather=jnp.asarray(S.SUN, jnp.int8), weather_turns=jnp.asarray(5, jnp.int8)
	)
	state = sync_weather_formes(state, engine, active_abilities(state, engine))
	assert int(state.species[1, 0]) == tables.species.index["cherrimsunshine"]
	assert int(state.types[1, 0, 0]) == tables.types.index["grass"]


def test_nature_power_pays_the_called_pressure_surcharge(world):
	_tables, _ = world
	state = start(
		world,
		[{"species": "sawsbuck", "moves": ["swordsdance", "naturepower"]}],
		[{"species": "dusclops", "ability": "pressure", "moves": ["tackle"]}],
	)
	before_pp = int(state.pp[0, 0, 1])
	before_hp = int(state.hp[1, 0])
	state = turn(world, state, [1, 0], [hit(0)])
	assert int(state.pp[0, 0, 1]) == before_pp - 2
	assert int(state.hp[1, 0]) < before_hp
