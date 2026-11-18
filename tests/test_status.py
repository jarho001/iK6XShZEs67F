"""Rare major-status rules exercised at decision boundaries."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.data import build_engine_tables
from pokejax.engine.randomness import HIT, ORDER, THAW, UNFORCED
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.residual import residual
from pokejax.engine.state import BURN, FREEZE, ONGOING, SLEEP, TOXIC
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import team_pool_from_data

KEY = jax.random.key(19)


@pytest.fixture(scope="module")
def setup():
	tables = load_tables()
	t = build_engine_tables(tables)
	fixture = json.loads(Path("tests/fixtures/reset.json").read_text())
	pool = team_pool_from_data(
		{
			"format": "gen5randombattle",
			"count": len(fixture["teamSets"]),
			"seed": 0,
			"showdownCommit": fixture["showdownCommit"],
			"teams": fixture["teamSets"],
		},
		tables,
	)
	teams = teams_from_pool(pool, 0, 1)
	s = reset(KEY, teams._replace(ability=jnp.zeros_like(teams.ability)), t)
	s = s._replace(
		stats=jnp.full_like(s.stats, 100).at[:, :, 0].set(160),
		hp=jnp.full_like(s.hp, 160),
		types=jnp.full_like(s.types, -1).at[:, :, 0].set(tables.types.index["normal"]),
		moves=jnp.full_like(s.moves, tables.moves.index["tackle"]),
		pp=jnp.full_like(s.pp, 20),
		item=jnp.zeros_like(s.item),
		item_public=jnp.zeros_like(s.item_public),
		item_consumed=jnp.zeros_like(s.item_consumed),
		item_knocked_off=jnp.zeros_like(s.item_knocked_off),
		item_revealed=jnp.zeros_like(s.item_revealed),
	)
	return tables, t, s


def test_sleep_clause_ignores_self_inflicted_sleep(setup):
	tables, t, s = setup
	s = s._replace(
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["spore"]),
		status=s.status.at[1, 1].set(SLEEP),
		sleep_from_foe=s.sleep_from_foe.at[1, 1].set(1),
	)
	forced = UNFORCED.at[HIT, :, 0].set(1)
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.status[1, 0]) == 0
	s = s._replace(sleep_from_foe=s.sleep_from_foe.at[1, 1].set(0))
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.status[1, 0]) == SLEEP


def test_toxic_stage_resets_on_switch_in(setup):
	_, t, s = setup
	s = s._replace(
		status=s.status.at[0, 1].set(TOXIC), status_counter=s.status_counter.at[0, 1].set(9)
	)
	out = jax.jit(step)(s, jnp.array([4, 4]), KEY, UNFORCED, t)
	assert int(out.active[0]) == 1
	assert int(out.status_counter[0, 1]) == 1
	assert int(out.hp[0, 1]) == 150


def test_fire_hit_thaws_target(setup):
	tables, t, s = setup
	s = s._replace(
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["flamethrower"]),
		status=s.status.at[1, 0].set(FREEZE),
		stats=s.stats.at[0, 0, 5].set(200),
	)
	forced = UNFORCED.at[HIT, :, 0].set(1).at[THAW, 1, 0].set(0)
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.status[1, 0]) == 0
	assert int(out.pp[1, 0, 0]) == 19


def test_residual_faints_both_and_stops_on_battle_end(setup):
	_, t, s = setup
	s = s._replace(hp=s.hp.at[:, 0].set(1), status=s.status.at[:, 0].set(BURN))
	out = jax.jit(residual)(s, True, KEY, UNFORCED.at[ORDER, 0, 2].set(0), t)
	np.testing.assert_array_equal(out.hp[:, 0], [0, 0])
	assert int(out.winner) == ONGOING
	# Once the first side's last member faints, fieldEvent ends immediately.
	s = s._replace(hp=s.hp.at[:, 1:].set(0))
	out = jax.jit(residual)(s, True, KEY, UNFORCED.at[ORDER, 0, 2].set(0), t)
	np.testing.assert_array_equal(out.hp[:, 0], [0, 1])
	assert int(out.winner) == 1


def test_sleep_restores_duration_on_switch_in(setup):
	_, t, s = setup
	s = s._replace(
		status=s.status.at[0, 1].set(SLEEP),
		status_counter=s.status_counter.at[0, 1].set(1),
		sleep_start=s.sleep_start.at[0, 1].set(4),
	)
	out = jax.jit(step)(s, jnp.array([4, 4]), KEY, UNFORCED, t)
	assert int(out.status_counter[0, 1]) == 4


def test_thunder_wave_ground_immunity_and_electric_status_immunity(setup):
	tables, t, s = setup
	s = s._replace(
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["thunderwave"]),
		types=s.types.at[1, 0, 0].set(tables.types.index["ground"]),
	)
	forced = UNFORCED.at[HIT, :, 0].set(1)
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.status[1, 0]) == 0
	s = s._replace(types=s.types.at[1, 0, 0].set(tables.types.index["electric"]))
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.status[1, 0]) == 4


def test_scald_defrosts_user_but_not_target_in_gen5(setup):
	tables, t, s = setup
	s = s._replace(
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["scald"]),
		status=s.status.at[:, 0].set(FREEZE),
		stats=s.stats.at[0, 0, 5].set(200),
	)
	forced = UNFORCED.at[HIT, :, 0].set(1).at[THAW, :, 0].set(0)
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.status[0, 0]) == 0
	assert int(out.status[1, 0]) == FREEZE
	assert int(out.pp[1, 0, 0]) == 20


def test_rest_heals_and_sleeps_only_user(setup):
	tables, t, s = setup
	s = s._replace(
		moves=s.moves.at[0, 0, 0].set(tables.moves.index["rest"]),
		hp=s.hp.at[0, 0].set(50),
		stats=s.stats.at[0, 0, 5].set(200),
	)
	forced = UNFORCED.at[HIT, 1, 0].set(0)
	out = jax.jit(step)(s, jnp.array([0, 0]), KEY, forced, t)
	assert int(out.hp[0, 0]) == 160
	assert int(out.status[0, 0]) == SLEEP
	assert int(out.status[1, 0]) == 0
	assert int(out.status_counter[0, 0]) == 3
