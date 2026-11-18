"""Focused tests for Gen 5 ability interactions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.engine.randomness import (
	ACTION_ORDER,
	CONFUSION_DURATION,
	CRIT,
	DAMAGE_ROLL,
	HIT,
	HIT_COUNT,
	ORDER,
	SECONDARY,
	SWITCH_HANDLER_ORDER,
	UNFORCED,
)
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import (
	CONFUSION,
	FLASH_FIRE,
	HAIL,
	HEALTHY,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	MOVE,
	NONE,
	RAIN,
	SUN,
)
from pokejax.engine.step import step
from pokejax.engine.switching import run_switch_in

KEY = jax.random.key(12)
RUN = jax.jit(step)
FORCED = (
	UNFORCED.at[ORDER, 0, ACTION_ORDER]
	.set(0)
	.at[HIT, :, :]
	.set(1)
	.at[CRIT, :, :]
	.set(0)
	.at[DAMAGE_ROLL, :, :]
	.set(0)
	.at[SECONDARY, :, :]
	.set(1)
)


@pytest.fixture
def battle(tables, synthetic_pool):
	teams = teams_from_pool(synthetic_pool, 0, 1)
	engine = build_engine_tables(tables)
	teams = teams._replace(ability=jnp.zeros_like(teams.ability), item=jnp.zeros_like(teams.item))
	state = reset(KEY, teams, engine)
	return (
		teams,
		engine,
		state._replace(
			stats=jnp.full_like(state.stats, 100).at[:, :, 0].set(1000),
			hp=jnp.full_like(state.hp, 1000),
			level=jnp.full_like(state.level, 100),
			types=jnp.full_like(state.types, -1),
			request=jnp.array([MOVE, NONE], dtype=state.request.dtype),
		),
	)


def abilities(state, tables, source, target):
	return state._replace(
		ability=state.ability.at[:, 0].set(
			jnp.asarray(
				[tables.abilities.index[source], tables.abilities.index[target]],
				state.ability.dtype,
			)
		)
	)


def attack(state, engine, tables, move, forced=FORCED, actions=(0, 0)):
	state = state._replace(moves=state.moves.at[0, 0, 0].set(tables.moves.index[move]))
	return RUN(state, jnp.array(actions), KEY, forced, engine)


def test_mold_breaker_bypasses_and_does_not_reveal_levitate(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "levitate")
	assert observe(state, 0, engine).opponent.ability[0] == 0
	immune = attack(state, engine, tables, "earthquake")
	assert immune.hp[1, 0] == 1000
	assert observe(immune, 0, engine).opponent.ability[0] == tables.abilities.index["levitate"]

	state = abilities(state, tables, "moldbreaker", "levitate")
	out = attack(state, engine, tables, "earthquake")
	assert out.hp[1, 0] == 914
	assert observe(out, 0, engine).opponent.ability[0] == 0


@pytest.mark.parametrize("first", [0, 1])
def test_simultaneous_lead_intimidate_and_weather(battle, tables, first):
	teams, engine, _ = battle
	# The two fixture leads are the same species. Equalize their input stats too.
	teams = jax.tree.map(lambda field: jnp.stack([field[0], field[0]]), teams)
	forced = FORCED.at[ORDER, 0, SWITCH_HANDLER_ORDER].set(first)
	leads = [tables.abilities.index["intimidate"]] * 2
	state = reset(KEY, abilities(teams, tables, "intimidate", "intimidate"), engine, forced)
	np.testing.assert_array_equal(state.boosts[:, 0], [-1, -1])
	np.testing.assert_array_equal(state.ability_public[:, 0], leads)

	teams = abilities(teams, tables, "drizzle", "drought")
	state = reset(KEY, teams, engine, forced)
	assert state.weather == (SUN if first == 0 else RAIN)
	assert state.weather_turns == 0


def test_download_uses_boosted_defenses_and_ignores_eviolite(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "download", "")
	state = state._replace(
		stats=state.stats.at[1, 0, 4].set(130),
		item=state.item.at[1, 0].set(tables.items.index["eviolite"]),
	)
	out, _ = run_switch_in(state, jnp.array([True, False]), 0, engine)
	np.testing.assert_array_equal(out.boosts[0, :3], [1, 0, 0])
	boosted = state._replace(boosts=state.boosts.at[1, 1].set(1))
	out, _ = run_switch_in(boosted, jnp.array([True, False]), 0, engine)
	np.testing.assert_array_equal(out.boosts[0, :3], [0, 0, 1])


def test_flash_fire_absorbs_before_accuracy_and_boosts_attack(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "flashfire")
	miss = FORCED.at[HIT, 0, 0].set(0)
	absorb = attack(state, engine, tables, "flamethrower", miss)
	assert absorb.hp[1, 0] == 1000
	assert absorb.volatiles[1, FLASH_FIRE] == 1
	fire = tables.moves.index["flamethrower"]
	absorb = absorb._replace(
		request=jnp.array([NONE, MOVE], jnp.int8),
		moves=absorb.moves.at[1, 0, 0].set(fire),
	)
	out = RUN(absorb, jnp.array([0, 0]), KEY, FORCED.at[SECONDARY, :, :].set(0), engine)
	assert out.hp[0, 0] == 879


def test_wonder_guard_blocks_neutral_nonstatus_damage(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "wonderguard")
	bug = tables.types.index["bug"]
	state = state._replace(types=state.types.at[1, 0, 0].set(bug))
	neutral = attack(state, engine, tables, "surf")
	assert neutral.hp[1, 0] == 1000
	fire = attack(state, engine, tables, "flamethrower", FORCED.at[SECONDARY, :, :].set(0))
	assert fire.hp[1, 0] == 838


def test_sheer_force_removes_secondary_and_life_orb_recoil(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "sheerforce", "")
	state = state._replace(item=state.item.at[0, 0].set(tables.items.index["lifeorb"]))
	out = attack(state, engine, tables, "flamethrower")
	assert out.status[1, 0] == HEALTHY
	assert out.hp[0, 0] == 1000
	assert out.hp[1, 0] == 862


def test_unaware_ignores_boosts_and_skill_link_maximizes_hits(battle, tables):
	_, engine, state = battle
	unaware = abilities(state, tables, "", "unaware")
	boosted = unaware._replace(boosts=state.boosts.at[0, 0].set(6))
	out = attack(boosted, engine, tables, "tackle")
	plain = attack(state, engine, tables, "tackle")
	assert out.hp[1, 0] == plain.hp[1, 0]
	link = abilities(state, tables, "skilllink", "")
	out = attack(link, engine, tables, "rockblast", FORCED.at[HIT_COUNT, 0, 0].set(2))
	assert out.hp[1, 0] == 885


def test_voluntary_switches_intimidate_the_foe_present_at_each_switch(battle, tables):
	_, engine, state = battle
	intimidate = tables.abilities.index["intimidate"]
	state = state._replace(
		ability=state.ability.at[:, 1].set(intimidate),
		request=jnp.array([MOVE, MOVE], jnp.int8),
	)
	out = RUN(state, jnp.array([4, 4]), KEY, FORCED, engine)
	np.testing.assert_array_equal(out.boosts[:, 0], [-1, 0])


def test_winning_locked_move_leaves_fatigue_confusion_and_lum(battle, tables):
	_, engine, state = battle
	outrage = tables.moves.index["outrage"]
	lum = tables.items.index["lumberry"]
	state = state._replace(
		hp=state.hp.at[1].set(0).at[1, 0].set(1),
		item=state.item.at[0, 0].set(lum),
		volatiles=state.volatiles.at[0, LOCKED_MOVE]
		.set(outrage)
		.at[0, LOCKED_REMAINING]
		.set(1)
		.at[0, LOCKED_TRUE_DURATION]
		.set(1),
	)
	out = attack(state, engine, tables, "outrage", FORCED.at[CONFUSION_DURATION, 0, 0].set(2))
	assert out.winner == 0
	assert out.volatiles[0, CONFUSION] == 4
	assert out.item[0, 0] == lum


@pytest.mark.parametrize("move", ["dig", "dive"])
def test_dig_and_dive_block_hail_damage(battle, tables, move):
	_, engine, state = battle
	out = attack(state._replace(weather=jnp.int8(HAIL)), engine, tables, move)
	np.testing.assert_array_equal(out.hp[:, 0], [1000, 938])


def test_own_tempo_reveals_when_direct_confusion_is_blocked(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "owntempo")
	out = attack(state, engine, tables, "confuseray")
	assert out.volatiles[1, CONFUSION] == 0
	assert observe(out, 0, engine).opponent.ability[0] == tables.abilities.index["owntempo"]


def test_marvel_scale_applies_to_hit_after_poison(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "marvelscale")
	out = attack(state, engine, tables, "twineedle")
	# 23 on hit one, 16 after poisoning raises Defense, then 125 poison damage.
	assert out.hp[1, 0] == 836


def test_lum_can_cure_the_first_hit_and_poison_lands_on_the_second(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "marvelscale")
	lum = tables.items.index["lumberry"]
	state = state._replace(item=state.item.at[1, 0].set(lum))
	out = attack(state, engine, tables, "twineedle")
	assert out.item_consumed[1, 0]
	# 23 twice because Lum cured the first hit's poison, then 125 residual.
	assert out.hp[1, 0] == 829


def test_winning_recoil_move_consumes_sitrus_before_battle_ends(battle, tables):
	_, engine, state = battle
	sitrus = tables.items.index["sitrusberry"]
	state = state._replace(
		hp=state.hp.at[0, 0].set(501).at[1].set(0).at[1, 0].set(30),
		item=state.item.at[0, 0].set(sitrus),
	)
	out = attack(state, engine, tables, "bravebird")
	assert out.winner == 0
	assert out.item_consumed[0, 0]
	assert out.hp[0, 0] == 741


def test_switch_restores_a_temporary_ability_and_keeps_its_public_reveal(battle, tables):
	teams, engine, _ = battle
	state = reset(KEY, abilities(teams, tables, "levitate", ""), engine)
	trace = tables.abilities.index["trace"]
	state = state._replace(
		ability=state.ability.at[0, 0].set(trace),
		ability_public=state.ability_public.at[0, 0].set(trace),
		request=jnp.array([MOVE, NONE], jnp.int8),
	)
	out = RUN(state, jnp.array([4, 0]), KEY, FORCED, engine)
	assert out.ability[0, 0] == tables.abilities.index["levitate"]
	assert observe(out, 1, engine).opponent.ability[0] == trace
