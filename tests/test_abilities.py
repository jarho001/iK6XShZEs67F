"""Focused tests for Gen 5 ability interactions."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.engine.randomness import (
	ABILITY_CHANCE,
	ACTION_ORDER,
	CONFUSION_DAMAGE_ROLL,
	CONFUSION_DURATION,
	CONFUSION_HIT,
	CRIT,
	DAMAGE_ROLL,
	FULL_PARALYSIS,
	HIT,
	HIT_COUNT,
	ORDER,
	SECONDARY,
	SWITCH_HANDLER_ORDER,
	UNFORCED,
)
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.state import (
	ATK,
	BURN,
	CONFUSION,
	CURSE,
	DEF,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	FLASH_FIRE,
	HAIL,
	HEALTHY,
	HP,
	LOCKED_MOVE,
	LOCKED_REMAINING,
	LOCKED_TRUE_DURATION,
	MOVE,
	MUST_RECHARGE,
	NONE,
	PARALYSIS,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	POISON,
	RAIN,
	SAND,
	SPA,
	SPD,
	SPE,
	SPIKES,
	SUN,
	SWITCH,
	TOXIC,
	TRUANT,
	TWO_TURN_MOVE,
)
from pokejax.engine.step import step
from pokejax.engine.switching import run_switch_in, switch_members

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


def test_sturdy_saves_a_full_hp_one_hit_and_triggers_on_a_multihit_move(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "sturdy")
	sash = tables.items.index["focussash"]
	state = state._replace(
		stats=state.stats.at[0, 0, ATK].set(1000).at[1, 0, DEF].set(1),
		hp=state.hp.at[1, 0].set(1000),
		item=state.item.at[1, 0].set(sash),
	)
	one_hit = attack(state, engine, tables, "tackle")
	assert int(one_hit.hp[1, 0]) == 1
	assert int(one_hit.ability_public[1, 0]) == tables.abilities.index["sturdy"]
	assert int(one_hit.item[1, 0]) == sash

	multi = attack(
		state,
		engine,
		tables,
		"tailslap",
		FORCED.at[HIT_COUNT, 0, 0].set(2),
	)
	assert int(multi.hp[1, 0]) == 0
	assert int(multi.ability_public[1, 0]) == tables.abilities.index["sturdy"]
	assert int(multi.item[1, 0]) == sash


def test_blaze_and_choice_specs_round_once_at_the_stat_event_boundary(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "blaze", "")
	fire = tables.types.index["fire"]
	water = tables.types.index["water"]
	state = state._replace(
		level=state.level.at[0, 0].set(80),
		hp=state.hp.at[0, 0].set(49).at[1, 0].set(293),
		stats=state.stats.at[0, 0, HP].set(256).at[0, 0, SPA].set(221).at[1, 0, SPD].set(230),
		item=state.item.at[0, 0].set(tables.items.index["choicespecs"]),
		types=state.types.at[0, 0, 0].set(fire).at[1, 0, 0].set(water),
	)
	forced = FORCED.at[DAMAGE_ROLL, 0, 0].set(2).at[SECONDARY, 0, 0].set(0)
	out = attack(state, engine, tables, "fireblast", forced)
	assert int(out.hp[1, 0]) == 163


def test_magic_guard_blocks_weather_status_and_hazard_damage(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "magicguard", "")
	for weather in (HAIL, SAND):
		out = attack(state._replace(weather=jnp.int8(weather)), engine, tables, "tackle")
		assert int(out.hp[0, 0]) == 1000
	for status in (BURN, POISON, TOXIC):
		poisoned = state._replace(
			status=state.status.at[0, 0].set(status),
			status_counter=state.status_counter.at[0, 0].set(3),
		)
		assert int(attack(poisoned, engine, tables, "tackle").hp[0, 0]) == 1000

	hazard = state._replace(
		status=state.status.at[0, 0].set(HEALTHY),
		weather=jnp.int8(0),
		ability=state.ability.at[0].set(tables.abilities.index["magicguard"]),
		hazards=state.hazards.at[0, SPIKES].set(1),
	)
	out = RUN(hazard, jnp.array([4, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 1]) == 1000


def test_magic_guard_blocks_life_orb_recoil_rocky_helmet_and_crash(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "magicguard", "")
	life_orb = state._replace(item=state.item.at[0, 0].set(tables.items.index["lifeorb"]))
	assert int(attack(life_orb, engine, tables, "tackle").hp[0, 0]) == 1000

	brave_bird = attack(state, engine, tables, "bravebird")
	assert int(brave_bird.hp[0, 0]) == 1000

	helm = state._replace(item=state.item.at[1, 0].set(tables.items.index["rockyhelmet"]))
	assert int(attack(helm, engine, tables, "tackle").hp[0, 0]) == 1000

	miss = FORCED.at[HIT, 0, 0].set(0)
	crash = attack(state, engine, tables, "highjumpkick", miss)
	assert int(crash.hp[0, 0]) == 1000


def test_magic_guard_blocks_trapping_and_curse_residuals(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "magicguard", "")
	state = state._replace(
		volatiles=state.volatiles.at[0, PARTIALLY_TRAPPED].set(2).at[0, PARTIAL_TRAP_SOURCE].set(1),
		active_turns=state.active_turns.at[1, 0].set(1),
	)
	assert int(attack(state, engine, tables, "tackle").hp[0, 0]) == 1000

	state = state._replace(
		volatiles=state.volatiles.at[0, PARTIALLY_TRAPPED].set(0).at[0, CURSE].set(1),
	)
	assert int(attack(state, engine, tables, "tackle").hp[0, 0]) == 1000


def test_magic_guard_does_not_block_struggle_recoil(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "magicguard", "")
	struggle = tables.moves.index["struggle"]
	state = state._replace(moves=state.moves.at[0, 0, 0].set(struggle))
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 750


def test_ghost_curse_cost_is_direct_and_non_ghost_curse_boosts(battle, tables):
	_, engine, state = battle
	ghost = tables.types.index["ghost"]
	state = abilities(state, tables, "magicguard", "")
	state = state._replace(
		types=state.types.at[0, 0, 0].set(jnp.asarray(ghost, dtype=state.types.dtype)),
		moves=state.moves.at[0, 0, 0].set(tables.moves.index["curse"]),
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.hp[0, 0]) == 500
	assert int(out.volatiles[1, CURSE]) == 1
	assert int(out.hp[1, 0]) == 750

	plain = state._replace(
		types=state.types.at[0, 0].set(jnp.array([-1, -1], dtype=state.types.dtype)),
		ability=state.ability.at[0, 0].set(tables.abilities.index["noability"]),
	)
	out = RUN(plain, jnp.array([0, 0]), KEY, FORCED, engine)
	np.testing.assert_array_equal(out.boosts[0, :5], [1, 1, 0, 0, -1])
	assert int(out.hp[0, 0]) == 1000
	assert int(out.volatiles[1, CURSE]) == 0


def test_regenerator_heals_after_uturn_and_phazing(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "regenerator", "")
	state = state._replace(
		hp=state.hp.at[0, 0].set(500),
		moves=state.moves.at[0, 0, 0].set(tables.moves.index["uturn"]),
		request=jnp.array([MOVE, MOVE], jnp.int8),
	)
	paused = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(paused.request[0]) == SWITCH
	finished = RUN(paused, jnp.array([4, 0]), KEY, FORCED, engine)
	assert int(finished.active[0]) == 1
	assert int(finished.hp[0, 0]) == 833
	assert int(finished.ability_public[0, 0]) == 0

	phaze = abilities(state, tables, "", "regenerator")
	phaze = phaze._replace(
		active=jnp.array([0, 0]),
		hp=phaze.hp.at[1, 0].set(500),
		moves=phaze.moves.at[0, 0, 0].set(tables.moves.index["roar"]),
		request=jnp.array([MOVE, NONE], jnp.int8),
	)
	finished = RUN(phaze, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(finished.active[1]) != 0
	assert int(finished.hp[1, 0]) == 833
	assert int(finished.ability_public[1, 0]) == 0


def test_natural_cure_reveals_on_a_midturn_switch(battle, tables):
	_, engine, state = battle
	state = state._replace(
		ability=state.ability.at[0].set(tables.abilities.index["naturalcure"]),
		status=state.status.at[0, 0].set(BURN),
		hp=state.hp.at[0, 0].set(500),
		moves=state.moves.at[0, 0, 0].set(tables.moves.index["uturn"]),
		request=jnp.array([MOVE, MOVE], jnp.int8),
	)
	paused = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	finished = RUN(paused, jnp.array([4, 0]), KEY, FORCED, engine)
	assert int(finished.status[0, 0]) == HEALTHY
	assert int(finished.ability_public[0, 0]) == tables.abilities.index["naturalcure"]


def test_truant_loafs_on_two_turn_continuation_and_recharge(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "truant", "")
	first = attack(state, engine, tables, "fly")
	assert int(first.volatiles[0, TWO_TURN_MOVE]) == tables.moves.index["fly"]
	assert int(first.volatiles[0, TRUANT]) == 1
	second = RUN(first, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(second.volatiles[0, TWO_TURN_MOVE]) == 0
	assert int(second.hp[1, 0]) == 1000
	assert int(second.pp[0, 0, 0]) == int(first.pp[0, 0, 0])
	assert int(second.ability_public[0, 0]) == tables.abilities.index["truant"]

	confused = first._replace(volatiles=first.volatiles.at[0, CONFUSION].set(3))
	forced = FORCED.at[CONFUSION_HIT, 0, 0].set(1)
	loaf = RUN(confused, jnp.array([0, 0]), KEY, forced, engine)
	assert int(loaf.volatiles[0, CONFUSION]) == 3
	assert int(loaf.volatiles[0, TRUANT]) == 0

	beam = attack(state, engine, tables, "hyperbeam")
	assert int(beam.volatiles[0, MUST_RECHARGE]) == 1
	recharge = RUN(beam, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(recharge.volatiles[0, MUST_RECHARGE]) == 0
	assert int(recharge.volatiles[0, TRUANT]) == 0
	assert int(recharge.pp[0, 0, 0]) == int(beam.pp[0, 0, 0])


def test_aftermath_and_rough_skin_can_make_a_double_faint(battle, tables):
	_, engine, state = battle
	for ability, hp_cost in (("aftermath", 250), ("roughskin", 125)):
		state = abilities(state, tables, "", ability)
		state = state._replace(
			hp=state.hp.at[0].set(0).at[0, 0].set(hp_cost - 1).at[1].set(0).at[1, 0].set(1),
			stats=state.stats.at[:, :, HP].set(1000),
		)
		out = attack(state, engine, tables, "tackle")
		assert int(out.hp[0, 0]) == 0
		assert int(out.hp[1, 0]) == 0
		assert int(out.winner) == 0


def test_moxie_does_not_boost_when_rough_skin_causes_a_double_faint(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "moxie", "roughskin")
	state = state._replace(
		hp=state.hp.at[0].set(0).at[0, 0].set(100).at[1].set(0).at[1, 0].set(1),
		stats=state.stats.at[:, :, HP].set(1000).at[0, 0, ATK].set(1000),
	)
	out = attack(state, engine, tables, "tackle")
	assert int(out.hp[0, 0]) == 0
	assert int(out.hp[1, 0]) == 0
	assert int(out.boosts[0, ATK - 1]) == 0


def test_shadow_tag_does_not_trap_shadow_tag_users_and_ghosts_are_trapped(battle, tables):
	_, engine, state = battle
	shadow = tables.abilities.index["shadowtag"]
	state = state._replace(
		ability=state.ability.at[:, 0].set(shadow),
		request=jnp.array([MOVE, MOVE], dtype=state.request.dtype),
	)
	assert np.all(np.asarray(legal_actions(state, engine)[:, 4:]))

	state = state._replace(ability=state.ability.at[1, 0].set(tables.abilities.index["noability"]))
	assert not np.any(np.asarray(legal_actions(state, engine)[1, 4:]))

	arena = tables.abilities.index["arenatrap"]
	ghost = tables.types.index["ghost"]
	state = state._replace(
		ability=state.ability.at[0, 0].set(tables.abilities.index["noability"]).at[1, 0].set(arena),
		types=state.types.at[0, 0].set(jnp.array([ghost, -1], dtype=state.types.dtype)),
	)
	assert not np.any(np.asarray(legal_actions(state, engine)[0, 4:]))


def test_magnet_pull_traps_steel_but_arena_trap_does_not_trap_flying(battle, tables):
	_, engine, state = battle
	state = state._replace(
		ability=state.ability.at[1, 0].set(tables.abilities.index["magnetpull"]),
		types=state.types.at[0, 0].set(
			jnp.array([tables.types.index["steel"], -1], dtype=state.types.dtype)
		),
	)
	assert not np.any(np.asarray(legal_actions(state, engine)[0, 4:]))

	arena = state._replace(
		ability=state.ability.at[1, 0].set(tables.abilities.index["arenatrap"]),
		types=state.types.at[0, 0].set(
			jnp.array([tables.types.index["flying"], -1], dtype=state.types.dtype)
		),
	)
	assert np.any(np.asarray(legal_actions(arena, engine)[0, 4:]))


def test_prankster_status_move_can_match_priority_and_act_first(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "prankster", "")
	state = state._replace(
		stats=state.stats.at[0, 0, SPE].set(200).at[1, 0, SPE].set(100),
		moves=state.moves.at[0, 0, 0]
		.set(tables.moves.index["thunderwave"])
		.at[1, 0, 0]
		.set(tables.moves.index["quickattack"]),
		request=jnp.array([MOVE, MOVE], jnp.int8),
	)
	forced = FORCED.at[FULL_PARALYSIS, 1, 0].set(1)
	out = RUN(state, jnp.array([0, 0]), KEY, forced, engine)
	assert int(out.status[1, 0]) == PARALYSIS
	assert int(out.hp[0, 0]) == 1000


def test_weak_armor_uses_the_gen5_one_speed_stage_per_hit_rule(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "weakarmor")
	out = attack(state, engine, tables, "tackle")
	assert int(out.boosts[1, SPE - 1]) == 1


def test_cursed_body_stores_disable_move_and_counter(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "", "cursedbody")
	state = state._replace(
		moves=state.moves.at[0, 0, 1].set(tables.moves.index["surf"]),
		pp=state.pp.at[0, 0, 1].set(10),
	)
	forced = FORCED.at[ABILITY_CHANCE, 1, 0].set(1)
	out = attack(state, engine, tables, "sludgebomb", forced)
	assert int(out.volatiles[0, DISABLE_MOVE]) == tables.moves.index["sludgebomb"]
	assert int(out.volatiles[0, DISABLE_REMAINING]) == 3
	assert int(out.ability_public[1, 0]) == tables.abilities.index["cursedbody"]
	assert not bool(legal_actions(out, engine)[0, 0])
	assert bool(legal_actions(out, engine)[0, 1])
	assert int(out.ability_public[1, 0]) == tables.abilities.index["cursedbody"]


def test_mummy_reveals_the_transfer_and_both_protocol_abilities(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "levitate", "mummy")
	out = attack(state, engine, tables, "tackle")
	mummy = tables.abilities.index["mummy"]
	assert int(out.ability[0, 0]) == mummy
	assert int(out.ability_public[0, 0]) == mummy
	assert int(out.ability_public[1, 0]) == mummy


def test_mummy_disables_rock_head_before_recoil(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "rockhead", "mummy")
	out = attack(state, engine, tables, "headsmash")
	assert int(out.ability[0, 0]) == tables.abilities.index["mummy"]
	assert int(out.hp[0, 0]) < int(state.hp[0, 0])


def test_mummy_restores_life_orb_recoil_after_sheer_force_hit(battle, tables):
	_, engine, state = battle
	state = abilities(state, tables, "sheerforce", "mummy")
	state = state._replace(
		item=state.item.at[0, 0].set(tables.items.index["lifeorb"]),
		moves=state.moves.at[0, 0, 0].set(tables.moves.index["ironhead"]),
	)
	out = RUN(state, jnp.array([0, 0]), KEY, FORCED, engine)
	assert int(out.ability[0, 0]) == tables.abilities.index["mummy"]
	assert int(out.hp[0, 0]) == 900


def test_switch_resets_color_change_to_the_team_type(battle, tables):
	_, engine, state = battle
	rock = jnp.asarray(tables.types.index["rock"], dtype=state.types.dtype)
	state = abilities(state, tables, "", "colorchange")
	state = state._replace(types=state.types.at[1, 0].set(jnp.array([rock, -1])))
	state = switch_members(
		state,
		jnp.array([False, True]),
		jnp.array([0, 1]),
		tables=engine,
	)
	np.testing.assert_array_equal(state.types[1, 0], state.base_types[1, 0])


def test_confusion_self_hit_uses_focus_sash_only_for_a_lethal_hit(battle, tables):
	_, engine, state = battle
	sash = tables.items.index["focussash"]
	state = state._replace(
		hp=state.hp.at[0, 0].set(1),
		stats=state.stats.at[0, 0, HP].set(1),
		item=state.item.at[0, 0].set(sash),
		volatiles=state.volatiles.at[0, CONFUSION].set(5),
	)
	forced = FORCED.at[CONFUSION_HIT, 0, 0].set(1).at[CONFUSION_DAMAGE_ROLL, 0, 0].set(0)
	out = RUN(state, jnp.array([0, 4]), KEY, forced, engine)
	assert int(out.hp[0, 0]) == 1
	assert int(out.item[0, 0]) == 0
	assert bool(out.item_consumed[0, 0])
	assert bool(out.item_revealed[0, 0])
	assert int(out.volatiles[0, CONFUSION]) == 4

	sturdy = RUN(abilities(state, tables, "sturdy", ""), jnp.array([0, 4]), KEY, forced, engine)
	assert int(sturdy.hp[0, 0]) == 1
	assert int(sturdy.item[0, 0]) == sash
	assert int(sturdy.ability_public[0, 0]) == tables.abilities.index["sturdy"]

	state = state._replace(hp=state.hp.at[0, 0].set(268), stats=state.stats.at[0, 0, HP].set(268))
	out = RUN(state, jnp.array([0, 4]), KEY, forced, engine)
	assert 0 < int(out.hp[0, 0]) < 268
	assert int(out.item[0, 0]) == sash
	assert not bool(out.item_consumed[0, 0])
