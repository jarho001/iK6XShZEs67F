"""Gen 5 weather, entry hazards, and Rapid Spin from the pinned Showdown source."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import test_status

from pokejax.engine.damage import accuracy_hit, move_damage_bases
from pokejax.engine.field import apply_move_field_effects, apply_switch_hazards, effective_move_type
from pokejax.engine.items import gather_active_items
from pokejax.engine.move_effects import apply_healing
from pokejax.engine.randomness import (
	CRIT,
	DAMAGE_ROLL,
	HIT,
	ORDER,
	SWITCH_HANDLER_ORDER,
	UNFORCED,
	WEATHER_ORDER,
)
from pokejax.engine.residual import residual
from pokejax.engine.state import (
	FREEZE,
	HAIL,
	HEALTHY,
	MOVE,
	NONE,
	POISON,
	RAIN,
	SAND,
	SIDE1,
	SPIKES,
	SUN,
	SWITCH,
	TOXIC,
	TOXIC_SPIKES,
)
from pokejax.engine.status import inflict_status
from pokejax.engine.step import step

KEY = jax.random.key(51)
FORCED = (
	UNFORCED.at[HIT, :, 0]
	.set(1)
	.at[CRIT, :, 0]
	.set(0)
	.at[DAMAGE_ROLL, :, 0]
	.set(0)
	.at[ORDER, 0, 0]
	.set(0)
	.at[ORDER, 0, WEATHER_ORDER]
	.set(0)
	.at[ORDER, 0, SWITCH_HANDLER_ORDER]
	.set(0)
)
RUN = jax.jit(step)


@pytest.fixture(scope="module")
def setup():
	return test_status.setup.__wrapped__()


def set_move(state, tables, side, move):
	return state._replace(moves=state.moves.at[side, 0, 0].set(tables.moves.index[move]))


def isolated(state):
	return state._replace(request=state.request.at[1].set(NONE))


def test_weather_replaces_but_same_weather_does_not_refresh(setup):
	tables, t, state = setup
	rain = isolated(set_move(state, tables, 0, "raindance"))
	out = RUN(rain, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.weather) == RAIN
	assert int(out.weather_turns) == 4

	old_rain = rain._replace(weather=jnp.int8(RAIN), weather_turns=jnp.int8(2))
	out = RUN(old_rain, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.weather) == RAIN
	assert int(out.weather_turns) == 1

	to_sun = isolated(set_move(old_rain, tables, 0, "sunnyday"))
	out = RUN(to_sun, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.weather) == SUN
	assert int(out.weather_turns) == 4


def test_weather_modifier_accuracy_weather_ball_and_rock_special_defense(setup):
	tables, t, state = setup
	water = tables.types.index["water"]
	fire = tables.types.index["fire"]
	normal = tables.types.index["normal"]
	state = state._replace(
		level=jnp.full_like(state.level, 100),
		stats=jnp.full_like(state.stats, 100).at[:, :, 0].set(160),
		types=state.types.at[:, 0].set(jnp.array([normal, -1], jnp.int8)),
	)
	surf = tables.moves.index["surf"]

	item_rows = gather_active_items(state, t)
	clear = move_damage_bases(state, 0, surf, KEY, FORCED, t, item_rows, False)[0]
	rain = state._replace(weather=jnp.int8(RAIN))
	sun = state._replace(weather=jnp.int8(SUN))
	assert int(clear) == 81
	assert int(move_damage_bases(rain, 0, surf, KEY, FORCED, t, item_rows, False)[0]) == 121
	assert int(move_damage_bases(sun, 0, surf, KEY, FORCED, t, item_rows, False)[0]) == 40
	assert int(t.weather_damage_by_type[RAIN, water]) == 6144
	assert int(t.weather_damage_by_type[RAIN, fire]) == 2048

	weather_ball = tables.moves.index["weatherball"]
	ball_sun = sun._replace(moves=sun.moves.at[0, 0, 0].set(weather_ball))
	assert int(effective_move_type(ball_sun, weather_ball, t)) == fire
	assert (
		int(move_damage_bases(ball_sun, 0, weather_ball, KEY, FORCED, t, item_rows, False)[0])
		== 129
	)

	thunder = tables.moves.index["thunder"]
	hurricane = tables.moves.index["hurricane"]
	blizzard = tables.moves.index["blizzard"]
	forced_miss = FORCED.at[HIT, 0, 0].set(0)
	assert bool(accuracy_hit(rain, 0, thunder, KEY, forced_miss, t))
	assert not bool(accuracy_hit(sun, 0, hurricane, KEY, forced_miss, t))
	assert bool(
		accuracy_hit(state._replace(weather=jnp.int8(HAIL)), 0, blizzard, KEY, forced_miss, t)
	)

	rock = tables.types.index["rock"]
	rock_target = state._replace(
		types=state.types.at[1, 0].set(jnp.array([rock, -1], jnp.int8)),
		weather=jnp.int8(SAND),
	)
	flamethrower = tables.moves.index["flamethrower"]
	assert int(
		move_damage_bases(rock_target, 0, flamethrower, KEY, FORCED, t, item_rows, False)[0]
	) < int(move_damage_bases(state, 0, flamethrower, KEY, FORCED, t, item_rows, False)[0])


def test_weather_residual_immunities_and_expiration_order(setup):
	tables, t, state = setup
	rock = tables.types.index["rock"]
	ice = tables.types.index["ice"]
	normal = tables.types.index["normal"]
	state = state._replace(
		types=state.types.at[0, 0].set(jnp.array([rock, -1], jnp.int8)),
		weather=jnp.int8(SAND),
		weather_turns=jnp.int8(2),
	)
	out = jax.jit(residual)(state, True, KEY, FORCED, t)
	assert int(out.hp[0, 0]) == 160
	assert int(out.hp[1, 0]) == 150
	assert int(out.weather_turns) == 1

	hail = state._replace(
		types=state.types.at[0, 0]
		.set(jnp.array([normal, -1], jnp.int8))
		.at[1, 0]
		.set(jnp.array([ice, -1], jnp.int8)),
		weather=jnp.int8(HAIL),
		weather_turns=jnp.int8(1),
	)
	out = jax.jit(residual)(hail, True, KEY, FORCED, t)
	assert int(out.weather) == 0
	assert int(out.weather_turns) == 0
	assert int(out.hp[0, 0]) == 160
	assert int(out.hp[1, 0]) == 160


def test_weather_healing_uses_showdown_weather_factors(setup):
	tables, t, state = setup
	synthesis = tables.moves.index["synthesis"]
	state = state._replace(
		stats=state.stats.at[0, 0, 0].set(515),
		hp=state.hp.at[0, 0].set(10),
	)
	state, _ = apply_healing(state, 0, synthesis, True, True, t)
	assert int(state.hp[0, 0]) == 267  # modify(515, 0.5) rounds the half down.

	sunny = state._replace(hp=state.hp.at[0, 0].set(10), weather=jnp.int8(SUN))
	sunny, _ = apply_healing(sunny, 0, synthesis, True, True, t)
	assert int(sunny.hp[0, 0]) == 354  # modify(515, 0.667) yields 344.

	rain = state._replace(hp=state.hp.at[0, 0].set(10), weather=jnp.int8(RAIN))
	rain, _ = apply_healing(rain, 0, synthesis, True, True, t)
	assert int(rain.hp[0, 0]) == 139


def test_sun_blocks_freeze_status(setup):
	_, t, state = setup
	sunny = state._replace(weather=jnp.int8(SUN))
	sunny = inflict_status(sunny, 1, FREEZE, True, KEY, FORCED, t)
	assert int(sunny.status[1, 0]) == HEALTHY

	clear = state._replace(weather=jnp.int8(0))
	clear = inflict_status(clear, 1, FREEZE, True, KEY, FORCED, t)
	assert int(clear.status[1, 0]) == FREEZE


def test_spikes_layers_damage_and_toxic_spikes_switch_rules(setup):
	tables, t, state = setup
	spikes_move = tables.moves.index["spikes"]
	for expected in (1, 2, 3, 3):
		state = apply_move_field_effects(
			state, 0, spikes_move, True, True, True, KEY, FORCED, t, gather_active_items(state, t)
		)
		assert int(state.hazards[1, SPIKES]) == expected

	for layers, expected_hp in ((1, 140), (2, 134), (3, 120)):
		incoming = state._replace(
			hp=state.hp.at[0, 1].set(160),
			hazards=state.hazards.at[0, SPIKES].set(layers),
		)
		incoming = apply_switch_hazards(incoming, 0, 1, True, t)
		assert int(incoming.hp[0, 1]) == expected_hp

	toxic_move = tables.moves.index["toxicspikes"]
	state = state._replace(hazards=state.hazards.at[1, TOXIC_SPIKES].set(2))
	state = apply_move_field_effects(
		state, 0, toxic_move, True, True, True, KEY, FORCED, t, gather_active_items(state, t)
	)
	assert int(state.hazards[1, TOXIC_SPIKES]) == 2
	poison = tables.types.index["poison"]
	poison_switch = state._replace(
		types=state.types.at[1, 1].set(jnp.array([poison, -1], jnp.int8))
	)
	poison_switch = apply_switch_hazards(poison_switch, 1, 1, True, t)
	assert int(poison_switch.hazards[1, TOXIC_SPIKES]) == 0
	assert int(poison_switch.status[1, 1]) == HEALTHY

	flying = tables.types.index["flying"]
	flying_switch = state._replace(
		types=state.types.at[1, 1].set(jnp.array([flying, -1], jnp.int8))
	)
	flying_switch = apply_switch_hazards(flying_switch, 1, 1, True, t)
	assert int(flying_switch.hazards[1, TOXIC_SPIKES]) == 2
	assert int(flying_switch.status[1, 1]) == HEALTHY

	steel = tables.types.index["steel"]
	steel_switch = state._replace(types=state.types.at[1, 1].set(jnp.array([steel, -1], jnp.int8)))
	steel_switch = apply_switch_hazards(steel_switch, 1, 1, True, t)
	assert int(steel_switch.status[1, 1]) == HEALTHY

	plain = state._replace(hazards=state.hazards.at[1, TOXIC_SPIKES].set(1))
	plain_switch = apply_switch_hazards(plain, 1, 1, True, t)
	assert int(plain_switch.status[1, 1]) == POISON
	toxic = state._replace(hazards=state.hazards.at[1, TOXIC_SPIKES].set(2))
	toxic_switch = apply_switch_hazards(toxic, 1, 1, True, t)
	assert int(toxic_switch.status[1, 1]) == TOXIC

	poison = tables.types.index["poison"]
	poison_with_hazards = state._replace(
		types=state.types.at[1, 1].set(jnp.array([poison, -1], jnp.int8)),
		hp=state.hp.at[1, 1].set(40),
		hazards=state.hazards.at[1].set(jnp.array([3, 2], jnp.int8)),
	)
	toxic_first = poison_with_hazards._replace(
		hazard_order=state.hazard_order.at[1].set(jnp.array([2, 1], jnp.int8))
	)
	toxic_first = apply_switch_hazards(toxic_first, 1, 1, True, t)
	assert int(toxic_first.hazards[1, TOXIC_SPIKES]) == 0

	spikes_first = poison_with_hazards._replace(
		hazard_order=state.hazard_order.at[1].set(jnp.array([1, 2], jnp.int8))
	)
	spikes_first = apply_switch_hazards(spikes_first, 1, 1, True, t)
	assert int(spikes_first.hazards[1, TOXIC_SPIKES]) == 2


def test_hazard_faint_on_replacement_ends_battle_and_rapid_spin_clears(setup):
	tables, t, state = setup
	state = state._replace(
		hp=jnp.zeros_like(state.hp).at[0, 1].set(40).at[1, 0].set(160),
		hazards=state.hazards.at[0, SPIKES].set(3),
		request=jnp.array([SWITCH, NONE], jnp.int8),
	)
	out = RUN(state, jnp.array([4, 0]), KEY, FORCED, t)
	assert int(out.hp[0, 1]) == 0
	assert int(out.winner) == SIDE1

	rapid = isolated(set_move(state, tables, 0, "rapidspin"))
	rapid = rapid._replace(
		hp=rapid.hp.at[:, 0].set(160),
		hazards=rapid.hazards.at[0].set(jnp.array([2, 1], jnp.int8)),
		request=rapid.request.at[0].set(MOVE),
	)
	out = RUN(rapid, jnp.array([0, 0]), KEY, FORCED, t)
	assert int(out.hp[1, 0]) < 160
	assert np.all(np.asarray(out.hazards[0]) == 0)
