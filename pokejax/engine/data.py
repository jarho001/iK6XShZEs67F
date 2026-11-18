"""Device tables built once on the host; compiled engine code reads only arrays."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.support import CONFUSION_MOVES, SECONDARY_MOVES
from pokejax.tables import Tables

from .items import build_item_hook_table
from .state import (
	BURN,
	FREEZE,
	HAIL,
	NO_WEATHER,
	NUM_WEATHERS,
	PARALYSIS,
	POISON,
	RAIN,
	SAND,
	SLEEP,
	SUN,
	TOXIC,
)
from .status import PARTY_CURE, REFRESH, REST, TRANSFER


class EngineTables(NamedTuple):
	species_base_stats: jax.Array
	species_types: jax.Array
	species_plain_types: jax.Array
	species_shedinja: jax.Array
	move_type: jax.Array
	move_category: jax.Array
	move_base_power: jax.Array
	move_accuracy: jax.Array
	move_always_hits: jax.Array
	move_priority: jax.Array
	move_crit_ratio: jax.Array
	move_crit_always: jax.Array
	move_drain: jax.Array
	move_recoil: jax.Array
	move_recoil_basehp: jax.Array
	move_heal: jax.Array
	move_heal_weather: jax.Array
	move_damage_kind: jax.Array
	move_fixed_damage: jax.Array
	move_hit_min: jax.Array
	move_hit_max: jax.Array
	move_self_faint: jax.Array
	move_crash_damage: jax.Array
	move_pain_split: jax.Array
	move_roost: jax.Array
	move_max_pp: jax.Array
	move_flags: jax.Array
	move_status: jax.Array
	move_status_self: jax.Array
	move_effect: jax.Array
	move_defrost: jax.Array
	move_thaws_target: jax.Array
	move_checks_immunity: jax.Array
	move_boosts: jax.Array
	move_boost_when: jax.Array
	move_boost_target: jax.Array
	move_boost_effect: jax.Array
	move_confusion: jax.Array
	move_secondary_chance: jax.Array
	move_secondary_self: jax.Array
	move_secondary_status: jax.Array
	move_secondary_status_choices: jax.Array
	move_secondary_boosts: jax.Array
	move_secondary_volatile: jax.Array
	move_secondary_user_species: jax.Array
	status_immune: jax.Array
	type_exponent: jax.Array
	type_immune: jax.Array
	nature_multipliers: jax.Array
	struggle_id: jax.Array
	multitype_id: jax.Array
	fire_type: jax.Array
	flying_type: jax.Array
	ground_type: jax.Array
	normal_type: jax.Array
	move_weather: jax.Array
	move_hazard: jax.Array
	move_rapid_spin: jax.Array
	move_weather_accuracy: jax.Array
	move_weather_type: jax.Array
	move_weather_power: jax.Array
	move_weather_heal: jax.Array
	weather_damage_by_type: jax.Array
	weather_spdef_boost: jax.Array
	weather_residual_immune: jax.Array
	weather_status_immune: jax.Array
	poison_type: jax.Array
	move_stall_kind: jax.Array
	move_volatile_status: jax.Array
	move_self_volatile: jax.Array
	move_breaks_protect: jax.Array
	move_protects: jax.Array
	move_charge: jax.Array
	move_turn_volatile_ids: jax.Array
	move_charge_semi: jax.Array
	move_charge_boosts: jax.Array
	move_charge_sun_skip: jax.Array
	move_fakeout: jax.Array
	move_target_foe: jax.Array
	move_spread: jax.Array
	move_hits_semi: jax.Array
	move_semi_power: jax.Array
	move_weather_power_scale: jax.Array
	move_magnitude: jax.Array
	move_self_switch: jax.Array
	move_force_switch: jax.Array
	move_contact: jax.Array
	species_nfe: jax.Array
	species_num: jax.Array
	item_hooks: object


def build_engine_tables(tables: Tables) -> EngineTables:
	"""Apply Gen 5 Pokemon constructor PP Ups (including the Trump Card exception)."""
	moves = tables.moves
	pp = moves.pp.astype(np.int32)
	no_boost = moves.no_pp_boosts | (np.asarray(moves.ids) == "trumpcard")
	max_pp = np.where(no_boost, pp, pp * 8 // 5).astype(np.uint8)
	# data/conditions.ts arceus.onType: in Gen 5 all forms are Normal
	# without Multitype. The Pokedex's displayed form types are not actual types.
	plain_types = tables.species.types.copy()
	plain_types[tables.species.num == 493] = [tables.types.index["normal"], -1]
	status = np.zeros(len(moves.ids), np.int8)
	for names, value in (
		(("toxic",), TOXIC),
		(("thunderwave", "glare", "stunspore"), PARALYSIS),
		(("willowisp",), BURN),
		(("spore", "sleeppowder", "hypnosis", "darkvoid", "lovelykiss"), SLEEP),
		(("rest",), SLEEP),
	):
		for name in names:
			status[moves.index[name]] = value
	effect = np.zeros(len(moves.ids), np.int8)
	for names, value in (
		(("rest",), REST),
		(("healbell", "aromatherapy"), PARTY_CURE),
		(("psychoshift",), TRANSFER),
		(("refresh",), REFRESH),
	):
		for name in names:
			effect[moves.index[name]] = value
	# Only mechanics in this task are enabled; other exported boosts can have
	# additional effects that belong to later tasks.
	boost_moves = (
		"agility",
		"bulkup",
		"calmmind",
		"coil",
		"cosmicpower",
		"dragondance",
		"honeclaws",
		"nastyplot",
		"quiverdance",
		"rockpolish",
		"shellsmash",
		"shiftgear",
		"swordsdance",
		"tailglow",
		"workup",
		"swagger",
		"flatter",
	)
	drop_moves = (
		"closecombat",
		"dracometeor",
		"leafstorm",
		"overheat",
		"psychoboost",
		"superpower",
		"vcreate",
	)
	boosts = np.zeros_like(moves.boosts)
	when = np.zeros(len(moves.ids), np.int8)
	boost_target = np.zeros(len(moves.ids), np.int8)
	for name in boost_moves:
		idx = moves.index[name]
		boosts[idx], when[idx] = moves.boosts[idx], 1
		boost_target[idx] = moves.target[idx] != moves.target_names.index("self")
	for name in drop_moves:
		idx = moves.index[name]
		boosts[idx], when[idx] = moves.self_boosts[idx], 2
	boost_effect = np.zeros(len(moves.ids), np.int8)
	for value, name in enumerate(("haze", "clearsmog", "bellydrum"), 1):
		boost_effect[moves.index[name]] = value
	immune = np.zeros((7, len(tables.types.ids)), np.bool_)
	for value, names in (
		(BURN, ("fire",)),
		(POISON, ("poison", "steel")),
		(TOXIC, ("poison", "steel")),
		(FREEZE, ("ice",)),
	):
		for name in names:
			immune[value, tables.types.index[name]] = True
	defrost = moves.flags[:, moves.flag_names.index("defrost")]
	# Gen 5 explicitly removes Scald's later-generation thawsTarget property.
	thaws_target = np.zeros(len(moves.ids), np.bool_)
	secondary_chance = moves.secondary_chance.copy()
	secondary_chance[~np.isin(moves.ids, tuple(SECONDARY_MOVES))] = 0
	user_species = np.zeros_like(moves.secondary_status, dtype=np.int16)
	user_species[moves.index["chatter"]] = tables.species.index["chatot"]
	weather_moves = np.zeros(len(moves.ids), np.int8)
	for name, weather in (
		("raindance", RAIN),
		("sunnyday", SUN),
		("sandstorm", SAND),
		("hail", HAIL),
	):
		weather_moves[moves.index[name]] = weather
	hazards = np.zeros(len(moves.ids), np.int8)
	hazards[moves.index["spikes"]] = 1
	hazards[moves.index["toxicspikes"]] = 2
	weather_accuracy = np.full((len(moves.ids), NUM_WEATHERS), -1, np.int16)
	weather_always = -2
	for name in ("thunder", "hurricane"):
		weather_accuracy[moves.index[name], RAIN] = weather_always
		weather_accuracy[moves.index[name], SUN] = 50
	weather_accuracy[moves.index["blizzard"], HAIL] = weather_always
	weather_type = np.full((len(moves.ids), NUM_WEATHERS), -1, np.int8)
	weather_power = np.ones((len(moves.ids), NUM_WEATHERS), np.uint8)
	weather_ball = moves.index["weatherball"]
	for weather, type_name in (
		(SUN, "fire"),
		(RAIN, "water"),
		(SAND, "rock"),
		(HAIL, "ice"),
	):
		weather_type[weather_ball, weather] = tables.types.index[type_name]
		weather_power[weather_ball, weather] = 2
	weather_power_scale = np.full((len(moves.ids), NUM_WEATHERS), 4096, np.int16)
	solarbeam = moves.index["solarbeam"]
	weather_power_scale[solarbeam, (RAIN, SAND, HAIL)] = 2048
	weather_heal = np.broadcast_to(
		moves.heal[:, None, :], (len(moves.ids), NUM_WEATHERS, 2)
	).astype(np.int16, copy=True)
	for name in ("moonlight", "morningsun", "synthesis"):
		row = moves.index[name]
		weather_heal[row, NO_WEATHER] = (2048, 4096)
		weather_heal[row, SUN] = (2732, 4096)
		for weather in (RAIN, SAND, HAIL):
			weather_heal[row, weather] = (1024, 4096)
	weather_damage = np.full((NUM_WEATHERS, len(tables.types.ids)), 4096, np.int16)
	weather_damage[RAIN, tables.types.index["water"]] = 6144
	weather_damage[RAIN, tables.types.index["fire"]] = 2048
	weather_damage[SUN, tables.types.index["fire"]] = 6144
	weather_damage[SUN, tables.types.index["water"]] = 2048
	weather_spdef = np.zeros((NUM_WEATHERS, len(tables.types.ids)), np.bool_)
	weather_spdef[SAND, tables.types.index["rock"]] = True
	weather_immune = np.zeros((NUM_WEATHERS, len(tables.types.ids)), np.bool_)
	for type_name in ("rock", "ground", "steel"):
		weather_immune[SAND, tables.types.index[type_name]] = True
	weather_immune[HAIL, tables.types.index["ice"]] = True
	weather_status_immune = np.zeros((NUM_WEATHERS, 7), np.bool_)
	weather_status_immune[SUN, FREEZE] = True
	stall_kind = np.zeros(len(moves.ids), np.uint8)
	stall_kind[moves.volatile_status == 1] = 1
	stall_kind[moves.volatile_status == 2] = 2
	stall_kind[moves.index["quickguard"]] = 3
	stall_kind[moves.index["wideguard"]] = 4
	charge = moves.flags[:, moves.flag_names.index("charge")]
	move_turn_volatile_ids = np.asarray(
		[
			moves.index[name]
			for name in (
				"fly",
				"dig",
				"bounce",
				"dive",
				"shadowforce",
				"skullbash",
				"skyattack",
				"razorwind",
				"solarbeam",
				"freezeshock",
				"iceburn",
			)
		],
		dtype=np.int16,
	)
	charge_semi = np.zeros(len(moves.ids), np.uint8)
	for name in ("fly", "bounce"):
		charge_semi[moves.index[name]] = 1
	charge_semi[moves.index["dig"]] = 2
	charge_semi[moves.index["dive"]] = 3
	charge_semi[moves.index["shadowforce"]] = 4
	charge_boosts = np.zeros((len(moves.ids), 7), np.int8)
	charge_boosts[moves.index["skullbash"], 1] = 1
	hits_semi = np.zeros((len(moves.ids), 5), np.bool_)
	semi_power = np.zeros_like(hits_semi)
	for semi, names in (
		(1, ("gust", "twister", "skyuppercut", "thunder", "hurricane", "smackdown")),
		(2, ("earthquake", "magnitude")),
		(3, ("surf", "whirlpool")),
	):
		for name in names:
			hits_semi[moves.index[name], semi] = True
	for semi, names in (
		(1, ("gust", "twister")),
		(2, ("earthquake", "magnitude")),
		(3, ("surf", "whirlpool")),
	):
		for name in names:
			semi_power[moves.index[name], semi] = True
	target_foe = np.isin(
		np.asarray(moves.target_names)[moves.target],
		("normal", "any", "randomNormal", "allAdjacent", "allAdjacentFoes"),
	)
	spread = np.isin(
		np.asarray(moves.target_names)[moves.target], ("allAdjacent", "allAdjacentFoes")
	)
	return EngineTables(
		*map(
			jnp.asarray,
			(
				tables.species.base_stats,
				tables.species.types,
				plain_types,
				np.asarray(tables.species.ids) == "shedinja",
				moves.type,
				moves.category,
				moves.base_power,
				moves.accuracy,
				moves.always_hits,
				moves.priority,
				moves.crit_ratio,
				moves.crit_always,
				moves.drain,
				moves.recoil,
				moves.recoil_basehp,
				moves.heal,
				moves.heal_weather,
				moves.damage_kind,
				moves.fixed_damage,
				moves.hit_min,
				moves.hit_max,
				moves.self_faint,
				moves.crash_damage,
				moves.pain_split,
				moves.roost,
				max_pp,
				moves.flags,
				status,
				np.asarray(moves.ids) == "rest",
				effect,
				defrost,
				thaws_target,
				np.asarray(moves.ids) == "thunderwave",
				boosts,
				when,
				boost_target,
				boost_effect,
				moves.confusion & np.isin(moves.ids, tuple(CONFUSION_MOVES)),
				secondary_chance,
				moves.secondary_self,
				moves.secondary_status,
				moves.secondary_status_choices,
				moves.secondary_boosts,
				moves.secondary_volatile,
				user_species,
				immune,
				tables.types.exponent,
				tables.types.immune,
				tables.natures.multipliers,
				np.asarray(moves.index["struggle"], dtype=np.int32),
				np.asarray(tables.abilities.index["multitype"], dtype=np.int32),
				np.asarray(tables.types.index["fire"], dtype=np.int32),
				np.asarray(tables.types.index["flying"], dtype=np.int32),
				np.asarray(tables.types.index["ground"], dtype=np.int32),
				np.asarray(tables.types.index["normal"], dtype=np.int32),
				weather_moves,
				hazards,
				np.asarray(moves.ids) == "rapidspin",
				weather_accuracy,
				weather_type,
				weather_power,
				weather_heal,
				weather_damage,
				weather_spdef,
				weather_immune,
				weather_status_immune,
				np.asarray(tables.types.index["poison"], dtype=np.int32),
				stall_kind,
				moves.volatile_status,
				moves.self_volatile,
				moves.breaks_protect,
				moves.flags[:, moves.flag_names.index("protect")],
				charge,
				move_turn_volatile_ids,
				charge_semi,
				charge_boosts,
				np.asarray(moves.ids) == "solarbeam",
				np.asarray(moves.ids) == "fakeout",
				target_foe,
				spread,
				hits_semi,
				semi_power,
				weather_power_scale,
				np.asarray(moves.ids) == "magnitude",
				np.isin(np.asarray(moves.ids), ("uturn", "voltswitch")),
				np.isin(
					np.asarray(moves.ids),
					("roar", "whirlwind", "dragontail", "circlethrow"),
				),
				moves.flags[:, moves.flag_names.index("contact")],
				tables.species.nfe,
				tables.species.num,
			),
		),
		item_hooks=build_item_hook_table(tables),
	)
