"""Device tables built once on the host; compiled engine code reads only arrays."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.support import CONFUSION_MOVES, SECONDARY_MOVES
from pokejax.tables import Tables

from .abilities import build_ability_hooks
from .items import build_item_hook_table
from .move_data import MOVE_FIELDS, MoveTable, build_move_table
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


class _EngineTables(NamedTuple):
	species_base_stats: jax.Array
	species_types: jax.Array
	species_plain_types: jax.Array
	species_shedinja: jax.Array
	move_max_pp: jax.Array
	move_flags: jax.Array
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
	weather_damage_by_type: jax.Array
	weather_spdef_boost: jax.Array
	weather_residual_immune: jax.Array
	weather_status_immune: jax.Array
	poison_type: jax.Array
	move_turn_volatile_ids: jax.Array
	move_charge_semi: jax.Array
	species_nfe: jax.Array
	species_num: jax.Array
	item_hooks: object
	ability_hooks: jax.Array
	ice_type: jax.Array
	fighting_type: jax.Array
	captivate_id: jax.Array
	anticipation_type: jax.Array
	move_table: MoveTable


class EngineTables(_EngineTables):
	__slots__ = ()

	def __getattr__(self, name):
		if name in MOVE_FIELDS:
			return self.move_table.column(name)
		raise AttributeError(name)

	def _replace(self, **changes):
		# Keep column-based overrides consistent with the packed lookup.
		columns = {name: changes.pop(name) for name in MOVE_FIELDS if name in changes}
		if columns:
			table = changes.get("move_table", self.move_table)
			changes["move_table"] = table.replace_columns(columns)
		return super()._replace(**changes)


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
	# Anticipation asks only whether a base move type can hit super effectively.
	exponents = np.pad(tables.types.exponent, ((0, 0), (1, 0)))
	immunities = np.pad(tables.types.immune, ((0, 0), (1, 0)))
	anticipation_type = (exponents[:, :, None] + exponents[:, None, :]) > 0
	anticipation_type &= ~(immunities[:, :, None] | immunities[:, None, :])
	move_table = build_move_table(
		move_type=moves.type,
		move_category=moves.category,
		move_base_power=moves.base_power,
		move_accuracy=moves.accuracy,
		move_always_hits=moves.always_hits,
		move_priority=moves.priority,
		move_crit_ratio=moves.crit_ratio,
		move_crit_always=moves.crit_always,
		move_drain=moves.drain,
		move_recoil=moves.recoil,
		move_recoil_basehp=moves.recoil_basehp,
		move_heal=moves.heal,
		move_heal_weather=moves.heal_weather,
		move_damage_kind=moves.damage_kind,
		move_fixed_damage=moves.fixed_damage,
		move_hit_min=moves.hit_min,
		move_hit_max=moves.hit_max,
		move_self_faint=moves.self_faint,
		move_crash_damage=moves.crash_damage,
		move_pain_split=moves.pain_split,
		move_roost=moves.roost,
		move_status=status,
		move_status_self=np.asarray(moves.ids) == "rest",
		move_effect=effect,
		move_defrost=defrost,
		move_thaws_target=thaws_target,
		move_checks_immunity=np.asarray(moves.ids) == "thunderwave",
		move_boosts=boosts,
		move_boost_when=when,
		move_boost_target=boost_target,
		move_boost_effect=boost_effect,
		move_confusion=moves.confusion & np.isin(moves.ids, tuple(CONFUSION_MOVES)),
		move_secondary_chance=secondary_chance,
		move_secondary_self=moves.secondary_self,
		move_secondary_status=moves.secondary_status,
		move_secondary_status_choices=moves.secondary_status_choices,
		move_secondary_boosts=moves.secondary_boosts,
		move_secondary_volatile=moves.secondary_volatile,
		move_secondary_user_species=user_species,
		move_weather=weather_moves,
		move_hazard=hazards,
		move_rapid_spin=np.asarray(moves.ids) == "rapidspin",
		move_weather_accuracy=weather_accuracy,
		move_weather_type=weather_type,
		move_weather_power=weather_power,
		move_weather_heal=weather_heal,
		move_stall_kind=stall_kind,
		move_volatile_status=moves.volatile_status,
		move_self_volatile=moves.self_volatile,
		move_breaks_protect=moves.breaks_protect,
		move_protects=moves.flags[:, moves.flag_names.index("protect")],
		move_charge=charge,
		move_charge_boosts=charge_boosts,
		move_charge_sun_skip=np.asarray(moves.ids) == "solarbeam",
		move_fakeout=np.asarray(moves.ids) == "fakeout",
		move_target_foe=target_foe,
		move_spread=spread,
		move_hits_semi=hits_semi,
		move_semi_power=semi_power,
		move_weather_power_scale=weather_power_scale,
		move_magnitude=np.asarray(moves.ids) == "magnitude",
		move_self_switch=np.isin(np.asarray(moves.ids), ("uturn", "voltswitch")),
		move_force_switch=np.isin(
			np.asarray(moves.ids),
			("roar", "whirlwind", "dragontail", "circlethrow"),
		),
		move_contact=moves.flags[:, moves.flag_names.index("contact")],
		move_punch=moves.flags[:, moves.flag_names.index("punch")],
		move_sheer_force=np.any(moves.secondary_chance > 0, axis=1),
	)
	return EngineTables(
		species_base_stats=jnp.asarray(tables.species.base_stats),
		species_types=jnp.asarray(tables.species.types),
		species_plain_types=jnp.asarray(plain_types),
		species_shedinja=jnp.asarray(np.asarray(tables.species.ids) == "shedinja"),
		move_max_pp=jnp.asarray(max_pp),
		move_flags=jnp.asarray(moves.flags),
		status_immune=jnp.asarray(immune),
		type_exponent=jnp.asarray(tables.types.exponent),
		type_immune=jnp.asarray(tables.types.immune),
		nature_multipliers=jnp.asarray(tables.natures.multipliers),
		struggle_id=jnp.asarray(np.asarray(moves.index["struggle"], dtype=np.int32)),
		multitype_id=jnp.asarray(np.asarray(tables.abilities.index["multitype"], dtype=np.int32)),
		fire_type=jnp.asarray(np.asarray(tables.types.index["fire"], dtype=np.int32)),
		flying_type=jnp.asarray(np.asarray(tables.types.index["flying"], dtype=np.int32)),
		ground_type=jnp.asarray(np.asarray(tables.types.index["ground"], dtype=np.int32)),
		normal_type=jnp.asarray(np.asarray(tables.types.index["normal"], dtype=np.int32)),
		weather_damage_by_type=jnp.asarray(weather_damage),
		weather_spdef_boost=jnp.asarray(weather_spdef),
		weather_residual_immune=jnp.asarray(weather_immune),
		weather_status_immune=jnp.asarray(weather_status_immune),
		poison_type=jnp.asarray(np.asarray(tables.types.index["poison"], dtype=np.int32)),
		move_turn_volatile_ids=jnp.asarray(move_turn_volatile_ids),
		move_charge_semi=jnp.asarray(charge_semi),
		species_nfe=jnp.asarray(tables.species.nfe),
		species_num=jnp.asarray(tables.species.num),
		item_hooks=build_item_hook_table(tables),
		ability_hooks=build_ability_hooks(tables),
		ice_type=jnp.asarray(tables.types.index["ice"], jnp.int32),
		fighting_type=jnp.asarray(tables.types.index["fighting"], jnp.int32),
		captivate_id=jnp.asarray(moves.index["captivate"], jnp.int32),
		anticipation_type=jnp.asarray(anticipation_type),
		move_table=move_table,
	)
