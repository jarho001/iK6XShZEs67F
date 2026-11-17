"""Device tables built once on the host; compiled engine code reads only arrays."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.support import CONFUSION_MOVES, SECONDARY_MOVES
from pokejax.tables import Tables

from .state import BURN, FREEZE, PARALYSIS, POISON, SLEEP, TOXIC
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
	normal_type: jax.Array


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
				np.asarray(tables.types.index["normal"], dtype=np.int32),
			),
		)
	)
