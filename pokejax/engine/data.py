"""Device tables built once on the host; compiled engine code reads only arrays."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
	move_max_pp: jax.Array
	move_flags: jax.Array
	move_status: jax.Array
	move_status_self: jax.Array
	move_effect: jax.Array
	move_defrost: jax.Array
	move_thaws_target: jax.Array
	move_checks_immunity: jax.Array
	status_immune: jax.Array
	type_exponent: jax.Array
	type_immune: jax.Array
	nature_multipliers: jax.Array
	struggle_id: jax.Array
	multitype_id: jax.Array
	fire_type: jax.Array


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
				max_pp,
				moves.flags,
				status,
				np.asarray(moves.ids) == "rest",
				effect,
				defrost,
				thaws_target,
				np.asarray(moves.ids) == "thunderwave",
				immune,
				tables.types.exponent,
				tables.types.immune,
				tables.natures.multipliers,
				np.asarray(moves.index["struggle"], dtype=np.int32),
				np.asarray(tables.abilities.index["multitype"], dtype=np.int32),
				np.asarray(tables.types.index["fire"], dtype=np.int32),
			),
		)
	)
