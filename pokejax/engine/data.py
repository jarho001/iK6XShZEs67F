"""Device tables built once on the host; compiled engine code reads only arrays."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.tables import Tables


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
	type_exponent: jax.Array
	type_immune: jax.Array
	nature_multipliers: jax.Array
	struggle_id: jax.Array
	multitype_id: jax.Array


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
				tables.types.exponent,
				tables.types.immune,
				tables.natures.multipliers,
				np.asarray(moves.index["struggle"], dtype=np.int32),
				np.asarray(tables.abilities.index["multitype"], dtype=np.int32),
			),
		)
	)
