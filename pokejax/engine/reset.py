"""Host team validation and pure reset to the first turn (Showdown turn 1)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.mechanics.stats import derive_stored_stats
from pokejax.tables import Tables
from pokejax.teams import TeamPool

from .abilities import active_abilities
from .data import EngineTables
from .items import gather_active_items, update_active_items
from .randomness import UNFORCED
from .state import (
	FIRST_TURN,
	MOVE,
	MOVE_SLOTS,
	NUM_BOOSTS,
	NUM_SIDES,
	NUM_STATS,
	NUM_VOLATILES,
	ONGOING,
	TEAM_SLOTS,
	BattleState,
)
from .switching import run_switch_in, switch_handler_first


class Teams(NamedTuple):
	"""One battle's [2, 6, ...] inputs, optionally prefixed by a batch axis."""

	species: jax.Array
	level: jax.Array
	moves: jax.Array
	ability: jax.Array
	item: jax.Array
	nature: jax.Array
	gender: jax.Array
	evs: jax.Array
	ivs: jax.Array
	happiness: jax.Array


def teams_from_pool(pool: TeamPool, side0_index, side1_index) -> Teams:
	"""Select scalar indices or broadcastable index arrays; side follows batch axes."""
	indices = np.stack(np.broadcast_arrays(side0_index, side1_index), axis=-1)
	if not np.issubdtype(indices.dtype, np.integer):
		raise ValueError("Team indices must be integers")
	if np.any((indices < 0) | (indices >= len(pool.team_size))):
		raise ValueError("Team index out of range")
	if np.any(pool.team_size[indices] != TEAM_SLOTS):
		raise ValueError("Expected six members per team")
	return Teams(*(jnp.asarray(getattr(pool, name)[indices]) for name in Teams._fields))


def check_teams(teams: Teams, tables: Tables):
	"""Check shapes, integer ranges and nonempty, trailing-padded moves on the CPU.

	This checks representation, not format legality or engine mechanic support.
	Empty ability/item rows are allowed for stripped differential battles.
	"""
	shape = np.shape(teams.species)
	if len(shape) not in (2, 3) or shape[-2:] != (NUM_SIDES, TEAM_SLOTS):
		raise ValueError("Teams must have shape [2, 6] or [B, 2, 6]")
	ranges = {
		"species": (1, len(tables.species.ids) - 1),
		"level": (1, 100),
		"moves": (0, len(tables.moves.ids) - 1),
		"ability": (0, len(tables.abilities.ids) - 1),
		"item": (0, len(tables.items.ids) - 1),
		"nature": (0, len(tables.natures.ids) - 1),
		"gender": (0, 2),
		"evs": (0, 255),
		"ivs": (0, 31),
		"happiness": (0, 255),
	}
	for name, (minimum, maximum) in ranges.items():
		array = np.asarray(getattr(teams, name))
		trailing = (
			(MOVE_SLOTS,) if name == "moves" else (NUM_STATS,) if name in ("evs", "ivs") else ()
		)
		if array.shape != (*shape, *trailing):
			raise ValueError(f"Incorrect {name} shape")
		if not np.issubdtype(array.dtype, np.integer):
			raise ValueError(f"{name} must contain integers")
		if np.any((array < minimum) | (array > maximum)):
			raise ValueError(f"{name} must be in {minimum}..{maximum}")
	moves = np.asarray(teams.moves)
	if np.any(moves[..., 0] == 0) or np.any((moves[..., :-1] == 0) & (moves[..., 1:] != 0)):
		raise ValueError("Each member needs 1..4 moves with empty slots at the end")


def reset(key, teams: Teams, engine_tables: EngineTables, forced=UNFORCED) -> BattleState:
	"""Derive compact member data and resolve the lead switch-ins."""
	t = engine_tables
	stats = derive_stored_stats(
		t.species_base_stats[teams.species].astype(jnp.int32),
		teams.level.astype(jnp.int32),
		teams.ivs.astype(jnp.int32),
		teams.evs.astype(jnp.int32),
		t.nature_multipliers[teams.nature].astype(jnp.int32),
		t.species_shedinja[teams.species],
	).astype(jnp.int16)
	max_pp = t.move_max_pp[teams.moves]
	initial_types = jnp.where(
		(teams.ability == t.multitype_id)[..., None],
		t.species_types[teams.species],
		t.species_plain_types[teams.species],
	)
	state = BattleState(
		species=teams.species.astype(jnp.int16),
		level=teams.level.astype(jnp.uint8),
		types=initial_types,
		roost_types=initial_types,
		stats=stats,
		hp=stats[..., 0],
		moves=teams.moves.astype(jnp.int16),
		pp=max_pp,
		max_pp=max_pp,
		ability=teams.ability.astype(jnp.int16),
		ability_base=teams.ability.astype(jnp.int16),
		ability_public=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int16),
		item=teams.item.astype(jnp.int16),
		item_consumed=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.bool_),
		item_knocked_off=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.bool_),
		item_revealed=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.bool_),
		item_public=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int16),
		nature=teams.nature.astype(jnp.int8),
		gender=teams.gender.astype(jnp.int8),
		evs=teams.evs.astype(jnp.uint8),
		ivs=teams.ivs.astype(jnp.uint8),
		happiness=teams.happiness.astype(jnp.uint8),
		status=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int8),
		status_counter=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int8),
		sleep_start=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int8),
		sleep_from_foe=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.uint8),
		seen_order=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int8).at[:, 0].set(1),
		revealed_moves=jnp.zeros((NUM_SIDES, TEAM_SLOTS, MOVE_SLOTS), jnp.uint8),
		active=jnp.zeros(NUM_SIDES, jnp.int8),
		active_turns=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.uint8).at[:, 0].set(1),
		boosts=jnp.zeros((NUM_SIDES, NUM_BOOSTS), jnp.int8),
		volatiles=jnp.zeros((NUM_SIDES, NUM_VOLATILES), jnp.int16),
		weather=jnp.asarray(0, jnp.int8),
		weather_turns=jnp.asarray(0, jnp.int8),
		hazards=jnp.zeros((NUM_SIDES, 2), jnp.int8),
		hazard_order=jnp.zeros((NUM_SIDES, 2), jnp.int8),
		request=jnp.full(NUM_SIDES, MOVE, jnp.int8),
		turn_stage=jnp.asarray(0, jnp.int8),
		turn_actions=jnp.zeros(NUM_SIDES, jnp.int8),
		turn_moves=jnp.zeros(NUM_SIDES, jnp.int16),
		turn_pending=jnp.zeros(NUM_SIDES, jnp.bool_),
		turn_first=jnp.asarray(0, jnp.int8),
		turn=jnp.asarray(FIRST_TURN, jnp.int32),
		winner=jnp.asarray(ONGOING, jnp.int8),
	)
	items = gather_active_items(state, t)
	rows = active_abilities(state, t)
	first = switch_handler_first(state, key, forced, items, rows)
	state, _ = run_switch_in(state, jnp.ones(NUM_SIDES, jnp.bool_), first, t, rows=rows)
	return update_active_items(state, True, items, rows)
