"""Host team validation and pure reset to the first turn (Showdown turn 1)."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.mechanics.stats import derive_stored_stats
from pokejax.tables import Tables
from pokejax.teams import TeamPool

from .abilities import RESIDUAL, RESIDUAL_SLOW_START, active_abilities
from .data import EngineTables
from .events import EVENT_FIELDS
from .forms import foe_view
from .illusion import enter_illusion
from .items import gather_active_items, update_active_items
from .state import (
	FIRST_TURN,
	MOVE,
	MOVE_SLOTS,
	NUM_BOOSTS,
	NUM_SIDES,
	NUM_STATS,
	NUM_VOLATILES,
	ONGOING,
	SLOW_START,
	TEAM_SLOTS,
	BattleState,
)
from .switching import copy_on_entry, run_switch_in, switch_handler_first


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


class PreparedTeams(NamedTuple):
	"""Member inputs that depend only on one team, prepared once per team (prepare_teams)."""

	stats: jax.Array  # int16 [..., 2, 6, 6]: stored HP, Atk, Def, SpA, SpD, Spe
	max_pp: jax.Array  # [..., 2, 6, 4]
	types: jax.Array  # [..., 2, 6, 2]: initial types; a Multitype holder has its species' types
	pirouette_stats: jax.Array  # int16 [..., 2, 6, 5]: Atk..Spe in Meloetta's Pirouette forme


def prepare_teams(teams: Teams, t: EngineTables) -> PreparedTeams:
	"""Derive stats, max PP, initial types and Pirouette stats; batched over leading axes."""
	stats = derive_stored_stats(
		t.species_base_stats[teams.species].astype(jnp.int32),
		teams.level.astype(jnp.int32),
		teams.ivs.astype(jnp.int32),
		teams.evs.astype(jnp.int32),
		t.nature_multipliers[teams.nature].astype(jnp.int32),
		t.species_shedinja[teams.species],
	).astype(jnp.int16)
	max_pp = t.move_max_pp[teams.moves]
	pirouette_stats = derive_stored_stats(
		jnp.broadcast_to(
			t.species_base_stats[t.meloetta_forms[1]].astype(jnp.int32),
			(*stats.shape[:-1], NUM_STATS),
		),
		teams.level.astype(jnp.int32),
		teams.ivs.astype(jnp.int32),
		teams.evs.astype(jnp.int32),
		t.nature_multipliers[teams.nature].astype(jnp.int32),
		False,
	)[..., 1:].astype(jnp.int16)
	types = jnp.where(
		(teams.ability == t.multitype_id)[..., None],
		t.species_types[teams.species],
		t.species_plain_types[teams.species],
	)
	return PreparedTeams(stats, max_pp, types, pirouette_stats)


def reset(
	key, teams: Teams, engine_tables: EngineTables, forced=None, prepared=None
) -> BattleState:
	"""Derive compact member data and resolve the lead switch-ins.

	prepared (from prepare_teams) replaces the per-team derivation; the state is identical.
	"""
	t = engine_tables
	if prepared is None:
		prepared = prepare_teams(teams, t)
	stats, max_pp = prepared.stats, prepared.max_pp
	initial_types, pirouette_stats = prepared.types, prepared.pirouette_stats
	lead_rows = t.ability_hooks[teams.ability[:, 0].astype(jnp.int32)]
	lead_slow_start = lead_rows[:, RESIDUAL] == RESIDUAL_SLOW_START
	initial_volatiles = jnp.zeros((NUM_SIDES, NUM_VOLATILES), jnp.int16)
	initial_volatiles = initial_volatiles.at[:, SLOW_START].set(5 * lead_slow_start)
	state = BattleState(
		species=teams.species.astype(jnp.int16),
		level=teams.level.astype(jnp.uint8),
		types=initial_types,
		base_types=initial_types,
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
		last_item=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int16),
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
		volatiles=initial_volatiles,
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
		turn_damage=jnp.zeros((NUM_SIDES, 3), jnp.int16),
		turn_hit=jnp.zeros(NUM_SIDES, jnp.bool_),
		focus_lost=jnp.zeros(NUM_SIDES, jnp.bool_),
		last_success=jnp.asarray(0, jnp.int16),
		fainted_last=jnp.zeros(NUM_SIDES, jnp.bool_),
		alive_mark=jnp.ones(stats.shape[:2], jnp.bool_),
		last_move=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int16),
		wish_turns=jnp.zeros(NUM_SIDES, jnp.int8),
		wish_hp=jnp.zeros(NUM_SIDES, jnp.int16),
		healing_wish=jnp.zeros(NUM_SIDES, jnp.bool_),
		trick_room=jnp.asarray(0, jnp.int8),
		species_base=teams.species.astype(jnp.int16),
		saved_stats=jnp.zeros((NUM_SIDES, NUM_STATS - 1), jnp.int16),
		saved_moves=jnp.zeros((NUM_SIDES, MOVE_SLOTS), jnp.int16),
		saved_pp=jnp.zeros((NUM_SIDES, MOVE_SLOTS), jnp.uint8),
		saved_max_pp=jnp.zeros((NUM_SIDES, MOVE_SLOTS), jnp.uint8),
		saved_revealed=jnp.zeros((NUM_SIDES, MOVE_SLOTS), jnp.uint8),
		pirouette_stats=pirouette_stats,
		shown_species=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int16),
		party=jnp.broadcast_to(jnp.arange(TEAM_SLOTS, dtype=jnp.int8), (NUM_SIDES, TEAM_SLOTS)),
		event=jnp.zeros((NUM_SIDES, EVENT_FIELDS), jnp.int8),
		event_move=jnp.zeros(NUM_SIDES, jnp.int16),
		weather_age=jnp.asarray(0, jnp.int8),
		trick_room_age=jnp.asarray(0, jnp.int8),
		wish_age=jnp.zeros(NUM_SIDES, jnp.int8),
		hp_seen=stats[..., 0],
		item_gone=jnp.zeros((NUM_SIDES, TEAM_SLOTS), jnp.int8),
		trap_known=jnp.zeros(NUM_SIDES, jnp.bool_),
	)
	both = jnp.ones(NUM_SIDES, jnp.bool_)
	state = enter_illusion(state, both, t)
	# Both leads are active before their switch-in handlers run (the queued runSwitch actions),
	# so each Imposter copies the other lead, and each Trace copies it at once.
	state, _ = copy_on_entry(
		state, both, jnp.zeros((2,), jnp.bool_), t, foe_view(state, state.active[::-1])
	)
	items = gather_active_items(state, t)
	rows = active_abilities(state, t)
	first = switch_handler_first(state, key, forced, items, rows)
	state, _ = run_switch_in(state, both, first, t, rows=rows)
	return update_active_items(state, True, items, rows, tables=t)
