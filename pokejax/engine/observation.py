"""Policy input at a decision point, for one battle and one perspective.

Own members retain fixed team slots for actions.bench_slots. Opponent members
are packed in first-appearance order, followed by zero rows with unseen=1.
Revealed opponent moves are sorted by descending ID with trailing zeros;
even the owner's move-slot positions are private. No private slot indices escape.

IDs are int16, with 0 reserved for padding (type IDs are engine IDs + 1).
All features are float32: level / 100, stored stats / 1000, boosts / 6,
turn / 1000; HP and PP are fractions, flags and requests are 0/1.
Opponent types describe the visible species, independent of hidden abilities;
later type-changing effects need explicit public type tracking.

Pinned Showdown config/formats.ts:4609-4613 enables HP Percentage Mod for
gen5randombattle; data/rulesets.ts:1336-1340 sets reportPercentages.
sim/pokemon.ts:2096-2102 uses ceil(100*hp/maxhp), capped at 99 if damaged;
:2082 reports zero on faint. We expose that integer percentage divided by 100.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .actions import legal_actions
from .state import (
	CONFUSION,
	LOCKED_MOVE,
	MUST_RECHARGE,
	NONE,
	ONGOING,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	TEAM_SLOTS,
	TURN_IDLE,
	TWO_TURN_MOVE,
)


class OwnObservation(NamedTuple):
	species: jax.Array  # int16 [6]
	types: jax.Array  # int16 [6, 2]
	moves: jax.Array  # int16 [6, 4]
	level: jax.Array  # float32 [6]
	pp: jax.Array  # float32 [6, 4]
	hp: jax.Array  # float32 [6]
	status: jax.Array  # int16 [6], major status ID (0 means healthy)
	fainted: jax.Array  # float32 [6]
	is_active: jax.Array  # float32 [6]
	stats: jax.Array  # float32 [6, 6], hp/atk/def/spa/spd/spe
	boosts: jax.Array  # float32 [7], atk/def/spa/spd/spe/accuracy/evasion
	confused: jax.Array  # float32 scalar; duration is private
	charging: jax.Array  # float32 scalar; public two-turn state, no duration
	recharging: jax.Array  # float32 scalar
	locked: jax.Array  # float32 scalar; public locked-move state
	trapped: jax.Array  # float32 scalar; source must still be active


class OpponentObservation(NamedTuple):
	species: jax.Array  # int16 [6]
	types: jax.Array  # int16 [6, 2]
	moves: jax.Array  # int16 [6, 4], revealed IDs descending, then zeros
	level: jax.Array  # float32 [6]
	hp: jax.Array  # float32 [6], protocol percentage / 100
	status: jax.Array  # int16 [6], zero for unseen members
	fainted: jax.Array  # float32 [6]
	is_active: jax.Array  # float32 [6]
	unseen: jax.Array  # float32 [6]
	boosts: jax.Array  # float32 [7], active only
	confused: jax.Array  # float32 scalar; duration is private
	charging: jax.Array  # float32 scalar; public two-turn state, no duration
	recharging: jax.Array  # float32 scalar
	locked: jax.Array  # float32 scalar; public locked-move state
	trapped: jax.Array  # float32 scalar; source must still be active


class Observation(NamedTuple):
	own: OwnObservation
	opponent: OpponentObservation
	turn: jax.Array  # float32 scalar
	request: jax.Array  # float32 [3], NONE/MOVE/SWITCH
	mid_turn: jax.Array  # float32 scalar; saved turn work remains after this choice
	legal_actions: jax.Array  # bool [9], actions.py encoding
	weather: jax.Array  # int16 scalar; public WEATHER_NAMES ID
	weather_turns: jax.Array  # float32 scalar; zero for permanent/clear
	hazards: jax.Array  # float32 [own/opponent, Spikes/Toxic Spikes] layer counts


def observe(state, side, engine_tables) -> Observation:
	"""Pure, branch-free single-perspective observation; jit/vmap externally."""
	foe = 1 - side

	def f(x):
		return x.astype(jnp.float32)

	def public_turn_state(member_side):
		source_side = 1 - member_side
		source_slot = state.volatiles[member_side, PARTIAL_TRAP_SOURCE].astype(jnp.int32) - 1
		source_active = source_slot == state.active[source_side]
		source_alive = state.hp[source_side, jnp.clip(source_slot, 0, TEAM_SLOTS - 1)] > 0
		trapped = (
			(state.volatiles[member_side, PARTIALLY_TRAPPED] > 0) & source_active & source_alive
		)
		return (
			f(state.volatiles[member_side, TWO_TURN_MOVE] > 0),
			f(state.volatiles[member_side, MUST_RECHARGE] > 0),
			f(state.volatiles[member_side, LOCKED_MOVE] > 0),
			f(trapped),
		)

	own_turn_state = public_turn_state(side)
	foe_turn_state = public_turn_state(foe)

	own = OwnObservation(
		state.species[side],
		state.types[side].astype(jnp.int16) + 1,
		state.moves[side],
		f(state.level[side]) / 100,
		f(state.pp[side]) / jnp.maximum(f(state.max_pp[side]), 1),
		f(state.hp[side]) / jnp.maximum(f(state.stats[side, :, 0]), 1),
		state.status[side].astype(jnp.int16),
		f(state.hp[side] == 0),
		f(jnp.arange(TEAM_SLOTS) == state.active[side]),
		f(state.stats[side]) / 1000,
		f(state.boosts[side]) / 6,
		f(state.volatiles[side, CONFUSION] > 0),
		*own_turn_state,
	)
	# A public rank selects one private slot without exposing the slot itself.
	ranks = jnp.arange(1, TEAM_SLOTS + 1)
	match = state.seen_order[foe, :, None] == ranks[None, :]
	slots = jnp.argmax(match, axis=0)
	seen = jnp.any(match, axis=0)
	species = jnp.where(seen, state.species[foe, slots], 0)
	hp = state.hp[foe, slots].astype(jnp.int32)
	maximum = jnp.maximum(state.stats[foe, slots, 0].astype(jnp.int32), 1)
	percent = (100 * hp + maximum - 1) // maximum
	percent = jnp.where(hp < maximum, jnp.minimum(percent, 99), percent)
	moves = jnp.where(state.revealed_moves[foe, slots] != 0, state.moves[foe, slots], 0)
	# Sorting by -id puts padding last; reverse sign restores descending IDs.
	moves = -jnp.sort(-moves, axis=-1)
	opponent = OpponentObservation(
		species,
		jnp.where(seen[:, None], engine_tables.species_types[species].astype(jnp.int16) + 1, 0),
		jnp.where(seen[:, None], moves, 0),
		jnp.where(seen, f(state.level[foe, slots]) / 100, 0),
		jnp.where(seen, f(percent) / 100, 0),
		jnp.where(seen, state.status[foe, slots], 0).astype(jnp.int16),
		f(seen & (hp == 0)),
		f(seen & (slots == state.active[foe])),
		f(~seen),
		f(state.boosts[foe]) / 6,
		f(state.volatiles[foe, CONFUSION] > 0),
		*foe_turn_state,
	)
	request = jnp.where(state.winner == ONGOING, state.request[side], NONE)
	return Observation(
		own,
		opponent,
		f(state.turn) / 1000,
		f(jnp.arange(3) == request),
		f(state.turn_stage > TURN_IDLE),
		legal_actions(state, engine_tables)[side],
		state.weather.astype(jnp.int16),
		f(state.weather_turns),
		f(state.hazards[jnp.stack((side, foe))]),
	)
