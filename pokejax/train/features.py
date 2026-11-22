"""Public damage and matchup features for the policy network (network.features in the configuration).

They are computed from the Observation of one learner and the static tables every player has (the
move table, the type chart and the species base stats), with the formulas of the max-damage baseline
(pokejax.baselines: move score, boosts, the defence estimate) and of its matchup estimate, which
heuristics shares. The features read the observation only, so a feature changes only when the
observation does.

For a batch of observations with leading axes [...]:

- move features [..., MOVE_COUNT, MOVE_FEATURES]: one row per move slot of the active member:
  power / 100, accuracy, expected hits / 3, physical, status, STAB, effectiveness / 2 against the
  opponent's current types, immune, legal, PP fraction, the max-damage score relative to the best
  legal move, and log(1 + score) / 5. Hidden Power counts as power 70 of type Normal (its type is
  not in the observation).
- switch features [..., BENCH_COUNT, SWITCH_FEATURES]: one row per switch action, in action order:
  legal, the member's HP fraction, its offence / 2 and defence / 2 (poke-env's estimate), the speed
  tier, the matchup / 4, the matchup minus the best legal matchup, over 4, and the threat / 2.
- threat [...]: the largest damage estimate of the opponent's revealed moves against the active
  member, as the damage of power x STAB x effectiveness divided by THREAT_SCALE.
- the switch features carry the threat of each bench member.
"""

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import (
	HP_FRACTION_COEFFICIENT,
	M_ACCURACY,
	M_HIDDEN_POWER,
	M_HITS,
	M_PHYSICAL,
	M_POWER,
	M_TYPE,
	SPEED_TIER_COEFFICIENT,
	boost_factor,
	build_baseline_tables,
	stat_estimation,
)
from pokejax.engine.state import ATK, BOOST_ATK, BOOST_DEF, BOOST_SPA, BOOST_SPD, DEF, SPA, SPD, SPE
from pokejax.tables import CATEGORY_INDEX, Tables, load_tables

MOVE_COUNT = 4
BENCH_COUNT = 5
TEAM_SLOTS = 6
MOVE_FEATURES = 12
SWITCH_FEATURES = 8
# Stand-in for the power of Hidden Power, as in pokejax.baselines.
HIDDEN_POWER = 70.0
# Damage estimate scale of the threat feature (power x STAB x effectiveness).
THREAT_SCALE = 200.0
STAB_MULTIPLIER = 1.5


class PublicTables(NamedTuple):
	"""Static tables as numpy arrays: the type chart and the move and species columns."""

	chart: np.ndarray  # float32 [T + 1, T + 1]: attacking x defending; index T is no type
	move_type: np.ndarray  # int32 [M]: engine type index (T for none)
	move_power: np.ndarray  # float32 [M]
	move_accuracy: np.ndarray  # float32 [M]: always-hit moves 1.0
	move_hits: np.ndarray  # float32 [M]: expected hits
	move_physical: np.ndarray  # float32 [M]
	move_status: np.ndarray  # float32 [M]
	species_base: np.ndarray  # float32 [S, 6]: hp, atk, def, spa, spd, spe


def build_public_tables(host: Tables) -> PublicTables:
	"""The public tables of pokejax.tables, with the columns the baselines use."""
	baseline = build_baseline_tables(host)
	rows = np.asarray(baseline.moves, np.float32)
	power = rows[:, M_POWER].copy()
	power[rows[:, M_HIDDEN_POWER] > 0] = HIDDEN_POWER
	return PublicTables(
		chart=np.asarray(baseline.chart, np.float32),
		move_type=rows[:, M_TYPE].astype(np.int32),
		move_power=power,
		move_accuracy=rows[:, M_ACCURACY].copy(),
		move_hits=rows[:, M_HITS].copy(),
		move_physical=rows[:, M_PHYSICAL].copy(),
		move_status=(np.asarray(host.moves.category) == CATEGORY_INDEX["Status"]).astype(
			np.float32
		),
		species_base=np.asarray(host.species.base_stats, np.float32),
	)


@functools.cache
def public_tables() -> PublicTables:
	"""The public tables of the default data file (built once per process, outside any trace)."""
	with jax.ensure_compile_time_eval():
		return build_public_tables(load_tables())


def _pick(x, onehot):
	"""The member of x [lead..., 6, rest...] that the one-hot [lead..., 6] selects."""
	weights = onehot.reshape(onehot.shape + (1,) * (x.ndim - onehot.ndim))
	return jnp.sum(x * weights, axis=onehot.ndim - 1)


def _gather(x, slots):
	"""Team members x [lead..., 6, rest...] at the slots [lead..., 5]: [lead..., 5, rest...]."""
	lead = slots.shape[:-1]
	rest = x.shape[len(lead) + 1 :]
	index = jnp.broadcast_to(
		slots.reshape(lead + (BENCH_COUNT,) + (1,) * len(rest)), lead + (BENCH_COUNT,) + rest
	)
	return jnp.take_along_axis(x, index, axis=len(lead))


def _type_index(code, none):
	"""An observation type code (0 none, else engine ID + 1) as an engine type index (none: T)."""
	return jnp.where(code > 0, code - 1, none)


def action_features(obs, tables: PublicTables):
	"""(move features [..., 4, 12], switch features [..., 5, 8], threat of the active [...])."""
	own, foe = obs.own, obs.opponent
	f32, i32 = jnp.float32, jnp.int32
	chart = jnp.asarray(tables.chart)
	none = chart.shape[0] - 1
	move_type_table = jnp.asarray(tables.move_type)
	species_base = jnp.asarray(tables.species_base)
	own_is_active = own.is_active.astype(f32)
	foe_is_active = foe.is_active.astype(f32)

	def pick(x, onehot):
		return jnp.round(_pick(x.astype(f32), onehot)).astype(i32)

	def pick_float(x, onehot):
		return _pick(x.astype(f32), onehot)

	# The active members and their public numbers.
	active = jnp.argmax(own.is_active, axis=-1).astype(i32)
	own_types_all = _type_index(own.types.astype(i32), none)  # [..., 6, 2]
	own_types = pick(own_types_all, own_is_active)  # [..., 2]
	foe_types_all = _type_index(foe.types.astype(i32), none)
	foe_types = pick(foe_types_all, foe_is_active)  # [..., 2]
	foe_species = pick(foe.species, foe_is_active)  # [...]
	foe_base = species_base[foe_species]  # [..., 6]
	own_attack = pick_float(own.stats[..., ATK], own_is_active) * 1000.0
	own_special = pick_float(own.stats[..., SPA], own_is_active) * 1000.0
	own_boost = jnp.round(own.boosts * 6.0)  # [..., 7]
	foe_boost = jnp.round(foe.boosts * 6.0)
	foe_defence = stat_estimation(foe_base[..., DEF], foe_boost[..., BOOST_DEF])
	foe_spirit = stat_estimation(foe_base[..., SPD], foe_boost[..., BOOST_SPD])
	attack = own_attack * boost_factor(own_boost[..., BOOST_ATK])
	special = own_special * boost_factor(own_boost[..., BOOST_SPA])

	# Move slots of the active member.
	moves = pick(own.moves, own_is_active)  # [..., 4]
	valid = (moves > 0).astype(f32)
	move_type = move_type_table[moves]
	power = jnp.asarray(tables.move_power)[moves]
	accuracy = jnp.asarray(tables.move_accuracy)[moves]
	hits = jnp.asarray(tables.move_hits)[moves]
	physical = jnp.asarray(tables.move_physical)[moves]
	status = jnp.asarray(tables.move_status)[moves]
	stab = jnp.any(
		(move_type[..., :, None] == own_types[..., None, :]) & (own_types[..., None, :] != none),
		axis=-1,
	)
	effectiveness = chart[move_type, foe_types[..., 0:1]] * chart[move_type, foe_types[..., 1:2]]
	ratio = jnp.where(
		physical > 0,
		attack[..., None] / foe_defence[..., None],
		special[..., None] / foe_spirit[..., None],
	)
	stab_factor = jnp.where(stab, STAB_MULTIPLIER, 1.0)
	score = valid * power * stab_factor * effectiveness * accuracy * hits * ratio
	legal_moves = obs.legal_actions[..., :MOVE_COUNT]
	best = jnp.max(jnp.where(legal_moves, score, 0.0), axis=-1, keepdims=True)
	relative = jnp.where(best > 0, score / jnp.maximum(best, 1e-6), 0.0)
	pp = pick_float(own.pp, own_is_active)
	move_features = jnp.stack(
		[
			power / 100.0 * valid,
			accuracy * valid,
			hits / 3.0 * valid,
			physical * valid,
			status * valid,
			stab.astype(f32) * valid,
			effectiveness / 2.0 * valid,
			(effectiveness == 0).astype(f32) * valid,
			legal_moves.astype(f32),
			pp,
			relative,
			jnp.log1p(score) / 5.0,
		],
		axis=-1,
	)

	# Threat: the opponent's revealed moves against every own member (per team slot).
	foe_moves = pick(foe.moves, foe_is_active)  # [..., 4]
	revealed = foe_moves > 0
	foe_move_type = move_type_table[foe_moves]
	foe_power = jnp.asarray(tables.move_power)[foe_moves]
	foe_stab = jnp.any(
		(foe_move_type[..., :, None] == foe_types[..., None, :])
		& (foe_types[..., None, :] != none),
		axis=-1,
	)
	against0 = own_types_all[..., None, :, 0]  # [..., 1, 6]
	against1 = own_types_all[..., None, :, 1]
	foe_effect = (
		chart[foe_move_type[..., :, None], against0] * chart[foe_move_type[..., :, None], against1]
	)  # [..., 4, 6]
	foe_damage = jnp.where(
		revealed[..., :, None],
		foe_power[..., :, None]
		* jnp.where(foe_stab, STAB_MULTIPLIER, 1.0)[..., :, None]
		* foe_effect,
		0.0,
	)
	threat_by_slot = foe_damage.max(axis=-2) / THREAT_SCALE  # [..., 6]
	threat_active = _pick(threat_by_slot, own_is_active)

	# Switch actions: the bench members in action order (bench_slots of the engine).
	choices = jnp.arange(BENCH_COUNT, dtype=i32)
	slots = choices + (choices >= active[..., None]).astype(i32)  # [..., 5]
	member_types = _gather(own_types_all, slots)  # [..., 5, 2]
	member_species = _gather(own.species.astype(i32), slots)  # [..., 5]
	member_hp = _gather(own.hp.astype(f32), slots)  # [..., 5]
	legal_switch = obs.legal_actions[..., MOVE_COUNT:]
	foe0 = foe_types[..., 0, None, None]
	foe1 = foe_types[..., 1, None, None]
	offence_each = chart[member_types, foe0] * chart[member_types, foe1]
	offence = jnp.where(member_types != none, offence_each, 0.0).max(axis=-1)
	defence_foe = foe_types[..., None, :]  # [..., 1, 2]
	defence_each = (
		chart[defence_foe, member_types[..., 0:1]] * chart[defence_foe, member_types[..., 1:2]]
	)  # [..., 5, 2]
	defence = jnp.where(defence_foe != none, defence_each, 0.0).max(axis=-1)
	own_speed = species_base[member_species][..., SPE]
	foe_speed = foe_base[..., SPE][..., None]
	speed_term = SPEED_TIER_COEFFICIENT * (own_speed > foe_speed).astype(f32) - (
		SPEED_TIER_COEFFICIENT * (foe_speed > own_speed).astype(f32)
	)
	foe_hp = pick_float(foe.hp, foe_is_active)[..., None]
	matchup = offence - defence + speed_term + HP_FRACTION_COEFFICIENT * (member_hp - foe_hp)
	any_switch = legal_switch.any(axis=-1)[..., None]
	best_switch = jnp.max(jnp.where(legal_switch, matchup, -1e9), axis=-1, keepdims=True)
	relative_switch = jnp.where(any_switch, matchup - best_switch, 0.0)
	threat_switch = _gather(threat_by_slot, slots)
	switch_features = jnp.stack(
		[
			legal_switch.astype(f32),
			member_hp,
			offence / 2.0,
			defence / 2.0,
			speed_term,
			matchup / 4.0,
			relative_switch / 4.0,
			threat_switch,
		],
		axis=-1,
	)
	return move_features, switch_features, threat_active
