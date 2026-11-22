"""Four baseline policies as JAX functions of the battle state: (state, side, key) -> action.

Every policy returns an action in pokejax.engine.actions encoding, picked from the engine's true legal
mask (hidden traps included), so none is ever refused or replaced. They are branch-free (no cond or
switch, only gathers and elementwise selects) and vmap over a batch of battles. The optional fourth
argument of a call is that mask ([9] bool); Env.engine_mask(es) has it already, so a rollout passes it
in instead of making each policy compute it again.

- random: uniform over the legal actions (what OpponentEnv has always used).
- random-move: uniform over the legal moves; a uniform legal switch only when no move is legal (a
  forced replacement).
- max-damage: scores each legal move as
  base power x STAB x type effectiveness x accuracy x expected hits x attack / defence,
  picks the highest (ties to the lowest move slot) and switches only when no move is legal, to the
  bench member with the best poke-env matchup estimate.
- heuristics: poke-env's SimpleHeuristicsPlayer.choose_singles_move (poke-env commit bab3673,
  src/poke_env/player/baselines.py, Gen 5 singles, no Dynamax or Terastallization).

What they read is what the player knows (the facts that pokejax.engine.observation shows;
tests/test_baselines.py checks that they are the same): the own team in full (species, types, HP,
stats, moves, boosts, Hidden Power's type) and of the opponent only its shown species, current
types, protocol HP percentage, boosts, number of members seen fainted and side conditions.

Interpretations of poke-env (reference/baselines.cjs makes the same ones):

- Move data is the table row of pokejax.tables: base power, type, category, accuracy (always-hit
  moves 1.0) and expected hits as Move.expected_hits (a fixed count; Triple Kick 1 + 2 x 0.9 +
  3 x 0.81; 2-5 hit moves (2 + 3) / 3 + (4 + 5) / 6). Hidden Power is the exception: the Gen 5 table
  row has power 0 and type Normal, while poke-env (and a player who reads the request) knows the
  typed move, power 70, so it counts as power 70 of the type the own IVs give.
- opponent.damage_multiplier(move) and the matchup's type multipliers are type-chart products over
  the defender's current types; an immunity gives 0. Roost's Flying removal is not shown, so it is
  not used.
- _stat_estimation's boost factor is (2 + b) / 2 above +1 and 2 / (2 - b) otherwise, so +1 counts as
  2.0 (poke-env's own quirk, kept). Base stats are those of the current species (a Transformed
  member has the copied species). Max-damage uses the same factor for the user's own stat, which
  is its real stat from the request.
- Python compares float64 sums, whose ties are exact; here scores are float32 and two scores count
  as tied when they differ by at most TIE_TOLERANCE (relative above 1.0), because mathematically
  equal sums differ in the last bit under either arithmetic. The first of the tied candidates wins,
  in request order: moves by slot, switches in Showdown's side.pokemon order, which
  BattleState.party tracks (the bench members in party order, whatever their team slot).
  Thresholds (matchup > 0, matchup < -2) use the same tolerance.
- available_switches are the engine's legal switch actions, so a hidden trap (Shadow Tag, Arena
  Trap, Magnet Pull; poke-env cannot know) removes them, and a forced replacement offers the
  living members that are not on the field. At a forced replacement the fainted (or pivoting)
  member is still poke-env's active Pokemon (its request entry says active), so there is no
  random fallback; the matchup is against the opponent's active member, fainted or not.
- n_remaining_mons counts the own members with HP left, n_opp_remaining_mons is 6 minus the
  opponent members seen fainted. battle.side_conditions is truthy when the own side has Spikes or
  Toxic Spikes (the only side conditions that occur in the pool; Stealth Rock and Sticky Web do
  not, so their branch cannot be reached). The anti-hazard moves are Rapid Spin and Defog.
- The setup branch tests move.target == "self", but Move.target is a Target enum member, which
  never equals the string: at the pinned commit the branch never runs. It is ported but off
  (SETUP_BRANCH); heuristics_action(..., setup_branch=True) runs it (boosts summing to at least 2
  on the user, whose lowest raised stat is below +6, at full HP and a positive matchup).
- A random fallback (poke-env picks a random order when it has no Pokemon or no option) never
  occurs: both actives always exist and a side with no choice has the single no-op action.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import MOVE_ACTIONS, NUM_ACTIONS, legal_actions
from pokejax.engine.data import EngineTables
from pokejax.engine.events import protocol_percent
from pokejax.engine.state import (
	ATK,
	BOOST_ATK,
	BOOST_DEF,
	BOOST_SPA,
	BOOST_SPD,
	DEF,
	ILLUSION_SLOT,
	NUM_BOOSTS,
	ROOST,
	SPA,
	SPD,
	SPE,
	TEAM_SLOTS,
	TRANSFORMED,
	BattleState,
)
from pokejax.engine.variable_moves import hidden_power_for
from pokejax.policy_names import POLICY_NAMES
from pokejax.tables import Tables, load_tables

# poke-env SimpleHeuristicsPlayer constants.
SPEED_TIER_COEFFICIENT = 0.1
HP_FRACTION_COEFFICIENT = 0.4
SWITCH_OUT_MATCHUP_THRESHOLD = -2.0
SETUP_BRANCH = False  # see the module docstring: poke-env's own condition is never true
HIDDEN_POWER_BASE_POWER = 70
TIE_TOLERANCE = 1e-5
# Move.expected_hits.
TRIPLE_KICK_HITS = 1 + 2 * 0.9 + 3 * 0.81
MULTI_HIT_EXPECTED = (2 + 3) / 3 + (4 + 5) / 6
ANTI_HAZARD_MOVES = ("rapidspin", "defog")
# The hazard a move sets: 1 Spikes, 2 Toxic Spikes (hazard layers are indexed from 0).
HAZARD_MOVES = {"spikes": 1, "toxicspikes": 2}

# Columns of BaselineTables.moves; the last NUM_BOOSTS columns flag the stats the move raises.
(
	M_TYPE,
	M_POWER,
	M_ACCURACY,
	M_HITS,
	M_PHYSICAL,
	M_HAZARD,
	M_ANTI_HAZARD,
	M_HIDDEN_POWER,
	M_SETUP,
	M_RAISES,
) = range(10)
MOVE_COLUMNS = M_RAISES + NUM_BOOSTS


class BaselineTables(NamedTuple):
	"""What the baselines read besides the engine tables, built once on the host."""

	chart: jax.Array  # float32 [T + 1, T + 1]: attacker x defender multiplier; index T is "no type"
	moves: jax.Array  # float32 [M, MOVE_COLUMNS], see the M_ columns


def build_baseline_tables(tables: Tables) -> BaselineTables:
	"""Type multipliers and per-move scoring columns from the host tables."""
	types, moves = tables.types, tables.moves
	count = len(types.ids)
	multiplier = np.where(types.immune, 0.0, 2.0 ** types.exponent.astype(np.float64))
	chart = np.ones((count + 1, count + 1), np.float32)
	chart[:count, :count] = multiplier
	rows = np.zeros((len(moves.ids), MOVE_COLUMNS), np.float32)
	rows[:, M_TYPE] = moves.type
	rows[:, M_POWER] = moves.base_power
	rows[:, M_ACCURACY] = np.where(moves.always_hits, 1.0, moves.accuracy / 100.0)
	expected = np.where(moves.hit_min == moves.hit_max, moves.hit_min, MULTI_HIT_EXPECTED)
	expected[moves.hit_min == 0] = 1.0  # the empty row
	for name in ("triplekick", "tripleaxel"):
		if name in moves.index:
			expected[moves.index[name]] = TRIPLE_KICK_HITS
	rows[:, M_HITS] = expected
	rows[:, M_PHYSICAL] = moves.category == 0
	for name, kind in HAZARD_MOVES.items():
		rows[moves.index[name], M_HAZARD] = kind
	for name in ANTI_HAZARD_MOVES:
		if name in moves.index:
			rows[moves.index[name], M_ANTI_HAZARD] = 1
	rows[moves.index["hiddenpower"], M_HIDDEN_POWER] = 1
	# poke-env's setup test: boosts that sum to at least 2 on a move that targets the user.
	targets_self = np.asarray(moves.target_names)[moves.target] == "self"
	boosts = moves.boosts.astype(np.int32)
	rows[:, M_SETUP] = targets_self & (boosts.sum(axis=1) >= 2)
	rows[:, M_RAISES : M_RAISES + NUM_BOOSTS] = boosts > 0
	return BaselineTables(jnp.asarray(chart), jnp.asarray(rows))


class Facts(NamedTuple):
	"""What a policy knows at a decision point, for one battle and one side."""

	active: jax.Array  # int32: own active team slot
	own_types: jax.Array  # int32 [6, 2]: current types per team slot (-1 none)
	own_species: jax.Array  # int32 [6]
	own_fraction: jax.Array  # float32 [6]: HP / max HP
	own_hp_full: jax.Array  # bool: the active is at full HP
	own_attack: jax.Array  # float32: the active's Atk (Transform undone, as the request has it)
	own_special_attack: jax.Array  # float32
	own_boosts: jax.Array  # int32 [7]
	own_remaining: jax.Array  # int32: members with HP left
	own_hazards: jax.Array  # bool: Spikes or Toxic Spikes on the own side
	foe_species: jax.Array  # int32: the species shown for the opponent's active
	foe_types: jax.Array  # int32 [2]: its current types (-1 none)
	foe_fraction: jax.Array  # float32: protocol percentage / 100
	foe_boosts: jax.Array  # int32 [7]
	foe_remaining: jax.Array  # int32: 6 minus the members seen fainted
	foe_hazards: jax.Array  # int32 [2]: Spikes, Toxic Spikes layers on the opponent's side


def public_facts(state: BattleState, side, tables: EngineTables) -> Facts:
	"""The facts of observation.observe that the baselines use, read from the state."""
	foe = 1 - side
	active = state.active[side].astype(jnp.int32)
	foe_active = state.active[foe].astype(jnp.int32)
	slots = jnp.arange(TEAM_SLOTS)
	# Own types: the current ones, Roost's Flying removal undone (it is not announced).
	roosting = (state.volatiles[side, ROOST] > 0) & (slots == active)
	own_types = jnp.where(roosting[:, None], state.roost_types[side], state.types[side])
	# A Transformed member has its own stats in the request (the saved ones), not the copied ones.
	reverted = state.volatiles[side, TRANSFORMED] > 0
	stats = jnp.where(reverted, state.saved_stats[side], state.stats[side, active, 1:])
	max_hp = state.stats[side, :, 0]
	hp = state.hp[side]
	# The opponent's active: shown species (an Illusion disguise stays), current types (the
	# disguise's own while it holds) and protocol HP percentage.
	foe_species = state.shown_species[foe, foe_active].astype(jnp.int32)
	foe_live = jnp.where(
		state.volatiles[foe, ROOST] > 0,
		state.roost_types[foe, foe_active],
		state.types[foe, foe_active],
	)
	disguised = state.volatiles[foe, ILLUSION_SLOT] > 0
	foe_types = jnp.where(disguised, tables.species_types[foe_species], foe_live)
	foe_percent = protocol_percent(state.hp[foe, foe_active], state.stats[foe, foe_active, 0])
	# Members seen fainted: an off-field member shows the HP it had when it left.
	public_hp = jnp.where(slots == foe_active, state.hp[foe], state.hp_seen[foe])
	seen_fainted = (state.seen_order[foe] > 0) & (public_hp == 0)
	return Facts(
		active=active,
		own_types=own_types.astype(jnp.int32),
		own_species=state.species[side].astype(jnp.int32),
		own_fraction=hp.astype(jnp.float32) / jnp.maximum(max_hp, 1).astype(jnp.float32),
		own_hp_full=hp[active] == max_hp[active],
		own_attack=stats[ATK - 1].astype(jnp.float32),
		own_special_attack=stats[SPA - 1].astype(jnp.float32),
		own_boosts=state.boosts[side].astype(jnp.int32),
		own_remaining=jnp.sum(hp > 0).astype(jnp.int32),
		own_hazards=jnp.any(state.hazards[side] > 0),
		foe_species=foe_species,
		foe_types=foe_types.astype(jnp.int32),
		foe_fraction=foe_percent.astype(jnp.float32) / 100.0,
		foe_boosts=state.boosts[foe].astype(jnp.int32),
		foe_remaining=(TEAM_SLOTS - jnp.sum(seen_fainted)).astype(jnp.int32),
		foe_hazards=state.hazards[foe].astype(jnp.int32),
	)


# Scoring helpers.


def first_max(scores, valid, tolerance=TIE_TOLERANCE):
	"""Index of the first valid score within the tolerance of the best one (0 if none is valid)."""
	masked = jnp.where(valid, scores, -jnp.inf)
	best = jnp.max(masked)
	near = valid & (masked >= best - tolerance * jnp.maximum(1.0, jnp.abs(best)))
	return jnp.argmax(near).astype(jnp.int32)


def boost_factor(stage):
	"""The factor of poke-env's _stat_estimation: +1 counts as 2.0, as it does there."""
	stage = stage.astype(jnp.float32)
	return jnp.where(stage > 1, (2.0 + stage) / 2.0, 2.0 / (2.0 - stage))


def stat_estimation(base, stage):
	"""poke-env's _stat_estimation: ((2 x base + 31) + 5) x the boost factor."""
	return ((2.0 * base.astype(jnp.float32) + 31.0) + 5.0) * boost_factor(stage)


def _type_index(chart, types):
	"""Map -1 (no type) to the chart's neutral last index."""
	return jnp.where(types < 0, chart.shape[0] - 1, types)


def type_multiplier(chart, attack, defending):
	"""Attack type [...] against a defender's two types [2] (-1 none): the product, 0 if immune."""
	index = _type_index(chart, defending)
	return chart[attack, index[0]] * chart[attack, index[1]]


def estimate_matchups(facts: Facts, data: BaselineTables, tables: EngineTables):
	"""poke-env _estimate_matchup of every own team slot [6] against the opponent's active."""
	chart = data.chart
	none = chart.shape[0] - 1
	foe = _type_index(chart, facts.foe_types)
	own = _type_index(chart, facts.own_types)
	# What each own type does to the opponent, and what each opponent type does to each own member.
	to_foe = chart[:, foe[0]] * chart[:, foe[1]]
	offence = jnp.max(jnp.where(own == none, 0.0, to_foe[own]), axis=1)
	from_foe = chart[foe][:, own[:, 0]] * chart[foe][:, own[:, 1]]
	defence = jnp.max(jnp.where((foe == none)[:, None], 0.0, from_foe), axis=0)
	speed = tables.species_base_stats[facts.own_species, SPE].astype(jnp.int32)
	foe_speed = tables.species_base_stats[facts.foe_species, SPE].astype(jnp.int32)
	score = offence - defence
	score += SPEED_TIER_COEFFICIENT * (speed > foe_speed)
	score -= SPEED_TIER_COEFFICIENT * (foe_speed > speed)
	score += facts.own_fraction * HP_FRACTION_COEFFICIENT
	return score - facts.foe_fraction * HP_FRACTION_COEFFICIENT


class MoveFacts(NamedTuple):
	"""The four move slots of the active member, as the policies score them."""

	rows: jax.Array  # float32 [4, MOVE_COLUMNS]
	power: jax.Array  # float32 [4]: Hidden Power counts as HIDDEN_POWER_BASE_POWER
	stab: jax.Array  # float32 [4]: 1.5 when the move's type is one of the user's current types
	effectiveness: jax.Array  # float32 [4]: against the opponent's current types
	physical: jax.Array  # bool [4]


def move_facts(state, side, facts: Facts, data: BaselineTables, tables: EngineTables) -> MoveFacts:
	rows = data.moves[state.moves[side, facts.active].astype(jnp.int32)]
	hidden = rows[:, M_HIDDEN_POWER] > 0
	# Hidden Power's type is the own IVs'; every other move has its table type.
	move_type = jnp.where(
		hidden, hidden_power_for(state, side, tables)[0], rows[:, M_TYPE].astype(jnp.int32)
	)
	power = jnp.where(hidden, float(HIDDEN_POWER_BASE_POWER), rows[:, M_POWER])
	stab = jnp.any(move_type[:, None] == facts.own_types[facts.active][None, :], axis=1)
	return MoveFacts(
		rows,
		power,
		jnp.where(stab, 1.5, 1.0),
		type_multiplier(data.chart, move_type, facts.foe_types),
		rows[:, M_PHYSICAL] > 0,
	)


def _foe_defence(facts: Facts, tables: EngineTables):
	"""The estimated Def and SpD of the opponent's active, from its shown species and boosts."""
	base = tables.species_base_stats[facts.foe_species]
	return (
		stat_estimation(base[DEF], facts.foe_boosts[BOOST_DEF]),
		stat_estimation(base[SPD], facts.foe_boosts[BOOST_SPD]),
	)


def max_damage_scores(state, side, facts: Facts, data, tables) -> jax.Array:
	"""base power x STAB x effectiveness x accuracy x hits x (attack with boost / foe's Def or SpD)."""
	moves = move_facts(state, side, facts, data, tables)
	foe_def, foe_spd = _foe_defence(facts, tables)
	attack = facts.own_attack * boost_factor(facts.own_boosts[BOOST_ATK])
	special = facts.own_special_attack * boost_factor(facts.own_boosts[BOOST_SPA])
	ratio = jnp.where(moves.physical, attack / foe_def, special / foe_spd)
	score = moves.power * moves.stab * moves.effectiveness
	return score * moves.rows[:, M_ACCURACY] * moves.rows[:, M_HITS] * ratio


def heuristics_damage_scores(state, side, facts: Facts, data, tables) -> jax.Array:
	"""poke-env's damage score of each move slot (estimated stats on both sides)."""
	moves = move_facts(state, side, facts, data, tables)
	foe_def, foe_spd = _foe_defence(facts, tables)
	base = tables.species_base_stats[facts.own_species[facts.active]]
	attack = stat_estimation(base[ATK], facts.own_boosts[BOOST_ATK])
	special = stat_estimation(base[SPA], facts.own_boosts[BOOST_SPA])
	ratio = jnp.where(moves.physical, attack / foe_def, special / foe_spd)
	score = moves.power * moves.stab * ratio
	return score * moves.rows[:, M_ACCURACY] * moves.rows[:, M_HITS] * moves.effectiveness


def bench_candidates(state, side, facts: Facts, mask):
	"""The bench in request order: team slots [5], their switch actions [5], whether each is legal.

	Showdown lists side.pokemon with the active first and a switch-in swapping positions;
	BattleState.party is that order, so index 0 is the active member.
	"""
	slots = state.party[side, 1:].astype(jnp.int32)
	actions = MOVE_ACTIONS + slots - (slots > facts.active)
	return slots, actions, mask[actions]


def best_switch(matchups, slots, actions, legal):
	"""The legal bench member with the best matchup (first in request order on a tie)."""
	return actions[first_max(matchups[slots], legal)]


class RandomPolicy(NamedTuple):
	"""Uniform over the engine's legal actions."""

	tables: EngineTables

	def __call__(self, state, side, key, mask=None):
		mask = legal_actions(state, self.tables)[side] if mask is None else mask
		return jax.random.categorical(key, jnp.where(mask, 0.0, -jnp.inf))


class RandomMovePolicy(NamedTuple):
	"""Uniform over the legal moves; over the legal switches only when no move is legal."""

	tables: EngineTables

	def __call__(self, state, side, key, mask=None):
		mask = legal_actions(state, self.tables)[side] if mask is None else mask
		is_move = jnp.arange(NUM_ACTIONS) < MOVE_ACTIONS
		pool = jnp.where(jnp.any(mask & is_move), mask & is_move, mask)
		return jax.random.categorical(key, jnp.where(pool, 0.0, -jnp.inf))


def max_damage_action(state, side, mask, tables, data) -> jax.Array:
	facts = public_facts(state, side, tables)
	moves_legal = mask[:MOVE_ACTIONS]
	move = first_max(max_damage_scores(state, side, facts, data, tables), moves_legal)
	slots, actions, legal = bench_candidates(state, side, facts, mask)
	switch = best_switch(estimate_matchups(facts, data, tables), slots, actions, legal)
	return jnp.where(jnp.any(moves_legal), move, jnp.where(jnp.any(legal), switch, 0)).astype(
		jnp.int32
	)


class MaxDamagePolicy(NamedTuple):
	"""The module docstring's max-damage rule."""

	tables: EngineTables
	data: BaselineTables

	def __call__(self, state, side, key, mask=None):
		mask = legal_actions(state, self.tables)[side] if mask is None else mask
		return max_damage_action(state, side, mask, self.tables, self.data)


def heuristics_action(state, side, mask, tables, data, setup_branch=SETUP_BRANCH) -> jax.Array:
	"""poke-env SimpleHeuristicsPlayer.choose_singles_move, see the module docstring."""
	facts = public_facts(state, side, tables)
	matchups = estimate_matchups(facts, data, tables)
	slots, switch_actions, switch_legal = bench_candidates(state, side, facts, mask)
	moves_legal = mask[:MOVE_ACTIONS]
	any_switch = jnp.any(switch_legal)

	# _should_switch_out: a decent switch-in and a good reason to leave.
	boosts = facts.own_boosts
	reason = (
		(boosts[BOOST_DEF] <= -3)
		| (boosts[BOOST_SPD] <= -3)
		| ((boosts[BOOST_ATK] <= -3) & (facts.own_attack >= facts.own_special_attack))
		| ((boosts[BOOST_SPA] <= -3) & (facts.own_attack <= facts.own_special_attack))
		| (matchups[facts.active] < SWITCH_OUT_MATCHUP_THRESHOLD - TIE_TOLERANCE)
	)
	decent = jnp.any(switch_legal & (matchups[slots] > TIE_TOLERANCE))
	use_moves = jnp.any(moves_legal) & (~(decent & reason) | ~any_switch)

	# The move branch: entry hazards and their removal, setup moves, then the damage score.
	rows = data.moves[state.moves[side, facts.active].astype(jnp.int32)]
	hazard = rows[:, M_HAZARD].astype(jnp.int32)
	foe_has = facts.foe_hazards[jnp.clip(hazard - 1, 0, 1)] > 0
	set_hazard = (facts.foe_remaining >= 3) & (hazard > 0) & ~foe_has
	remove_hazard = facts.own_hazards & (rows[:, M_ANTI_HAZARD] > 0) & (facts.own_remaining >= 2)
	hazard_move = moves_legal & (set_hazard | remove_hazard)
	damage = first_max(heuristics_damage_scores(state, side, facts, data, tables), moves_legal)
	chosen = jnp.where(jnp.any(hazard_move), jnp.argmax(hazard_move), damage)
	if setup_branch:
		raises = rows[:, M_RAISES : M_RAISES + NUM_BOOSTS] > 0
		lowest = jnp.min(jnp.where(raises, boosts[None, :], 99), axis=1)
		ready = facts.own_hp_full & (matchups[facts.active] > TIE_TOLERANCE)
		setup = moves_legal & (rows[:, M_SETUP] > 0) & (lowest < 6) & ready
		chosen = jnp.where(
			jnp.any(hazard_move), chosen, jnp.where(jnp.any(setup), jnp.argmax(setup), chosen)
		)
	switch = best_switch(matchups, slots, switch_actions, switch_legal)
	return jnp.where(use_moves, chosen, jnp.where(any_switch, switch, 0)).astype(jnp.int32)


class HeuristicsPolicy(NamedTuple):
	"""poke-env's SimpleHeuristicsPlayer."""

	tables: EngineTables
	data: BaselineTables

	def __call__(self, state, side, key, mask=None):
		mask = legal_actions(state, self.tables)[side] if mask is None else mask
		return heuristics_action(state, side, mask, self.tables, self.data)


def make_policy(name: str, tables: EngineTables, host_tables: Tables | None = None):
	"""A policy by name (POLICY_NAMES) for these engine tables.

	host_tables are the pokejax.tables the engine tables were built from (loaded when omitted);
	random and random-move need none.
	"""
	if name == "random":
		return RandomPolicy(tables)
	if name == "random-move":
		return RandomMovePolicy(tables)
	if name not in POLICY_NAMES:
		raise ValueError(f"Unknown policy {name!r}; choose one of {POLICY_NAMES}")
	data = build_baseline_tables(load_tables() if host_tables is None else host_tables)
	return (MaxDamagePolicy if name == "max-damage" else HeuristicsPolicy)(tables, data)
