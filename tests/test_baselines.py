"""The four baseline policies: rules on constructed states, legality and privacy on recorded battles,
and exact agreement with the Node policies (reference/baselines.cjs) on recorded decision points.
"""

import copy
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax import baselines as B
from pokejax.baselines import (
	M_ACCURACY,
	M_HIDDEN_POWER,
	M_HITS,
	M_SETUP,
	POLICY_NAMES,
	HeuristicsPolicy,
	MaxDamagePolicy,
	RandomMovePolicy,
	RandomPolicy,
	build_baseline_tables,
	heuristics_action,
	make_policy,
	max_damage_action,
	public_facts,
)
from pokejax.baselines_diff import COMPARED_POLICIES, compare_file
from pokejax.diff import load_recording, replay_histories
from pokejax.engine import state as S
from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.tables import load_tables

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/policy-pool.jsonl.gz"


def test_fixture_agrees_with_the_node_policies():
	"""Max-damage and heuristics choose the same action in JAX and in Node at every decision point."""
	result = compare_file(FIXTURE)
	assert result.battles == 40
	assert result.points > 10000
	assert not result.mismatches, result.report()
	assert result.fallbacks == {name: 0 for name in COMPARED_POLICIES}


# The 300-battle recording in work/diff/ (git-ignored) runs only with --large.
@pytest.mark.large
def test_large_recording():
	path = ROOT / "work/diff/policy-pool.jsonl.gz"
	if not path.exists():
		pytest.skip(f"Regenerate {path} with reference/record-battles.cjs --policies")
	result = compare_file(path)
	assert result.battles == 300
	assert not result.mismatches, result.report()
	assert result.fallbacks == {name: 0 for name in COMPARED_POLICIES}


def test_a_changed_node_choice_is_reported_where_it_is():
	header, battles = load_recording(FIXTURE)
	header = dict(header, count=1)
	battle = copy.deepcopy(battles[3])
	step = battle["steps"][5]
	step["policies"]["heuristics"][1] = (step["policies"]["heuristics"][1] + 1) % 9
	result = compare_file(FIXTURE, recording=(header, [battle]))
	assert [(m.step, m.side, m.policy) for m in result.mismatches] == [(5, 1, "heuristics")]
	assert result.by_policy["heuristics"] == 1


@pytest.fixture(scope="module")
def world():
	"""Engine tables and every decision-point state of the recorded battles, flattened."""
	tables = load_tables()
	engine = build_engine_tables(tables)
	header, battles = load_recording(FIXTURE)
	history, lengths = replay_histories(header, battles, tables, project=lambda states, t: states)
	valid = np.arange(lengths.max() + 1)[None, :] <= lengths[:, None]
	states = jax.tree.map(lambda x: jnp.asarray(x[valid]), history)
	return tables, engine, build_baseline_tables(tables), states, header, battles


def test_every_policy_picks_a_legal_action_at_every_recorded_state(world):
	tables, engine, _, states, *_ = world
	count = states.winner.shape[0]
	assert count == 2649
	keys = jax.random.split(jax.random.key(1), count)

	@jax.jit
	def choose(states, keys):
		mask = jax.vmap(legal_actions, in_axes=(0, None))(states, engine)
		out = []
		for name in POLICY_NAMES:
			policy = make_policy(name, engine, tables)
			out.append(
				jnp.stack(
					[
						jax.vmap(lambda s, k, m, p=policy, side=side: p(s, side, k, m[side]))(
							states, keys, mask
						)
						for side in (0, 1)
					],
					axis=1,
				)
			)
		return mask, jnp.stack(out)

	mask, actions = jax.device_get(choose(states, keys))
	for index, name in enumerate(POLICY_NAMES):
		legal = np.take_along_axis(mask, actions[index][..., None], axis=-1)[..., 0]
		assert legal.all(), f"{name} chose an illegal action"
	# The deterministic ones moved at least once per battle in each kind.
	moves, switches = (actions[3] < 4).sum(), (actions[3] >= 4).sum()
	assert moves > 0 and switches > 0


def test_policies_called_without_a_mask_use_the_engine_mask(world):
	tables, engine, _, states, *_ = world
	state = jax.tree.map(lambda x: x[40], states)
	mask = legal_actions(state, engine)
	key = jax.random.key(4)
	for side in (0, 1):
		for name in POLICY_NAMES:
			policy = make_policy(name, engine, tables)
			call = jax.jit(
				lambda s, side, k, m, policy=policy: policy(s, side, k, m), static_argnums=1
			)
			assert call(state, side, key, None) == call(state, side, key, mask[side])


def test_policy_objects_run_the_documented_functions(world):
	_, engine, data, states, *_ = world

	@jax.jit
	def both(states):
		mask = jax.vmap(legal_actions, in_axes=(0, None))(states, engine)
		out = []
		for side in (0, 1):
			objects = (
				jax.vmap(
					lambda s, m, side=side: MaxDamagePolicy(engine, data)(s, side, None, m[side])
				)(states, mask),
				jax.vmap(
					lambda s, m, side=side: HeuristicsPolicy(engine, data)(s, side, None, m[side])
				)(states, mask),
			)
			functions = (
				jax.vmap(lambda s, m, side=side: max_damage_action(s, side, m[side], engine, data))(
					states, mask
				),
				jax.vmap(lambda s, m, side=side: heuristics_action(s, side, m[side], engine, data))(
					states, mask
				),
			)
			out.append(jnp.stack([jnp.all(a == b) for a, b in zip(objects, functions)]))
		return jnp.stack(out)

	assert np.asarray(both(states)).all()


# What the policies know is the observation's facts.


def test_the_facts_are_those_of_the_observation(world):
	"""public_facts reads the state; every fact equals the matching leaf of observe()."""
	_, engine, _, states, *_ = world

	@jax.jit
	def both(states):
		def one(state, side):
			return public_facts(state, side, engine), observe(state, side, engine)

		return [jax.vmap(lambda s, side=side: one(s, side))(states) for side in (0, 1)]

	for facts, obs in jax.device_get(both(states)):
		own, foe = obs.own, obs.opponent
		active = np.argmax(own.is_active, axis=-1)
		foe_row = np.argmax(foe.is_active, axis=-1)
		rows = np.arange(len(active))
		np.testing.assert_array_equal(facts.active, active)
		np.testing.assert_array_equal(facts.own_species, own.species)
		np.testing.assert_array_equal(facts.own_types + 1, own.types)
		np.testing.assert_allclose(facts.own_fraction, own.hp, atol=1e-6)
		# The observation divides in float32 (on the GPU 300 / 300 can be one ulp short of 1).
		np.testing.assert_array_equal(facts.own_hp_full, own.hp[rows, active] >= 1 - 1e-6)
		np.testing.assert_allclose(facts.own_attack, own.stats[rows, active, 1] * 1000, atol=1e-3)
		np.testing.assert_allclose(
			facts.own_special_attack, own.stats[rows, active, 3] * 1000, atol=1e-3
		)
		np.testing.assert_allclose(facts.own_boosts, own.boosts * 6, atol=1e-5)
		np.testing.assert_array_equal(facts.own_remaining, (own.fainted == 0).sum(-1))
		np.testing.assert_array_equal(facts.own_hazards, (obs.hazards[:, 0] > 0).any(-1))
		np.testing.assert_array_equal(facts.foe_species, foe.species[rows, foe_row])
		np.testing.assert_array_equal(facts.foe_types + 1, foe.types[rows, foe_row])
		np.testing.assert_allclose(facts.foe_fraction, foe.hp[rows, foe_row], atol=1e-6)
		np.testing.assert_allclose(facts.foe_boosts, foe.boosts * 6, atol=1e-5)
		np.testing.assert_array_equal(facts.foe_remaining, 6 - foe.fainted.sum(-1))
		np.testing.assert_array_equal(facts.foe_hazards, obs.hazards[:, 1])


def hide_private_data(foe):
	"""A function that changes everything about the opponent that its player cannot see."""

	def change(s):
		unseen = s.seen_order[foe] == 0
		updates = {}
		for name in (
			"pp", "max_pp", "evs", "ivs", "ability", "ability_base", "item", "nature", "happiness",
			"gender",
		):  # fmt: skip
			x = getattr(s, name)
			updates[name] = x.at[foe].set(jnp.zeros_like(x[foe]))
		# Every stored stat changes, public HP does not; a benched member's true HP is private.
		updates.update(
			stats=s.stats.at[foe].set(s.stats[foe] * 2),
			hp=s.hp.at[foe].set(s.hp[foe] * 2),
			hp_seen=s.hp_seen.at[foe].set(s.hp_seen[foe] * 2),
		)
		for name in ("species", "level", "types", "status"):
			x = getattr(s, name)
			mask = unseen.reshape((6,) + (1,) * (x.ndim - 2))
			updates[name] = x.at[foe].set(jnp.where(mask, jnp.ones_like(x[foe]), x[foe]))
		updates["moves"] = s.moves.at[foe].set(
			jnp.where(s.revealed_moves[foe] != 0, s.moves[foe], 1)
		)
		# Counters keep their presence and lose their value.
		volatiles = s.volatiles
		for column in (S.ENCORE_REMAINING, S.TAUNT, S.DISABLE_REMAINING, S.YAWN, S.SUBSTITUTE):
			present = volatiles[foe, column] > 0
			volatiles = volatiles.at[foe, column].set(
				jnp.where(present, 3, 0).astype(volatiles.dtype)
			)
		updates["volatiles"] = volatiles
		return s._replace(**updates)

	return change


@pytest.mark.parametrize("side", [0, 1])
def test_the_policies_do_not_read_hidden_information(world, side):
	"""Two states that differ only in what the opponent's player never showed give the same
	facts and the same deterministic choices."""
	_, engine, data, states, *_ = world
	foe = 1 - side
	changed = jax.jit(jax.vmap(hide_private_data(foe)))(states)

	# The legal mask is the engine's (a hidden trap on the foe forbids switches); it is an input.
	mask = jax.vmap(legal_actions, in_axes=(0, None))(states, engine)

	@jax.jit
	def read(states):
		facts = jax.vmap(lambda s: public_facts(s, side, engine))(states)
		choices = jnp.stack(
			[
				jax.vmap(lambda s, m: max_damage_action(s, side, m[side], engine, data))(
					states, mask
				),
				jax.vmap(lambda s, m: heuristics_action(s, side, m[side], engine, data))(
					states, mask
				),
			]
		)
		return facts, choices

	facts, choices = jax.device_get(read(states))
	other_facts, other_choices = jax.device_get(read(changed))
	for name, x, y in zip(facts._fields, facts, other_facts, strict=True):
		np.testing.assert_array_equal(x, y, err_msg=name)
	np.testing.assert_array_equal(choices, other_choices)
	# The test has teeth: what is public does change the facts.
	shown = jax.jit(jax.vmap(lambda s: s._replace(hp=s.hp.at[foe].set(s.hp[foe] // 2))))(states)
	assert not np.array_equal(facts.foe_fraction, jax.device_get(read(shown))[0].foe_fraction)
	assert np.any(np.asarray(states.seen_order[:, foe]) == 0)


# The tables.


def test_baseline_tables_follow_poke_env_moves(world):
	tables, _, data, *_ = world
	rows = np.asarray(data.moves)
	column = lambda name, c: rows[tables.moves.index[name], c]
	assert column("tackle", M_HITS) == 1
	assert column("doublekick", M_HITS) == 2
	assert column("furyattack", M_HITS) == pytest.approx((2 + 3) / 3 + (4 + 5) / 6)
	assert column("triplekick", M_HITS) == pytest.approx(1 + 2 * 0.9 + 3 * 0.81)
	assert column("swift", M_ACCURACY) == 1  # accuracy true
	assert column("stoneedge", M_ACCURACY) == pytest.approx(0.8)
	assert column("hiddenpower", M_HIDDEN_POWER) == 1 and column("tackle", M_HIDDEN_POWER) == 0
	# Boosts that sum to at least 2 on a move that targets the user (poke-env's setup test).
	setup = {name for name in tables.moves.ids if name and column(name, M_SETUP)}
	assert {
		"swordsdance",
		"nastyplot",
		"shellsmash",
		"calmmind",
		"dragondance",
		"quiverdance",
	} <= setup
	assert not setup & {"growl", "toxic", "tackle", "swagger", "bellydrum", "curse"}


def test_unknown_policy_names_are_refused(world):
	_, engine, *_ = world
	with pytest.raises(ValueError, match="Unknown policy"):
		make_policy("best", engine)
	assert isinstance(make_policy("random", engine), RandomPolicy)
	assert isinstance(make_policy("random-move", engine), RandomMovePolicy)


# Rules on constructed states. Every case starts from the first decision point of a recorded battle
# (both leads active, nothing has happened) and changes the fields that the rule reads.

# Stats of the constructed members: HP 300, every other stat 200, so attack and defence ratios are
# those of the base stats alone.
STATS = [300, 200, 200, 200, 200, 200]


@pytest.fixture(scope="module")
def start(world):
	"""The first state, without abilities or items: nothing traps, locks or reveals anything."""
	state = jax.tree.map(lambda x: x[0], world[3])
	return state._replace(
		ability=jnp.zeros_like(state.ability),
		ability_base=jnp.zeros_like(state.ability_base),
		item=jnp.zeros_like(state.item),
	)


@pytest.fixture(scope="module")
def act(world):
	"""act(state, side, policy, mask=None, setup=False) -> the policy's action as an int."""
	_, engine, data, *_ = world

	@partial(jax.jit, static_argnames=("name", "setup"))
	def run(state, side, mask, name, setup):
		if mask is None:
			mask = legal_actions(state, engine)[side]
		if name == "max-damage":
			return max_damage_action(state, side, mask, engine, data)
		return heuristics_action(state, side, mask, engine, data, setup)

	return lambda state, side, name="heuristics", mask=None, setup=False: int(
		run(state, side, mask, name, setup)
	)


def set_moves(state, tables, side, names, slot=None):
	"""The active member (or slot) knows exactly these moves, with PP."""
	slot = int(state.active[side]) if slot is None else slot
	moves = jnp.asarray([tables.moves.index[n] for n in names] + [0] * (4 - len(names)))
	known = moves > 0
	return state._replace(
		moves=state.moves.at[side, slot].set(moves.astype(state.moves.dtype)),
		pp=state.pp.at[side, slot].set(jnp.where(known, 10, 0).astype(state.pp.dtype)),
		max_pp=state.max_pp.at[side, slot].set(jnp.where(known, 10, 0).astype(state.max_pp.dtype)),
	)


def set_species(state, tables, engine, side, slot, name):
	"""A member of this species (shown as such to the opponent) at full HP."""
	species = tables.species.index[name]
	types = engine.species_types[species]
	return state._replace(
		species=state.species.at[side, slot].set(species),
		species_base=state.species_base.at[side, slot].set(species),
		shown_species=state.shown_species.at[side, slot].set(species),
		types=state.types.at[side, slot].set(types),
		base_types=state.base_types.at[side, slot].set(types),
		roost_types=state.roost_types.at[side, slot].set(types),
		stats=state.stats.at[side, slot].set(jnp.asarray(STATS, state.stats.dtype)),
		hp=state.hp.at[side, slot].set(STATS[0]),
		hp_seen=state.hp_seen.at[side, slot].set(STATS[0]),
		status=state.status.at[side, slot].set(0),
	)


def team(state, tables, engine, side, *names):
	"""Slots 0.. of a side are these species (the rest keep theirs); every member at full HP."""
	for slot, name in enumerate(names):
		state = set_species(state, tables, engine, side, slot, name)
	return state


def duel(start, tables, engine, mine, foe):
	return team(team(start, tables, engine, 0, mine), tables, engine, 1, foe)


def foes_fainted(state, foe, count):
	"""The first count bench members of the foe have been seen and have fainted."""
	slots = jnp.arange(1, count + 1)
	return state._replace(
		seen_order=state.seen_order.at[foe, slots].set((slots + 1).astype(state.seen_order.dtype)),
		hp=state.hp.at[foe, slots].set(0),
		hp_seen=state.hp_seen.at[foe, slots].set(0),
	)


def test_stat_estimation_and_boost_factor_are_poke_envs():
	base = jnp.asarray(100)
	for stage, wanted in ((0, 236), (2, 472), (1, 472), (3, 590), (-1, 157.3333), (-6, 59)):
		assert float(B.stat_estimation(base, jnp.asarray(stage))) == pytest.approx(wanted, rel=1e-5)
	# poke-env's own unit test: a base Atk of 84 at -1 is 136; +1 counts as +2 there.
	assert float(B.stat_estimation(jnp.asarray(84), jnp.asarray(-1))) == pytest.approx(136)
	assert float(B.boost_factor(jnp.asarray(1))) == 2.0
	assert float(B.boost_factor(jnp.asarray(6))) == 4.0


def test_ties_within_the_tolerance_go_to_the_first_candidate():
	valid = jnp.asarray([True, True, True, True])
	assert int(B.first_max(jnp.asarray([1.0, 3.0, 3.0, 2.0]), valid)) == 1
	assert int(B.first_max(jnp.asarray([1.0, 3.0 - 1e-6, 3.0, 2.0]), valid)) == 1
	assert int(B.first_max(jnp.asarray([1.0, 3.0 - 1e-3, 3.0, 2.0]), valid)) == 2
	assert int(B.first_max(jnp.asarray([5.0, 3.0, 3.0, 2.0]), valid.at[0].set(False))) == 1
	assert int(B.first_max(jnp.asarray([1e4, 1e4 + 0.05, 5.0, 2.0]), valid)) == 0  # relative
	assert int(B.first_max(jnp.asarray([0.0, 0.0, 0.0, 0.0]), valid)) == 0
	assert int(B.first_max(jnp.asarray([1.0, 2.0]), jnp.asarray([False, False]))) == 0


def test_the_matchup_estimate_is_poke_envs(world, start):
	tables, engine, data, *_ = world
	s = team(
		start, tables, engine, 0, "gengar", "snorlax", "gyarados", "snorlax", "snorlax", "snorlax"
	)
	s = team(s, tables, engine, 1, "machamp")
	matchups = B.estimate_matchups(public_facts(s, 0, engine), data, engine)
	# Offence minus defence, a tenth for speed, 0.4 per HP fraction (equal here).
	assert float(matchups[0]) == pytest.approx(
		1.0 - 0.0 + 0.1, abs=1e-5
	)  # Gengar: immune to Fighting
	assert float(matchups[1]) == pytest.approx(1.0 - 2.0 - 0.1, abs=1e-5)  # Snorlax
	assert float(matchups[2]) == pytest.approx(2.0 - 0.5 + 0.1, abs=1e-5)  # Gyarados: Flying hits
	same = B.estimate_matchups(
		public_facts(duel(start, tables, engine, "snorlax", "snorlax"), 0, engine), data, engine
	)
	assert float(same[0]) == pytest.approx(0, abs=1e-6)
	hurt = s._replace(hp=s.hp.at[0, 1].set(150))
	assert float(
		B.estimate_matchups(public_facts(hurt, 0, engine), data, engine)[1]
	) == pytest.approx(-1.1 - 0.2, abs=1e-5)
	foe_hurt = s._replace(hp=s.hp.at[1, 0].set(150))
	assert float(
		B.estimate_matchups(public_facts(foe_hurt, 0, engine), data, engine)[0]
	) == pytest.approx(1.1 + 0.2, abs=1e-5)


def test_max_damage_picks_the_best_score_and_ties_go_to_the_lowest_slot(world, start, act):
	tables, engine, *_ = world
	s = duel(start, tables, engine, "charizard", "venusaur")
	s = set_moves(s, tables, 0, ["tackle", "earthquake", "airslash", "flamethrower"])
	assert act(s, 0, "max-damage") == 3 and act(s, 0) == 3
	# A status move scores zero.
	s2 = set_moves(s, tables, 0, ["splash", "swordsdance", "tackle", "toxic"])
	assert act(s2, 0, "max-damage") == 2
	# Thunderbolt and Ice Beam: equal power, accuracy and category, no STAB, neutral on Snorlax.
	tie = set_moves(
		duel(start, tables, engine, "starmie", "snorlax"),
		tables,
		0,
		["splash", "thunderbolt", "icebeam"],
	)
	assert act(tie, 0, "max-damage") == 1 and act(tie, 0) == 1
	# Stone Edge (100 at 80%) and Iron Head (80 at 100%) are 80 either way, but not in float.
	for names in (["splash", "stoneedge", "ironhead"], ["splash", "ironhead", "stoneedge"]):
		noise = set_moves(duel(start, tables, engine, "mew", "snorlax"), tables, 0, names)
		assert act(noise, 0, "max-damage") == 1 and act(noise, 0) == 1
	# A move without PP is not a candidate.
	spent = tie._replace(pp=tie.pp.at[0, 0, 1].set(0))
	assert act(spent, 0, "max-damage") == 2


def test_max_damage_uses_stab_effectiveness_accuracy_hits_and_the_stats(world, start, act):
	tables, engine, *_ = world
	# Hits: Double Kick (30 x 2) against Tackle (40 x 1), both neutral on Pikachu.
	s = set_moves(
		duel(start, tables, engine, "mew", "pikachu"), tables, 0, ["tackle", "doublekick"]
	)
	assert act(s, 0, "max-damage") == 1
	# Accuracy: Hydro Pump (120 at 80% = 96) beats Surf (95) and Blizzard (120 at 70% = 84).
	s = set_moves(
		duel(start, tables, engine, "mew", "snorlax"), tables, 0, ["blizzard", "surf", "hydropump"]
	)
	assert act(s, 0, "max-damage") == 2
	# STAB: Flamethrower (95 x 1.5) beats Thunderbolt (95) on a neutral foe.
	s = set_moves(
		duel(start, tables, engine, "charizard", "snorlax"),
		tables,
		0,
		["thunderbolt", "flamethrower"],
	)
	assert act(s, 0, "max-damage") == 1
	# An immunity scores zero: Earthquake cannot hit a Flying type, Tackle can.
	s = set_moves(
		duel(start, tables, engine, "charizard", "pidgeot"), tables, 0, ["earthquake", "tackle"]
	)
	assert act(s, 0, "max-damage") == 1
	# Stats: Body Slam (85, physical) against Hyper Voice (90, special) on a foe with equal
	# defences: the higher power. A +3 Special Defence on the foe turns it to the physical move.
	s = set_moves(duel(start, tables, engine, "mew", "mew"), tables, 0, ["bodyslam", "hypervoice"])
	assert act(s, 0, "max-damage") == 1
	assert act(s._replace(boosts=s.boosts.at[1, S.BOOST_SPD].set(3)), 0, "max-damage") == 0
	# The user's own boosts count with poke-env's factor: +1 on Attack doubles it, so a -3 Attack
	# (x 0.4) sends the choice to the special move, and a +1 Attack to the physical one.
	assert act(s._replace(boosts=s.boosts.at[0, S.BOOST_ATK].set(-3)), 0, "max-damage") == 1
	assert act(s._replace(boosts=s.boosts.at[0, S.BOOST_ATK].set(1)), 0, "max-damage") == 0


def test_heuristics_sets_hazards_while_the_foe_has_none(world, start, act):
	tables, engine, *_ = world
	s = set_moves(duel(start, tables, engine, "forretress", "snorlax"), tables, 0,
		["gyroball", "spikes", "toxicspikes", "explosion"])  # fmt: skip
	assert act(s, 0) == 1
	foe_spikes = s._replace(hazards=s.hazards.at[1, 0].set(1))
	assert act(foe_spikes, 0) == 2
	both = foe_spikes._replace(hazards=foe_spikes.hazards.at[1, 1].set(1))
	assert act(both, 0) == 3  # Explosion, the strongest attack, as no hazard is left to set
	# Layers of Spikes already there: no more (poke-env sets a hazard only once).
	assert act(s._replace(hazards=s.hazards.at[1, 0].set(3)), 0) == 2
	# Two members left on the foe's side: the hazards are not worth it; three: they are.
	assert act(foes_fainted(s, 1, 4), 0) == 3
	assert act(foes_fainted(s, 1, 3), 0) == 1
	# max-damage never sets them.
	assert act(s, 0, "max-damage") == 3


def test_heuristics_removes_hazards_with_rapid_spin_while_two_members_are_left(world, start, act):
	tables, engine, *_ = world
	s = set_moves(
		duel(start, tables, engine, "starmie", "snorlax"),
		tables,
		0,
		["surf", "rapidspin", "tackle"],
	)
	assert act(s, 0) == 0
	hurt = s._replace(hazards=s.hazards.at[0, 1].set(2))
	assert act(hurt, 0) == 1  # Toxic Spikes on its own side
	assert act(hurt, 0, "max-damage") == 0
	# The last member alone, with the others fainted, does not bother; two left do.
	assert act(hurt._replace(hp=hurt.hp.at[0, 1:].set(0)), 0) == 0
	assert act(hurt._replace(hp=hurt.hp.at[0, 2:].set(0)), 0) == 1


def test_heuristics_leaves_when_the_stats_collapsed_and_a_decent_switch_exists(world, start, act):
	tables, engine, *_ = world
	s = team(
		start, tables, engine, 0, "gengar", "snorlax", "gyarados", "snorlax", "snorlax", "snorlax"
	)
	s = set_moves(team(s, tables, engine, 1, "machamp"), tables, 0, ["shadowball", "splash"])
	assert act(s, 0) < 4
	for stat in (S.BOOST_DEF, S.BOOST_SPD):
		low = s._replace(boosts=s.boosts.at[0, stat].set(-3))
		assert act(low, 0) == 4 + 1  # Gyarados is slot 2, the second bench member
		assert act(low, 0, "max-damage") < 4
		# A drop of two is not enough.
		assert act(s._replace(boosts=s.boosts.at[0, stat].set(-2)), 0) < 4
	# The lowered offence counts only for the stat that the member uses more.
	stats = s.stats.at[0, 0, S.ATK].set(300).at[0, 0, S.SPA].set(200)
	assert act(s._replace(stats=stats, boosts=s.boosts.at[0, S.BOOST_ATK].set(-3)), 0) == 5
	assert act(s._replace(stats=stats, boosts=s.boosts.at[0, S.BOOST_SPA].set(-3)), 0) < 4
	# A hidden trap (the mask has no switch) keeps it in whatever the reason.
	low = s._replace(boosts=s.boosts.at[0, S.BOOST_DEF].set(-3))
	assert act(low, 0, mask=legal_actions(low, engine)[0].at[4:].set(False)) < 4
	# No decent switch (every bench member is worse than even): it stays.
	stuck = team(
		low, tables, engine, 0, "gengar", "snorlax", "snorlax", "snorlax", "snorlax", "snorlax"
	)
	assert act(stuck, 0) < 4


def test_heuristics_leaves_a_matchup_below_minus_two(world, start, act):
	tables, engine, *_ = world
	# Charizard against Golem: offence 0.5, defence 4 (Rock hits Fire and Flying), speed +0.1.
	s = team(
		start,
		tables,
		engine,
		0,
		"charizard",
		"blastoise",
		"snorlax",
		"snorlax",
		"snorlax",
		"snorlax",
	)
	s = set_moves(team(s, tables, engine, 1, "golem"), tables, 0, ["flamethrower", "splash"])
	assert act(s, 0) == 4  # Blastoise (slot 1), Water hits Rock and Ground for four times
	assert act(s, 0, "max-damage") < 4
	# No decent switch-in: it fights on.
	alone = team(s, tables, engine, 0, "charizard", "snorlax")
	assert act(alone, 0) < 4
	# A matchup of -1.1 is not below -2.
	fine = team(s, tables, engine, 0, "snorlax", "blastoise")
	assert act(set_moves(fine, tables, 0, ["tackle"]), 0) < 4


def forced_replacement(state, side=0):
	"""The side's active member fainted and it must send in another one; the foe waits."""
	request = state.request.at[side].set(S.SWITCH).at[1 - side].set(S.NONE)
	return state._replace(request=request, hp=state.hp.at[side, state.active[side]].set(0))


def test_a_forced_replacement_sends_the_best_matchup_and_ties_follow_the_request(world, start, act):
	tables, engine, *_ = world
	# Bench slots 3 and 5 are identical Gyarados, the rest Snorlax: all but Gyarados are worse.
	s = team(
		start,
		tables,
		engine,
		0,
		"magikarp",
		"snorlax",
		"snorlax",
		"gyarados",
		"snorlax",
		"gyarados",
	)
	s = forced_replacement(team(s, tables, engine, 1, "machamp"))
	for name in ("heuristics", "max-damage"):
		assert act(s, 0, name) == 4 + 2  # slot 3: the third bench member (slots 1, 2, 3)
		# Showdown's request lists the party, a switch-in swapping two positions: here slot 5
		# comes before slot 3, so it wins the tie, though its team slot is higher.
		party = jnp.asarray([0, 1, 5, 3, 4, 2], s.party.dtype)
		assert act(s._replace(party=s.party.at[0].set(party)), 0, name) == 4 + 4
	# The active member may be any slot: the bench skips it, and the fainted ones.
	moved = team(
		s, tables, engine, 0, "snorlax", "snorlax", "snorlax", "magikarp", "snorlax", "gyarados"
	)
	moved = moved._replace(
		active=moved.active.at[0].set(3),
		party=moved.party.at[0].set(jnp.asarray([3, 1, 2, 0, 4, 5], moved.party.dtype)),
		hp=moved.hp.at[0, 3].set(0).at[0, 4].set(0),
	)
	assert act(moved, 0) == 4 + 4  # slot 5: k = 5 - 1
	only = s._replace(hp=s.hp.at[0, 1:5].set(0))
	assert act(only, 0) == 4 + 4


def test_a_side_without_a_choice_plays_the_noop(world, start, act):
	waiting = start._replace(request=start.request.at[0].set(S.NONE))
	for name in ("heuristics", "max-damage"):
		assert act(waiting, 0, name) == 0


def test_struggle_and_forced_moves_are_action_zero(world, start, act):
	tables, engine, *_ = world
	s = set_moves(
		duel(start, tables, engine, "charizard", "venusaur"), tables, 0, ["spikes", "flamethrower"]
	)
	spent = s._replace(pp=s.pp.at[0, 0].set(0))
	assert legal_actions(spent, engine)[0][:4].tolist() == [True, False, False, False]
	for name in ("heuristics", "max-damage"):
		assert act(spent, 0, name) == 0  # Struggle, whatever slot 0 happens to name
	recharge = s._replace(volatiles=s.volatiles.at[0, S.MUST_RECHARGE].set(1))
	assert act(recharge, 0) == 0


def test_hidden_power_counts_as_the_typed_power_70_move(world, start, act):
	tables, engine, *_ = world
	from pokejax.mechanics.hidden_power import hidden_power

	def ivs_for(name):
		for bits in range(64):
			ivs = jnp.asarray([30 + ((bits >> i) & 1) for i in range(6)], jnp.int32)
			if int(engine.hidden_power_type[hidden_power(ivs)[0]]) == tables.types.index[name]:
				return ivs.astype(jnp.uint8)
		raise AssertionError(name)

	s = set_moves(
		duel(start, tables, engine, "mew", "venusaur"), tables, 0, ["tackle", "hiddenpower"]
	)
	fire = s._replace(ivs=s.ivs.at[0, 0].set(ivs_for("fire")))
	assert act(fire, 0, "max-damage") == 1 and act(fire, 0) == 1  # 70 x 2 against 40
	water = s._replace(ivs=s.ivs.at[0, 0].set(ivs_for("water")))
	assert act(water, 0, "max-damage") == 0 and act(water, 0) == 0  # 70 x 0.5 against 40


def test_the_setup_branch_is_off_as_in_poke_env_and_works_when_asked(world, start, act):
	tables, engine, *_ = world
	s = set_moves(
		duel(start, tables, engine, "gengar", "machamp"), tables, 0, ["shadowball", "swordsdance"]
	)
	assert act(s, 0) == 0
	assert act(s, 0, setup=True) == 1
	# It needs full HP, a positive matchup and a raised stat below +6.
	assert act(s._replace(hp=s.hp.at[0, 0].set(299)), 0, setup=True) == 0
	assert act(s._replace(boosts=s.boosts.at[0, S.BOOST_ATK].set(6)), 0, setup=True) == 0
	bad = set_moves(
		duel(start, tables, engine, "snorlax", "machamp"), tables, 0, ["bodyslam", "swordsdance"]
	)
	assert act(bad, 0, setup=True) == 0


@pytest.mark.parametrize("name", ["random", "random-move"])
def test_the_random_policies_draw_from_the_right_set(world, start, name):
	tables, engine, *_ = world
	policy = make_policy(name, engine, tables)
	s = set_moves(
		duel(start, tables, engine, "mew", "snorlax"), tables, 0, ["tackle", "splash", "surf"]
	)
	keys = jax.random.split(jax.random.key(7), 300)
	draw = jax.jit(jax.vmap(lambda state, key: policy(state, 0, key), in_axes=(None, 0)))
	seen = set(np.asarray(draw(s, keys)).tolist())
	assert {0, 1, 2} <= seen <= {0, 1, 2, 4, 5, 6, 7, 8}
	assert (name == "random") == bool(seen & {4, 5, 6, 7, 8})
	# At a forced replacement only switches are legal, and random-move picks among them.
	forced = forced_replacement(s)
	seen = set(np.asarray(draw(forced, keys)).tolist())
	assert seen == {4, 5, 6, 7, 8}
