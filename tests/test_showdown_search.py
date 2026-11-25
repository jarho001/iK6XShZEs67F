"""Check the own-side stream adapter against new pinned, forced-outcome battles."""

import json
import subprocess
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.diff import replay_histories
from pokejax.engine import state as s
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.env import Env, EnvConfig
from pokejax.search.belief import Catalogue
from pokejax.search.method1 import Method1, _worlds
from pokejax.search.rebuild import OWN_FIELDS, _same_leaf
from pokejax.search.transition import observed_mask
from pokejax.showdown.player import stack_observations
from pokejax.showdown.search import stack_views
from pokejax.showdown.search_evaluate import draw_setups
from pokejax.teams import save_npz, team_pool_from_data

ROOT = Path(__file__).resolve().parents[1]
KEY = jax.random.key(491)
WORLDS = 4
CORE = ("active", "boosts", "hp", "pp", "status", "ability", "item", "species")


@pytest.fixture(scope="module")
def recorded():
	output = subprocess.check_output(
		[
			"node",
			"tests/support/record-search-showdown.cjs",
			"tests/fixtures/diff-pool.jsonl.gz",
		],
		cwd=ROOT,
	)
	return json.loads(output)


def oracle_fields(truth, tables):
	def members(name, convert=lambda value: value):
		return [convert(member[name]) for member in truth["members"]]

	return {
		"active": truth["active"],
		"boosts": truth["boosts"],
		"hp": members("hp"),
		"pp": members("pp", lambda moves: [*moves, *[0] * (4 - len(moves))]),
		"status": members("status", s.STATUS_NAMES.index),
		"ability": members("ability", lambda name: tables.abilities.index.get(name, 0)),
		"item": members("item", lambda name: tables.items.index.get(name, 0)),
		"species": members("species", lambda name: tables.species.index[name]),
	}


def test_own_stream_matches_pinned_state_at_every_request(recorded, tables):
	_, battles = recorded
	seen = Counter()
	for battle in battles:
		snapshots = [battle["initialState"], *(step["state"] for step in battle["steps"])]
		for side, requests in enumerate(battle["requests"]):
			for item, snapshot in zip(requests, snapshots[:-1], strict=True):
				truth = snapshot["sides"][side]
				for name, expected in oracle_fields(truth, tables).items():
					np.testing.assert_array_equal(item["own"][name], expected)
					seen[name] += 1
				# The oracle uses the actual trap when listing actions. A player
				# can still see a possible (maybeTrapped) switch in its request.
				assert all(
					not enabled or item["legal"][i]
					for i, enabled in enumerate(truth["legalActions"])
				)
	assert min(seen.values()) > 850
	print(f"{min(seen.values())} recorded decision points with exact {', '.join(CORE)}")


def test_own_stream_matches_forced_engine_replay(recorded, tables):
	header, battles = recorded
	history, lengths = replay_histories(header, battles, tables, project=lambda states, _: states)
	mismatches = Counter()
	reconciled = Counter()
	approximated = Counter()
	volatile_differences = Counter()
	for i, (battle, length) in enumerate(zip(battles, lengths, strict=True)):
		snapshots = [battle["initialState"], *(step["state"] for step in battle["steps"])]
		for side, requests in enumerate(battle["requests"]):
			for ply, item in enumerate(requests[:length]):
				truth = oracle_fields(snapshots[ply]["sides"][side], tables)
				own = item["own"]
				for name in OWN_FIELDS:
					engine = getattr(history, name)[i, ply, side]
					streamed = own[name]
					if name == "volatiles":
						streamed = np.zeros_like(engine)
						for key, value in own[name].items():
							streamed[getattr(s, key)] = value
						for index in np.flatnonzero(engine != streamed):
							volatile_differences[int(index)] += 1
						continue
					if np.array_equal(engine, streamed):
						continue
					if name in truth and not np.array_equal(engine, truth[name]):
						np.testing.assert_array_equal(streamed, truth[name])
						reconciled[name] += 1
					elif name == "wish_hp" and not own["wish_turns"]:
						# The Wish effect cannot read the stale amount.
						approximated["inactive_wish_hp"] += 1
					elif name in ("status_counter", "sleep_start", "sleep_from_foe"):
						# Sleep duration is private. Inactive sleep bookkeeping is
						# gated by current status, including the Sleep Clause check.
						changed = np.asarray(streamed) != engine
						status = np.asarray(own["status"])[changed]
						assert np.all(status != s.TOXIC)
						approximated[name] += 1
					elif name == "ability_public":
						# Only a transformed own observation reads this field for
						# the own ability. Method 1 never evaluates from the foe seat.
						assert not item["observation"]["own"]["transformed"]
						approximated[name] += 1
					else:
						mismatches[name] += 1
	assert not mismatches, mismatches
	assert volatile_differences.keys() <= {
		s.CONFUSION,
		s.LOCKED_TRUE_DURATION,
		s.TAUNT,
		s.STALL,
		s.STALL_USED,
		s.FAKE_OUT_USED,
	}, volatile_differences
	print(f"Verified engine disagreements {reconciled}, approximations {approximated}")
	print(f"Volatile differences {volatile_differences}")


def stack(rows):
	return jax.tree.map(lambda *leaves: np.stack(leaves), *rows)


def test_retained_worlds_reproduce_pinned_observation_and_mask(recorded, tables, tmp_path):
	header, battles = recorded
	pool_sets = [team for battle in battles for team in battle["teams"]]
	pool = team_pool_from_data(dict(header, teams=pool_sets, count=len(pool_sets)), tables)
	path = tmp_path / "pool.npz"
	save_npz(pool, path)
	catalogue = Catalogue.load(path, tables)
	engine = build_engine_tables(tables)
	history, _ = replay_histories(header, battles, tables, project=lambda states, _: states)
	example = jax.tree.map(lambda array: array[0, 0], history)
	template = observe(example, 0, engine)
	dtypes = [getattr(example, name).dtype for name in OWN_FIELDS]
	method = Method1(None, engine, catalogue)
	items, histories, particles = [], [], []
	for battle in battles:
		for side, requests in enumerate(battle["requests"]):
			belief = method.new_beliefs([battle["index"], side], 1)[0]
			for item in requests:
				obs = stack_observations([item["observation"]], template)
				obs = jax.tree.map(lambda x: x[0], obs)
				belief_history = belief.update(obs, False)
				candidates, count, _ = belief.sample(WORLDS)
				if count and sum(item["legal"]) > 1:
					items.append(item)
					histories.append(belief_history)
					particles.append(candidates)
	observation = stack_observations([item["observation"] for item in items], template)
	view = stack_views(items, observation, dtypes)
	worlds, valid = _worlds(
		view,
		stack(particles),
		stack(histories),
		jnp.asarray([item["previous"] for item in items]),
		jnp.zeros((len(items), len(tables.moves.ids) + 9)),
		jax.random.split(KEY, len(items)),
		method.data,
		engine,
		protocol=True,
	)
	rows, samples = np.where(np.asarray(valid))
	assert len(rows) > 1500
	state = jax.tree.map(lambda value: value[rows, samples], worlds.state)
	sides = view.side[rows]
	mask = observed_mask(worlds)[rows, samples, sides]
	np.testing.assert_array_equal(mask, view.obs.legal_actions[rows])

	@jax.jit
	def observations_equal(state, sides, expected):
		actual = jax.vmap(observe, in_axes=(0, 0, None))(state, sides, engine)
		return all_leaves(expected, actual)

	expected = jax.tree.map(lambda value: value[rows], view.obs)
	assert observations_equal(state, sides, expected)
	print(f"{len(rows)} of {valid.size} candidate worlds reproduce every observation leaf and mask")


def all_leaves(expected, actual):
	return jnp.all(
		jnp.asarray(
			jax.tree.leaves(jax.tree.map(lambda e, a: _same_leaf(e, a, True), expected, actual))
		)
	)


def test_evaluation_draws_identical_team_pairs_and_seats(recorded, tables, tmp_path):
	header, battles = recorded
	teams = [team for battle in battles for team in battle["teams"]]
	pool = team_pool_from_data(dict(header, teams=teams, count=len(teams)), tables)
	path = tmp_path / "setups.npz"
	save_npz(pool, path)
	env = Env(EnvConfig(batch_size=4, observe="none"), path, tables=tables)
	for head in (False, True):
		specs, engine = draw_setups(env, 8, 4, 491, head, True)
		for batch_index, (states, seats, _) in enumerate(engine):
			setup = specs[batch_index * 4 : batch_index * 4 + 4]
			indices = [row["teams"] for row in setup]
			np.testing.assert_array_equal(states.species_base, pool.species[indices])
			np.testing.assert_array_equal(seats, [row["learner"] for row in setup])
			if head:
				assert indices[:2] == indices[2:]
				assert [row["seedIndex"] for row in setup[:2]] == [
					row["seedIndex"] for row in setup[2:]
				]
