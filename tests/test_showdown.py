"""The Showdown player: the bridge protocol, the batched policy and the comparison's team pairs.

These tests play a few battles through reference/showdown-player.cjs (the pinned Showdown checkout must be
set up with reference/setup.cjs) with a small randomly initialised network on the CPU. The large test plays
the checkpoint of the Phase 4 run (work/runs/rtx3060ti) and runs only with --large.
"""

import gzip
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from train_support import league_config, tiny_config, write_pool

from pokejax.baselines import make_policy
from pokejax.env import Env, EnvConfig
from pokejax.sampler import sample_team_indices
from pokejax.showdown.compare import evaluation_pairs
from pokejax.showdown.player import (
	Checkpoint,
	Policy,
	RunResult,
	battle_specs,
	compare_first_observations,
	engine_observations,
	load_checkpoint,
	pad_size,
	run_battles,
	summarize,
)
from pokejax.tables import load_tables
from pokejax.train.league import Layout, League
from pokejax.train.network import PokePolicy, initial_carry, vocabulary
from pokejax.train.ppo import EVALUATION_SEED, Trainer

ROOT = Path(__file__).resolve().parents[1]
MAX_STEPS = 40  # short battles: some end, some are truncated
CONCURRENCY = 3

pytestmark = pytest.mark.usefixtures("cpu_only")


def random_checkpoint(env: Env, seed: int = 0) -> Checkpoint:
	config = tiny_config("network.features=true", "network.action_head=factored")
	model = PokePolicy(vocabulary(load_tables()), config.network)
	example = engine_observations(env, np.zeros((1, 2), np.int32), np.zeros(1, np.int32))
	params = model.init(jax.random.key(seed), example, initial_carry(config.network, (1,)))[
		"params"
	]
	return Checkpoint(config=config, params=params, path=Path("random"))


@pytest.fixture(scope="module")
def player(tables, tmp_path_factory):
	directory = tmp_path_factory.mktemp("showdown")
	npz = write_pool(directory, tables)
	env = Env(EnvConfig(batch_size=1, observe="none"), npz, tables=tables)
	policy = Policy(random_checkpoint(env), env, slots=CONCURRENCY, seed=5)
	return env, policy, npz.with_name("teams.json")


def play(player, count, *, opponent="max-damage", seed=3, greedy=False, record=None) -> tuple:
	env, policy, pool_json = player
	pairs = np.asarray(sample_team_indices(jax.random.key(seed), env.team_indices, count))
	learners = np.arange(count) % 2
	run = run_battles(
		policy,
		battle_specs(pairs, learners),
		pool_json=pool_json,
		opponent=opponent,
		seed=seed,
		max_steps=MAX_STEPS,
		concurrency=CONCURRENCY,
		greedy=greedy,
		record=record,
	)
	return run, pairs, learners


def test_battles_play_to_the_end_without_errors(player):
	run, pairs, learners = play(player, 6)
	assert sorted(r["index"] for r in run.results) == list(range(6))
	assert all(not r["errors"] for r in run.results), run.results
	assert run.illegal_picks == 0
	assert sum(r["maskMismatches"] for r in run.results) == 0
	assert run.decisions == sum(r["decisions"] for r in run.results)
	for result in run.results:
		index = result["index"]
		assert result["learner"] == learners[index]
		assert result["teams"] == pairs[index].tolist()
		if result["truncated"]:
			assert result["steps"] == MAX_STEPS and result["winner"] is None
		else:
			assert result["winner"] in (0, 1, "tie")
	summary = summarize(run, "max-damage", "sampled")
	assert summary.counted == 6 and summary.errors == 0
	assert summary.wins + summary.losses + summary.ties + summary.truncated == 6


def test_first_observations_match_the_engine(player):
	env, _, _ = player
	run, pairs, learners = play(player, 6, opponent="random")
	check = compare_first_observations(env, pairs, learners, run.first)
	assert check["battles"] == 6
	assert check["identical"], check
	# Integers and flags are equal bit for bit; a float quotient may be one unit in the last place off.
	assert check["max_ulp"] <= 1
	assert not check["leaves_mismatched"]


def test_a_lead_draw_is_accepted_when_another_draw_matches(tables, tmp_path_factory):
	"""Two Kyogre leads (Drizzle) tie in speed: which of them the protocol names is a random draw.

	The engine's reset draws that order from its key, so a battle can differ from Showdown under the default
	draw and match under another one; the check must accept it and name the draw.
	"""
	from pokejax.teams import load_team_pool, save_npz

	fixture = json.loads((ROOT / "tests/fixtures/reset.json").read_text())
	teams = [list(team) for team in fixture["teamSets"]]
	lead = dict(
		teams[0][0],
		species="kyogre",
		ability="drizzle",
		item="",
		gender="N",
		moves=["surf", "icebeam", "thunder", "calmmind"],
		level=70,
	)
	twin = dict(lead)  # the same stats: the two leads tie in speed
	teams += [[lead, *teams[0][1:]], [twin, *teams[1][1:]]]
	directory = tmp_path_factory.mktemp("lead-draw")
	source = directory / "teams.json"
	source.write_text(
		json.dumps(
			{
				"format": "gen5randombattle",
				"count": len(teams),
				"seed": 0,
				"showdownCommit": fixture["showdownCommit"],
				"teams": teams,
			}
		)
	)
	npz = directory / "teams.npz"
	save_npz(load_team_pool(source, tables), npz)
	env = Env(EnvConfig(batch_size=1, observe="none"), npz, tables=tables)
	policy = Policy(random_checkpoint(env), env, slots=CONCURRENCY, seed=5)
	count = 16
	pairs = np.tile(np.array([[len(teams) - 2, len(teams) - 1]]), (count, 1))
	learners = np.arange(count) % 2
	run = run_battles(
		policy,
		battle_specs(pairs, learners),
		pool_json=source,
		opponent="random",
		seed=9,
		max_steps=MAX_STEPS,
		concurrency=CONCURRENCY,
		greedy=True,
	)
	check = compare_first_observations(env, pairs, learners, run.first)
	assert check["identical"], check
	assert check["battles_mismatched"] == 0
	assert check["battles_needing_another_draw"], "no battle differed under the default draw"
	assert all(1 <= draw <= 16 for draw in check["battles_needing_another_draw"].values())


def test_the_same_seed_plays_the_same_battles(player):
	first, _, _ = play(player, 5, seed=11, greedy=True)
	second, _, _ = play(player, 5, seed=11, greedy=True)
	key = lambda r: r["index"]
	assert sorted(first.results, key=key) == sorted(second.results, key=key)
	assert first.decisions == second.decisions
	for a, b in zip(first.first, second.first, strict=True):
		assert np.array_equal(a["legal_actions"], b["legal_actions"])
		assert np.array_equal(a["own"]["hp"], b["own"]["hp"])


def test_records_hold_every_decision(player, tmp_path):
	path = tmp_path / "battles.jsonl.gz"
	play(player, 4, record=path)
	with gzip.open(path, "rt", encoding="utf-8") as handle:
		records = [json.loads(line) for line in handle]
	assert sorted(record["index"] for record in records) == list(range(4))
	for record in records:
		assert record["type"] == "battle"
		assert len(record["steps"]) == record["result"]["decisions"]
		# Every step that did not end the battle carries both sides' requests.
		for step in record["steps"][:-1]:
			assert {entry["side"] for entry in step["requests"]} == {0, 1}
		for step in record["steps"]:
			assert len(step["legal"]) == 9 and "observation" in step


def test_evaluation_pairs_are_the_engine_evaluation_draws(tables, tmp_path_factory):
	"""The comparison's pairs and sides are the ones Trainer.evaluate plays (its keys and its sampler)."""
	with jax.default_device(jax.devices("cpu")[0]):
		npz = write_pool(tmp_path_factory.mktemp("pairs"), tables)
		trainer = Trainer(league_config(), npz)
		pairs, learners = evaluation_pairs(trainer, 8)
		batch = trainer.config.evaluation.batch
		env = trainer.env.replace(batch_size=batch, keep_final=False)
		policy = make_policy("random", env.tables, tables)
		league = League(
			env,
			Layout({"random": batch}),
			trainer.model,
			trainer.config.network,
			trainer.config.pool.size,
			{"random": policy},
		)
		key = jax.random.key(trainer.config.run.seed + EVALUATION_SEED)
		species = np.asarray(env.pool.species)
		for index in range(8 // batch):
			learner = jnp.arange(batch, dtype=jnp.int32) % 2
			state = league.reset(jax.random.fold_in(key, index), learner)
			rows = slice(index * batch, (index + 1) * batch)
			assert np.array_equal(learners[rows], np.asarray(learner))
			assert np.array_equal(np.asarray(state.env.state.species), species[pairs[rows]])


def test_batches_are_padded_to_fixed_sizes():
	assert [pad_size(n) for n in (1, 64, 65, 300, 4096)] == [64, 64, 128, 512, 4096]
	assert pad_size(5000) == 5000


def test_summary_counts_truncations_and_ties_apart():
	results = [
		{
			"index": 0,
			"learner": 0,
			"winner": 0,
			"ended": True,
			"truncated": False,
			"steps": 10,
			"turns": 5,
			"decisions": 10,
			"refusals": 1,
			"illegal": 0,
			"maskMismatches": 0,
			"knownTrapDecisions": 0,
			"turnLimit": False,
			"errors": [],
			"teams": [0, 1],
		},
		{
			"index": 1,
			"learner": 0,
			"winner": 1,
			"ended": True,
			"truncated": False,
			"steps": 20,
			"turns": 9,
			"decisions": 20,
			"refusals": 0,
			"illegal": 0,
			"maskMismatches": 0,
			"knownTrapDecisions": 0,
			"turnLimit": False,
			"errors": [],
			"teams": [0, 1],
		},
		{
			"index": 2,
			"learner": 1,
			"winner": "tie",
			"ended": True,
			"truncated": False,
			"steps": 30,
			"turns": 1001,
			"decisions": 30,
			"refusals": 0,
			"illegal": 0,
			"maskMismatches": 0,
			"knownTrapDecisions": 0,
			"turnLimit": True,
			"errors": [],
			"teams": [0, 1],
		},
		{
			"index": 3,
			"learner": 1,
			"winner": None,
			"ended": False,
			"truncated": True,
			"steps": 40,
			"turns": 12,
			"decisions": 40,
			"refusals": 0,
			"illegal": 0,
			"maskMismatches": 0,
			"knownTrapDecisions": 0,
			"turnLimit": False,
			"errors": [],
			"teams": [0, 1],
		},
	]
	run = RunResult(
		results=results,
		first=[],
		seconds=2.0,
		rounds=3,
		decisions=100,
		illegal_picks=0,
		node_seconds=1.5,
	)
	summary = summarize(run, "random", "greedy")
	assert (summary.wins, summary.losses, summary.ties, summary.truncated) == (1, 1, 1, 1)
	assert summary.turn_limit_ties == 1
	assert summary.win_rate == pytest.approx(0.25)
	assert summary.refusals == 1 and summary.battles_per_second == pytest.approx(2.0)


@pytest.mark.large
def test_large_checkpoint_plays_all_baselines():
	"""The Phase 4 checkpoint on 64 battles against each baseline, both policies (needs --large)."""
	path = ROOT / "work/runs/rtx3060ti/checkpoint.npz"
	if not path.exists():
		pytest.skip(f"no checkpoint at {path}")
	pool = ROOT / "work/teams/gen5-teams.npz"
	pool_json = ROOT / "work/teams/gen5-teams.json"
	env = Env(EnvConfig(batch_size=1, observe="none"), pool)
	checkpoint = load_checkpoint(path.parent, env)
	policy = Policy(checkpoint, env, slots=64, seed=1)
	for opponent in ("random", "random-move", "max-damage", "heuristics"):
		for greedy in (False, True):
			pairs = np.asarray(sample_team_indices(jax.random.key(4), env.team_indices, 64))
			learners = np.arange(64) % 2
			run = run_battles(
				policy,
				battle_specs(pairs, learners),
				pool_json=pool_json,
				opponent=opponent,
				seed=4,
				max_steps=1000,
				concurrency=64,
				greedy=greedy,
			)
			summary = summarize(run, opponent, "greedy" if greedy else "sampled")
			assert (
				summary.errors == 0 and summary.mask_mismatches == 0 and summary.illegal_picks == 0
			)
			assert compare_first_observations(env, pairs, learners, run.first)["identical"]
