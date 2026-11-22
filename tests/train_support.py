"""Helpers of the training tests: a tiny configuration and a small team pool (no Showdown needed)."""

import json
from contextlib import contextmanager
from pathlib import Path

import jax

from pokejax.teams import load_team_pool, save_npz
from pokejax.train.config import load_config

REPO = Path(__file__).resolve().parents[1]

# Small enough to compile and run on the CPU in a test: four battles of eight steps, cut at three
# environment steps so that every rollout holds truncated battles and episode boundaries.
TINY_OVERRIDES = (
	"run.updates=4",
	"rollout.batch_size=4",
	"rollout.rollout_length=8",
	"rollout.sequence_length=4",
	"ppo.minibatches=2",
	"ppo.epochs=2",
	"network.core_width=16",
	"network.trunk_width=16",
	"network.member_width=8",
	"network.species_embedding=4",
	"network.move_embedding=4",
	"network.item_embedding=2",
	"network.ability_embedding=2",
	"network.type_embedding=2",
	"network.code_embedding=2",
	"env.max_steps=3",
	"evaluation.batch=4",
	"evaluation.battles=4",
	"evaluation.final_battles=4",
)


# A league of every group, small enough for a test: 8 battle rows split 4 selfplay, 2 pool (one
# row for each of the two members), 1 random and 1 max-damage row. Battles are cut at 400 steps, so
# some end within a rollout of 256 steps; pool insertions happen after every update.
LEAGUE_OVERRIDES = (
	"rollout.batch_size=8",
	"rollout.rollout_length=256",
	"rollout.sequence_length=4",
	"ppo.minibatches=2",
	"env.max_steps=400",
	"opponent.shares={selfplay=0.5, pool=0.25, random=0.125, max-damage=0.125}",
	"pool.size=2",
	"pool.every=1",
	"evaluation.greedy=true",
	'evaluation.opponents=["max-damage"]',
)


@contextmanager
def on_cpu():
	"""Run JAX on the CPU: GPU scatter-adds (embedding gradients) are not bit-reproducible."""
	with jax.default_device(jax.devices("cpu")[0]):
		yield


def tiny_config(*overrides: str):
	return load_config(None, [*TINY_OVERRIDES, *overrides])


def league_config(*overrides: str):
	return tiny_config(*LEAGUE_OVERRIDES, *overrides)


def write_pool(directory: Path, tables) -> Path:
	"""The two teams of tests/fixtures/reset.json as an .npz pool in directory."""
	fixture = json.loads((REPO / "tests/fixtures/reset.json").read_text())
	source = directory / "teams.json"
	source.write_text(
		json.dumps(
			{
				"format": "gen5randombattle",
				"count": len(fixture["teamSets"]),
				"seed": 0,
				"showdownCommit": fixture["showdownCommit"],
				"teams": fixture["teamSets"],
			}
		)
	)
	path = directory / "teams.npz"
	save_npz(load_team_pool(source, tables), path)
	return path
