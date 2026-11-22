"""The trained policy on Showdown: the checkpoint's network run on batches of waiting battles.

A run keeps one recurrent carry per battle slot of the bridge (Node's concurrent battles). A battle that
starts in a slot resets that slot's carry (its first decision says so), and every decision of a round
runs in one jitted call over the padded batch. Actions are sampled from the masked logits, or the argmax
when greedy. The observations come from Node as JSON; they are converted to the engine's Observation
leaves (same names, shapes and dtypes) and checked against the engine's observation of the same teams
at each battle's first decision.

run_battles plays a list of battles {index, teams, learner} through one bridge and summarises them.
"""

import contextlib
import gzip
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.observation import Observation, observe
from pokejax.engine.reset import Teams
from pokejax.engine.reset import reset as reset_battle
from pokejax.env import Env
from pokejax.showdown.bridge import Bridge
from pokejax.tables import load_tables
from pokejax.train.checkpoint import CONFIG_ENTRY, CheckpointError
from pokejax.train.config import TrainConfig
from pokejax.train.evaluation import summarize_battles
from pokejax.train.network import PokePolicy, initial_carry, reset_carry, vocabulary

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_JSON = ROOT / "work/teams/gen5-teams.json"
# Batch sizes that jitted steps are compiled for; a round pads up to the next one.
PAD_SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
CHECKPOINT_FILE = "checkpoint.npz"


def pad_size(count: int) -> int:
	for size in PAD_SIZES:
		if count <= size:
			return size
	return count


def engine_observations(env: Env, pairs, learners, key: int = 0) -> Observation:
	"""The engine's observation of each battle's first decision: teams pairs [B, 2], learner sides [B].

	The battle is the reset of the pool teams (as Env.new_battles makes it). The reset draws the order of
	the lead switch-ins when their speeds tie, so key chooses one draw (see compare_first_observations).
	"""
	pairs = jnp.asarray(pairs, jnp.int32)
	teams = Teams(*(column[pairs] for column in env.pool))
	prepared = jax.tree.map(lambda column: column[pairs], env.prepared)
	keys = jax.random.split(jax.random.key(key), pairs.shape[0])
	states = jax.vmap(lambda k, t, p: reset_battle(k, t, env.tables, None, p))(
		keys, teams, prepared
	)
	sides = jnp.asarray(learners, jnp.int32)
	return jax.vmap(observe, in_axes=(0, 0, None))(states, sides, env.tables)


def stack_observations(values: list, template):
	"""Observations as JSON (one dict per battle) stacked as template leaves with a leading axis."""
	if hasattr(template, "_fields"):
		return type(template)(
			*(
				stack_observations([v[name] for v in values], getattr(template, name))
				for name in template._fields
			)
		)
	return np.asarray(values, dtype=template.dtype)


def leaves(value, prefix: str = "") -> dict:
	"""Leaves of nested NamedTuples by dotted field path."""
	if hasattr(value, "_fields"):
		out = {}
		for name in value._fields:
			out.update(leaves(getattr(value, name), f"{prefix}{name}."))
		return out
	return {prefix[:-1]: value}


# How many draws of the lead switch-in order to try for a battle whose first observation differs.
DRAW_RETRIES = 16


def _differences(engine, node, count: int):
	"""Per battle: whether a leaf differs at all, and whether a difference is more than one float ulp.

	Integers and flags must be equal. A float leaf may differ by one float32 unit in the last place: XLA
	compiles the engine's float32 divisions (level / 100, stat / 1000, hp / max hp ...) as a
	multiplication by a reciprocal, so a quotient can be one unit off the correctly rounded double
	quotient that the bridge's JSON carries.
	"""
	engine_leaves, node_leaves = leaves(engine), leaves(node)
	bit_differing = np.zeros(count, bool)
	mismatched = np.zeros(count, bool)
	by_leaf, ulp_leaves, max_ulp = {}, {}, 0
	for name, expected in engine_leaves.items():
		e = np.asarray(expected).reshape(count, -1)
		n = np.asarray(node_leaves[name]).reshape(count, -1)
		differs = np.any(e != n, axis=1)
		if e.dtype.kind == "f":
			ulps = np.abs(e.view(np.int32).astype(np.int64) - n.view(np.int32).astype(np.int64))
			max_ulp = max(max_ulp, int(ulps.max(initial=0)))
			one_ulp = np.any(ulps == 1, axis=1)
			if one_ulp.any():
				ulp_leaves[name] = int(one_ulp.sum())
			over = np.any(ulps > 1, axis=1)
			mismatched |= over
			if over.any():
				by_leaf[name] = int(over.sum())
		else:
			mismatched |= differs
			if differs.any():
				by_leaf[name] = int(differs.sum())
		bit_differing |= differs
	return bit_differing, mismatched, by_leaf, ulp_leaves, max_ulp


def _select(tree, index):
	"""The rows of every leaf of a batched observation at the given battle indices."""
	return jax.tree.map(lambda leaf: np.asarray(leaf)[index], tree)


def compare_first_observations(env: Env, pairs, learners, node_first: list[dict]) -> dict:
	"""The bridge's first observations against the engine's, battle by battle.

	The engine's reset orders the two lead switch-ins at random when their speeds tie. If both leads
	have an ability that announces weather (Drizzle, Drought, Sand Stream, Snow Warning), the order
	decides whose ability the protocol names, so the first observation is a function of that draw
	(Showdown's own draw is a different one). A battle that differs under the default draw is therefore
	accepted when another draw of the engine reproduces the bridge's observation, and the report names
	the draw that did. A battle that no draw reproduces is a mismatch. The float rule is in _differences.
	"""
	count = len(node_first)
	pairs = np.asarray(pairs)
	learners = np.asarray(learners)
	engine = engine_observations(env, pairs, learners)
	node = stack_observations(node_first, engine)
	bit_differing, mismatched, by_leaf, ulp_leaves, max_ulp = _differences(engine, node, count)
	pending = np.flatnonzero(mismatched)
	other_draws = {}
	for draw in range(1, DRAW_RETRIES + 1):
		if pending.size == 0:
			break
		# A draw is the same for every battle: the batch is reset with its own key, and the rows are picked.
		retry = engine_observations(env, pairs, learners, key=draw)
		_, still, _, _, _ = _differences(
			_select(retry, pending), _select(node, pending), pending.size
		)
		for battle in pending[~still]:
			other_draws[int(battle)] = draw
		pending = pending[still]
	return {
		"battles": count,
		"leaves": len(leaves(engine)),
		"bit_identical": not bit_differing.any(),
		"battles_bit_differing": int(bit_differing.sum()),
		"leaves_one_ulp": ulp_leaves,
		"max_ulp": max_ulp,
		"battles_needing_another_draw": other_draws,
		"battles_mismatched": int(pending.size),
		"leaves_mismatched": by_leaf,
		"identical": pending.size == 0,
	}


def _restore_params(data, template):
	leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(template)
	values = []
	for key_path, leaf in leaves_with_paths:
		name = ".params" + jax.tree_util.keystr(key_path)
		if name not in data.files:
			raise CheckpointError(f"checkpoint has no entry {name}")
		array = data[name]
		if array.shape != leaf.shape or array.dtype != leaf.dtype:
			raise CheckpointError(
				f"{name} is {array.dtype}{array.shape}, expected {leaf.dtype}{leaf.shape}"
			)
		values.append(jnp.asarray(array))
	return jax.tree_util.tree_unflatten(treedef, values)


@dataclass
class Checkpoint:
	config: TrainConfig
	params: object
	path: Path


def checkpoint_path(run_dir) -> Path:
	path = Path(run_dir)
	path = path / CHECKPOINT_FILE if path.is_dir() else path
	if not path.exists():
		raise CheckpointError(f"no checkpoint at {path}")
	return path


def read_config(run_dir) -> TrainConfig:
	"""The configuration a run saved with its checkpoint (no parameters are read)."""
	with np.load(checkpoint_path(run_dir), allow_pickle=False) as data:
		if CONFIG_ENTRY not in data.files:
			raise CheckpointError(f"{checkpoint_path(run_dir)} has no configuration")
		return TrainConfig.from_dict(json.loads(str(data[CONFIG_ENTRY])))


def load_checkpoint(run_dir, env: Env) -> Checkpoint:
	"""The parameters and the configuration a run saved (its checkpoint.npz, or the file itself)."""
	path = checkpoint_path(run_dir)
	with np.load(path, allow_pickle=False) as data:
		config = TrainConfig.from_dict(json.loads(str(data[CONFIG_ENTRY])))
		model = PokePolicy(vocabulary(load_tables()), config.network)
		example = engine_observations(env, np.zeros((1, 2), np.int32), np.zeros(1, np.int32))
		carry = initial_carry(config.network, (1,))
		template = model.init(jax.random.key(0), example, carry)["params"]
		params = _restore_params(data, template)
	return Checkpoint(config=config, params=params, path=path)


class Policy:
	"""The network on batches of decisions, with one recurrent carry per bridge slot."""

	def __init__(self, checkpoint: Checkpoint, env: Env, slots: int, seed: int):
		self.config = checkpoint.config
		self.model = PokePolicy(vocabulary(load_tables()), self.config.network)
		self.params = checkpoint.params
		self.slots = slots
		self.template = engine_observations(env, np.zeros((1, 2), np.int32), np.zeros(1, np.int32))
		self.hidden = initial_carry(self.config.network, (slots + 1,))  # the last row is padding
		self.base_key = jax.random.key(seed)
		self.rounds = 0
		self._step = jax.jit(self._forward)

	def _forward(self, params, hidden, slots, reset, obs, key, greedy):
		carry = reset_carry(jax.tree.map(lambda h: h[slots], hidden), reset)
		after, logits, _ = self.model.apply({"params": params}, obs, carry)
		sampled = jax.random.categorical(key, logits)
		action = jnp.where(greedy, jnp.argmax(logits, axis=-1), sampled).astype(jnp.int32)
		hidden = jax.tree.map(lambda h, a: h.at[slots].set(a), hidden, after)
		return hidden, action

	def act(self, items: list[dict], greedy: bool) -> np.ndarray:
		count = len(items)
		size = pad_size(count)
		slots = np.full(size, self.slots, np.int32)
		slots[:count] = [item["slot"] for item in items]
		reset = np.zeros(size, bool)
		reset[:count] = [item["reset"] for item in items]
		observations = [item["observation"] for item in items]
		observations += [observations[0]] * (size - count)  # padding: any legal observation
		obs = stack_observations(observations, self.template)
		key = jax.random.fold_in(self.base_key, self.rounds)
		self.rounds += 1
		self.hidden, actions = self._step(
			self.params, self.hidden, slots, reset, obs, key, jnp.bool_(greedy)
		)
		return np.asarray(actions)[:count]


@dataclass
class RunResult:
	results: list[dict]  # one per battle, as reference/showdown-player.cjs reports it
	first: list[dict]  # the first observation of each battle, by battle index
	seconds: float
	rounds: int
	decisions: int
	illegal_picks: int  # actions the observed mask forbids (never, by construction)
	node_seconds: float


@contextlib.contextmanager
def _record_writer(path: Path | None):
	"""A text file for the per-battle records (gzip when the name ends in .gz), or None."""
	if path is None:
		yield None
		return
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	opener = gzip.open if path.suffix == ".gz" else open
	with opener(path, "wt", encoding="utf-8") as handle:
		yield handle


def run_battles(
	policy: Policy,
	battles: list[dict],
	*,
	pool_json: Path,
	opponent: str,
	seed: int,
	max_steps: int,
	concurrency: int,
	greedy: bool,
	record: Path | None = None,
) -> RunResult:
	"""Play the battles through one Node bridge; the results come back in the order they finish."""
	bridge = Bridge(
		pool_json,
		battles,
		opponent=opponent,
		seed=seed,
		max_steps=max_steps,
		concurrency=concurrency,
		record=record is not None,
	)
	results, first = [], []
	rounds = decisions = illegal = 0
	started = time.perf_counter()
	with _record_writer(record) as writer:
		try:
			message = bridge.receive()
			while True:
				results.extend(message.get("finished", []))
				if writer is not None:
					for line in message.get("records", []):
						writer.write(line + "\n")
				if message["type"] == "done":
					node_seconds = float(message["stats"]["seconds"])
					break
				items = message["items"]
				for item in items:
					if item["first"]:
						first.append(item)
				actions = policy.act(items, greedy)
				for item, action in zip(items, actions, strict=True):
					illegal += 0 if item["legal"][int(action)] else 1
				bridge.answer(items, actions)
				rounds += 1
				decisions += len(items)
				message = bridge.receive()
		finally:
			bridge.close()
	seconds = time.perf_counter() - started
	first.sort(key=lambda item: item["battle"])
	first_by_battle = [item["observation"] for item in first]
	return RunResult(
		results=results,
		first=first_by_battle,
		seconds=seconds,
		rounds=rounds,
		decisions=decisions,
		illegal_picks=illegal,
		node_seconds=node_seconds,
	)


def battle_specs(pairs, learners) -> list[dict]:
	"""One bridge battle per row: its pool team indices and the learner's side."""
	return [
		{"index": index, "teams": [int(a), int(b)], "learner": int(side)}
		for index, ((a, b), side) in enumerate(
			zip(np.asarray(pairs), np.asarray(learners), strict=True)
		)
	]


@dataclass
class Summary:
	opponent: str
	policy: str
	battles: int
	counted: int  # battles without errors (the win rate's denominator)
	wins: int
	losses: int
	ties: int  # the battle ended as a tie: Showdown's turn limit, or a draw
	turn_limit_ties: int
	truncated: int  # the step cap ended the battle (not counted as a win)
	errors: int
	refusals: int  # switches Showdown refused because of a hidden trap
	mask_mismatches: int  # decisions whose observed mask differs from the request's
	known_trap_decisions: int  # decisions where a refusal removed the switches from the mask
	illegal_picks: int  # Python's actions the observed mask forbids
	mean_steps: float
	win_rate: float
	ci95: list
	seconds: float
	battles_per_second: float
	decisions: int
	rounds: int
	node_seconds: float
	first_check: dict = field(default_factory=dict)

	def as_dict(self) -> dict:
		return dict(self.__dict__)


def summarize(run: RunResult, opponent: str, policy: str) -> Summary:
	results = run.results
	errored = [r for r in results if r["errors"]]
	counted = [r for r in results if not r["errors"]]
	wins = sum(1 for r in counted if r["winner"] == r["learner"])
	ties = sum(1 for r in counted if r["winner"] == "tie")
	losses = len(counted) - wins - ties - sum(1 for r in counted if r["truncated"])
	truncated = sum(1 for r in counted if r["truncated"])
	mean = sum(r["steps"] for r in counted) / max(len(counted), 1)
	summary = summarize_battles(opponent, len(counted), wins, truncated, mean, policy=policy)
	return Summary(
		opponent=opponent,
		policy=policy,
		battles=len(results),
		counted=len(counted),
		wins=wins,
		losses=losses,
		ties=ties,
		turn_limit_ties=sum(1 for r in counted if r["turnLimit"]),
		truncated=truncated,
		errors=len(errored),
		refusals=sum(r["refusals"] for r in results),
		mask_mismatches=sum(r["maskMismatches"] for r in results),
		known_trap_decisions=sum(r["knownTrapDecisions"] for r in results),
		illegal_picks=run.illegal_picks,
		mean_steps=mean,
		win_rate=summary["win_rate"],
		ci95=summary["ci95"],
		seconds=run.seconds,
		battles_per_second=len(results) / run.seconds,
		decisions=run.decisions,
		rounds=run.rounds,
		node_seconds=run.node_seconds,
	)
