"""Compare the JAX baselines with the Node ones at every decision point of recorded battles.

A recording made with `node reference/record-battles.cjs --profile pool --policies` carries, at every decision
point (initial and after each step), the action that the Node max-damage and heuristics policies
(reference/baselines.cjs) choose for both sides from that player's own view (reference/player-view.cjs)
and the engine's legal choices. This replays every battle through the engine with the recorded forced
outcomes, has the JAX policies (pokejax.baselines) choose from the replayed state and legal mask, and
compares the actions exactly: the policies are deterministic, so they must agree. Choices that a policy
makes at random (none of these two does) are counted apart in the recording's stats.

Run `python -m pokejax.baselines_diff PATH` (plain JSONL or gzip). Step numbers in reports are zero-based;
step -1 is the initial decision point.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import build_baseline_tables, heuristics_action, max_damage_action
from pokejax.diff import load_recording, replay_histories
from pokejax.engine.actions import legal_actions
from pokejax.tables import load_tables

COMPARED_POLICIES = ("max-damage", "heuristics")


@lru_cache(maxsize=1)
def _host_tables():
	return load_tables()


def jax_choices(states, tables, setup_branch=False):
	"""The JAX policies' actions at a batch of states: int32 [batch, policy, side]."""
	data = build_baseline_tables(_host_tables())
	masks = jax.vmap(legal_actions, in_axes=(0, None))(states, tables)

	def choose(name, side):
		def one(state, mask):
			if name == "max-damage":
				return max_damage_action(state, side, mask[side], tables, data)
			return heuristics_action(state, side, mask[side], tables, data, setup_branch)

		return jax.vmap(one)(states, masks)

	return jnp.stack(
		[
			jnp.stack([choose(name, side) for side in range(2)], axis=-1)
			for name in COMPARED_POLICIES
		],
		axis=1,
	)


def node_choices(battle):
	"""The recorded Node actions of a battle: int [decision points, policy, side]."""
	points = [battle["initialPolicies"]] + [record["policies"] for record in battle["steps"]]
	return np.asarray([[point[name] for name in COMPARED_POLICIES] for point in points])


@dataclass(frozen=True)
class Mismatch:
	battle: int
	step: int  # -1 is the initial decision point
	side: int
	policy: str
	engine: int
	node: int

	def __str__(self):
		point = "initial" if self.step == -1 else f"step {self.step}"
		return (
			f"Battle {self.battle}, {point}, side {self.side}, {self.policy}: "
			f"JAX action {self.engine}, Node action {self.node}"
		)


@dataclass(frozen=True)
class PolicyResult:
	battles: int
	points: int  # decision points x 2 sides x policies compared
	mismatches: tuple[Mismatch, ...]
	by_policy: Counter
	fallbacks: dict  # random fallbacks counted by the recorder, per policy
	seconds: float

	def report(self, limit=40):
		lines = [str(mismatch) for mismatch in self.mismatches[:limit]]
		if len(self.mismatches) > limit:
			lines.append(f"... {len(self.mismatches) - limit} more")
		lines += [f"  {count:7d}  {name}" for name, count in self.by_policy.most_common()]
		lines.append(
			f"{len(self.mismatches)} of {self.points} choices differ over {self.battles} battles; "
			f"random fallbacks {self.fallbacks}; {self.seconds:.1f}s (including compilation)"
		)
		return "\n".join(lines)


def compare_file(path, *, recording=None, keep=100000) -> PolicyResult:
	"""Replay every battle of a --policies recording and compare the actions at every decision point."""
	started = perf_counter()
	header, battles = load_recording(path) if recording is None else recording
	if "policies" not in header:
		raise ValueError("Recording has no policy choices; record it with --policies")
	if tuple(header["policies"]["names"]) != COMPARED_POLICIES:
		raise ValueError(f"Recording compares {header['policies']['names']}")
	project = partial(jax_choices, setup_branch=bool(header["policies"].get("setupBranch")))
	actual, lengths = replay_histories(header, battles, _host_tables(), project=project)
	mismatches = []
	by_policy = Counter()
	points = 0
	fallbacks = Counter()
	for b, battle in enumerate(battles):
		count = int(lengths[b]) + 1
		points += count * 2 * len(COMPARED_POLICIES)
		fallbacks.update(battle["stats"].get("policyFallbacks", {}))
		wanted = node_choices(battle)
		if wanted.shape != actual[b, :count].shape:
			raise ValueError(
				f"Battle {battle['index']}: {wanted.shape} vs {actual[b, :count].shape}"
			)
		for address in np.argwhere(actual[b, :count] != wanted):
			point, policy, side = (int(i) for i in address)
			by_policy[COMPARED_POLICIES[policy]] += 1
			if len(mismatches) < keep:
				mismatches.append(
					Mismatch(
						battle["index"],
						point - 1,
						side,
						COMPARED_POLICIES[policy],
						int(actual[b, point, policy, side]),
						int(wanted[point, policy, side]),
					)
				)
	mismatches.sort(key=lambda m: (m.battle, m.step, m.side, m.policy))
	return PolicyResult(
		len(battles),
		points,
		tuple(mismatches),
		by_policy,
		dict(fallbacks),
		perf_counter() - started,
	)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("path", type=Path)
	parser.add_argument("--limit", type=int, default=40)
	args = parser.parse_args()
	result = compare_file(args.path)
	print(result.report(args.limit))
	raise SystemExit(bool(result.mismatches))


if __name__ == "__main__":
	main()
