"""Compare the engine observation with the one the Node player view builds from the protocol.

A recording made with `node reference/record-battles.cjs --views` carries, at every decision point
(initial and after each step), the observation of both players as reference/player-view.cjs builds
it from that player's own request and protocol lines only. This replays every battle through the
engine with the recorded forced outcomes, observes both sides at every decision point and compares
every leaf: integers and flags exactly, floats to 1e-6.

Run `python -m pokejax.obs_diff PATH` (plain JSONL or gzip). Step numbers in reports are zero-based;
step -1 is the initial decision point, and the side is the player the observation belongs to.
"""

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.diff import load_recording, replay_histories
from pokejax.engine.observation import observe
from pokejax.tables import load_tables

FLOAT_TOLERANCE = 1e-6
# Own leaves that the player cannot know at the last decision point, which has no request: HP
# of members that left the field in the last step (Regenerator's heal prints nothing) and the
# stats after a forme change in it (stats come only with a request).
TERMINAL_UNKNOWN = ("own.hp", "own.stats")  # hp: members off the field; stats: all
# The opponent's roster arrays, which Illusion makes ambiguous (see ambiguous_from).
ROSTER_FIELDS = tuple(
	f"opponent.{name}"
	for name in (
		"species",
		"types",
		"moves",
		"level",
		"hp",
		"status",
		"fainted",
		"is_active",
		"unseen",
		"ability",
		"item",
		"item_revealed",
		"item_consumed",
		"item_knocked_off",
	)
)


def observe_both(states, tables):
	"""Observation of side 0 and side 1 for a batch of states: leaves are [batch, 2, ...]."""
	both = jax.vmap(observe, in_axes=(None, 0, None))
	return jax.vmap(lambda state: both(state, jnp.arange(2), tables))(states)


def flatten(value, prefix=""):
	"""Leaves of nested NamedTuples (engine) or dicts (JSON) by dotted field path."""
	if hasattr(value, "_fields"):
		value = dict(zip(value._fields, value, strict=True))
	if isinstance(value, dict):
		out = {}
		for name, child in value.items():
			out.update(flatten(child, f"{prefix}{name}."))
		return out
	return {prefix[:-1]: value}


def node_history(battle):
	"""The player view observations of a battle: {path: array [decision points, 2, ...]}."""
	points = [battle["initialViews"]] + [record["views"] for record in battle["steps"]]
	flat = [[flatten(view) for view in point] for point in points]
	return {
		path: np.asarray([[view[path] for view in point] for point in flat], dtype=np.float64)
		for path in flat[0][0]
	}


def ambiguous_from(battle, side):
	"""First decision point from which the opponent's roster is ambiguous to `side`, or None.

	An Illusion holder enters under another member's name and details. The protocol cannot say
	whether a later entry of that name is the disguised holder or the member it copies (it names
	the holder only when the disguise breaks, and a holder can change disguise between entries),
	so which roster entry a switch-in belongs to is not a function of what the player received.
	A client merges entries by name; the engine keeps its fixed slots. From the first decision
	point at which the holder is on the field, the roster arrays are therefore not compared.
	"""
	foe = 1 - side
	holders = {
		slot for slot, member in enumerate(battle["teams"][foe]) if member["ability"] == "illusion"
	}
	actives = [battle["initialActive"]] + [record["active"] for record in battle["steps"]]
	for point, active in enumerate(actives):
		if active[foe] in holders:
			return point
	return None


@dataclass(frozen=True)
class Mismatch:
	battle: int
	step: int  # -1 is the initial decision point
	side: int
	path: str
	engine: object
	node: object

	def __str__(self):
		point = "initial" if self.step == -1 else f"step {self.step}"
		return (
			f"Battle {self.battle}, {point}, side {self.side}, {self.path}: "
			f"engine={self.engine!r}, player view={self.node!r}"
		)


@dataclass(frozen=True)
class ObservationResult:
	battles: int
	points: int
	mismatches: tuple[Mismatch, ...]
	by_path: Counter
	skipped: Counter  # differing roster leaves left out as ambiguous (see ambiguous_from)
	seconds: float

	def report(self, limit=40):
		lines = [str(mismatch) for mismatch in self.mismatches[:limit]]
		if len(self.mismatches) > limit:
			lines.append(f"... {len(self.mismatches) - limit} more")
		lines += [f"  {count:7d}  {path}" for path, count in self.by_path.most_common()]
		if self.skipped:
			lines.append(
				f"Not compared, roster ambiguous under Illusion: {sum(self.skipped.values())} leaves "
				f"that differ ({dict(self.skipped)})"
			)
		lines.append(
			f"{len(self.mismatches)} mismatched leaves over {self.battles} battles, "
			f"{self.points} decision points x 2 sides; {self.seconds:.1f}s (including compilation)"
		)
		return "\n".join(lines)


def engine_history(header, battles):
	"""The engine observations of both sides at every decision point: [battles, points, 2, ...]."""
	if "playerViews" not in header:
		raise ValueError("Recording has no player views; record it with --views")
	actual, lengths = replay_histories(header, battles, load_tables(), project=observe_both)
	return flatten(actual), lengths


def compare_file(path, *, recording=None, history=None, keep=100000) -> ObservationResult:
	"""Replay every battle of a --views recording and compare all observation leaves.

	history is a precomputed engine_history of the same recording.
	"""
	started = perf_counter()
	header, battles = load_recording(path) if recording is None else recording
	engine, lengths = engine_history(header, battles) if history is None else history
	mismatches = []
	by_path = Counter()
	points = 0
	skipped = Counter()
	for b, battle in enumerate(battles):
		expected = node_history(battle)
		count = int(lengths[b]) + 1
		points += count
		ambiguous = [ambiguous_from(battle, side) for side in range(2)]
		if expected.keys() != engine.keys():
			raise ValueError(f"Field mismatch: {sorted(expected.keys() ^ engine.keys())}")
		for name, wanted in expected.items():
			leaf = engine[name][b, :count]
			got = leaf.astype(np.float64)
			if wanted.shape != got.shape:
				raise ValueError(f"{name}: engine {got.shape}, player view {wanted.shape}")
			if leaf.dtype.kind == "f":
				different = np.abs(got - wanted) > FLOAT_TOLERANCE
			else:
				different = got != wanted
			if name == "own.stats":
				different[count - 1] = False
			elif name in TERMINAL_UNKNOWN:
				different[count - 1] &= engine["own.is_active"][b, count - 1] != 0
			if name in ROSTER_FIELDS:
				for side, start in enumerate(ambiguous):
					if start is not None:
						skipped[name] += int(different[start:, side].sum())
						different[start:, side] = False
			for address in np.argwhere(different):
				point, side, *axes = (int(i) for i in address)
				index = tuple(axes)
				by_path[name] += 1
				if len(mismatches) < keep:
					where = "".join(f"[{i}]" for i in index)
					mismatches.append(
						Mismatch(
							battle["index"],
							point - 1,
							side,
							name + where,
							got[(point, side, *index)].item(),
							wanted[(point, side, *index)].item(),
						)
					)
	mismatches.sort(key=lambda m: (m.battle, m.step, m.side, m.path))
	return ObservationResult(
		len(battles), points, tuple(mismatches), by_path, skipped, perf_counter() - started
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
