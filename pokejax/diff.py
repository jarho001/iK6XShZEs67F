"""Replay decision-point JSONL recordings against the Phase 1 engine.

Run ``python -m pokejax.diff PATH`` (plain JSONL or gzip). A reusable compiled
scan replays fixed-size chunks; host comparison reports the first mismatch per battle.
Step numbers in reports are zero-based; ``initial`` precedes every action.
"""

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import legal_actions
from pokejax.engine.data import build_engine_tables
from pokejax.engine.randomness import UNFORCED, forced_from_records
from pokejax.engine.reset import check_teams, reset, teams_from_pool
from pokejax.engine.state import (
	CONFUSION,
	ENDURE,
	FLINCH,
	HP,
	LEECH_SEED,
	LOCKED_MOVE,
	MOVE,
	MUST_RECHARGE,
	NONE,
	ONGOING,
	PARTIALLY_TRAPPED,
	PROTECT,
	QUICK_GUARD,
	ROOST,
	SMACKDOWN,
	STALL,
	STATUS_NAMES,
	SWITCH,
	TIE,
	TWO_TURN_MOVE,
	VOLATILE_NAMES,
	WEATHER_NAMES,
	WIDE_GUARD,
)
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import FORMAT, MOVE_SLOTS, team_pool_from_data

KEY = jax.random.key(0)
ROOT = Path(__file__).resolve().parents[1]
REQUESTS = {"none": NONE, "move": MOVE, "switch": SWITCH}
NULL_TYPE = -3
BATTLE_CHUNK = 32
STEP_CHUNK = 32


def comparable(state, tables):
	"""Map engine state to numeric snapshot fields; supports batches."""
	# Compare effective current types. The saved Roost restoration copy is internal state.
	base_volatiles = jnp.stack(
		(
			state.volatiles[..., CONFUSION] > 0,
			state.volatiles[..., FLINCH] > 0,
			state.volatiles[..., ROOST] > 0,
			state.volatiles[..., LEECH_SEED] > 0,
			state.volatiles[..., PARTIALLY_TRAPPED] > 0,
			state.volatiles[..., PROTECT] > 0,
			state.volatiles[..., ENDURE] > 0,
			state.volatiles[..., STALL] > 0,
			state.volatiles[..., TWO_TURN_MOVE] != 0,
			state.volatiles[..., LOCKED_MOVE] > 0,
			state.volatiles[..., MUST_RECHARGE] > 0,
		),
		axis=-1,
	)
	charge_volatiles = state.volatiles[..., TWO_TURN_MOVE, None] == tables.move_turn_volatile_ids
	smackdown = state.volatiles[..., SMACKDOWN] > 0
	return {
		"active": state.active,
		"legalActions": jax.vmap(legal_actions, in_axes=(0, None))(state, tables),
		"status": state.status,
		"statusCounter": state.status_counter[
			jnp.arange(state.active.shape[0])[:, None], jnp.arange(2), state.active
		],
		"types": state.types[
			jnp.arange(state.active.shape[0])[:, None], jnp.arange(2), state.active
		],
		"item": state.item,
		"ability": state.ability,
		"ended": state.winner != ONGOING,
		"boosts": state.boosts,
		"volatiles": jnp.concatenate(
			(base_volatiles, charge_volatiles, smackdown[..., None]), axis=-1
		),
		"sideConditions": jnp.stack(
			(
				state.volatiles[..., QUICK_GUARD] > 0,
				state.volatiles[..., WIDE_GUARD] > 0,
			),
			axis=-1,
		),
		"confusionCounter": state.volatiles[..., CONFUSION],
		"request": state.request,
		"hp": state.hp,
		"maxhp": state.stats[..., HP],
		"pp": state.pp,
		"fainted": state.hp == 0,
		"turn": state.turn,
		"winner": state.winner,
		"weather": state.weather,
		"weatherTurns": state.weather_turns,
		"hazards": state.hazards,
	}


def recorded_state(snapshot, tables):
	"""Convert oracle snapshots without deriving fainted or any compared value."""
	sides = snapshot["sides"]
	members = [side["members"] for side in sides]
	winner = snapshot["winner"]
	return {
		"active": np.asarray([side["active"] for side in sides]),
		"legalActions": np.asarray([side["legalActions"] for side in sides]),
		"status": np.asarray(
			[
				[
					STATUS_NAMES.index(m["status"]) if m["status"] in STATUS_NAMES else -1
					for m in team
				]
				for team in members
			]
		),
		"statusCounter": np.asarray([side["statusCounter"] for side in sides]),
		"types": np.asarray(
			[
				(
					[NULL_TYPE, NULL_TYPE]
					if side["types"] is None
					else [tables.types.index.get(t, -2) for t in side["types"]]
					+ [-1] * (2 - len(side["types"]))
				)
				for side in sides
			]
		),
		"item": np.asarray(
			[
				[tables.items.index.get(m["item"], -1) if m["item"] else 0 for m in team]
				for team in members
			]
		),
		"ability": np.asarray(
			[[tables.abilities.index.get(m["ability"], -1) for m in team] for team in members]
		),
		"ended": np.asarray(snapshot["ended"]),
		"boosts": np.asarray([side["boosts"] for side in sides]),
		"volatiles": np.asarray(
			[[name in (side["volatiles"] or []) for name in VOLATILE_NAMES] for side in sides]
		),
		"sideConditions": np.asarray(
			[
				[name in side["sideConditions"] for name in ("quickguard", "wideguard")]
				for side in sides
			]
		),
		"confusionCounter": np.asarray(
			[side.get("volatileCounters", {}).get("confusion", 0) for side in sides]
		),
		"request": np.asarray([REQUESTS[side["request"]] for side in sides]),
		**{
			name: np.asarray([[member[name] for member in team] for team in members])
			for name in ("hp", "maxhp", "fainted")
		},
		"pp": np.asarray(
			[
				[member["pp"] + [0] * (MOVE_SLOTS - len(member["pp"])) for member in team]
				for team in members
			]
		),
		"turn": np.asarray(snapshot["turn"]),
		"winner": np.asarray(ONGOING if winner is None else TIE if winner == "tie" else winner),
		"weather": np.asarray(WEATHER_NAMES.index(snapshot["weather"])),
		"weatherTurns": np.asarray(snapshot["weatherTurns"]),
		"hazards": np.asarray(
			[
				[
					side["sideConditions"].get(name, {}).get("layers", 0)
					for name in ("spikes", "toxicspikes")
				]
				for side in sides
			]
		),
	}


@jax.jit
def _initial(teams, tables):
	states = jax.vmap(reset, in_axes=(None, 0, None))(KEY, teams, tables)
	return states, comparable(states, tables)


@jax.jit
def _replay(states, actions, forced, valid, offset, tables):

	def advance(states, inputs):
		actions, forced, valid, index = inputs
		out = jax.vmap(step, in_axes=(0, 0, None, 0, None))(
			states, actions, jax.random.fold_in(KEY, index), forced, tables
		)
		# Even a turn-limited recording can be padded without advancing its state.
		out = jax.tree.map(
			lambda new, old: jnp.where(valid.reshape((-1,) + (1,) * (new.ndim - 1)), new, old),
			out,
			states,
		)
		return out, comparable(out, tables)

	states, history = jax.lax.scan(
		advance,
		states,
		(
			jnp.swapaxes(actions, 0, 1),
			jnp.swapaxes(forced, 0, 1),
			jnp.swapaxes(valid, 0, 1),
			jnp.arange(actions.shape[1]) + offset,
		),
	)
	return states, jax.tree.map(lambda rest: jnp.swapaxes(rest, 0, 1), history)


@dataclass(frozen=True)
class Mismatch:
	battle: int
	step: int  # -1 is initial state
	field: str
	engine: object
	showdown: object
	actions: object
	outcomes: object

	def __str__(self):
		point = "initial" if self.step == -1 else f"step {self.step}"
		return (
			f"Battle {self.battle}, {point}, {self.field}: "
			f"engine={self.engine!r}, Showdown={self.showdown!r}\n"
			f"  actions={self.actions}, outcomes={self.outcomes}"
		)


@dataclass(frozen=True)
class ReplayResult:
	battles: int
	steps: int
	mismatches: tuple[Mismatch, ...]
	seconds: float

	def report(self):
		return "\n".join(
			[str(mismatch) for mismatch in self.mismatches]
			+ [
				(
					f"{self.battles - len(self.mismatches)}/{self.battles} battles match; "
					f"{self.steps} decisions; {self.seconds:.3f}s (including compilation)"
				)
			]
		)


def load_recording(path):
	"""Reject incompatible or incomplete recordings before compiling anything."""
	path = Path(path)
	opener = gzip.open if path.suffix == ".gz" else open
	with opener(path, "rt", encoding="utf-8") as source:
		header, *battles = [json.loads(line) for line in source if line.strip()]
	if header["type"] != "header" or header["schemaVersion"] != 2:
		raise ValueError("Expected differential JSONL schema v2 header")
	lock = json.loads((ROOT / "reference-lock.json").read_text())
	if header["showdownCommit"] != lock["commit"] or header["format"] != FORMAT:
		raise ValueError("Recording does not use the pinned Showdown commit and format")
	if not battles or len(battles) != header["count"]:
		raise ValueError("Battle count does not match header")
	if len({battle["index"] for battle in battles}) != len(battles):
		raise ValueError("Duplicate battle indices")
	for battle in battles:
		if battle["type"] != "battle":
			raise ValueError("Expected a battle record")
		if battle["initialOverrides"] != []:
			raise ValueError(
				f"Battle {battle['index']}: unknown initialOverrides: {battle['initialOverrides']!r}"
			)
	return header, battles


def replay_file(path, *, recording=None) -> ReplayResult:
	"""Replay ALL battles in fixed chunks, then compare exactly on the host."""
	started = perf_counter()
	header, battles = load_recording(path) if recording is None else recording
	tables = load_tables()
	pool = team_pool_from_data(
		dict(
			header,
			teams=[team for battle in battles for team in battle["teams"]],
			count=2 * len(battles),
		),
		tables,
	)
	indices = np.arange(len(battles)) * 2
	teams = teams_from_pool(pool, indices, indices + 1)
	check_teams(teams, tables)
	lengths = np.asarray([len(battle["steps"]) for battle in battles])
	shape = (len(battles), int(lengths.max()))
	actions = np.zeros((*shape, 2), np.int32)
	forced = np.full((*shape, *UNFORCED.shape), -1, np.int8)
	for b, battle in enumerate(battles):
		for t, record in enumerate(battle["steps"]):
			choice = np.asarray(record["actions"])
			if (
				choice.shape != (2,)
				or choice.dtype.kind not in "iu"
				or np.any((choice < 0) | (choice > 8))
			):
				raise ValueError(f"Battle {battle['index']} step {t}: invalid actions")
			actions[b, t] = choice
			forced[b, t] = forced_from_records(record["outcomes"], host=True)
	valid = np.arange(shape[1])[None] < lengths[:, None]
	engine_tables = build_engine_tables(tables)
	parts = []
	for start in range(0, len(battles), BATTLE_CHUNK):
		end = min(start + BATTLE_CHUNK, len(battles))
		indices = np.minimum(np.arange(start, start + BATTLE_CHUNK), end - 1)
		chunk_teams = jax.tree.map(lambda x, indices=indices: x[indices], teams)
		states, initial = _initial(chunk_teams, engine_tables)
		histories = [initial]
		for offset in range(0, shape[1], STEP_CHUNK):
			stop = min(offset + STEP_CHUNK, shape[1])
			chunk_actions = np.zeros((BATTLE_CHUNK, STEP_CHUNK, 2), np.int32)
			chunk_forced = np.full((BATTLE_CHUNK, STEP_CHUNK, *UNFORCED.shape), -1, np.int8)
			chunk_valid = np.zeros((BATTLE_CHUNK, STEP_CHUNK), bool)
			n = stop - offset
			chunk_actions[:, :n] = actions[indices, offset:stop]
			chunk_forced[:, :n] = forced[indices, offset:stop]
			chunk_valid[: end - start, :n] = valid[indices[: end - start], offset:stop]
			states, history = _replay(
				states, chunk_actions, chunk_forced, chunk_valid, np.int32(offset), engine_tables
			)
			histories.append(history)
		host_history = jax.device_get(histories)
		parts.append(
			jax.tree.map(
				lambda first, *rest, count=end - start: np.concatenate(
					[first[:, None], *rest], axis=1
				)[:count, : shape[1] + 1],
				*host_history,
			)
		)
	actual = jax.tree.map(lambda *xs: np.concatenate(xs, axis=0), *parts)
	mismatches = []
	for b, battle in enumerate(battles):
		snapshots = [battle["initialState"]] + [record["state"] for record in battle["steps"]]
		expected = [recorded_state(snapshot, tables) for snapshot in snapshots]
		# Check padding too: it must keep the last recorded state, even after a win.
		expected += [expected[-1]] * (shape[1] + 1 - len(expected))
		fields = {name: np.stack([state[name] for state in expected]) for name in actual}
		first = None
		for name, wanted in fields.items():
			different = actual[name][b] != wanted
			if name == "types":
				different &= wanted != NULL_TYPE
			where = np.argwhere(different)
			if where.size:
				address = tuple(where[0])
				point, *axes = address
				if first is None or point < first[0]:
					first = (point, name, address, axes)
		# Unmodeled fields retain explicit defaults; hazards are represented by layers,
		# while all other side conditions stay required to be absent.
		# Active aliases are checked independently so no snapshot field is ignored.
		for point, snapshot in enumerate(snapshots):
			extra = {
				"pseudoWeather": (
					{"sleepclausemod": {"layers": 0, "turns": 0}},
					snapshot["pseudoWeather"],
				),
			}
			for side_index, side in enumerate(snapshot["sides"]):
				member = side["members"][side["active"]]
				for key in ("status", "item", "ability"):
					extra[f"sides[{side_index}].{key}"] = (member[key], side[key])
				if side["volatiles"] is not None:
					extra[f"sides[{side_index}].volatiles"] = (
						[],
						[name for name in side["volatiles"] if name not in VOLATILE_NAMES],
					)
				extra[f"sides[{side_index}].volatileCounters"] = (
					{},
					{
						name: value
						for name, value in side.get("volatileCounters", {}).items()
						if name != "confusion"
					},
				)
				extra[f"sides[{side_index}].sideConditions"] = (
					{},
					{
						name: condition
						for name, condition in side["sideConditions"].items()
						if name not in ("spikes", "toxicspikes", "quickguard", "wideguard")
					},
				)
			for name, (engine, wanted) in extra.items():
				if engine != wanted and (first is None or point < first[0]):
					first = (point, name, None, (engine, wanted))
		if first is not None:
			point, name, address, axes = first
			record = battle["steps"][point - 1] if 0 < point <= lengths[b] else None
			mismatches.append(
				Mismatch(
					battle["index"],
					int(point) - 1,
					name + ("".join(f"[{i}]" for i in axes) if address is not None else ""),
					actual[name][b][address].item() if address is not None else axes[0],
					fields[name][address].item() if address is not None else axes[1],
					record["actions"] if record else None,
					record["outcomes"] if record else None,
				)
			)
	return ReplayResult(
		len(battles), int(lengths.sum()), tuple(mismatches), perf_counter() - started
	)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("path", type=Path)
	args = parser.parse_args()
	result = replay_file(args.path)
	print(result.report())
	raise SystemExit(bool(result.mismatches))


if __name__ == "__main__":
	main()
