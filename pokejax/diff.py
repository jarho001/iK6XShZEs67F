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
	AUTOTOMIZE,
	CHOICE_MOVE,
	CONFUSION,
	CURSE,
	DESTINY_BOND,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	ENCORE_MOVE,
	ENCORE_REMAINING,
	ENDURE,
	FLASH_FIRE,
	FLINCH,
	HP,
	ILLUSION_SLOT,
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
	SLOW_START,
	SMACKDOWN,
	STALL,
	STATUS_NAMES,
	SUBSTITUTE,
	SWITCH,
	TAUNT,
	TIE,
	TRANSFORMED,
	TURN_IDLE,
	TWO_TURN_MOVE,
	UNBURDEN,
	VOLATILE_NAMES,
	WEATHER_NAMES,
	WIDE_GUARD,
	YAWN,
)
from pokejax.engine.state import (
	TRUANT as TRUANT_VOLATILE,
)
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import FORMAT, MOVE_SLOTS, team_pool_from_data

KEY = jax.random.key(0)
ROOT = Path(__file__).resolve().parents[1]
REQUESTS = {"none": NONE, "move": MOVE, "switch": SWITCH}
NULL_TYPE = -3
# Volatile names compared as presence flags, in the order of comparable()["volatiles"].
VOLATILE_KEYS = VOLATILE_NAMES + ("substitute", "encore", "taunt", "yawn", "destinybond")
# Volatile counters that the recorder reports for these volatiles and that are compared above.
COUNTER_KEYS = ("confusion", "disable", "slowstart", "substitute", "encore", "taunt", "yawn")
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
	choice_lock = state.volatiles[..., CHOICE_MOVE] > 0
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
		"itemConsumed": state.item_consumed,
		"itemKnockedOff": state.item_knocked_off,
		"ability": state.ability,
		"abilityPublic": state.ability_public,
		"ended": state.winner != ONGOING,
		"species": state.species,
		"shownSpecies": state.shown_species,
		"transformed": state.volatiles[..., TRANSFORMED] > 0,
		"illusion": _disguise_species(state, tables),
		"boosts": state.boosts,
		"volatiles": jnp.concatenate(
			(
				base_volatiles,
				charge_volatiles,
				smackdown[..., None],
				choice_lock[..., None],
				state.volatiles[..., FLASH_FIRE, None] > 0,
				state.volatiles[..., UNBURDEN, None] > 0,
				state.volatiles[..., TRUANT_VOLATILE, None] > 0,
				state.volatiles[..., DISABLE_REMAINING, None] > 0,
				state.volatiles[..., CURSE, None] > 0,
				state.volatiles[..., AUTOTOMIZE, None] > 0,
				state.volatiles[..., SUBSTITUTE, None] > 0,
				state.volatiles[..., ENCORE_REMAINING, None] > 0,
				state.volatiles[..., TAUNT, None] > 0,
				state.volatiles[..., YAWN, None] > 0,
				state.volatiles[..., DESTINY_BOND, None] > 0,
			),
			axis=-1,
		),
		"sideConditions": jnp.stack(
			(
				state.volatiles[..., QUICK_GUARD] > 0,
				state.volatiles[..., WIDE_GUARD] > 0,
			),
			axis=-1,
		),
		"confusionCounter": state.volatiles[..., CONFUSION],
		"disableMove": state.volatiles[..., DISABLE_MOVE],
		"disableCounter": state.volatiles[..., DISABLE_REMAINING],
		"substituteHp": state.volatiles[..., SUBSTITUTE],
		"encoreMove": state.volatiles[..., ENCORE_MOVE],
		"encoreCounter": state.volatiles[..., ENCORE_REMAINING],
		"tauntCounter": state.volatiles[..., TAUNT],
		"yawnCounter": state.volatiles[..., YAWN],
		"wishTurns": state.wish_turns,
		"wishHp": state.wish_hp,
		"healingWish": state.healing_wish,
		"trickRoom": state.trick_room,
		"slowStartCounter": state.volatiles[..., SLOW_START],
		"request": state.request,
		"midTurn": state.turn_stage > TURN_IDLE,
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


def _disguise_species(state, tables):
	"""Species of each active's Illusion disguise (zero when none is shown)."""
	slot = state.volatiles[..., ILLUSION_SLOT].astype(jnp.int32) - 1
	picked = jnp.take_along_axis(
		state.species, jnp.clip(slot, 0, state.species.shape[-1] - 1)[..., None], axis=-1
	)[..., 0]
	return jnp.where(slot >= 0, picked, 0)


def recorded_state(snapshot, tables, initial_species=None):
	"""Convert oracle snapshots without deriving fainted or any compared value.

	Schema-v2 files written before task 14 have no species, shownSpecies, transformed or
	illusion fields: species then defaults to the team's starting species, transformed and
	illusion to none, and shownSpecies is not compared at all.
	"""
	sides = snapshot["sides"]
	members = [side["members"] for side in sides]
	winner = snapshot["winner"]
	species_known = all("species" in member for team in members for member in team)
	shown_known = all("shownSpecies" in member for team in members for member in team)
	extra = {}
	if species_known:
		extra["species"] = np.asarray(
			[[tables.species.index.get(m["species"], -1) for m in team] for team in members]
		)
	elif initial_species is not None:
		extra["species"] = initial_species
	if shown_known:
		extra["shownSpecies"] = np.asarray(
			[[tables.species.index.get(m["shownSpecies"], -1) for m in team] for team in members]
		)
	extra["transformed"] = np.asarray([bool(side.get("transformed", False)) for side in sides])
	extra["illusion"] = np.asarray(
		[tables.species.index.get(side.get("illusion") or "", 0) for side in sides]
	)
	return {
		**extra,
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
		"itemConsumed": np.asarray(
			[[member.get("itemConsumed", False) for member in team] for team in members]
		),
		"itemKnockedOff": np.asarray(
			[[member.get("itemKnockedOff", False) for member in team] for team in members]
		),
		"ability": np.asarray(
			[[tables.abilities.index.get(m["ability"], -1) for m in team] for team in members]
		),
		"abilityPublic": np.asarray(
			[
				[tables.abilities.index.get(m.get("abilityPublic", ""), -1) for m in team]
				for team in members
			]
		),
		"ended": np.asarray(snapshot["ended"]),
		"boosts": np.asarray([side["boosts"] for side in sides]),
		"volatiles": np.asarray(
			[[name in (side["volatiles"] or []) for name in VOLATILE_KEYS] for side in sides]
		),
		"substituteHp": np.asarray(
			[side.get("volatileCounters", {}).get("substitute", 0) for side in sides]
		),
		"encoreMove": np.asarray(
			[
				tables.moves.index.get(side.get("volatileMoves", {}).get("encore", ""), 0)
				for side in sides
			]
		),
		"encoreCounter": np.asarray(
			[side.get("volatileCounters", {}).get("encore", 0) for side in sides]
		),
		"tauntCounter": np.asarray(
			[side.get("volatileCounters", {}).get("taunt", 0) for side in sides]
		),
		"yawnCounter": np.asarray(
			[side.get("volatileCounters", {}).get("yawn", 0) for side in sides]
		),
		"wishTurns": np.asarray(
			[side.get("slotConditions", {}).get("wish", {}).get("turns", 0) for side in sides]
		),
		"wishHp": np.asarray(
			[side.get("slotConditions", {}).get("wish", {}).get("hp", 0) for side in sides]
		),
		"healingWish": np.asarray(
			["healingwish" in side.get("slotConditions", {}) for side in sides]
		),
		"trickRoom": np.asarray(snapshot["pseudoWeather"].get("trickroom", {}).get("turns", 0)),
		"sideConditions": np.asarray(
			[
				[name in side["sideConditions"] for name in ("quickguard", "wideguard")]
				for side in sides
			]
		),
		"confusionCounter": np.asarray(
			[side.get("volatileCounters", {}).get("confusion", 0) for side in sides]
		),
		"disableCounter": np.asarray(
			[side.get("volatileCounters", {}).get("disable", 0) for side in sides]
		),
		"slowStartCounter": np.asarray(
			[side.get("volatileCounters", {}).get("slowstart", 0) for side in sides]
		),
		"disableMove": np.asarray(
			[
				tables.moves.index.get(side.get("volatileMoves", {}).get("disable", ""), 0)
				for side in sides
			]
		),
		"request": np.asarray([REQUESTS[side["request"]] for side in sides]),
		"midTurn": np.asarray(any(side["request"] == "switch" for side in sides)),
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
def _initial(teams, tables, forced):
	states = jax.vmap(reset, in_axes=(None, 0, None, 0))(KEY, teams, tables, forced)
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
	initial_forced = np.stack(
		[forced_from_records(battle.get("initialOutcomes", []), host=True) for battle in battles]
	)
	parts = []
	for start in range(0, len(battles), BATTLE_CHUNK):
		end = min(start + BATTLE_CHUNK, len(battles))
		indices = np.minimum(np.arange(start, start + BATTLE_CHUNK), end - 1)
		chunk_teams = jax.tree.map(lambda x, indices=indices: x[indices], teams)
		states, initial = _initial(chunk_teams, engine_tables, initial_forced[indices])
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
		initial_species = (
			np.asarray(
				[
					[tables.species.index.get(m["species"], 0) for m in team]
					for team in battle["teams"]
				]
			)
			if "species" in battle["teams"][0][0]
			else None
		)
		expected = [recorded_state(snapshot, tables, initial_species) for snapshot in snapshots]
		# Check padding too: it must keep the last recorded state, even after a win.
		expected += [expected[-1]] * (shape[1] + 1 - len(expected))
		fields = {
			name: np.stack([state[name] for state in expected])
			for name in actual
			if name in expected[0]
		}
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
					{k: v for k, v in snapshot["pseudoWeather"].items() if k != "trickroom"},
				),
			}
			for side_index, side in enumerate(snapshot["sides"]):
				member = side["members"][side["active"]]
				for key in ("status", "item", "ability"):
					extra[f"sides[{side_index}].{key}"] = (member[key], side[key])
				if side["volatiles"] is not None:
					extra[f"sides[{side_index}].volatiles"] = (
						[],
						[name for name in side["volatiles"] if name not in VOLATILE_KEYS],
					)
				extra[f"sides[{side_index}].volatileCounters"] = (
					{},
					{
						name: value
						for name, value in side.get("volatileCounters", {}).items()
						if name not in COUNTER_KEYS
					},
				)
				extra[f"sides[{side_index}].volatileMoves"] = (
					{},
					{
						name: value
						for name, value in side.get("volatileMoves", {}).items()
						if name not in ("disable", "encore")
					},
				)
				extra[f"sides[{side_index}].slotConditions"] = (
					{},
					{
						name: condition
						for name, condition in side.get("slotConditions", {}).items()
						if name not in ("wish", "healingwish")
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
