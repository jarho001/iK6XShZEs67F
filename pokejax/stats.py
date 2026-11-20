"""Compare whole-battle random-policy distributions from Showdown and PokeJAX.

Counting contract: a move use is a selected legal move (even if a status
prevents execution); switches are selected switch actions, including forced
replacements. A status or confusion is inflicted when first visible at a
decision boundary, and a faint is counted when a member first becomes fainted.
Both simulators emit those same decision-boundary facts; this module applies
the counts once to both sets of battle traces.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import legal_actions, move_availability
from pokejax.engine.data import build_engine_tables
from pokejax.engine.items import showdown_item_consumed
from pokejax.engine.reset import check_teams, reset, teams_from_pool
from pokejax.engine.state import (
	BURN,
	CONFUSION,
	FREEZE,
	LOCKED_MOVE,
	MOVE,
	MUST_RECHARGE,
	ONGOING,
	PARALYSIS,
	POISON,
	SLEEP,
	SWITCH,
	TIE,
	TOXIC,
	TWO_TURN_MOVE,
)
from pokejax.engine.step import step
from pokejax.tables import load_tables
from pokejax.teams import FORMAT, MOVE_SLOTS, team_pool_from_data

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "work/teams/gen5-teams.json"
DEFAULT_STATS_DIR = ROOT / "work/stats"
DEFAULT_BATTLES = 4000
BATTLE_BATCH = 128
MAX_DECISIONS = 3000
STATUS_CODES = (BURN, POISON, TOXIC, PARALYSIS, SLEEP, FREEZE)
STATUS_NAMES = ("brn", "psn", "tox", "par", "slp", "frz")


def _team_pairs_hash(team_pairs):
	encoded = json.dumps(team_pairs, sort_keys=True, separators=(",", ":")).encode()
	return hashlib.sha256(encoded).hexdigest()


def count_battle_summary(battle):
	"""Apply the shared counting rules to a battle's raw decision trace."""
	counts = {
		"winner": "tie" if battle["winner"] in ("tie", TIE) else int(battle["winner"]),
		"turns": int(battle["turns"]),
		"moves": {},
		"statuses": dict.fromkeys(STATUS_NAMES, 0),
		"faints": 0,
		"switches": [0, 0],
		"struggle": 0,
		"confusion": 0,
		"itemsConsumed": {},
	}
	previous_status = np.full((2, 6), "", dtype=object)
	previous_fainted = np.zeros((2, 6), dtype=np.bool_)
	previous_confusion = np.zeros(2, dtype=np.bool_)
	previous_item_consumed = np.zeros((2, 6), dtype=np.bool_)
	team_pair = battle.get("teamPair", [[], []])
	for event in battle["decisions"]:
		moves = event["moves"]
		switches = np.asarray(event["switches"], dtype=np.bool_)
		statuses = np.asarray(event["statuses"], dtype=object).reshape(2, 6)
		fainted = np.asarray(event["fainted"], dtype=np.bool_).reshape(2, 6)
		confusion = np.asarray(event["confusion"], dtype=np.bool_)
		item_consumed = np.asarray(
			event.get("itemConsumed", np.zeros((2, 6), dtype=np.bool_)), dtype=np.bool_
		).reshape(2, 6)
		for move in moves:
			if move is not None:
				counts["moves"][move] = counts["moves"].get(move, 0) + 1
				if move == "struggle":
					counts["struggle"] += 1
		counts["switches"] = [
			count + int(selected) for count, selected in zip(counts["switches"], switches)
		]
		counts["faints"] += int(np.count_nonzero(~previous_fainted & fainted))
		counts["confusion"] += int(np.count_nonzero(~previous_confusion & confusion))
		for name in STATUS_NAMES:
			counts["statuses"][name] += int(
				np.count_nonzero((statuses == name) & (previous_status != name))
			)
		newly_consumed = item_consumed & ~previous_item_consumed
		for side, slot in zip(*np.nonzero(newly_consumed), strict=True):
			if side < len(team_pair) and slot < len(team_pair[side]):
				item = _item_id(team_pair[side][slot].get("item", ""))
				if item:
					counts["itemsConsumed"][item] = counts["itemsConsumed"].get(item, 0) + 1
		previous_status = statuses
		previous_fainted = fainted
		previous_confusion = confusion
		previous_item_consumed = item_consumed
	return counts


def _item_id(value):
	"""Normalize a Showdown display item name to its engine table ID."""
	return "".join(character.lower() for character in value if character.isalnum())


def _rollout_batch(teams, policy_keys, engine_tables):
	"""Run a batch and return raw decision traces for shared host-side counting."""
	batch_size = teams.species.shape[0]
	states = jax.vmap(reset, in_axes=(0, 0, None))(policy_keys, teams, engine_tables)
	trace_moves = jnp.full((MAX_DECISIONS, batch_size, 2), -1, dtype=jnp.int16)
	trace_switches = jnp.zeros((MAX_DECISIONS, batch_size, 2), dtype=jnp.bool_)
	trace_statuses = jnp.zeros((MAX_DECISIONS, batch_size, 2, 6), dtype=jnp.int8)
	trace_fainted = jnp.zeros((MAX_DECISIONS, batch_size, 2, 6), dtype=jnp.bool_)
	trace_confusion = jnp.zeros((MAX_DECISIONS, batch_size, 2), dtype=jnp.bool_)
	trace_item_consumed = jnp.zeros((MAX_DECISIONS, batch_size, 2, 6), dtype=jnp.bool_)
	lengths = jnp.zeros((batch_size,), dtype=jnp.int32)
	indices = jnp.arange(batch_size)
	sides = jnp.arange(2)[None, :]

	def condition(carry):
		state, _, decision, *_ = carry
		return (decision < MAX_DECISIONS) & jnp.any(state.winner == ONGOING)

	def advance(carry):
		(
			state,
			keys,
			decision,
			lengths,
			moves_trace,
			switches_trace,
			statuses_trace,
			fainted_trace,
			confusion_trace,
			item_consumed_trace,
		) = carry
		was_ongoing = state.winner == ONGOING
		mask = jax.vmap(legal_actions, in_axes=(0, None))(state, engine_tables)
		key_parts = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
		next_keys = key_parts[:, 0]
		move_policy_keys = jax.vmap(lambda key: jax.random.split(key, 2))(key_parts[:, 1])
		step_keys = key_parts[:, 2]
		logits = jnp.where(mask, 0.0, -jnp.inf)
		actions = jax.vmap(
			lambda legal, action_keys: jax.vmap(
				lambda player_legal, action_key: jax.random.categorical(
					action_key, jnp.where(player_legal, 0.0, -jnp.inf)
				)
			)(legal, action_keys)
		)(logits == 0.0, move_policy_keys).astype(jnp.int32)
		# The trace names the move the step executes: Struggle when no move is usable (PP,
		# Disable, Taunt, Encore) or the Choice lock's move has no PP, as in step().
		active_moves, usable, choice_active, choice_usable = jax.vmap(
			move_availability, in_axes=(0, None)
		)(state, engine_tables)
		slot = jnp.minimum(actions, MOVE_SLOTS - 1)
		selected_moves = active_moves[indices[:, None], sides, slot]
		locked = state.volatiles[indices[:, None], sides, LOCKED_MOVE]
		charging = state.volatiles[indices[:, None], sides, TWO_TURN_MOVE]
		recharging = state.volatiles[indices[:, None], sides, MUST_RECHARGE] > 0
		forced_move = (locked > 0) | (charging > 0) | recharging
		choice_struggle = choice_active & ~jnp.any(choice_usable, axis=-1)
		struggle = (~jnp.any(usable, axis=-1) | choice_struggle) & ~forced_move
		selected_moves = jnp.where(struggle, engine_tables.struggle_id, selected_moves)
		selected_moves = jnp.where(locked > 0, locked, selected_moves)
		selected_moves = jnp.where(charging > 0, charging, selected_moves)
		selected_moves = jnp.where(recharging, -2, selected_moves)
		is_move = was_ongoing[:, None] & (state.request == MOVE) & (actions < 4)
		can_switch = (state.request == MOVE) | (state.request == SWITCH)
		is_switch = was_ongoing[:, None] & can_switch & (actions >= 4)
		moves_trace = moves_trace.at[decision].set(
			jnp.where(is_move, selected_moves, -1).astype(jnp.int16)
		)
		switches_trace = switches_trace.at[decision].set(is_switch)
		new_state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
			state, actions, step_keys, None, engine_tables
		)
		statuses = jnp.where(new_state.hp > 0, new_state.status, 0).astype(jnp.int8)
		fainted = new_state.hp == 0
		confusion = new_state.volatiles[:, :, CONFUSION] > 0
		statuses_trace = statuses_trace.at[decision].set(statuses)
		fainted_trace = fainted_trace.at[decision].set(fainted)
		confusion_trace = confusion_trace.at[decision].set(confusion)
		item_consumed_trace = item_consumed_trace.at[decision].set(
			showdown_item_consumed(new_state)
		)
		return (
			new_state,
			next_keys,
			decision + 1,
			lengths + was_ongoing.astype(jnp.int32),
			moves_trace,
			switches_trace,
			statuses_trace,
			fainted_trace,
			confusion_trace,
			item_consumed_trace,
		)

	initial = (
		states,
		policy_keys,
		jnp.asarray(0, dtype=jnp.int32),
		lengths,
		trace_moves,
		trace_switches,
		trace_statuses,
		trace_fainted,
		trace_confusion,
		trace_item_consumed,
	)
	final = jax.lax.while_loop(condition, advance, initial)
	state, _, _decisions, lengths, moves, switches, statuses, fainted, confusion, item_consumed = (
		final
	)
	return (
		state.winner,
		state.turn,
		lengths,
		moves,
		switches,
		statuses,
		fainted,
		confusion,
		item_consumed,
	)


_rollout_batch = jax.jit(_rollout_batch)


def _teams_for_pairs(team_pairs, tables):
	flat_teams = [team for pair in team_pairs for team in pair]
	pool = team_pool_from_data(
		{
			"format": FORMAT,
			"count": len(flat_teams),
			"seed": 0,
			"showdownCommit": json.loads((ROOT / "reference-lock.json").read_text())["commit"],
			"teams": flat_teams,
		},
		tables,
	)
	indices = np.arange(len(team_pairs), dtype=np.int32) * 2
	teams = teams_from_pool(pool, indices, indices + 1)
	check_teams(teams, tables)
	return teams


def engine_summaries(showdown, *, seed, tables=None):
	"""Run the engine on the exact team pairs saved by the Showdown runner."""
	team_pairs = showdown["teamPairs"]
	count = len(team_pairs)
	if count != showdown["count"]:
		raise ValueError("Showdown summary team-pair count does not match metadata")
	tables = load_tables() if tables is None else tables
	engine_tables = build_engine_tables(tables)
	teams = _teams_for_pairs(team_pairs, tables)
	move_ids = tuple(str(move) for move in tables.moves.ids)
	base_key = jax.random.key(seed & 0xFFFFFFFF)
	all_battles = []
	started = perf_counter()
	batch_size = min(BATTLE_BATCH, count)
	for start in range(0, count, BATTLE_BATCH):
		stop = min(start + BATTLE_BATCH, count)
		chunk = jax.tree.map(lambda values, start=start, stop=stop: values[start:stop], teams)
		actual_size = stop - start
		if actual_size < batch_size:
			chunk = jax.tree.map(
				lambda values, batch_size=batch_size, actual_size=actual_size: jnp.concatenate(
					(values, jnp.repeat(values[:1], batch_size - actual_size, axis=0)), axis=0
				),
				chunk,
			)
		keys = jax.vmap(lambda index: jax.random.fold_in(base_key, index))(
			jnp.arange(start, start + batch_size, dtype=jnp.uint32)
		)
		result = jax.block_until_ready(_rollout_batch(chunk, keys, engine_tables))
		winner, turns, lengths, moves, switches, statuses, fainted, confusion, item_consumed = map(
			jax.device_get, result
		)
		if np.any(np.asarray(winner[:actual_size]) == ONGOING):
			raise RuntimeError(f"Engine battle exceeded {MAX_DECISIONS} decisions")
		for local in range(actual_size):
			trace = {
				"winner": int(winner[local]),
				"turns": int(turns[local]),
				"teamPair": team_pairs[start + local],
				"decisions": [],
			}
			for decision in range(int(lengths[local])):
				trace["decisions"].append(
					{
						"moves": [
							"recharge"
							if moves[decision, local, side] == -2
							else None
							if moves[decision, local, side] < 0
							else move_ids[int(moves[decision, local, side])]
							for side in range(2)
						],
						"switches": switches[decision, local].tolist(),
						"statuses": [
							("", *STATUS_NAMES)[int(status)]
							for status in statuses[decision, local].reshape(-1)
						],
						"fainted": fainted[decision, local].reshape(-1).tolist(),
						"confusion": confusion[decision, local].tolist(),
						"itemConsumed": item_consumed[decision, local].tolist(),
					}
				)
			all_battles.append(count_battle_summary(trace))
	return {
		"schemaVersion": 1,
		"simulator": "engine",
		"format": FORMAT,
		"seed": seed,
		"count": count,
		"teamPairsHash": _team_pairs_hash(team_pairs),
		"policy": "uniform over the legal-action mask for both sides at every request",
		"seconds": perf_counter() - started,
		"battles": all_battles,
	}


def _betacf(a, b, x):
	max_iterations = 200
	epsilon = 3e-14
	floor = 1e-300
	qab = a + b
	qap = a + 1.0
	qam = a - 1.0
	c = 1.0
	d = 1.0 - qab * x / qap
	d = 1.0 / (floor if abs(d) < floor else d)
	h = d
	for iteration in range(1, max_iterations + 1):
		m2 = 2 * iteration
		term = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
		d = 1.0 + term * d
		d = floor if abs(d) < floor else d
		c = 1.0 + term / c
		c = floor if abs(c) < floor else c
		d = 1.0 / d
		h *= d * c
		term = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
		d = 1.0 + term * d
		d = floor if abs(d) < floor else d
		c = 1.0 + term / c
		c = floor if abs(c) < floor else c
		d = 1.0 / d
		delta = d * c
		h *= delta
		if abs(delta - 1.0) < epsilon:
			break
	return h


def _regularized_beta(a, b, x):
	if x <= 0:
		return 0.0
	if x >= 1:
		return 1.0
	front = math.exp(
		math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
	)
	if x < (a + 1.0) / (a + b + 2.0):
		return front * _betacf(a, b, x) / a
	return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _student_t_pvalue(statistic, degrees):
	if not math.isfinite(statistic):
		return 0.0
	if statistic == 0:
		return 1.0
	x = degrees / (degrees + statistic * statistic)
	return min(1.0, max(0.0, _regularized_beta(degrees / 2.0, 0.5, x)))


def two_proportion_test(showdown, engine):
	"""Two-sided pooled two-proportion z test; returns difference, SE, p."""
	left = np.asarray(showdown, dtype=np.float64)
	right = np.asarray(engine, dtype=np.float64)
	if left.size != right.size or not left.size:
		raise ValueError("Proportion samples must have equal positive lengths")
	p_left, p_right = float(left.mean()), float(right.mean())
	difference = p_left - p_right
	pooled = (left.sum() + right.sum()) / (left.size + right.size)
	se = math.sqrt(pooled * (1.0 - pooled) * (2.0 / left.size))
	p_value = 1.0 if se == 0 else math.erfc(abs(difference / se) / math.sqrt(2.0))
	return difference, se, p_value


def welch_mean_test(showdown, engine):
	"""Welch test for per-battle means, robust to clustered/overdispersed counts."""
	left = np.asarray(showdown, dtype=np.float64)
	right = np.asarray(engine, dtype=np.float64)
	if left.size < 2 or right.size < 2:
		raise ValueError("Welch samples need at least two observations")
	mean_left, mean_right = float(left.mean()), float(right.mean())
	var_left, var_right = float(left.var(ddof=1)), float(right.var(ddof=1))
	left_term, right_term = var_left / left.size, var_right / right.size
	se = math.sqrt(left_term + right_term)
	difference = mean_left - mean_right
	if se == 0:
		return difference, se, 1.0 if difference == 0 else 0.0
	degrees = (left_term + right_term) ** 2 / (
		left_term * left_term / (left.size - 1) + right_term * right_term / (right.size - 1)
	)
	return difference, se, _student_t_pvalue(difference / se, degrees)


def ks_distribution_test(showdown, engine):
	"""Two-sample KS test using the standard asymptotic Kolmogorov tail."""
	left = np.sort(np.asarray(showdown, dtype=np.float64))
	right = np.sort(np.asarray(engine, dtype=np.float64))
	if left.size == 0 or right.size == 0:
		raise ValueError("Distribution samples must be nonempty")
	values = np.unique(np.concatenate((left, right)))
	difference = float(
		max(
			abs(
				np.searchsorted(left, value, side="right") / left.size
				- np.searchsorted(right, value, side="right") / right.size
			)
			for value in values
		)
	)
	effective = math.sqrt(left.size * right.size / (left.size + right.size))
	se = math.sqrt((left.size + right.size) / (left.size * right.size))
	lambda_value = (effective + 0.12 + 0.11 / effective) * difference
	if lambda_value < 0.4:
		p_value = 1.0
	else:
		p_value = 0.0
		for index in range(1, 101):
			term = (
				2.0
				* (-1.0) ** (index - 1)
				* math.exp(-2.0 * index * index * lambda_value * lambda_value)
			)
			p_value += term
			if abs(term) < 1e-14:
				break
		p_value = min(1.0, max(0.0, p_value))
	return difference, se, p_value


def holm_adjust(p_values):
	"""Return Holm-Bonferroni adjusted p values in input order."""
	values = np.asarray(p_values, dtype=np.float64)
	order = np.argsort(values, kind="stable")
	adjusted = np.empty_like(values)
	previous = 0.0
	for rank, index in enumerate(order):
		value = min(1.0, (len(values) - rank) * values[index])
		previous = max(previous, value)
		adjusted[index] = previous
	return adjusted.tolist()


def compare_summaries(showdown, engine, move_ids=None):
	"""Build all comparisons and apply one Holm correction across the report."""
	if len(showdown) != len(engine) or not showdown:
		raise ValueError("Both simulators need the same nonempty number of battles")
	rows = []
	for side in (0, 1):
		left = np.asarray([battle["winner"] == side for battle in showdown])
		right = np.asarray([battle["winner"] == side for battle in engine])
		diff, se, p = two_proportion_test(left, right)
		rows.append(
			_row(f"win.side{side}", left.mean(), right.mean(), diff, se, p, "two-proportion")
		)
	left = np.asarray([battle["winner"] == "tie" for battle in showdown])
	right = np.asarray([battle["winner"] == "tie" for battle in engine])
	diff, se, p = two_proportion_test(left, right)
	rows.append(_row("win.tie", left.mean(), right.mean(), diff, se, p, "two-proportion"))

	left_turns = np.asarray([battle["turns"] for battle in showdown])
	right_turns = np.asarray([battle["turns"] for battle in engine])
	diff, se, p = welch_mean_test(left_turns, right_turns)
	rows.append(_row("turns.mean", left_turns.mean(), right_turns.mean(), diff, se, p, "Welch t"))
	diff, se, p = ks_distribution_test(left_turns, right_turns)
	left_hist = {
		str(turn): float(np.count_nonzero(left_turns == turn) / left_turns.size)
		for turn in np.unique(left_turns)
	}
	right_hist = {
		str(turn): float(np.count_nonzero(right_turns == turn) / right_turns.size)
		for turn in np.unique(right_turns)
	}
	rows.append(_row("turns.distribution", left_hist, right_hist, diff, se, p, "two-sample KS"))

	def add_counts(name, left_values, right_values):
		left_values = np.asarray(left_values, dtype=np.float64)
		right_values = np.asarray(right_values, dtype=np.float64)
		diff, se, p = welch_mean_test(left_values, right_values)
		rows.append(_row(name, left_values.mean(), right_values.mean(), diff, se, p, "Welch count"))

	move_ids = sorted(
		set(move_ids or ()).union(
			*(battle["moves"].keys() for battle in showdown),
			*(battle["moves"].keys() for battle in engine),
		)
	)
	for move in move_ids:
		if move == "struggle":
			continue
		add_counts(
			f"move.{move}",
			[battle["moves"].get(move, 0) for battle in showdown],
			[battle["moves"].get(move, 0) for battle in engine],
		)
	for status in STATUS_NAMES:
		add_counts(
			f"status.{status}",
			[battle["statuses"][status] for battle in showdown],
			[battle["statuses"][status] for battle in engine],
		)
	item_ids = sorted(
		set().union(
			*(battle["itemsConsumed"].keys() for battle in showdown),
			*(battle["itemsConsumed"].keys() for battle in engine),
		)
	)
	for item in item_ids:
		add_counts(
			f"itemConsumed.{item}",
			[battle["itemsConsumed"].get(item, 0) for battle in showdown],
			[battle["itemsConsumed"].get(item, 0) for battle in engine],
		)
	add_counts(
		"faints", [battle["faints"] for battle in showdown], [battle["faints"] for battle in engine]
	)
	add_counts(
		"switches.total",
		[sum(battle["switches"]) for battle in showdown],
		[sum(battle["switches"]) for battle in engine],
	)
	for side in (0, 1):
		add_counts(
			f"switches.side{side}",
			[battle["switches"][side] for battle in showdown],
			[battle["switches"][side] for battle in engine],
		)
	add_counts(
		"struggle",
		[battle["struggle"] for battle in showdown],
		[battle["struggle"] for battle in engine],
	)
	add_counts(
		"confusion",
		[battle["confusion"] for battle in showdown],
		[battle["confusion"] for battle in engine],
	)
	adjusted = holm_adjust([row["p_value"] for row in rows])
	for row, p_adjusted in zip(rows, adjusted):
		row["p_adjusted"] = p_adjusted
		row["flagged"] = p_adjusted < 0.05
	return sorted(rows, key=lambda row: (row["p_adjusted"], row["p_value"], row["metric"]))


def _row(metric, showdown, engine, difference, standard_error, p_value, test):
	return {
		"metric": metric,
		"showdown": float(showdown) if isinstance(showdown, (np.number, int, float)) else showdown,
		"engine": float(engine) if isinstance(engine, (np.number, int, float)) else engine,
		"difference": float(difference),
		"standard_error": float(standard_error),
		"p_value": float(p_value),
		"test": test,
	}


def _read_json(path):
	return json.loads(Path(path).read_text(encoding="utf-8"))


def _validate_showdown(summary, battles, seed, teams):
	lock = _read_json(ROOT / "reference-lock.json")
	if summary.get("schemaVersion") != 1 or summary.get("simulator") != "showdown":
		raise ValueError("Expected a whole-battle Showdown summary schema v1")
	if summary.get("showdownCommit") != lock["commit"] or summary.get("format") != FORMAT:
		raise ValueError("Showdown summary does not use the pinned format and commit")
	if summary.get("count") != battles or len(summary.get("teamPairs", [])) != battles:
		raise ValueError("Showdown summary count differs from --battles")
	if len(summary.get("battles", [])) != battles:
		raise ValueError("Showdown summary has an incomplete battle list")
	if summary.get("seed") != seed:
		raise ValueError("Showdown summary seed differs from --seed")
	if summary.get("teams", "isolation") != teams:
		raise ValueError("Showdown summary team mode differs from --unchanged")


def _engine_summaries_from_raw(showdown, seed):
	raw = engine_summaries(showdown, seed=seed)
	return raw


def _write_json(path, value):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _print_report(rows, showdown_count, seconds):
	print(f"Whole-battle comparison: {showdown_count} battles per simulator; {seconds:.2f}s")
	print("metric\tShowdown\tEngine\tdiff\tSE\tp\tHolm p")

	def value_text(value):
		if isinstance(value, (float, int)):
			return f"{value:.6g}"
		if isinstance(value, dict):
			return f"turn histogram ({len(value)} bins)"
		return str(value)

	for row in rows:
		flag = " *" if row["flagged"] else ""
		print(
			f"{row['metric']}\t{value_text(row['showdown'])}\t{value_text(row['engine'])}\t"
			f"{row['difference']:.6g}\t"
			f"{row['standard_error']:.6g}\t{row['p_value']:.4g}\t{row['p_adjusted']:.4g}{flag}"
		)
	flagged = [row["metric"] for row in rows if row["flagged"]]
	print("Flagged after Holm correction: " + (", ".join(flagged) if flagged else "none"))


def run(args):
	started = perf_counter()
	stats_dir = Path(args.stats_dir)
	showdown_path = (
		Path(args.reuse_showdown) if args.reuse_showdown else stats_dir / "showdown.json"
	)
	engine_path = Path(args.reuse_engine) if args.reuse_engine else stats_dir / "engine.json"
	if args.reuse_showdown:
		showdown = _read_json(showdown_path)
	else:
		command = [
			"node",
			str(ROOT / "reference/whole-battle-stats.cjs"),
			"--battles",
			str(args.battles),
			"--seed",
			str(args.seed),
			"--pool",
			str(Path(args.team_pool).resolve()),
			"--out",
			str(showdown_path.resolve()),
		]
		if args.allowed:
			command.extend(("--allowed", str(Path(args.allowed).resolve())))
		if args.unchanged:
			command.append("--unchanged")
		subprocess.run(command, cwd=ROOT, check=True)
		showdown = _read_json(showdown_path)
	teams = "unchanged" if args.unchanged else "isolation"
	_validate_showdown(showdown, args.battles, args.seed, teams)
	showdown_counts = [
		count_battle_summary({**battle, "teamPair": showdown["teamPairs"][index]})
		for index, battle in enumerate(showdown["battles"])
	]
	team_hash = _team_pairs_hash(showdown["teamPairs"])
	if args.reuse_engine:
		engine_summary = _read_json(engine_path)
		if (
			engine_summary.get("simulator") != "engine"
			or engine_summary.get("count") != args.battles
			or engine_summary.get("seed") != args.seed
			or engine_summary.get("teamPairsHash") != team_hash
		):
			raise ValueError("Reused engine summary does not match the Showdown team pairs")
		engine_counts = engine_summary["battles"]
	else:
		engine_summary = _engine_summaries_from_raw(showdown, args.seed)
		_write_json(engine_path, engine_summary)
		engine_counts = engine_summary["battles"]
	rows = compare_summaries(showdown_counts, engine_counts, showdown.get("acceptedMoves", []))
	seconds = perf_counter() - started
	result = {
		"schemaVersion": 1,
		"battles": args.battles,
		"seed": args.seed,
		"format": FORMAT,
		"teams": teams,
		"showdownCommit": showdown["showdownCommit"],
		"teamPairsHash": team_hash,
		"allowed": showdown["allowed"],
		"acceptedMoves": showdown.get("acceptedMoves", []),
		"policy": showdown["policy"],
		"runtimeSeconds": seconds,
		"multipleComparisonCorrection": "Holm-Bonferroni",
		"countingRules": {
			"move": "selected legal move action, even if prevented from executing",
			"switch": "selected switch action, including forced replacements",
			"status": "first visible status at a decision boundary; per member",
			"faint": "first visible transition from alive to fainted; per member",
			"confusion": "first visible transition into active confusion",
			"itemConsumption": "first visible transition to consumed item, by initial held item",
		},
		"results": rows,
	}
	_print_report(rows, args.battles, seconds)
	_write_json(args.out, result)
	print(f"JSON report: {args.out}")
	return result


def main(argv=None):
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--battles", type=int, default=DEFAULT_BATTLES)
	parser.add_argument("--seed", type=int, default=1)
	parser.add_argument("--allowed", type=Path)
	parser.add_argument(
		"--unchanged",
		action="store_true",
		help="use pool teams exactly as generated (no isolation or replacements)",
	)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--stats-dir", type=Path, default=DEFAULT_STATS_DIR)
	parser.add_argument("--out", type=Path)
	parser.add_argument("--reuse-showdown", type=Path)
	parser.add_argument("--reuse-engine", type=Path)
	args = parser.parse_args(argv)
	if args.battles < 2:
		parser.error("--battles must be at least two for the battle-length test")
	if args.out is None:
		args.out = args.stats_dir / "report.json"
	if not args.allowed and args.reuse_showdown is None:
		pass
	if args.reuse_engine and not args.reuse_showdown:
		parser.error("--reuse-engine requires --reuse-showdown so its team pairs can be checked")
	try:
		run(args)
	except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
		print(f"stats: {error}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
