"""Engine pilot, confirmation and latency for player-view Method 1."""

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.state import NUM_SIDES, ONGOING, TIE
from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.search.belief import Catalogue
from pokejax.search.matches import MatchResult, choices, combine, play, report, start
from pokejax.search.method1 import Method1, SearchMetrics, infer
from pokejax.search.oracle import SearchConfig, forward
from pokejax.search.rebuild import views
from pokejax.search.signal import prefix
from pokejax.tables import load_tables
from pokejax.teams import load_team_pool, save_npz
from pokejax.train import headtohead, panel
from pokejax.train.matchup import baseline_contestant, network_contestant, paired_mean_interval
from pokejax.train.network import initial_carry

ARMS = {
	"A": {},
	"B": {"sample_strength": 4.0},
	"C": {"margin": 0.02},
	"O-diagnostic": {},
}
OPPONENTS = ("sampled-raw", "max-damage", "heuristics")


def emit(out, case, **data):
	row = {"case": case, **data}
	line = json.dumps(row)
	print(json.dumps({k: v for k, v in row.items() if k != "game_data"}), flush=True)
	with out.open("a", encoding="utf-8") as file:
		file.write(line + "\n")


@jax.jit
def root_slice(tree, sides):
	return jax.tree.map(lambda x: x[jnp.arange(sides.size), sides], tree)


class EngineMatch:
	def __init__(self, env, player, method, arm, baseline, *, foe_method=None, sample_raw=False):
		self.env, self.player, self.method, self.arm, self.baseline = (
			env,
			player,
			method,
			arm,
			baseline,
		)
		self.foe_method = foe_method
		self.sample_raw = sample_raw

	def run(self, state, sides, key, roots=None, on_decision=None):
		tables = self.env.tables
		model, params = self.player.seat.model, self.player.params
		rows = jnp.arange(sides.size)
		game_key, sample_key = jax.random.split(key)
		search_key = jax.random.fold_in(key, 0x5EA2C)
		match = start(state, tables, game_key)
		carry = initial_carry(self.player.seat.network, (sides.size, NUM_SIDES))
		diagnostic = self.arm == "O-diagnostic"
		rebuilt = self.method is not None and not diagnostic
		beliefs = None
		if rebuilt:
			beliefs = self.method.new_beliefs(
				np.asarray(jax.random.key_data(search_key)), sides.size
			)
		previous_actions = jnp.zeros(sides.size, jnp.int32)
		refused = jnp.zeros(sides.size, jnp.bool_)
		if self.foe_method is not None:
			foe_sides = 1 - sides
			foe_key = jax.random.fold_in(search_key, 1)
			foe_beliefs = self.foe_method.new_beliefs(
				np.asarray(jax.random.key_data(foe_key)), sides.size
			)
			foe_previous, foe_refused = previous_actions, refused
		sample = self.sample_raw or bool(self.method and self.method.config.sample_strength)
		metrics = np.zeros((len(SearchMetrics._fields), sides.size))
		for ply in range(self.env.config.max_steps):
			# This simulator runs the real competitors. Their observations and
			# carries are kept separate at the rollout boundary below.
			after, logits, _, head = forward(
				model,
				params,
				match.position,
				carry,
				tables,
				self.method is not None or self.foe_method is not None,
			)
			own, foe, noise = choices(
				match,
				sides,
				logits,
				sample_key,
				jnp.int32(ply),
				self.baseline,
				self.method is not None and not sample,
			)
			decision = None
			if self.method is not None:
				keys = jax.random.split(jax.random.fold_in(search_key, ply), sides.size)
				our_after, our_logits, our_head = root_slice((after, logits, head), sides)
				noise = noise if sample else jnp.zeros_like(noise)
				if diagnostic:
					decision, measures = self.method.diagnostic_oracle(
						params, match.position, sides, our_after, our_logits, our_head, keys, noise
					)
				else:
					view = views(match.position, sides, tables, refused)
					decision, measures = self.method.choose(
						params,
						view,
						beliefs,
						our_after,
						our_logits,
						our_head,
						keys,
						noise,
						previous_actions,
					)
				own = decision.action
				metrics += np.asarray(measures)
				if on_decision is not None:
					on_decision(decision)
			if self.foe_method is not None:
				foe_keys = jax.random.split(jax.random.fold_in(foe_key, ply), sides.size)
				foe_view = views(match.position, foe_sides, tables, foe_refused)
				foe_after, foe_logits, foe_head = root_slice((after, logits, head), foe_sides)
				foe_decision, _ = self.foe_method.choose(
					params,
					foe_view,
					foe_beliefs,
					foe_after,
					foe_logits,
					foe_head,
					foe_keys,
					jnp.zeros_like(foe_logits),
					foe_previous,
				)
				foe = foe_decision.action
				foe_known = match.position.state.trap_known[rows, foe_sides]
				foe_moving = match.position.state.request[rows, foe_sides] != 0
				foe_previous = jnp.where(foe_moving & ~foe_view.has_held, foe, foe_previous)
			if roots is not None:
				view = views(match.position, sides, tables, refused)
				roots.append((view, root_slice(carry, sides), previous_actions))
			known = match.position.state.trap_known[rows, sides]
			held = match.position.has_held[rows, sides]
			moving = match.position.state.request[rows, sides] != 0
			match = play(match, sides, own, foe, decision, tables)
			refused = ~known & match.position.state.trap_known[rows, sides]
			if self.foe_method is not None:
				foe_refused = ~foe_known & match.position.state.trap_known[rows, foe_sides]
			previous_actions = jnp.where(moving & ~held, own, previous_actions)
			carry = after
			if bool((match.position.state.winner != ONGOING).all()):
				break
		winner = match.position.state.winner
		result = MatchResult(
			*jax.device_get(
				(
					winner == sides,
					winner == TIE,
					winner == ONGOING,
					match.length,
					match.searched,
					match.overrides,
					match.best_advantage,
					match.chosen_advantage,
				)
			)
		)
		return result, metrics


def match_cell(runner, setups, budget, label):
	began = time.monotonic()
	results, metrics = [], []
	limit = len(setups)
	warm_started = began
	for i, setup in enumerate(setups):
		started = time.monotonic()
		result, counts = runner.run(*setup)
		results.append(result)
		metrics.append(counts)
		elapsed = time.monotonic() - began
		if i == 3:
			# Estimate with warm batches and retain a complete setup/seat cluster.
			mean = (time.monotonic() - warm_started) / 3
			limit = min(limit, i + 1 + max(0, int((budget - elapsed) / mean)))
		if i < 4 or (i + 1) % 16 == 0 or i + 1 == limit:
			print(f"{label} {i + 1}/{limit} batches, {elapsed:.1f}s", flush=True)
		if i + 1 >= limit or elapsed + time.monotonic() - started > budget:
			break
		if i == 0:
			warm_started = time.monotonic()
	return combine(results), np.concatenate(metrics, axis=1), time.monotonic() - began


def summarize(out, stage, arm, opponent, cell, planned, seed, half, batch, raw=None):
	result, metrics, secs = cell
	if raw is not None:
		size = min(result.won.size, raw.won.size)
		result, raw = prefix(result, size), prefix(raw, size)
		metrics = metrics[:, :size]
	rows = report(result, seed, half=half, raw=raw)
	eligible, empty, invalid, particles, ess, worlds = metrics.sum(axis=1)
	denom = eligible - empty
	if eligible:
		rows.update(
			eligible=int(eligible),
			no_particle=int(empty),
			unrebuildable=int(invalid),
			fallback_rate=float((empty + invalid) / eligible),
			mean_particles=float(particles / eligible),
			mean_ess=float(ess / denom) if denom else None,
			mean_valid_worlds=float(worlds / denom) if denom else None,
		)
	emit(
		out,
		f"{stage}-{arm}-vs-{opponent}",
		stage=stage,
		arm=arm,
		opponent=opponent,
		seed=seed,
		diagnostic=arm == "O-diagnostic",
		planned_games=planned,
		seconds=secs,
		batch_games=batch,
		paired_setup_size=2 if half else 1,
		game_data={
			**{name: getattr(result, name).tolist() for name in result._fields},
			**{
				name: values.tolist()
				for name, values in zip(SearchMetrics._fields, metrics, strict=True)
			},
		},
		**rows,
	)
	return rows


def select(stats):
	def worst(arm):
		return max(-stats[arm][opponent]["delta"]["estimate"] for opponent in OPPONENTS[1:])

	candidates = [arm for arm in ("A", "B", "C") if worst(arm) <= 0.02 + 1e-12]
	if candidates:
		return (
			max(candidates, key=lambda a: stats[a]["sampled-raw"]["win_rate"]["estimate"]),
			"highest head-to-head among arms within two points of both baselines",
			candidates,
		)
	return min(("A", "B", "C"), key=worst), "smallest worst baseline loss", candidates


def saved_control(records, stage, arm, opponent, seed, batch, games):
	"""Read an unchanged raw or oracle cell from the same checkpoint/setup run."""
	label = f"{stage}-{arm}-vs-{opponent}"
	row = records.get(label)
	if row is None or (row["seed"], row["batch_games"], row["battles"]) != (seed, batch, games):
		return None
	data = row["game_data"]
	result = MatchResult(*(np.asarray(data[name]) for name in MatchResult._fields))
	stats = np.asarray([data[name] for name in SearchMetrics._fields])
	return result, stats, row["seconds"]


def matches(env, player, tables, catalogue, base, args):
	batch = args.batch
	controls = {}
	if args.reuse_controls:
		with args.reuse_controls.open(encoding="utf-8") as source:
			controls = {
				row["case"]: row
				for line in source
				if (row := json.loads(line)).get("arm") in ("raw", "O-diagnostic")
			}
	baselines = {name: baseline_contestant(name, env, tables).seat.policy for name in OPPONENTS[1:]}

	def stage(name, arms, seed, draws, games, oracle_draws):
		h2h = headtohead.battle_setups(env, draws, batch, seed)
		panels = panel.battle_setups(env, games, batch, seed)
		stats = {arm: {} for arm in arms}
		head_results = {}
		for opponent in OPPONENTS:
			setups = h2h if opponent == OPPONENTS[0] else panels
			half = batch // 2 if opponent == OPPONENTS[0] else None
			baseline = baselines.get(opponent)
			raw = saved_control(controls, name, "raw", opponent, seed, batch, len(setups) * batch)
			if raw is None:
				raw = match_cell(
					EngineMatch(env, player, None, "raw", baseline),
					setups,
					args.budget,
					f"{name}-raw-{opponent}",
				)
			summarize(args.out, name, "raw", opponent, raw, len(setups) * batch, seed, half, batch)
			for arm in arms:
				current = setups
				if arm == "O-diagnostic":
					if half is None and name == "confirm":
						continue
					if half is not None:
						current = setups[: oracle_draws // half]
				method = Method1(
					player.seat.model,
					env.tables,
					None if arm == "O-diagnostic" else catalogue,
					replace(base, **ARMS[arm]),
				)
				cell = saved_control(
					controls, name, arm, opponent, seed, batch, len(current) * batch
				)
				if cell is None:
					cell = match_cell(
						EngineMatch(env, player, method, arm, baseline),
						current,
						args.budget,
						f"{name}-{arm}-{opponent}",
					)
				if half is not None:
					head_results[arm] = cell[0]
				stats[arm][opponent] = summarize(
					args.out,
					name,
					arm,
					opponent,
					cell,
					len(current) * batch,
					seed,
					half,
					batch,
					raw[0],
				)
		if name == "confirm":
			selected = arms[0]
			oracle = head_results["O-diagnostic"].won
			rebuilt = head_results[selected].won
			count = min(oracle.size, rebuilt.size)
			difference = oracle[:count].astype(float) - rebuilt[:count]
			delta = paired_mean_interval(
				difference.reshape(-1, 2, batch // 2).mean(axis=1).ravel(), 2000, seed
			)
			emit(
				args.out,
				"confirm-oracle-gap",
				diagnostic=True,
				selected=selected,
				battles_per_arm=count,
				oracle_minus_rebuilt=delta,
			)
		return stats

	if args.stage in ("all", "pilot"):
		stats = stage(
			"pilot", tuple(ARMS), args.seed, args.pilot_draws, args.pilot_games, args.pilot_draws
		)
		selected, reason, qualifiers = select(stats)
		emit(args.out, "selected", arm=selected, reason=reason, qualifiers=qualifiers)
		if args.stage == "pilot":
			return
	else:
		selected = next(
			row["arm"]
			for line in reversed(args.out.read_text().splitlines())
			if (row := json.loads(line))["case"] == "selected"
		)
	stage(
		"confirm",
		(selected, "O-diagnostic"),
		args.seed + 1,
		args.confirm_draws,
		args.confirm_games,
		args.oracle_draws,
	)


def timings(env, player, catalogue, base, args):
	setup = headtohead.battle_setups(env, args.batch // 2, args.batch, args.seed)[0]
	roots = []
	EngineMatch(env, player, None, "raw", None).run(*setup, roots=roots)
	model, params = player.seat.model, player.params
	for worlds in (16, 32):
		config = replace(base, worlds=worlds, **ARMS[args.timing_arm])
		method = Method1(model, env.tables, catalogue, config)
		data = []
		cold = None
		for repeat in range(args.samples):
			row = repeat % args.batch
			belief = method.new_beliefs(args.seed + repeat, 1)
			for ply, (batch_view, batch_carry, batch_action) in enumerate(roots):
				view, carry, prior = jax.tree.map(
					lambda x, row=row: x[row : row + 1], (batch_view, batch_carry, batch_action)
				)
				if int(view.result[0]) != ONGOING:
					break
				key = jax.random.fold_in(jax.random.key(args.seed + repeat), ply)[None]
				started = time.perf_counter()
				after, logits, head = infer(model, params, view.obs, carry, view.has_held)
				if config.sample_strength:
					policy, search = jax.random.split(key[0])
					noise = jax.random.gumbel(policy, logits.shape, dtype=logits.dtype)
					key = search[None]
				else:
					noise = jnp.zeros_like(logits)
				decision, metrics = method.choose(
					params, view, belief, after, logits, head, key, noise, prior
				)
				jax.block_until_ready(decision)
				secs = time.perf_counter() - started
				if metrics.eligible[0]:
					if cold is None:
						cold = secs
					else:
						data.append(secs * 1000)
				if len(data) >= args.samples:
					break
			if len(data) >= args.samples:
				break
		emit(
			args.out,
			f"timing-{worlds}x2",
			arm=args.timing_arm,
			samples=len(data),
			cold_seconds=cold,
			warm_ms=np.percentile(data, [50, 95, 99]).tolist(),
		)


def ensure_draws(path, count, seed, tables):
	if path.is_file():
		return
	source = path.with_suffix(".json")
	subprocess.run(
		[
			"node",
			"reference/generate-teams.cjs",
			"--count",
			str(count),
			"--seed",
			str(seed),
			"--out",
			str(source),
		],
		check=True,
	)
	save_npz(load_team_pool(source, tables), path)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("stage", choices=("all", "timing", "pilot", "confirm"))
	parser.add_argument("checkpoint", type=Path)
	parser.add_argument("--out", type=Path, required=True)
	parser.add_argument(
		"--reuse-controls",
		type=Path,
		help="reuse complete raw and oracle cells from this checkpoint, pool and setup run",
	)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--evaluation-pool", type=Path, default=Path("work/s2-eval-draws.npz"))
	parser.add_argument("--extra", type=Path, default=Path("work/s2-catalogue-draws.npz"))
	parser.add_argument("--catalogue", type=Path, default=Path("work/s2-catalogue.npz"))
	parser.add_argument("--batch", type=int, default=32)
	parser.add_argument("--seed", type=int, default=2297)
	parser.add_argument("--samples", type=int, default=200)
	parser.add_argument("--timing-arm", choices=("A", "B", "C"), default="A")
	parser.add_argument("--pilot-draws", type=int, default=256)
	parser.add_argument("--pilot-games", type=int, default=512)
	parser.add_argument("--confirm-draws", type=int, default=4096)
	parser.add_argument("--confirm-games", type=int, default=4096)
	parser.add_argument("--oracle-draws", type=int, default=1024)
	parser.add_argument("--budget", type=float, default=6900)
	args = parser.parse_args()
	if args.batch < 2 or args.batch % 2:
		parser.error("--batch must be a positive even number")
	for n, multiple in (
		(args.pilot_draws, args.batch // 2),
		(args.pilot_games, args.batch),
		(args.confirm_draws, args.batch // 2),
		(args.confirm_games, args.batch),
		(args.oracle_draws, args.batch // 2),
	):
		if n <= 0 or n % multiple:
			parser.error("draw counts must be multiples of half a batch, panel counts of a batch")
	if args.oracle_draws > args.confirm_draws:
		parser.error("oracle draws must be a prefix of confirmation draws")
	if args.budget <= 0 or args.samples <= 0:
		parser.error("budget and sample counts must be positive")
	tables = load_tables()
	ensure_draws(args.extra, 40_000, 720023, tables)
	ensure_draws(args.evaluation_pool, 16_384, 720024, tables)
	player, train = network_contestant("480M checkpoint", args.checkpoint, args.team_pool, tables)
	env = Env(
		EnvConfig(args.batch, max_steps=train.env.max_steps, observe="none"),
		args.evaluation_pool,
		tables=tables,
	)
	catalogue = Catalogue.load(args.team_pool, tables, extra=args.extra, cache=args.catalogue)
	args.out.parent.mkdir(parents=True, exist_ok=True)
	base = SearchConfig(gamma=train.ppo.gamma, max_steps=train.env.max_steps)
	print(f"{jax.devices()[0].device_kind}. O is a diagnostic true-state oracle.", flush=True)
	if args.stage in ("all", "timing"):
		timings(env, player, catalogue, base, args)
	if args.stage != "timing":
		matches(env, player, tables, catalogue, base, args)


if __name__ == "__main__":
	main()
