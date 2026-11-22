"""python -m pokejax.showdown.compare: the engine's evaluation and Showdown, side by side.

For one checkpoint, every baseline and both action policies, the engine evaluates the learner with
pokejax.train.ppo.Trainer.evaluate (its own fixed battles), and Showdown plays the same number of battles
with the same team pairs and learner sides. The pairs and sides are the ones the engine's evaluation draws
(its key, its batch and its sampler), so the two sets of battles differ only in the simulator and in the
random outcomes. The table reports both win rates with their Wilson 95% intervals, the difference in points,
and the refused switches of each side (the engine's are counted by a replica of its evaluation loop, which
must reproduce the engine's win counts).

    python -m pokejax.showdown.compare --checkpoint work/runs/rtx3060ti --battles 4096 \
        --out work/codex/runs/p5t1-compare.json
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import make_policy
from pokejax.env import DEFAULT_POOL
from pokejax.policy_names import POLICY_NAMES
from pokejax.sampler import sample_team_indices
from pokejax.showdown.player import (
	DEFAULT_POOL_JSON,
	Policy,
	battle_specs,
	compare_first_observations,
	load_checkpoint,
	read_config,
	run_battles,
	summarize,
)
from pokejax.train.league import Layout, League
from pokejax.train.ppo import EVALUATION_SEED, Trainer

POLICIES = ("sampled", "greedy")
EXIT_POINTS = 3.0


def evaluation_pairs(trainer: Trainer, battles: int):
	"""The team pairs and learner sides of the engine's evaluation (Trainer.evaluate), battle by battle.

	Each evaluation batch draws its pairs from the key League.reset gives it, as the evaluator does.
	"""
	batch = trainer.config.evaluation.batch
	if battles % batch:
		raise ValueError(f"battles must be a multiple of the evaluation batch ({batch})")
	env = trainer.env.replace(batch_size=batch, keep_final=False)
	key = jax.random.key(trainer.config.run.seed + EVALUATION_SEED)
	pairs = []
	for index in range(battles // batch):
		_, reset_key, _ = jax.random.split(jax.random.fold_in(key, index), 3)
		index_key, _ = jax.random.split(reset_key)
		pairs.append(np.asarray(sample_team_indices(index_key, env.team_indices, batch)))
	learners = np.tile(np.arange(batch) % 2, battles // batch)
	return np.concatenate(pairs), learners


def engine_refusals(trainer: Trainer, params, opponent: str, battles: int, greedy: bool) -> dict:
	"""The engine's evaluation again, counting the learner's refused switches in the first battle of each row.

	The loop is that of Trainer._evaluate_batch (same keys, same steps); the wins it counts must equal
	Trainer.evaluate's, which is checked by the caller.
	"""
	batch = trainer.config.evaluation.batch
	env = trainer.env.replace(batch_size=batch, keep_final=False)
	policy = make_policy(opponent, env.tables, trainer.tables)
	league = League(
		env,
		Layout({opponent: batch}),
		trainer.model,
		trainer.config.network,
		trainer.config.pool.size,
		{opponent: policy},
	)
	limit = trainer.config.env.max_steps
	rows = jnp.arange(batch)

	def run_batch(params, key, greedy):
		learner = jnp.arange(batch, dtype=jnp.int32) % 2
		state = league.reset(key, learner)
		zeros = jnp.zeros(batch, jnp.bool_)
		start = (state, zeros, zeros, jnp.zeros(batch, jnp.int32), jnp.int32(0))

		def cond(carry):
			_, finished, _, _, count = carry
			return (~finished.all()) & (count < limit)

		def body(carry):
			state, finished, wins, refused, count = carry
			side = state.learner
			next_state, step = league.play(params, None, state, greedy)
			ended = step.terminated | step.truncated
			first = ended & ~finished
			refused = refused + jnp.where(~finished, step.rejected[rows, side], False)
			wins = jnp.where(first, step.won, wins)
			return next_state, finished | ended, wins, refused, count + 1

		_, _, wins, refused, _ = jax.lax.while_loop(cond, body, start)
		return wins.sum(), refused.sum()

	run = jax.jit(run_batch)
	wins = refused = 0
	for index in range(battles // batch):
		key = jax.random.key(trainer.config.run.seed + EVALUATION_SEED)
		won, count = run(params, jax.random.fold_in(key, index), jnp.bool_(greedy))
		wins += int(won)
		refused += int(count)
	return {"wins": wins, "refusals": refused}


def wilson_text(rate: float, ci) -> str:
	return f"{100 * rate:6.2f} [{100 * ci[0]:6.2f}, {100 * ci[1]:6.2f}]"


def table(rows: list[dict]) -> str:
	header = (
		f"{'Opponent':<13}{'Policy':<9}{'Engine win % [95% CI]':<30}{'Showdown win % [95% CI]':<31}"
		f"{'Diff':>8}  {'Refused: engine':>15} {'Showdown':>8}  {'Sd battles/s':>12}  "
		f"{'Errors':>6} {'Trunc':>5} {'Ties':>5}"
	)
	lines = [header]
	for row in rows:
		engine, showdown = row["engine"], row["showdown"]
		diff = 100 * (showdown["win_rate"] - engine["win_rate"])
		lines.append(
			f"{row['opponent']:<13}{row['policy']:<9}"
			f"{wilson_text(engine['win_rate'], engine['ci95']):<30}"
			f"{wilson_text(showdown['win_rate'], showdown['ci95']):<31}{diff:>+8.2f}  "
			f"{row['engine_refusals']:>15} {showdown['refusals']:>8}  "
			f"{showdown['battles_per_second']:>12.1f}  {showdown['errors']:>6} "
			f"{showdown['truncated']:>5} {showdown['ties']:>5}"
		)
	return "\n".join(lines)


def main(argv=None) -> int:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.showdown.compare", description=__doc__.split("\n")[0]
	)
	parser.add_argument("--checkpoint", type=Path, required=True)
	parser.add_argument("--battles", type=int, default=4096)
	parser.add_argument("--opponents", nargs="+", choices=POLICY_NAMES, default=list(POLICY_NAMES))
	parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
	parser.add_argument("--concurrency", type=int, default=512)
	parser.add_argument("--seed", type=int, default=None, help="default: the run's seed")
	parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
	parser.add_argument("--pool-json", type=Path, default=DEFAULT_POOL_JSON)
	parser.add_argument("--out", type=Path, default=None)
	args = parser.parse_args(argv)
	config = read_config(args.checkpoint)
	trainer = Trainer(config, args.pool)
	env = trainer.env
	checkpoint = load_checkpoint(args.checkpoint, env)
	pairs, learners = evaluation_pairs(trainer, args.battles)
	policy = Policy(checkpoint, env, slots=args.concurrency, seed=config.run.seed)
	seed = config.run.seed if args.seed is None else args.seed
	rows = []
	for opponent in args.opponents:
		for name in args.policies:
			greedy = name == "greedy"
			engine = trainer.evaluate(checkpoint.params, opponent, args.battles, greedy=greedy)
			replica = engine_refusals(trainer, checkpoint.params, opponent, args.battles, greedy)
			if replica["wins"] != engine["wins"]:
				raise SystemExit(
					f"the engine replica won {replica['wins']} battles, evaluate {engine['wins']}"
				)
			run = run_battles(
				policy,
				battle_specs(pairs, learners),
				pool_json=args.pool_json,
				opponent=opponent,
				seed=seed,
				max_steps=config.env.max_steps,
				concurrency=args.concurrency,
				greedy=greedy,
			)
			showdown = summarize(run, opponent, name)
			showdown.first_check = compare_first_observations(env, pairs, learners, run.first)
			row = {
				"opponent": opponent,
				"policy": name,
				"engine": engine,
				"engine_refusals": replica["refusals"],
				"showdown": showdown.as_dict(),
			}
			rows.append(row)
			print(
				f"{opponent} {name}: engine {100 * engine['win_rate']:.2f}% "
				f"({replica['refusals']} refused), Showdown {100 * showdown.win_rate:.2f}% "
				f"({showdown.refusals} refused) over {showdown.counted} battles in {showdown.seconds:.1f} s "
				f"({showdown.battles_per_second:.1f} battles/s); first observations identical up to one "
				f"float ulp: {showdown.first_check['identical']}",
				flush=True,
			)
	print(table(rows))
	gaps = [abs(100 * (r["showdown"]["win_rate"] - r["engine"]["win_rate"])) for r in rows]
	print(
		f"Largest gap: {max(gaps):.2f} points; within {EXIT_POINTS:g} points for every row: "
		f"{max(gaps) <= EXIT_POINTS}"
	)
	report = {
		"checkpoint": str(checkpoint.path),
		"battles": args.battles,
		"seed": seed,
		"concurrency": args.concurrency,
		"rows": rows,
	}
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
