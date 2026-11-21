"""The training loop: updates, metric lines, evaluations, checkpoints and the run summary.

A run directory (default work/runs/<name>, outside git) holds:

- config.json: the configuration the run started with;
- metrics.jsonl: one JSON object per line, "kind" update (every logging.every updates) or
  evaluation (every evaluation.every updates, and the final one with kind final_evaluation);
- checkpoint.npz: the learner (parameters, optimizer state, battles and keys, update count) and
  the configuration; written every checkpoint.every updates and at the end;
- result.json: the summary of the finished run.
"""

import json
import statistics
import time
from pathlib import Path

import jax

from pokejax.env import DEFAULT_POOL
from pokejax.train.checkpoint import CheckpointError, load_checkpoint, save_checkpoint
from pokejax.train.config import TrainConfig
from pokejax.train.ppo import Trainer

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"
CHECKPOINT_FILE = "checkpoint.npz"
RESULT_FILE = "result.json"
FINAL_OPPONENTS = ("random", "max-damage")


class RunError(RuntimeError):
	"""A run directory that cannot be started or resumed as asked."""


def _record(handle, record: dict) -> None:
	handle.write(json.dumps(record, allow_nan=False, sort_keys=True) + "\n")
	handle.flush()


def _truncate_metrics(path: Path, last_update: int) -> None:
	"""Drop the lines a resumed run writes again: later updates and the final evaluation."""
	if not path.exists():
		return
	kept = []
	for line in path.read_text(encoding="utf-8").splitlines():
		record = json.loads(line) if line else {}
		if record and record["kind"] != "final_evaluation" and record["update"] <= last_update:
			kept.append(line)
	path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")


def _peak_memory() -> int | None:
	stats = jax.devices()[0].memory_stats()
	return int(stats["peak_bytes_in_use"]) if stats and "peak_bytes_in_use" in stats else None


def run_training(
	config: TrainConfig, *, team_pool=DEFAULT_POOL, resume: bool = False, log=print
) -> dict:
	"""Train as the configuration says, writing the run directory; returns the summary."""
	run_dir = config.run_dir
	checkpoint_path = run_dir / CHECKPOINT_FILE
	metrics_path = run_dir / METRICS_FILE
	if resume:
		if not checkpoint_path.exists():
			raise RunError(f"nothing to resume: {checkpoint_path} does not exist")
	elif any((run_dir / name).exists() for name in (CONFIG_FILE, METRICS_FILE, CHECKPOINT_FILE)):
		raise RunError(f"{run_dir} already holds a run; resume it or choose another run name")

	trainer = Trainer(config, team_pool)
	learner = trainer.init()
	parameters = trainer.parameter_count(learner)
	cost = trainer.forward_cost(learner)
	if resume:
		try:
			learner = load_checkpoint(checkpoint_path, learner, config)
		except CheckpointError as error:
			raise RunError(str(error)) from error
		start = int(learner.update)
		_truncate_metrics(metrics_path, start)
		log(f"resuming {run_dir} after update {start}")
	else:
		run_dir.mkdir(parents=True, exist_ok=True)
		(run_dir / CONFIG_FILE).write_text(
			json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
		)
		start = 0

	rollout = config.rollout
	steps_per_update = rollout.batch_size * rollout.rollout_length
	environment_rate = trainer.environment_steps_per_second()
	log(
		f"{config.run.name}: {parameters} parameters, {cost['flops_per_battle_step']:.0f} "
		f"forward FLOPs per battle step, environment {environment_rate:,.0f} steps/s"
	)
	start_time = time.perf_counter()
	cold_seconds = 0.0
	warm_seconds: list[float] = []
	with metrics_path.open("a", encoding="utf-8") as metrics:
		for update in range(start + 1, config.run.updates + 1):
			tick = time.perf_counter()
			learner, values = trainer.update(learner)
			values = jax.device_get(jax.block_until_ready(values))
			elapsed = time.perf_counter() - tick
			# The first update of a process includes the compilation.
			if update == start + 1:
				cold_seconds = elapsed
			else:
				warm_seconds.append(elapsed)
			if update == 1 or update % config.logging.every == 0 or update == config.run.updates:
				record = {
					"kind": "update",
					"update": update,
					"env_steps": update * steps_per_update,
					"seconds": time.perf_counter() - start_time,
					"update_seconds": elapsed,
					"steps_per_second": steps_per_update / elapsed,
					"environment_steps_per_second": environment_rate,
				}
				record.update({name: float(value) for name, value in values.items()})
				_record(metrics, record)
				log(
					f"{update}: win {record['win_rate']:.3f} len {record['mean_battle_length']:.1f} "
					f"kl {record['approx_kl']:.4f} clip {record['clip_fraction']:.3f} "
					f"ent {record['entropy']:.3f} ev {record['explained_variance']:.3f} "
					f"{record['steps_per_second']:,.0f} steps/s"
				)
			if config.evaluation.every and update % config.evaluation.every == 0:
				record = {
					"kind": "evaluation",
					"update": update,
					"results": [
						trainer.evaluate(learner.params, name, config.evaluation.battles)
						for name in FINAL_OPPONENTS
					],
				}
				_record(metrics, record)
				log(_evaluation_line(update, record["results"]))
			if config.checkpoint.every and update % config.checkpoint.every == 0:
				save_checkpoint(checkpoint_path, learner, config)
		training_seconds = time.perf_counter() - start_time
		save_checkpoint(checkpoint_path, learner, config)

		final = [
			trainer.evaluate(learner.params, name, config.evaluation.final_battles)
			for name in FINAL_OPPONENTS
		]
		_record(
			metrics,
			{"kind": "final_evaluation", "update": config.run.updates, "results": final},
		)
	log(_evaluation_line(config.run.updates, final, final=True))

	env_steps = config.run.updates * steps_per_update
	updates_done = config.run.updates - start
	result = {
		"name": config.run.name,
		"device": str(jax.devices()[0]),
		"jax_version": jax.__version__,
		"config": config.to_dict(),
		"parameters": parameters,
		"forward_flops_per_battle_step": cost["flops_per_battle_step"],
		"updates": updates_done,
		"resumed_from_update": start,
		"updates_total": config.run.updates,
		"environment_steps": env_steps,
		"training_seconds": training_seconds,
		"first_update_seconds": cold_seconds,
		"steps_per_second": (
			steps_per_update * len(warm_seconds) / sum(warm_seconds) if warm_seconds else None
		),
		"steps_per_second_median": (
			steps_per_update / statistics.median(warm_seconds) if warm_seconds else None
		),
		"environment_steps_per_second": environment_rate,
		"peak_device_bytes": _peak_memory(),
		"final_evaluation": final,
	}
	(run_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
	return result


def _evaluation_line(update: int, results: list[dict], final: bool = False) -> str:
	name = "final evaluation" if final else "evaluation"
	parts = [
		f"{result['opponent']} {result['win_rate']:.4f} "
		f"[{result['ci95'][0]:.4f}, {result['ci95'][1]:.4f}] over {result['battles']}"
		for result in results
	]
	return f"{name} at {update}: " + "; ".join(parts)
