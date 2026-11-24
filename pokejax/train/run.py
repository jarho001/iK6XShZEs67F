"""The training loop: updates, metric lines, evaluations, checkpoints and the run summary.

A run directory (default work/runs/<name>, outside git) holds:

- config.json: the configuration the run started with;
- metrics.jsonl: one JSON object per line, "kind" update (every logging.every updates), evaluation
  (every evaluation.every updates: one result per configured baseline and action policy) and
  final_evaluation (at the end, with evaluation.final_battles);
- checkpoint.npz: the learner (parameters, optimizer state, battles and keys, the pool and its
  insertion count, the update count) and the configuration; written every checkpoint.every updates
  and at the end; checkpoint.keep names updates whose checkpoints are also kept as
  checkpoint-<update>.npz (never overwritten);
- result.json: the summary of the finished run.
"""

import json
import statistics
import time
from pathlib import Path

import jax

from pokejax.env import DEFAULT_POOL
from pokejax.train.checkpoint import (
	CheckpointError,
	load_checkpoint,
	read_checkpoint_config,
	resolve_decay_updates,
	save_checkpoint,
)
from pokejax.train.config import TrainConfig
from pokejax.train.ppo import Trainer

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.jsonl"
CHECKPOINT_FILE = "checkpoint.npz"
RESULT_FILE = "result.json"


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


def _keep_checkpoint(run_dir: Path, update: int, learner, config, team_digest, log) -> None:
	"""Keep the learner of this update as checkpoint-<update>.npz; an existing kept file stays."""
	path = run_dir / f"checkpoint-{update}.npz"
	if path.exists():
		return
	save_checkpoint(path, learner, config, team_digest)
	log(f"kept {path.name} at update {update}")


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

	try:
		saved = read_checkpoint_config(checkpoint_path) if resume else None
	except CheckpointError as error:
		raise RunError(str(error)) from error
	# The learning-rate horizon of a resumed run is the one it started with, unless configured.
	config = resolve_decay_updates(config, saved)
	trainer = Trainer(config, team_pool)
	layout = trainer.layout
	learner = trainer.init()
	parameters = trainer.parameter_count(learner)
	cost = trainer.forward_cost(learner)
	if resume:
		try:
			learner = load_checkpoint(checkpoint_path, learner, config, trainer.team_digest)
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
	lane_steps_per_update = layout.lanes * rollout.rollout_length
	environment_rate = trainer.environment_steps_per_second()
	groups = ", ".join(f"{name} {rows}" for name, rows in config.groups.items())
	log(
		f"{config.run.name}: {parameters} parameters, {cost['flops_per_policy_forward']:.0f} "
		f"FLOPs per policy forward; rows: {groups}; {layout.lanes} lanes; "
		f"environment {environment_rate:,.0f} steps/s"
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
					"lane_steps_per_second": lane_steps_per_update / elapsed,
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
				results = trainer.evaluate_all(learner.params, config.evaluation.battles)
				_record(metrics, {"kind": "evaluation", "update": update, "results": results})
				log(_evaluation_line(update, results, config.evaluation.battles))
			if config.checkpoint.every and update % config.checkpoint.every == 0:
				save_checkpoint(checkpoint_path, learner, config, trainer.team_digest)
			if update in config.checkpoint.keep:
				_keep_checkpoint(run_dir, update, learner, config, trainer.team_digest, log)
		training_seconds = time.perf_counter() - start_time
		save_checkpoint(checkpoint_path, learner, config, trainer.team_digest)

		final = trainer.evaluate_all(learner.params, config.evaluation.final_battles)
		_record(
			metrics,
			{"kind": "final_evaluation", "update": config.run.updates, "results": final},
		)
	log(_evaluation_line(config.run.updates, final, config.evaluation.final_battles, final=True))

	env_steps = config.run.updates * steps_per_update
	updates_done = config.run.updates - start
	result = {
		"name": config.run.name,
		"device": str(jax.devices()[0]),
		"jax_version": jax.__version__,
		"config": config.to_dict(),
		"groups": config.groups,
		"lanes": layout.lanes,
		"parameters": parameters,
		"flops_per_policy_forward": cost["flops_per_policy_forward"],
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
		"lane_steps_per_second": (
			lane_steps_per_update * len(warm_seconds) / sum(warm_seconds) if warm_seconds else None
		),
		"environment_steps_per_second": environment_rate,
		"peak_device_bytes": _peak_memory(),
		"final_evaluation": final,
	}
	(run_dir / RESULT_FILE).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
	return result


def _evaluation_line(update: int, results: list[dict], battles: int, final: bool = False) -> str:
	name = "final evaluation" if final else "evaluation"
	parts = [
		f"{result['opponent']} {result['policy']} {100 * result['win_rate']:.2f}% "
		f"[{100 * result['ci95'][0]:.2f}, {100 * result['ci95'][1]:.2f}]"
		for result in results
	]
	return f"{name} at {update} ({battles} battles each): " + "; ".join(parts)
