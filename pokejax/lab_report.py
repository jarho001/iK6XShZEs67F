"""Text of the lab kit: the preflight report (preflight/report.txt) and summary.md, the tables of a run.

Both read only the files a lab run writes, so they can be rebuilt from an archive on any machine.
"""

import itertools
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pokejax.records import (
	GIB,
	SWEEP_HEADERS,
	checkpoint_label,
	markdown_table,
	read_records,
	sweep_rows,
)

BASELINE_ORDER = ("random", "random-move", "max-damage", "heuristics")


def _yes_no(value: Any) -> str:
	return "ok" if value else "FAIL"


def advice(report: Mapping[str, Any]) -> list[str]:
	"""What to try for each failed check, in the words of lab/README.md."""
	checks = report["checks"]
	variant = checks["image_variant"]
	hints = []
	if checks["gpus_listed"] == 0 or checks.get("nvidia_smi_error"):
		hints.append(
			"- The container cannot see the GPUs. Start it with --gpus all (the NVIDIA "
			"Container Toolkit on the host) and check that nvidia-smi works on the host."
		)
	if checks.get("driver_ok_for_variant") is False:
		hints.append(
			f"- The image variant {variant} needs driver {checks['minimum_driver_for_variant']} "
			f"or newer; this host has {checks['driver']}. Build the cuda12 variant instead "
			"(lab/README.md, 'If the preflight fails')."
		)
	jax_probe = checks.get("jax", {})
	if not jax_probe.get("ok"):
		hints.append(
			f"- JAX could not use the GPUs: {jax_probe.get('error') or 'no GPU device'}. "
			"See jax-environment.txt in this folder. Try the other image variant: cuda12 "
			"for drivers from 525, cuda12-driver550 if cuda12 fails on a 550 driver."
		)
	for result in report.get("per_gpu", []):
		for error in result.get("errors", []):
			hints.append(f"- GPU {result.get('index')}: {error}")
	if not hints:
		hints.append("- Send this archive back; the owner's notes are in lab/README.md.")
	return hints


def preflight_text(report: Mapping[str, Any]) -> str:
	checks = report["checks"]
	jax_probe = checks.get("jax", {})
	needed = checks.get("minimum_driver_for_variant")
	lines = [
		"Preflight report",
		(
			"PASS: JAX runs on every GPU and compiles the engine."
			if report["ok"]
			else "FAIL: JAX cannot run the engine on this host. The GPU steps were not started."
		),
		f"Host: {report.get('host')}   time: {report.get('time')}",
		(
			f"Image variant: {checks['image_variant']} "
			f"({checks.get('image_variant_source', 'unknown source')}; "
			+ (f"needs driver {needed} or newer on Linux)" if needed else "no minimum known)")
		),
		(
			f"Driver: {checks['driver']}   CUDA in nvidia-smi: "
			f"{checks['cuda_version_nvidia_smi']}   GPUs listed: {checks['gpus_listed']}"
		),
	]
	for gpu in report.get("gpus", []):
		lines.append(
			f"  GPU {gpu['index']}: {gpu['name']}, {gpu['memory_mib']} MiB, "
			f"compute capability {gpu['compute_capability']}"
		)
	lines += ["", "Checks"]
	lines.append(
		"  nvidia-smi lists the GPUs: "
		+ _yes_no(checks["gpus_listed"] and not checks["nvidia_smi_error"])
	)
	driver_ok = checks.get("driver_ok_for_variant")
	lines.append(
		"  driver for the image variant: "
		+ ("unknown" if driver_ok is None else _yes_no(driver_ok))
	)
	devices = jax_probe.get("devices", [])
	lines.append(
		f"  jax.devices(): {_yes_no(jax_probe.get('ok'))} "
		f"({len(devices)} device(s), JAX {jax_probe.get('jax_version', '?')})"
	)
	for device in devices:
		lines.append(f"    {device['platform']} {device['kind']} (id {device['id']})")
	if jax_probe.get("error"):
		lines.append(f"    error: {jax_probe['error']}")
	for result in report.get("per_gpu", []):
		product = result.get("jitted_product", {})
		compile_ = result.get("engine_cold_compile", {})
		lines.append(
			f"  GPU {result.get('index')}: jitted product {_yes_no(product.get('ok'))}, "
			f"engine cold compile "
			+ (
				f"{_yes_no(compile_.get('ok'))} {compile_.get('compile_seconds', 0):.1f} s"
				if compile_
				else "FAIL"
			)
		)
	if not report["ok"]:
		lines += ["", "What to try"] + advice(report)
	lines += ["", "Bring back the whole archive; this report is preflight/report.txt inside it."]
	return "\n".join(lines) + "\n"


def _load(path: Path) -> dict[str, Any]:
	try:
		return json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}


def _steps_table(lab_dir: Path) -> str:
	"""Each step's last status in progress.log (complete, partial, failed), with its seconds and notes."""
	last: dict[str, dict[str, Any]] = {}
	try:
		lines = (lab_dir / "progress.log").read_text(encoding="utf-8").splitlines()
	except OSError:
		lines = []
	for line in lines:
		try:
			event = json.loads(line)
		except json.JSONDecodeError:
			continue
		if event.get("event") in ("complete", "partial", "failed"):
			last[event.get("step", "")] = event
	order = ("preflight", "facts", "bench", "sweep", "train", "panel", "pack")
	rows = []
	for step in sorted(last, key=lambda name: order.index(name) if name in order else 99):
		event = last[step]
		rows.append(
			[
				step,
				event.get("event", ""),
				f"{event.get('seconds', 0):.0f}",
				"; ".join(event.get("notes", [])),
			]
		)
	return markdown_table(["Step", "Status", "Seconds", "Notes"], rows) if rows else "(none)"


def _bench_table(lab_dir: Path) -> str:
	rows = []
	for path in sorted((lab_dir / "results").glob("bench-gpu*.jsonl")):
		for record in read_records(path):
			result = record.get("result") or {}
			memory = result.get("device_memory") or {}
			peak = memory.get("peak_bytes_in_use")
			ok = record.get("status") == "ok"
			rows.append(
				[
					str(record.get("gpu")),
					str(record.get("mode")),
					str(record.get("policy")),
					"yes" if record.get("observation") else "no",
					str(record.get("batch")),
					f"{result['steps_per_second']:,.0f}" if ok else "",
					f"{result['battles_per_second']:,.1f}" if ok else "",
					f"{result['compile_seconds']:.1f}" if ok else "",
					f"{peak / GIB:.2f}" if peak is not None else "",
					"ok" if ok else f"failed: {record.get('error') or record.get('returncode')}",
				]
			)
	rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], int(row[4])))
	headers = [
		"GPU",
		"Mode",
		"Policy",
		"Observed",
		"Batch",
		"Steps/s",
		"Battles/s",
		"Compile s",
		"Peak GiB",
		"Status",
	]
	return markdown_table(headers, rows) if rows else "(none)"


def _sweep_table(lab_dir: Path) -> str:
	records = []
	for path in sorted((lab_dir / "results").glob("sweep-gpu*.jsonl")):
		records += read_records(path)
	rows = sweep_rows(records)
	return markdown_table(SWEEP_HEADERS, rows) if rows else "(none)"


def attempts_text(attempts: Sequence[Mapping[str, Any]]) -> str:
	"""The attempts of a batch fallback in order, for example "16384 oom, 8192 ok"."""
	parts = []
	for attempt in attempts:
		if attempt.get("status") == "ok":
			outcome = "ok"
		else:
			outcome = str(attempt.get("error_class") or attempt.get("status"))
		parts.append(f"{attempt.get('batch_size')} {outcome}")
	return ", ".join(parts) or "none"


def _training_sections(lab_dir: Path) -> str:
	parts = []
	plan_rows, final_rows, curve_rows = [], [], []
	for run in sorted((lab_dir / "runs").glob("*-gpu*")):
		result = _load(run / "result.json")
		plan = _load(run / "lab-plan.json")
		if plan:
			plan_rows.append(
				[
					run.name,
					str(plan.get("batch_size", "")),
					str(plan.get("updates")),
					f"{plan.get('probe_update_seconds', 0):.3f}",
					str(plan.get("eval_every")),
					str(len(plan.get("keep", []))),
					attempts_text(plan.get("attempts", [])),
				]
			)
		if result:
			peak = result.get("peak_device_bytes")
			steps = result.get("steps_per_second_median")
			env_rate = result.get("environment_steps_per_second")
			for evaluation in result.get("final_evaluation", []):
				final_rows.append(
					[
						run.name,
						evaluation["opponent"],
						evaluation["policy"],
						f"{100 * evaluation['win_rate']:.2f}%",
						f"{100 * evaluation['ci95'][0]:.2f} to {100 * evaluation['ci95'][1]:.2f}",
						str(evaluation["battles"]),
						str(evaluation.get("truncated", "")),
					]
				)
			curve_rows.append(
				[
					run.name,
					str(plan.get("batch_size", "")),
					str(result.get("updates_total")),
					f"{result.get('training_seconds', 0):.0f}",
					f"{steps:,.0f}" if steps else "",
					f"{env_rate:,.0f}" if env_rate else "",
					f"{peak / GIB:.2f}" if peak else "",
				]
			)
	last_training: dict[Any, Mapping[str, Any]] = {}
	for record in read_records(lab_dir / "results" / "train.jsonl"):
		last_training[record.get("gpu")] = record
	if last_training:
		parts.append(
			"Training records (the last attempt of each GPU; its batch is the one the run used)\n\n"
			+ markdown_table(
				["GPU", "Status", "Batch", "Batch attempts"],
				[
					[
						str(gpu),
						str(record.get("status")),
						str(record.get("batch_size", "")),
						attempts_text(record.get("attempts", [])),
					]
					for gpu, record in sorted(last_training.items(), key=lambda item: str(item[0]))
				],
			)
		)
	if plan_rows:
		parts.append(
			"Plan (from a probe of the same configuration on each GPU, at the batch that ran)\n\n"
			+ markdown_table(
				[
					"Run",
					"Batch",
					"Updates",
					"Probe update s",
					"Eval every",
					"Kept checkpoints",
					"Batch attempts",
				],
				plan_rows,
			)
		)
	probes = read_records(lab_dir / "results" / "train-probe.jsonl")
	if probes:
		parts.append(
			"Training probes (one row per attempt; the batch halves on out of memory)\n\n"
			+ markdown_table(
				["GPU", "Batch", "Status", "Error class", "Error"],
				[
					[
						str(probe.get("gpu")),
						str(probe.get("batch_size", "")),
						str(probe.get("status")),
						str(probe.get("error_class") or ""),
						str(probe.get("error") or ""),
					]
					for probe in probes
				],
			)
		)
	if curve_rows:
		parts.append(
			"Runs\n\n"
			+ markdown_table(
				[
					"Run",
					"Batch",
					"Updates",
					"Training s",
					"Learner steps/s (median)",
					"Env steps/s",
					"Peak GiB",
				],
				curve_rows,
			)
		)
	if final_rows:
		parts.append(
			"Final evaluation (sampled policy unless noted)\n\n"
			+ markdown_table(
				["Run", "Opponent", "Policy", "Win rate", "95% CI (%)", "Battles", "Truncated"],
				final_rows,
			)
		)
	for run in sorted((lab_dir / "runs").glob("*-gpu*")):
		rows = []
		for record in read_records(run / "metrics.jsonl"):
			if record.get("kind") != "evaluation":
				continue
			cells = [f"{100 * r['win_rate']:.1f}%" for r in record.get("results", [])]
			labels = [f"{r['opponent']} {r['policy']}" for r in record.get("results", [])]
			rows.append([str(record.get("update"))] + cells)
			header = ["Update"] + labels
		if rows:
			parts.append(
				f"Evaluation during training, {run.name} (win rate)\n\n"
				+ markdown_table(header, rows)
			)
	return "\n\n".join(parts) if parts else "(none)"


def _sweep_arms_table(lab_dir: Path) -> str:
	"""The sweep of each arm (results/sweep-<arm>.jsonl): rows, parameters and rates per batch size."""
	rows = []
	for path in sorted((lab_dir / "results").glob("sweep-*.jsonl")):
		arm = path.name[len("sweep-") : -len(".jsonl")]
		for record in read_records(path):
			if record.get("kind") != "sweep":
				continue
			cell = record.get("cell") or {}
			ok = record.get("status") == "ok"
			peak = (record.get("device_memory") or {}).get("peak_bytes_in_use")
			environment = record.get("environment_steps_per_second")
			rows.append(
				[
					arm,
					str(cell.get("batch", "")),
					f"{record['parameters']:,}" if ok else "",
					f"{environment:,.0f}" if ok and environment is not None else "",
					f"{record['learner_steps_per_second']:,.0f}" if ok else "",
					f"{record['update_seconds']:.3f}" if ok else "",
					f"{record['compile_seconds']:.1f}" if ok else "",
					f"{peak / GIB:.2f}" if peak is not None else "",
					"ok" if ok else f"failed ({record.get('error_class') or record.get('error')})",
				]
			)
	rows.sort(key=lambda row: (row[0], int(row[1] or 0)))
	headers = [
		"Arm",
		"Batch",
		"Parameters",
		"Env steps/s",
		"Learner steps/s",
		"Update s",
		"Compile s",
		"Peak GiB",
		"Status",
	]
	return markdown_table(headers, rows) if rows else "(none)"


def _plans_table(lab_dir: Path) -> str:
	rows = []
	for path in sorted((lab_dir / "runs").glob("*-gpu*/lab-plan.json")):
		plan = _load(path)
		if not plan:
			continue
		spu = plan.get("steps_per_update", 0)
		rows.append(
			[
				path.parent.name,
				str(plan.get("gpu", "")),
				str(plan.get("batch_size", "")),
				str(plan.get("updates", "")),
				f"{plan.get('updates', 0) * spu:,}",
				", ".join(str(update) for update in plan.get("keep", [])),
				str(plan.get("eval_every", "")),
				f"{plan.get('window_seconds', 0):,.0f}",
				f"{plan.get('panel_reserve_seconds') or 0:,.0f}",
			]
		)
	headers = [
		"Run",
		"GPU",
		"Batch",
		"Updates",
		"Env steps",
		"Kept checkpoints (updates)",
		"Eval every",
		"Window s",
		"Panel reserve s",
	]
	return markdown_table(headers, rows) if rows else "(none)"


def _seconds_at(run_dir: Path, update: int) -> float | None:
	"""The wall seconds of a run at an update, interpolated between its logged updates."""
	points = sorted(
		(record["update"], record["seconds"])
		for record in read_records(run_dir / "metrics.jsonl")
		if record.get("kind") == "update"
	)
	if not points:
		return None
	for (low_update, low_seconds), (high_update, high_seconds) in itertools.pairwise(points):
		if low_update <= update <= high_update and high_update > low_update:
			share = (update - low_update) / (high_update - low_update)
			return low_seconds + share * (high_seconds - low_seconds)
	return points[-1][1] if update >= points[-1][0] else points[0][1]


def _opponent_order(names) -> list[str]:
	"""The baselines in the order of the lab's tables, then the other opponents by name."""
	names = list(names)
	first = [name for name in BASELINE_ORDER if name in names]
	return first + sorted(name for name in names if name not in BASELINE_ORDER)


def _panel_sections(lab_dir: Path) -> str:
	"""Per run: the panel's table per checkpoint (steps, wall minutes, win rates) and the paired
	differences between successive kept checkpoints, both action modes."""
	parts = []
	for plan_path in sorted((lab_dir / "runs").glob("*-gpu*/lab-plan.json")):
		run_name = plan_path.parent.name
		plan = _load(plan_path)
		panel = _load(lab_dir / "panel" / f"{run_name}.json")
		if not plan or not panel:
			continue
		final = _load(lab_dir / "panel" / f"{run_name}-final.json")
		opponents = _opponent_order(
			panel["results"][next(iter(panel["results"]))]["sampled"]["opponents"]
		)
		rows = []
		for update in plan.get("keep", []):
			steps = update * plan.get("steps_per_update", 0)
			label = checkpoint_label(run_name, steps)
			source = final if final and update == plan.get("updates") else panel
			report = source.get("results", {}).get(label)
			if report is None:
				continue
			sampled, greedy = report["sampled"], report["greedy"]
			low, high = sampled["panel_ci95"]["ci95"]
			seconds = _seconds_at(lab_dir / "runs" / run_name, update)
			rows.append(
				[
					label,
					f"{int(source.get('battles', 0)):,}",
					f"{steps:,}",
					f"{seconds / 60:.1f}" if seconds is not None else "",
				]
				+ [f"{100 * sampled['opponents'][name]['win_rate']:.2f}" for name in opponents]
				+ [
					f"{100 * sampled['panel_mean']:.2f} [{100 * low:.2f}, {100 * high:.2f}]",
					f"{100 * greedy['panel_mean']:.2f}",
				]
			)
		headers = ["Checkpoint", "Setups", "Steps", "Minutes"] + [
			name for name in opponents
		] + ["Panel mean, sampled [95%]", "Panel mean, greedy"]  # fmt: skip
		parts.append(
			f"{run_name} (win rate in %, sampled actions unless noted)\n\n"
			+ markdown_table(headers, rows)
		)
		diff_rows = []
		for entry in panel.get("differences", []):
			low, high = entry["panel"]["ci95"]
			cells = [
				f"{entry['a']} minus {entry['b']}",
				entry["mode"],
				f"{100 * entry['panel']['estimate']:+.2f} [{100 * low:+.2f}, {100 * high:+.2f}]",
			]
			for name in opponents:
				detail = entry["opponents"].get(name)
				cells.append(
					""
					if detail is None
					else f"{100 * detail['estimate']:+.2f} [{100 * detail['ci95'][0]:+.2f}, {100 * detail['ci95'][1]:+.2f}]"
				)
			diff_rows.append(cells)
		if diff_rows:
			headers = ["Difference", "Mode", "Panel mean [95%]"] + list(opponents)
			parts.append(
				f"{run_name}: paired differences between successive kept checkpoints (points)\n\n"
				+ markdown_table(headers, diff_rows)
			)
	return "\n\n".join(parts) if parts else "(none: no panel report yet)"


def _crossplay_section(lab_dir: Path) -> str:
	report = _load(lab_dir / "panel" / "crossplay.json")
	pairs = report.get("pairs", [])
	if not pairs:
		return "(none: no cross-play report yet)"
	labels = report.get("labels", [])
	parts = []
	for mode in ("sampled", "greedy"):
		cells = {(pair["first"], pair["second"]): pair["modes"][mode] for pair in pairs}
		rows = []
		for row_label in labels[:-1]:
			row = [row_label]
			for column_label in labels[1:]:
				cell = cells.get((row_label, column_label))
				if cell is None or labels.index(column_label) <= labels.index(row_label):
					row.append("")
				else:
					low, high = cell["ci95"]
					row.append(f"{100 * cell['win_rate']:.2f} [{100 * low:.2f}, {100 * high:.2f}]")
			rows.append(row)
		parts.append(
			f"Win rate of the row checkpoint against the column checkpoint, {mode} actions "
			"(each pair plays both seats on the same team draws; 95% Wilson interval)\n\n"
			+ markdown_table(["Row \\ column", *labels[1:]], rows)
		)
	detail_rows = []
	for pair in pairs:
		row = pair["modes"]["sampled"]
		paired = row["paired_draws"]
		detail_rows.append(
			[
				f"{pair['first']} against {pair['second']}",
				f"{100 * row['side0_win_rate']:.2f} / {100 * row['side1_win_rate']:.2f}",
				f"{100 * paired['estimate']:.2f} [{100 * paired['ci95'][0]:.2f}, {100 * paired['ci95'][1]:.2f}]",
				f"{row['truncated']}",
				f"{row['mean_battle_length']:.1f}",
			]
		)
	parts.append(
		"Sampled actions: side 0 / side 1 win rates, the paired-by-draw interval, truncated battles and "
		"mean length\n\n"
		+ markdown_table(
			["Pair", "Side 0 / side 1 %", "Paired by draw % [95%]", "Truncated", "Mean length"],
			detail_rows,
		)
	)
	return "\n\n".join(parts)


def _figures_section(lab_dir: Path) -> str:
	names = (
		sorted(path.name for path in (lab_dir / "plots").glob("*.svg"))
		if (lab_dir / "plots").exists()
		else []
	)
	if not names:
		return "(none)"
	return "\n".join(f"- plots/{name}" for name in names)


def summary_markdown(lab_dir: Path) -> str:
	facts = _load(lab_dir / "facts.json")
	preflight = _load(lab_dir / "preflight" / "preflight.json")
	gpus = facts.get("gpus") or preflight.get("gpus") or []
	gpu_lines = [
		f"- GPU {gpu.get('index')}: {gpu.get('name')}, {gpu.get('memory_mib')} MiB, "
		f"driver {gpu.get('driver')}"
		for gpu in gpus
	]
	bench = "run" if facts.get("lab_bench") else "not run (LAB_BENCH is not 1)"
	sections = [
		"# Lab run summary",
		"",
		"## System",
		"",
		f"- Time (UTC): {facts.get('time_utc', 'unknown')}",
		(
			f"- Host: {facts.get('host', 'unknown')}, CPU {facts.get('cpu', 'unknown')} "
			f"({facts.get('cpu_count', '?')} threads), RAM {_gib(facts.get('ram_bytes'))} GiB"
		),
		f"- Image variant: {facts.get('image_variant', 'unknown')}",
		f"- Commit: {facts.get('git_commit') or facts.get('kit_commit') or 'unknown'}",
		(
			f"- JAX {facts.get('packages', {}).get('jax', '?')}, jaxlib "
			f"{facts.get('packages', {}).get('jaxlib', '?')}, flax "
			f"{facts.get('packages', {}).get('flax', '?')}"
		),
		f"- Budget: LAB_HOURS {facts.get('lab_hours', '?')}, smoke {facts.get('smoke', '?')}",
		f"- Arms: {', '.join(facts.get('arms', [])) or 'unknown'}",
		f"- Benchmarks: {bench}",
		f"- Team pool sha256: {facts.get('team_pool_sha256', 'unknown')}",
		*gpu_lines,
		"",
		"## Steps",
		"",
		_steps_table(lab_dir),
		"",
		"## Preflight",
		"",
		f"Result: {'pass' if preflight.get('ok') else 'fail' if preflight else 'not run'}",
		"",
		"## Benchmarks (engine and environment, per GPU)",
		"",
		_bench_table(lab_dir),
		"",
		"## Sweep (each arm, one process per GPU; rows are the only size that changes)",
		"",
		_sweep_arms_table(lab_dir),
		"",
		"## Training plans (time budget, kept checkpoints)",
		"",
		_plans_table(lab_dir),
		"",
		"## Training runs (time budget)",
		"",
		_training_sections(lab_dir),
		"",
		"## Panel (fixed panel: four baselines and the two references, paired setups)",
		"",
		_panel_sections(lab_dir),
		"",
		"## Cross-play (the final checkpoints and the two references)",
		"",
		_crossplay_section(lab_dir),
		"",
		"## Figures",
		"",
		_figures_section(lab_dir),
		"",
	]
	return "\n".join(sections)


def _gib(value: Any) -> str:
	return f"{value / GIB:.0f}" if isinstance(value, (int, float)) else "unknown"
