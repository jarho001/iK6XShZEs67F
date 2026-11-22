"""The sizing sweep without a GPU: the grid and its cells, the resumable results file and the table."""

import os
import sys
from pathlib import Path

import pytest

from pokejax.records import (
	append_record,
	markdown_table,
	read_records,
	run_measurement,
	shard,
	sweep_rows,
)
from pokejax.sizing import (
	Cell,
	GridError,
	cell_config,
	classify_error,
	grid_cells,
	parse_grid,
	run_sweep,
)
from pokejax.train.config import ConfigError, load_config

SELFPLAY = Path(__file__).resolve().parents[1] / "configs" / "selfplay.toml"


def base():
	return load_config(SELFPLAY)


def test_parse_grid_reads_axes_and_rejects_bad_terms():
	assert parse_grid("batch=1024,4096 core=gru,lstm") == {
		"batch": (1024, 4096),
		"core": ("gru", "lstm"),
	}
	for bad in (
		"batch",
		"speed=1",
		"batch=1 batch=2",
		"batch=0",
		"batch=x",
		"core=rnn",
		"rollout=",
	):
		with pytest.raises(GridError):
			parse_grid(bad)


def test_grid_runs_one_axis_changes_before_combinations():
	axes = parse_grid("batch=1024,4096 rollout=64,128 width=1,2 core=gru,lstm")
	cells = grid_cells(axes, base())
	assert len(cells) == 16
	assert len({cell.key for cell in cells}) == 16
	assert cells[0] == Cell(1024, 64, 1, "gru")
	first = Cell(1024, 64, 1, "gru")
	changes = [
		sum(getattr(c, n) != getattr(first, n) for n in ("batch", "rollout", "width", "core"))
		for c in cells
	]
	assert changes == sorted(changes)


def test_axes_left_out_keep_the_configuration():
	cells = grid_cells({}, base())
	assert cells == [
		Cell(base().rollout.batch_size, base().rollout.rollout_length, 1, base().network.core)
	]


def test_cell_config_scales_widths_and_keeps_embeddings():
	config = cell_config(SELFPLAY, Cell(4096, 128, 2, "lstm"))
	network = base().network
	assert (config.rollout.batch_size, config.rollout.rollout_length) == (4096, 128)
	assert config.network.core == "lstm"
	assert config.network.core_width == 2 * network.core_width
	assert config.network.trunk_width == 2 * network.trunk_width
	assert config.network.member_width == 2 * network.member_width
	assert config.network.species_embedding == network.species_embedding


def test_invalid_cell_is_rejected_before_any_run():
	with pytest.raises(GridError, match="batch1000"):
		cell_config(SELFPLAY, Cell(1000, 64, 1, "gru"))
	with pytest.raises(ConfigError):
		load_config(SELFPLAY, ["rollout.batch_size=1000"])


def test_shard_takes_every_nth_cell():
	assert shard(range(7), 1, 3) == [1, 4]
	assert shard(range(7), 0, 1) == list(range(7))
	with pytest.raises(ValueError):
		shard([1], 2, 2)


def test_sweep_skips_recorded_cells_and_resumes(tmp_path):
	cells = [Cell(1024, 64, 1, "gru"), Cell(1024, 64, 1, "lstm"), Cell(1024, 128, 1, "gru")]
	out = tmp_path / "sweep.jsonl"
	append_record(out, {"key": cells[0].key, "status": "failed", "error_class": "oom"})
	calls = []

	def runner(**kwargs):
		calls.append(kwargs["key"])
		return {"key": kwargs["key"], "status": "ok"}

	counts = run_sweep(
		cells,
		config_path=SELFPLAY,
		team_pool=tmp_path / "pool.npz",
		updates=1,
		out=out,
		runner=runner,
		log=lambda text: None,
		log_dir=tmp_path,
	)
	assert calls == [cells[1].key, cells[2].key]
	assert counts == {"ran": 2, "resumed": 1, "left_for_deadline": []}
	again = run_sweep(
		cells,
		config_path=SELFPLAY,
		team_pool=tmp_path / "pool.npz",
		updates=1,
		out=out,
		runner=runner,
		log=lambda text: None,
		log_dir=tmp_path,
	)
	assert again["ran"] == 0 and again["resumed"] == 3


class _Clock:
	def __init__(self):
		self.now = 0.0

	def __call__(self):
		return self.now


def test_deadline_stops_cells_that_would_end_after_it(tmp_path):
	clock = _Clock()
	cells = [Cell(batch, 64, 1, "gru") for batch in (1024, 4096, 16384)]

	def runner(**kwargs):
		clock.now += 50.0
		return {"key": kwargs["key"], "status": "ok"}

	counts = run_sweep(
		cells,
		config_path=SELFPLAY,
		team_pool=tmp_path / "pool.npz",
		updates=1,
		out=tmp_path / "sweep.jsonl",
		deadline=120.0,
		runner=runner,
		clock=clock,
		log=lambda text: None,
		log_dir=tmp_path,
	)
	assert counts["ran"] == 2
	assert counts["left_for_deadline"] == [cells[2].key]


def test_partial_line_does_not_spoil_the_next_record(tmp_path):
	path = tmp_path / "results.jsonl"
	path.write_text('{"key": "cut', encoding="utf-8")
	assert read_records(path) == []
	append_record(path, {"key": "whole", "status": "ok", "value": float("nan")})
	records = read_records(path)
	assert records == [{"key": "whole", "status": "ok", "value": None}]


def test_error_classes():
	assert classify_error(RuntimeError("RESOURCE_EXHAUSTED: Out of memory allocating")) == "oom"
	assert classify_error(RuntimeError("CUDA_ERROR_OUT_OF_MEMORY while allocating")) == "oom"
	assert classify_error(ValueError("bad batch")) == "error"
	assert classify_error(ConfigError("bad")) == "config"


def test_run_measurement_parses_the_last_json_line(tmp_path):
	script = "print('progress'); print('{\"rate\": 3}')"
	measured = run_measurement(
		[sys.executable, "-c", script], gpu=None, log_path=tmp_path / "log.txt", timeout=60
	)
	assert measured["result"] == {"rate": 3}
	assert measured["returncode"] == 0 and not measured["timed_out"]


def test_sweep_table_marks_failures_and_units():
	records = [
		{
			"kind": "sweep",
			"gpu": 0,
			"cell": {"batch": 1024, "rollout": 64, "width": 1, "core": "gru"},
			"status": "ok",
			"parameters": 1000,
			"environment_steps_per_second": 2_000_000.0,
			"learner_steps_per_second": 50_000.0,
			"update_seconds": 1.5,
			"compile_seconds": 30.0,
			"device_memory": {"peak_bytes_in_use": 2 * 2**30},
		},
		{
			"kind": "sweep",
			"gpu": 0,
			"cell": {"batch": 16384, "rollout": 128, "width": 4, "core": "lstm"},
			"status": "failed",
			"error_class": "oom",
		},
	]
	rows = sweep_rows(records)
	assert rows[0][1:5] == ["1024", "64", "1", "gru"]
	assert rows[0][10] == "2.00" and rows[0][11] == "ok"
	assert rows[1][10] == "" and rows[1][11] == "failed (oom)"
	table = markdown_table(["a", "b"], [["x|y", "1"]])
	assert table.splitlines()[-1] == "| x/y | 1 |"


def test_environment_rate_is_measured_once_per_batch_and_rollout(tmp_path):
	cells = [
		Cell(1024, 64, 1, "gru"),
		Cell(1024, 64, 2, "gru"),
		Cell(1024, 64, 1, "lstm"),
		Cell(1024, 128, 1, "gru"),
	]
	out = tmp_path / "sweep.jsonl"
	asked = []

	def runner(**kwargs):
		asked.append((kwargs["key"], kwargs["environment"]))
		rate = 1000.0 if kwargs["environment"] else None
		return {"key": kwargs["key"], "status": "ok", "environment_steps_per_second": rate}

	run_sweep(
		cells,
		config_path=SELFPLAY,
		team_pool=tmp_path / "pool.npz",
		updates=1,
		out=out,
		runner=runner,
		log=lambda text: None,
		log_dir=tmp_path,
	)
	assert [flag for _, flag in asked] == [True, False, False, True]


def test_sweep_rows_show_the_shared_environment_rate():
	records = [
		{
			"kind": "sweep",
			"gpu": 0,
			"cell": {"batch": 1024, "rollout": 64, "width": 1, "core": "gru"},
			"status": "ok",
			"parameters": 1,
			"environment_steps_per_second": 5e5,
			"learner_steps_per_second": 1e4,
			"update_seconds": 1.0,
			"compile_seconds": 2.0,
		},
		{
			"kind": "sweep",
			"gpu": 0,
			"cell": {"batch": 1024, "rollout": 64, "width": 2, "core": "gru"},
			"status": "ok",
			"parameters": 2,
			"environment_steps_per_second": None,
			"learner_steps_per_second": 5e3,
			"update_seconds": 2.0,
			"compile_seconds": 2.0,
		},
	]
	rows = sweep_rows(records)
	assert [row[6] for row in rows] == ["500,000", "500,000"]


def test_bare_options_mean_the_sweep():
	from pokejax.sizing import build_parser, main

	saved = dict(os.environ)  # main() sets the allocator defaults in this process: undo them here
	try:
		args = build_parser().parse_args(["sweep", "--out", "x.jsonl", "--grid", "batch=1024"])
		assert args.command == "sweep" and args.grid == "batch=1024"
		assert main(["cells", "--grid", "batch=1024 core=gru"]) == 0
		with pytest.raises(SystemExit):
			main(["--grid", "batch=1000", "--out", "unused.jsonl"])
	finally:
		os.environ.clear()
		os.environ.update(saved)


def test_child_processes_get_the_xla_defaults_unless_set():
	from pokejax.records import ENVIRONMENT_DEFAULTS, xla_environment

	base = {"PATH": "/bin"}
	env = xla_environment(base, gpu=1)
	assert env["CUDA_VISIBLE_DEVICES"] == "1"
	for name, value in ENVIRONMENT_DEFAULTS:
		assert env[name] == value
	assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
	kept = xla_environment({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"})
	assert kept["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
	assert kept["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.90"
	assert "CUDA_VISIBLE_DEVICES" not in kept
	assert base == {"PATH": "/bin"}  # the caller's mapping is not changed


def test_gpu_memory_query_parses_one_line_and_tolerates_junk():
	from pokejax.records import parse_memory_used

	assert parse_memory_used("86\n") == 86
	assert parse_memory_used(" 1200 \n") == 1200
	assert parse_memory_used("") is None
	assert parse_memory_used("[Not Supported]") is None


def test_records_pool_is_the_env_pool():
	from pokejax.env import DEFAULT_POOL as ENV_POOL
	from pokejax.records import DEFAULT_POOL

	assert DEFAULT_POOL == ENV_POOL
