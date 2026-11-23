"""Training configuration: the default file, validation, overrides and their errors."""

from pathlib import Path

import pytest

from pokejax.train.cli import main
from pokejax.train.config import (
	ConfigError,
	TrainConfig,
	apply_overrides,
	load_config,
	parse_override,
)

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.toml"


def write_toml(tmp_path, text):
	path = tmp_path / "config.toml"
	path.write_text(text, encoding="utf-8")
	return path


def test_default_file_loads_and_is_the_default_run():
	config = load_config(DEFAULT)
	assert config.run.name == "default"
	assert config.network.core in ("gru", "lstm")
	assert config.run_dir == Path("work/runs/default")
	assert config.rollout.rollout_length % config.rollout.sequence_length == 0


def test_no_file_gives_the_defaults():
	assert load_config(None) == TrainConfig()


def test_dictionary_round_trip():
	config = load_config(DEFAULT, ["network.core=lstm", "ppo.learning_rate=1e-4"])
	assert TrainConfig.from_dict(config.to_dict()) == config


def test_unknown_table_and_key_are_errors(tmp_path):
	with pytest.raises(ConfigError, match="unknown table"):
		load_config(write_toml(tmp_path, "[nope]\nx = 1\n"))
	with pytest.raises(ConfigError, match="unknown key"):
		load_config(write_toml(tmp_path, "[ppo]\nlearning_rat = 1.0\n"))


@pytest.mark.parametrize(
	"text, message",
	[
		("[rollout]\nbatch_size = 1.5\n", "must be an integer"),
		("[rollout]\nbatch_size = true\n", "must be an integer"),
		("[rollout]\nbatch_size = 0\n", "at least 1"),
		('[ppo]\nlearning_rate = "fast"\n', "must be a number"),
		('[network]\ncore = "rnn"\n', "must be one of gru, lstm"),
		('[network]\nactivation = "sigmoid"\n', "must be one of relu, tanh, gelu"),
		('[opponent]\nbaseline = "oracle"\n', "must be one of"),
		("[ppo]\nclip = 1.5\n", "must be at most 1.0"),
		("[ppo]\ngamma = 0\n", "out of range"),
		("[env]\nshaping = 1\n", "true or false"),
		("[rollout]\nrollout_length = 10\nsequence_length = 4\n", "must divide"),
		(
			"[rollout]\nbatch_size = 3\nrollout_length = 8\nsequence_length = 4\n",
			"minibatches",
		),
		("[evaluation]\nbattles = 1000\nbatch = 512\n", "multiple of batch"),
		('[run]\nname = "../escape"\n', "not a plain name"),
		("[logging]\nevery = 0\n", "at least 1"),
	],
)
def test_invalid_values_are_errors(tmp_path, text, message):
	with pytest.raises(ConfigError, match=message):
		load_config(write_toml(tmp_path, text))


def test_overrides_set_single_values(tmp_path):
	config = load_config(
		DEFAULT,
		["network.core=lstm", "ppo.learning_rate=1e-4", "run.name=trial", "env.shaping=true"],
	)
	assert config.network.core == "lstm"
	assert config.ppo.learning_rate == 1e-4
	assert config.run.name == "trial"
	assert config.env.shaping is True
	assert config.run_dir == Path("work/runs/trial")


def test_override_text_is_checked():
	assert parse_override("ppo.learning_rate=1e-4") == ("ppo", "learning_rate", "1e-4")
	with pytest.raises(ConfigError, match="must look like"):
		parse_override("ppo.learning_rate")
	with pytest.raises(ConfigError, match="unknown table"):
		parse_override("nope.x=1")
	with pytest.raises(ConfigError, match="unknown key"):
		parse_override("ppo.nope=1")
	with pytest.raises(ConfigError, match="not a TOML value"):
		apply_overrides({}, ["ppo.learning_rate=[1"])


def test_the_command_line_rejects_bad_values(capsys):
	with pytest.raises(SystemExit) as error:
		main(["--updates", "0"])
	assert error.value.code == 2
	assert "run.updates" in capsys.readouterr().err


SELFPLAY = Path(__file__).resolve().parents[1] / "configs" / "selfplay.toml"


def test_the_selfplay_file_loads_with_every_group():
	config = load_config(SELFPLAY)
	assert config.run.name == "selfplay"
	assert config.groups == {
		"selfplay": 512,
		"pool": 256,
		"random": 64,
		"random-move": 64,
		"max-damage": 64,
		"heuristics": 64,
	}
	assert config.pool.size == 8 and config.pool.every == 50
	assert config.evaluation.opponents == ("random", "random-move", "max-damage", "heuristics")
	assert config.evaluation.greedy is True


def test_the_default_file_sends_every_row_to_the_baseline():
	config = load_config(DEFAULT)
	assert config.opponent.shares == {}
	assert config.groups == {"random": 1024}


@pytest.mark.parametrize(
	"overrides, message",
	[
		(["opponent.shares={selfplay=0.5, pool=0.25, random=0.5}"], "must sum to 1"),
		(["opponent.shares={oracle=1.0}"], "opponent.shares key must be one of"),
		(["opponent.shares={random=-0.5, pool=1.5}"], "out of range"),
		(["rollout.batch_size=1001", "opponent.shares={random=0.5, pool=0.5}"], "whole number"),
		(["rollout.batch_size=4", "opponent.shares={pool=0.25, random=0.75}"], "pool.size"),
		(['evaluation.opponents=["oracle"]'], "must be one of"),
		(['evaluation.opponents=["random", "random"]'], "must not repeat"),
		(
			["evaluation.batch=3", "evaluation.battles=3", "evaluation.final_battles=3"],
			"must be even",
		),
		(
			[
				"rollout.batch_size=2",
				"rollout.rollout_length=8",
				"rollout.sequence_length=4",
				"opponent.shares={selfplay=0.5, random=0.5}",
			],
			"minibatches",
		),
	],
)
def test_invalid_groups_and_evaluation_are_errors(overrides, message):
	with pytest.raises(ConfigError, match=message):
		load_config(None, overrides)


def test_group_rows_are_whole_rows_in_group_order():
	config = load_config(
		None,
		[
			"rollout.batch_size=8",
			"pool.size=2",
			"opponent.shares={heuristics=0.125, selfplay=0.5, pool=0.25, random=0.125}",
		],
	)
	assert list(config.groups) == ["selfplay", "pool", "random", "heuristics"]
	assert sum(config.groups.values()) == 8


def test_network_features_are_off_by_default():
	network = load_config(None).network
	assert network.features is False
	assert network.action_head == "dense" and network.action_width == 64


@pytest.mark.parametrize(
	"override, message",
	[
		("network.action_head=factored", "needs network.features"),
		("network.action_head=oracle", "must be one of"),
		("network.features=1", "must be true or false"),
		("network.action_width=0", "at least 1"),
	],
)
def test_invalid_network_features_are_errors(override, message):
	with pytest.raises(ConfigError, match=message):
		load_config(None, [override])


def test_factored_heads_load_with_features():
	network = load_config(None, ["network.features=true", "network.action_head=factored"]).network
	assert network.features and network.action_head == "factored"


def test_feature_set_is_full_by_default_and_needs_features():
	assert load_config(None).network.feature_set == "full"
	network = load_config(None, ["network.features=true", "network.feature_set=dex"]).network
	assert network.feature_set == "dex"
	assert load_config(None).to_dict()["network"]["feature_set"] == "full"


@pytest.mark.parametrize(
	"override, message",
	[
		("network.feature_set=dex", "needs network.features = true"),
		("network.feature_set=sparse", "must be one of"),
	],
)
def test_invalid_feature_sets_are_errors(override, message):
	with pytest.raises(ConfigError, match=message):
		load_config(None, [override])


def test_a_pool_turns_over_within_a_battle_or_the_config_is_refused():
	"""A pool battle may not outlive two insertions into its member: pool.size x pool.every x
	rollout.rollout_length must reach env.max_steps (train_support's league has 2 x 1 x 256 >= 400)."""
	from train_support import league_config

	assert league_config("env.max_steps=400").env.max_steps == 400
	with pytest.raises(ConfigError, match="may not outlive two insertions"):
		league_config("env.max_steps=600")
