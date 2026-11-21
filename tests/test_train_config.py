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
