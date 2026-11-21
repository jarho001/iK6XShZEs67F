"""Training configuration: a TOML file, validated dataclass sections and command-line overrides.

A file has these tables, and every key is optional (the defaults are the ones below):

- [run]: name, seed, updates, run_dir (default work/runs/<name>)
- [opponent]: baseline (random, random-move, max-damage or heuristics)
- [rollout]: batch_size (parallel battles), rollout_length (steps per update),
  sequence_length (steps per minibatch chunk; must divide rollout_length)
- [env]: max_steps (environment steps before a battle is truncated), shaping, shaping_coef
- [network]: core (gru or lstm), activation, core_width, trunk_width, member_width and the
  embedding sizes
- [ppo]: epochs, minibatches, learning_rate, learning_rate_end, lr_schedule (constant, linear or
  cosine), gamma, gae_lambda, clip, value_clip (0 turns value clipping off), value_coef,
  entropy_coef, max_grad_norm, adam_eps
- [logging]: every (updates between metric lines)
- [evaluation]: every (updates between evaluations, 0 for none), battles, final_battles, batch
  (battles per evaluation call; battles and final_battles must be multiples of it)
- [checkpoint]: every (updates between checkpoints, 0 for only the final one)

Unknown tables and keys are errors. Overrides are `section.key=value`, with the value read as
TOML (a string needs no quotes).
"""

import dataclasses
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pokejax.baselines import POLICY_NAMES

CORE_KINDS = ("gru", "lstm")
ACTIVATIONS = ("relu", "tanh", "gelu")
LR_SCHEDULES = ("constant", "linear", "cosine")
DEFAULT_RUN_ROOT = Path("work/runs")
RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


class ConfigError(ValueError):
	"""A configuration file or override that does not validate."""


def _check(condition: bool, message: str) -> None:
	if not condition:
		raise ConfigError(message)


def _check_int(value, minimum: int, name: str) -> None:
	_check(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
	_check(value >= minimum, f"{name} must be at least {minimum}")


def _check_float(value, low: float, high: float | None, name: str, low_open=False) -> None:
	_check(
		isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} must be a number"
	)
	_check(value > low if low_open else value >= low, f"{name} is out of range")
	_check(high is None or value <= high, f"{name} must be at most {high}")


def _check_choice(value, choices, name: str) -> None:
	_check(value in choices, f"{name} must be one of {', '.join(choices)}, not {value!r}")


@dataclass(frozen=True)
class RunSection:
	name: str = "default"
	seed: int = 0
	updates: int = 600
	run_dir: str = ""  # empty: work/runs/<name>

	def __post_init__(self):
		_check(bool(RUN_NAME.fullmatch(self.name)), f"run.name {self.name!r} is not a plain name")
		_check_int(self.seed, 0, "run.seed")
		_check_int(self.updates, 1, "run.updates")


@dataclass(frozen=True)
class OpponentSection:
	baseline: str = "random"

	def __post_init__(self):
		_check_choice(self.baseline, POLICY_NAMES, "opponent.baseline")


@dataclass(frozen=True)
class RolloutSection:
	batch_size: int = 1024
	rollout_length: int = 64
	sequence_length: int = 16

	def __post_init__(self):
		_check_int(self.batch_size, 1, "rollout.batch_size")
		_check_int(self.rollout_length, 1, "rollout.rollout_length")
		_check_int(self.sequence_length, 1, "rollout.sequence_length")
		_check(
			self.rollout_length % self.sequence_length == 0,
			"rollout.sequence_length must divide rollout.rollout_length",
		)


@dataclass(frozen=True)
class EnvSection:
	max_steps: int = 1000
	shaping: bool = False
	shaping_coef: float = 0.1

	def __post_init__(self):
		_check_int(self.max_steps, 1, "env.max_steps")
		_check(isinstance(self.shaping, bool), "env.shaping must be true or false")
		_check_float(self.shaping_coef, 0.0, None, "env.shaping_coef")


@dataclass(frozen=True)
class NetworkSection:
	core: str = "gru"
	activation: str = "relu"
	core_width: int = 256
	trunk_width: int = 256
	member_width: int = 64
	species_embedding: int = 32
	move_embedding: int = 16
	item_embedding: int = 8
	ability_embedding: int = 8
	type_embedding: int = 8
	code_embedding: int = 4

	def __post_init__(self):
		_check_choice(self.core, CORE_KINDS, "network.core")
		_check_choice(self.activation, ACTIVATIONS, "network.activation")
		for name in (
			"core_width",
			"trunk_width",
			"member_width",
			"species_embedding",
			"move_embedding",
			"item_embedding",
			"ability_embedding",
			"type_embedding",
			"code_embedding",
		):
			_check_int(getattr(self, name), 1, f"network.{name}")


@dataclass(frozen=True)
class PPOSection:
	epochs: int = 4
	minibatches: int = 4
	learning_rate: float = 3e-4
	learning_rate_end: float = 3e-5
	lr_schedule: str = "linear"
	gamma: float = 0.99
	gae_lambda: float = 0.95
	clip: float = 0.2
	value_clip: float = 0.2
	value_coef: float = 0.5
	entropy_coef: float = 0.01
	max_grad_norm: float = 0.5
	adam_eps: float = 1e-5

	def __post_init__(self):
		_check_int(self.epochs, 1, "ppo.epochs")
		_check_int(self.minibatches, 1, "ppo.minibatches")
		_check_choice(self.lr_schedule, LR_SCHEDULES, "ppo.lr_schedule")
		_check_float(self.learning_rate, 0.0, None, "ppo.learning_rate", low_open=True)
		_check_float(self.learning_rate_end, 0.0, None, "ppo.learning_rate_end")
		_check_float(self.gamma, 0.0, 1.0, "ppo.gamma", low_open=True)
		_check_float(self.gae_lambda, 0.0, 1.0, "ppo.gae_lambda")
		_check_float(self.clip, 0.0, 1.0, "ppo.clip", low_open=True)
		_check_float(self.value_clip, 0.0, None, "ppo.value_clip")
		_check_float(self.value_coef, 0.0, None, "ppo.value_coef")
		_check_float(self.entropy_coef, 0.0, None, "ppo.entropy_coef")
		_check_float(self.max_grad_norm, 0.0, None, "ppo.max_grad_norm", low_open=True)
		_check_float(self.adam_eps, 0.0, None, "ppo.adam_eps", low_open=True)


@dataclass(frozen=True)
class LoggingSection:
	every: int = 10

	def __post_init__(self):
		_check_int(self.every, 1, "logging.every")


@dataclass(frozen=True)
class EvaluationSection:
	every: int = 50
	battles: int = 1024
	final_battles: int = 4096
	batch: int = 512

	def __post_init__(self):
		_check_int(self.every, 0, "evaluation.every")
		_check_int(self.batch, 1, "evaluation.batch")
		_check_int(self.battles, 1, "evaluation.battles")
		_check_int(self.final_battles, 1, "evaluation.final_battles")
		_check(self.battles % self.batch == 0, "evaluation.battles must be a multiple of batch")
		_check(
			self.final_battles % self.batch == 0,
			"evaluation.final_battles must be a multiple of batch",
		)


@dataclass(frozen=True)
class CheckpointSection:
	every: int = 100

	def __post_init__(self):
		_check_int(self.every, 0, "checkpoint.every")


SECTIONS = {
	"run": RunSection,
	"opponent": OpponentSection,
	"rollout": RolloutSection,
	"env": EnvSection,
	"network": NetworkSection,
	"ppo": PPOSection,
	"logging": LoggingSection,
	"evaluation": EvaluationSection,
	"checkpoint": CheckpointSection,
}


@dataclass(frozen=True)
class TrainConfig:
	run: RunSection = RunSection()
	opponent: OpponentSection = OpponentSection()
	rollout: RolloutSection = RolloutSection()
	env: EnvSection = EnvSection()
	network: NetworkSection = NetworkSection()
	ppo: PPOSection = PPOSection()
	logging: LoggingSection = LoggingSection()
	evaluation: EvaluationSection = EvaluationSection()
	checkpoint: CheckpointSection = CheckpointSection()

	def __post_init__(self):
		chunks = self.rollout.rollout_length // self.rollout.sequence_length
		total = self.rollout.batch_size * chunks
		_check(
			total % self.ppo.minibatches == 0,
			f"batch_size x rollout_length / sequence_length = {total} sequences must divide "
			f"evenly into ppo.minibatches = {self.ppo.minibatches}",
		)

	@property
	def run_dir(self) -> Path:
		return Path(self.run.run_dir) if self.run.run_dir else DEFAULT_RUN_ROOT / self.run.name

	def to_dict(self) -> dict[str, Any]:
		return {section: dataclasses.asdict(getattr(self, section)) for section in SECTIONS}

	@classmethod
	def from_dict(cls, data: Mapping[str, Any]) -> "TrainConfig":
		unknown = set(data) - set(SECTIONS)
		_check(not unknown, f"unknown table(s): {', '.join(sorted(unknown))}")
		sections = {}
		for name, section_type in SECTIONS.items():
			sections[name] = _section(section_type, name, data.get(name, {}))
		return cls(**sections)


def _section(section_type, name: str, values) -> Any:
	_check(isinstance(values, Mapping), f"[{name}] must be a table")
	known = {field.name: field for field in fields(section_type)}
	unknown = set(values) - set(known)
	_check(not unknown, f"unknown key(s) in [{name}]: {', '.join(sorted(unknown))}")
	converted = {}
	for key, value in values.items():
		converted[key] = _coerce(value, known[key].type, f"{name}.{key}")
	return section_type(**converted)


def _coerce(value, kind, name: str):
	if kind is bool:
		_check(isinstance(value, bool), f"{name} must be true or false")
		return value
	if kind is int:
		_check(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
		return value
	if kind is float:
		_check(
			isinstance(value, (int, float)) and not isinstance(value, bool),
			f"{name} must be a number",
		)
		return float(value)
	if kind is str:
		_check(isinstance(value, str), f"{name} must be a string")
		return value
	raise TypeError(f"unsupported field type for {name}: {kind}")


def parse_override(text: str) -> tuple[str, str, str]:
	"""Split `section.key=value` into (section, key, raw value)."""
	path, separator, raw = text.partition("=")
	section, dot, key = path.partition(".")
	_check(bool(separator and dot), f"override {text!r} must look like section.key=value")
	_check(section in SECTIONS, f"override {text!r}: unknown table {section!r}")
	_check(key in {f.name for f in fields(SECTIONS[section])}, f"override {text!r}: unknown key")
	return section, key, raw


def apply_overrides(data: dict[str, Any], overrides) -> dict[str, Any]:
	"""A copy of the nested dict with each section.key=value override set (later ones win)."""
	result = {section: dict(values) for section, values in data.items()}
	for text in overrides:
		section, key, raw = parse_override(text)
		kind = {f.name: f.type for f in fields(SECTIONS[section])}[key]
		if kind is str:
			value = raw
		else:
			try:
				value = tomllib.loads(f"value = {raw}")["value"]
			except tomllib.TOMLDecodeError as error:
				raise ConfigError(f"override {text!r} is not a TOML value") from error
		result.setdefault(section, {})[key] = value
	return result


def load_config(path: Path | None = None, overrides=()) -> TrainConfig:
	"""The configuration of a TOML file (none: all defaults), with overrides applied."""
	data = {}
	if path is not None:
		try:
			data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
		except tomllib.TOMLDecodeError as error:
			raise ConfigError(f"{path}: {error}") from error
	return TrainConfig.from_dict(apply_overrides(data, overrides))
