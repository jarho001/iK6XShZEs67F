"""Resumable learner state: every leaf of the LearnerState as a named array in one .npz file.

The leaves are addressed by their pytree paths, so a checkpoint loads into a learner built from
the same configuration (the template), and every array comes back with its dtype and shape. PRNG
key leaves are stored as their key data. The configuration travels with the arrays; resuming
needs the same configuration, except for the settings that do not change the training (see
RESUME_FREE_SECTIONS and RESUME_FREE_KEYS).
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.train.config import TrainConfig

CONFIG_ENTRY = "__config__"
RESUME_FREE_SECTIONS = ("logging", "evaluation", "checkpoint")
RESUME_FREE_KEYS = (("run", "updates"), ("run", "run_dir"))


class CheckpointError(RuntimeError):
	"""A checkpoint that does not match the learner or the configuration it is resumed with."""


def _is_key(array) -> bool:
	return jax.dtypes.issubdtype(array.dtype, jax.dtypes.prng_key)


def _training_settings(data: dict) -> dict:
	"""The configuration with the settings that may change between a run and its resumption."""
	settings = {name: value for name, value in data.items() if name not in RESUME_FREE_SECTIONS}
	for section, key in RESUME_FREE_KEYS:
		settings[section] = {k: v for k, v in settings[section].items() if k != key}
	return settings


def save_checkpoint(path: Path, learner, config: TrainConfig) -> None:
	"""Write the learner and the configuration; atomically replaces any earlier file at path."""
	arrays = {}
	for key_path, leaf in jax.tree_util.tree_flatten_with_path(learner)[0]:
		name = jax.tree_util.keystr(key_path)
		if _is_key(leaf):
			leaf = jax.random.key_data(leaf)
		arrays[name] = np.asarray(leaf)
	arrays[CONFIG_ENTRY] = np.array(json.dumps(config.to_dict(), sort_keys=True))
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + ".tmp")
	with temporary.open("wb") as handle:
		np.savez(handle, **arrays)
	os.replace(temporary, path)


def load_checkpoint(path: Path, template, config: TrainConfig):
	"""The learner stored at path, with the template's structure; checks the configuration."""
	path = Path(path)
	if not path.exists():
		raise CheckpointError(f"no checkpoint at {path}")
	with np.load(path, allow_pickle=False) as data:
		if CONFIG_ENTRY not in data.files:
			raise CheckpointError(f"{path} has no configuration")
		saved = json.loads(str(data[CONFIG_ENTRY]))
		if _training_settings(saved) != _training_settings(config.to_dict()):
			raise CheckpointError(
				f"{path} was written with a different configuration; resume with the same one "
				"(only run.updates, run.run_dir and the logging, evaluation and checkpoint "
				"sections may change)"
			)
		leaves_with_paths, treedef = jax.tree_util.tree_flatten_with_path(template)
		leaves = []
		for key_path, leaf in leaves_with_paths:
			name = jax.tree_util.keystr(key_path)
			if name not in data.files:
				raise CheckpointError(f"{path} has no entry {name}")
			array = data[name]
			if _is_key(leaf):
				impl = jax.random.key_impl(leaf)
				leaves.append(jax.random.wrap_key_data(jnp.asarray(array), impl=impl))
				continue
			if array.shape != leaf.shape or array.dtype != leaf.dtype:
				raise CheckpointError(
					f"{path}: {name} is {array.dtype}{array.shape}, expected "
					f"{leaf.dtype}{leaf.shape}"
				)
			leaves.append(jnp.asarray(array))
	return jax.tree_util.tree_unflatten(treedef, leaves)
