"""Resumable learner state: every leaf of the LearnerState as a named array in one .npz file.

The leaves are addressed by their pytree paths, so a checkpoint loads into a learner built from
the same configuration (the template), and every array comes back with its dtype and shape. PRNG
key leaves are stored as their key data. The configuration travels with the arrays; resuming
needs the same configuration, except for the settings that do not change the training (see
RESUME_FREE_SECTIONS and RESUME_FREE_KEYS). The checkpoint also holds a digest of the team pool
the run trains on (pool_digest): a resume with another pool is refused, because the battles in
progress and the fresh ones would come from different teams.

The learning-rate schedule keeps its decay horizon across a resume: ppo.decay_updates is resolved
at the first start (to run.updates when it is zero) and saved with the configuration, so a resume
with a larger run.updates does not stretch the decay, which would raise the learning rate. A
configuration that sets ppo.decay_updates asks for a new schedule and gets it.
"""

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.train.config import TrainConfig

CONFIG_ENTRY = "__config__"
POOL_ENTRY = "__team_pool__"
RESUME_FREE_SECTIONS = ("logging", "evaluation", "checkpoint")
RESUME_FREE_KEYS = (("run", "updates"), ("run", "run_dir"), ("ppo", "decay_updates"))


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


def pool_digest(teams, team_indices) -> str:
	"""A SHA-256 of the team pool's arrays and the indices the sampler draws teams from."""
	digest = hashlib.sha256()
	for name, array in zip(teams._fields, teams, strict=True):
		host = np.ascontiguousarray(np.asarray(array))
		digest.update(f"{name}:{host.dtype}:{host.shape}".encode())
		digest.update(host.tobytes())
	digest.update(np.ascontiguousarray(np.asarray(team_indices)).tobytes())
	return digest.hexdigest()


def resolve_decay_updates(config: TrainConfig, saved: TrainConfig | None) -> TrainConfig:
	"""The configuration with ppo.decay_updates set.

	A value the configuration gives stands (a new schedule). Otherwise the saved configuration's
	decay horizon is kept (for a checkpoint written before the setting existed, its run.updates),
	and a fresh run takes its own run.updates.
	"""
	if config.ppo.decay_updates:
		return config
	if saved is None:
		kept = config.run.updates
	else:
		kept = saved.ppo.decay_updates or saved.run.updates
	return dataclasses.replace(config, ppo=dataclasses.replace(config.ppo, decay_updates=kept))


def read_checkpoint_config(path: Path) -> TrainConfig:
	"""The configuration a checkpoint was written with."""
	path = Path(path)
	if not path.exists():
		raise CheckpointError(f"no checkpoint at {path}")
	with np.load(path, allow_pickle=False) as data:
		if CONFIG_ENTRY not in data.files:
			raise CheckpointError(f"{path} has no configuration")
		return TrainConfig.from_dict(json.loads(str(data[CONFIG_ENTRY])))


def save_checkpoint(path: Path, learner, config: TrainConfig, team_digest: str) -> None:
	"""Write the learner, the configuration and the team pool digest; atomically replaces any
	earlier file at path."""
	arrays = {}
	for key_path, leaf in jax.tree_util.tree_flatten_with_path(learner)[0]:
		name = jax.tree_util.keystr(key_path)
		if _is_key(leaf):
			leaf = jax.random.key_data(leaf)
		arrays[name] = np.asarray(leaf)
	arrays[CONFIG_ENTRY] = np.array(json.dumps(config.to_dict(), sort_keys=True))
	arrays[POOL_ENTRY] = np.array(team_digest)
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	temporary = path.with_name(path.name + ".tmp")
	with temporary.open("wb") as handle:
		np.savez(handle, **arrays)
	os.replace(temporary, path)


def load_checkpoint(path: Path, template, config: TrainConfig, team_digest: str):
	"""The learner stored at path, with the template's structure; checks the configuration and
	the team pool."""
	path = Path(path)
	if not path.exists():
		raise CheckpointError(f"no checkpoint at {path}")
	with np.load(path, allow_pickle=False) as data:
		if CONFIG_ENTRY not in data.files:
			raise CheckpointError(f"{path} has no configuration")
		# A key a checkpoint predates reads as its default, so older checkpoints still resume.
		saved = TrainConfig.from_dict(json.loads(str(data[CONFIG_ENTRY]))).to_dict()
		if _training_settings(saved) != _training_settings(config.to_dict()):
			raise CheckpointError(
				f"{path} was written with a different configuration; resume with the same one "
				"(only run.updates, run.run_dir, ppo.decay_updates and the logging, evaluation and "
				"checkpoint sections may change)"
			)
		if POOL_ENTRY not in data.files:
			raise CheckpointError(
				f"{path} has no team pool digest (it predates the check); it cannot be resumed"
			)
		if str(data[POOL_ENTRY]) != team_digest:
			raise CheckpointError(f"{path} was trained on another team pool; resume with that pool")
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
