"""Deadline fallback must not return a late search result for a later request."""

from threading import Event
from types import SimpleNamespace
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from pokejax.showdown import search


class Opponent(NamedTuple):
	status: np.ndarray


class Observation(NamedTuple):
	legal_actions: np.ndarray
	opponent: Opponent


class Belief:
	def __init__(self):
		self.previous = None
		self.sleep_origin = np.full(6, -1)
		self.updates = 0

	def update(self, obs, *args, **kwargs):
		self.previous = SimpleNamespace(opponent=obs.opponent)
		self.updates += 1

	def sample(self, worlds):
		return None


class SlowMethod:
	config = SimpleNamespace(worlds=16)

	def __init__(self):
		self.entered = Event()
		self.release = Event()
		self.belief = None

	def new_beliefs(self, seed, count):
		self.belief = Belief()
		return [self.belief for _ in range(count)]

	def choose(self, params, view, beliefs, *args, **kwargs):
		beliefs[0].update(view.obs)
		self.entered.set()
		assert self.release.wait(5)
		metrics = SimpleNamespace(
			eligible=np.array([1]),
			no_particle=np.array([0]),
			unrebuildable=np.array([0]),
			particles=np.array([16]),
			valid_worlds=np.array([16]),
			ess=np.array([16.0]),
		)
		return SimpleNamespace(
			action=np.array([2]), raw=np.array([0]), searched=np.array([True])
		), metrics


def test_deadline_uses_current_raw_and_queues_history(monkeypatch):
	monkeypatch.setattr(search, "initial_carry", lambda *_: jnp.zeros((1, 1)))
	monkeypatch.setattr(search, "reset_carry", lambda carry, reset: carry)
	monkeypatch.setattr(search, "stack_observations", lambda values, template: values[0])
	monkeypatch.setattr(
		search,
		"infer",
		lambda model, params, obs, carry, held: (
			carry,
			jnp.asarray(obs.legal_actions),
			jnp.zeros((1, 2)),
		),
	)
	monkeypatch.setattr(
		search,
		"stack_views",
		lambda items, obs, dtypes: SimpleNamespace(
			obs=obs,
			result=np.array([0], np.int8),
			refused=np.array([False]),
			has_held=np.array([False]),
		),
	)
	method = SlowMethod()
	actor = search.Actor(None, None, None, None, [], 1, 1, method, deadline_seconds=0.05)

	def item(first, legal):
		return {
			"slot": 0,
			"battle": 1,
			"reset": first,
			"held": False,
			"refused": False,
			"previous": 0,
			"observation": Observation(np.array([legal]), Opponent(np.zeros((1, 6), np.int8))),
		}

	# Compile the same inference and indexing operations before starting the clock.
	method.release.set()
	actor.deadline_seconds = 10
	actor.act([item(True, [4, 1, 0])])
	method.release.clear()
	method.entered.clear()
	actor.stats.clear()
	actor.deadline_seconds = 0.05
	try:
		assert int(actor.act([item(True, [4, 1, 0])])[0]) == 0
		assert method.entered.is_set()
		assert int(actor.act([item(False, [1, 4, 0])])[0]) == 1
		assert actor.stats["deadline_misses"] == actor.stats["deadline_fallbacks"] == 2
	finally:
		method.release.set()
		actor.executor.shutdown(wait=True)
	assert method.belief.updates == 2
