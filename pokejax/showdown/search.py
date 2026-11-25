"""Run frozen Method 1 on one-sided pinned Showdown requests."""

import time
from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine import state as s
from pokejax.search.method1 import infer
from pokejax.search.rebuild import OWN_FIELDS, PlayerView
from pokejax.showdown.bridge import Bridge
from pokejax.showdown.player import _record_writer, stack_observations
from pokejax.train.network import initial_carry, reset_carry


def stack_views(items, observation, own_dtypes):
	"""Build the searcher's input from one player's messages and stable roster slots."""

	def field(item, name):
		if name != "volatiles":
			return item["own"][name]
		values = np.zeros(s.NUM_VOLATILES, np.int16)
		for key, value in item["own"][name].items():
			values[getattr(s, key)] = value
		return values

	return PlayerView(
		observation,
		tuple(
			np.asarray([field(item, name) for item in items], dtype=dtype)
			for name, dtype in zip(OWN_FIELDS, own_dtypes, strict=True)
		),
		np.asarray([item["side"] for item in items], np.int32),
		np.asarray([item["previous"] if item["held"] else 0 for item in items], np.int32),
		np.asarray([item["held"] for item in items], bool),
		np.asarray([item["refused"] for item in items], bool),
		np.full(len(items), s.ONGOING, np.int8),
	)


@jax.jit
def sample(logits, keys):
	return jax.vmap(jax.random.categorical)(keys, logits).astype(jnp.int32)


class Actor:
	"""Separate instances retain independent learner and opponent carries."""

	def __init__(self, model, params, network, template, own_dtypes, slots, seed, method=None):
		self.model, self.params, self.template, self.own_dtypes = (
			model,
			params,
			template,
			own_dtypes,
		)
		self.hidden = initial_carry(network, (slots,))
		self.seed = seed
		self.keys = jax.random.key(seed)
		self.ply = np.zeros(slots, np.uint32)
		self.beliefs = [None] * slots
		self.slots = slots
		self.method = method
		self.stats = Counter()

	def act(self, items):
		count = len(items)
		padded = items + [items[0]] * (self.slots - count)
		positions = np.asarray([item["slot"] for item in padded], np.int32)
		reset = np.asarray([item["reset"] for item in padded], bool)
		held = np.asarray([item.get("held", False) for item in padded], bool)
		self.ply[positions[reset]] = 0
		ply = self.ply[positions]
		self.ply[positions[:count]] += 1
		game_ids = np.asarray([item["battle"] for item in padded], np.uint32)
		keys = request_keys(self.keys, game_ids, ply)
		observation = stack_observations([item["observation"] for item in padded], self.template)
		carry = reset_carry(jax.tree.map(lambda leaf: leaf[positions], self.hidden), reset)
		after, logits, head = infer(self.model, self.params, observation, carry, held)
		self.hidden = jax.tree.map(
			lambda leaf, new: leaf.at[positions[:count]].set(new[:count]), self.hidden, after
		)
		if self.method is None:
			return np.asarray(sample(logits, keys))[:count]

		for item in items:
			if item["reset"]:
				slot = item["slot"]
				battle = item["battle"]
				self.beliefs[slot] = self.method.new_beliefs([battle, self.seed], 1)[0]
		beliefs = [self.beliefs[slot] for slot in positions[:count]]
		# Padding has a separate belief and a terminal result, so it cannot update an active row.
		if count < self.slots:
			beliefs += self.method.new_beliefs(0, self.slots - count)
		view = stack_views(padded, observation, self.own_dtypes)
		view.result[count:] = s.TIE
		previous = np.asarray([item["previous"] for item in padded], np.int32)
		decision, metrics = self.method.choose(
			self.params,
			view,
			beliefs,
			after,
			logits,
			head,
			keys,
			jnp.zeros_like(logits),
			previous,
			protocol=True,
		)
		action, raw, searched = jax.device_get((decision.action, decision.raw, decision.searched))
		prefix = slice(count)
		self.stats.update(
			eligible=int(metrics.eligible[prefix].sum()),
			no_particle=int(metrics.no_particle[prefix].sum()),
			unrebuildable=int(metrics.unrebuildable[prefix].sum()),
			rejected_worlds=int(
				(
					self.method.config.worlds * (metrics.particles[prefix] > 0)
					- metrics.valid_worlds[prefix]
				).sum()
			),
			attempted_worlds=int(self.method.config.worlds * (metrics.particles[prefix] > 0).sum()),
			searched=int(searched[prefix].sum()),
			overrides=int((action[prefix] != raw[prefix]).sum()),
			ess=float(metrics.ess[prefix].sum()),
		)
		return action[:count]


@jax.jit
def request_keys(key, game_ids, ply):
	return jax.vmap(lambda game, step: jax.random.fold_in(jax.random.fold_in(key, game), step))(
		game_ids, ply
	)


def play(learner, foe, specs, pool, opponent, seed, max_steps, record=None, *, audit=False):
	"""One Node session. Round latency includes messages, stream parsing, inference and rebuild."""
	started = time.perf_counter()
	bridge = Bridge(
		pool,
		specs,
		opponent=opponent,
		seed=seed,
		max_steps=max_steps,
		concurrency=learner.slots,
		record=record is not None,
		search=learner.method is not None,
		audit=audit,
	)
	results, latency, round_seconds = [], [], []
	with _record_writer(record) as writer:
		try:
			round_start = started
			next_progress = started + 30
			while True:
				message = bridge.receive()
				results.extend(message["finished"])
				if writer:
					for line in message["records"]:
						writer.write(line + "\n")
				if message["type"] == "done":
					break
				learner_items = [item for item in message["items"] if item["learner"]]
				foe_items = [item for item in message["items"] if not item["learner"]]
				actions = learner.act(learner_items)
				if foe_items:
					foe_actions = foe.act(foe_items)
					actions = np.concatenate((actions, foe_actions))
				bridge.answer(learner_items + foe_items, actions)
				now = time.perf_counter()
				elapsed = now - round_start
				round_seconds.append(elapsed)
				latency.extend([elapsed * 1_000] * len(learner_items))
				round_start = now
				if now >= next_progress:
					print(
						f"Showdown {len(results)}/{len(specs)} games, {now - started:.0f}s",
						flush=True,
					)
					next_progress = now + 30
		finally:
			bridge.close()
	return sorted(results, key=lambda result: result["index"]), latency, round_seconds
