"""Run frozen Method 1 on one-sided pinned Showdown requests."""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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

	def __init__(
		self,
		model,
		params,
		network,
		template,
		own_dtypes,
		slots,
		seed,
		method=None,
		*,
		deadline_seconds=None,
		greedy=False,
	):
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
		self.greedy = greedy
		self.stats = Counter()
		self.deadline_seconds = deadline_seconds
		self.executor = ThreadPoolExecutor(max_workers=1) if deadline_seconds is not None else None
		self.pending = None
		self.search_errors = []

	def act(self, items):
		started = time.monotonic()
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
			choices = jnp.argmax(logits, axis=-1) if self.greedy else sample(logits, keys)
			return np.asarray(choices)[:count]

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
		origins = [item.get("sleepOrigins", [-1] * 6) for item in padded]
		args = (
			self.params,
			view,
			beliefs,
			after,
			logits,
			head,
			keys,
			jnp.zeros_like(logits),
			previous,
		)
		if self.executor is None:
			decision, metrics = self.method.choose(*args, protocol=True, sleep_origins=origins)
			decision = jax.device_get(decision)
		else:
			# Only the worker owns the belief histories. A late search may finish,
			# but its choice is discarded; later observations are queued in order.
			raw = np.asarray(jnp.argmax(logits, axis=-1), np.int32)
			if self.pending is not None and self.pending.done():
				try:
					self.pending.result()
				except Exception as error:  # noqa: BLE001 - record late worker failures
					self.search_errors.append(repr(error))
					self.stats["search_errors"] += 1
				self.pending = None
			if self.pending is not None or time.monotonic() - started >= self.deadline_seconds:
				update = self.executor.submit(self._update_beliefs, view, beliefs, origins)
				update.add_done_callback(self._background_error)
				self.stats.update(deadline_misses=count, deadline_fallbacks=count)
				return raw[:count]
			self.pending = self.executor.submit(self._run_search, args, origins)
			try:
				remaining = max(0.0, self.deadline_seconds - (time.monotonic() - started))
				decision, metrics = self.pending.result(timeout=remaining)
				self.pending = None
			except TimeoutError:
				self.stats.update(deadline_misses=count, deadline_fallbacks=count)
				return raw[:count]
			except Exception as error:  # noqa: BLE001 - report worker failures and fall back
				self.search_errors.append(repr(error))
				self.stats.update(search_errors=1, error_fallbacks=count)
				self.pending = None
				return raw[:count]
		action, raw, searched = decision.action, decision.raw, decision.searched
		prefix = slice(count)
		if self.executor is None:
			self._record_sleep(beliefs[:count])
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

	def _background_error(self, future):
		error = future.exception()
		if error is not None:
			self.search_errors.append(repr(error))
			self.stats["search_errors"] += 1

	def _record_sleep(self, beliefs):
		for belief in beliefs:
			if belief.previous is None:
				continue
			sleeping = belief.previous.opponent.status == s.SLEEP
			self.stats.update(
				sleep_unknown=int((sleeping & (belief.sleep_origin < 0)).sum()),
				sleep_self=int((sleeping & (belief.sleep_origin == 0)).sum()),
				sleep_foe=int((sleeping & (belief.sleep_origin > 0)).sum()),
			)

	def _update_beliefs(self, view, beliefs, origins):
		for i, belief in enumerate(beliefs):
			obs = jax.tree.map(lambda x, i=i: x[i], view.obs)
			live = view.result[i] == s.ONGOING
			belief.update(
				obs,
				bool(view.refused[i] or view.has_held[i]),
				propose=live,
				sleep_origin=origins[i],
			)
			if live and not view.has_held[i] and obs.legal_actions.sum() > 1:
				belief.sample(self.method.config.worlds)
		self._record_sleep(beliefs)

	def _run_search(self, args, origins):
		decision, metrics = self.method.choose(*args, protocol=True, sleep_origins=origins)
		self._record_sleep(args[2])
		return jax.device_get((decision, metrics))


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
