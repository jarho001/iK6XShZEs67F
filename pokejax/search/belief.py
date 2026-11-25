"""Frequency-weighted random sets and a battle-local posterior over opposing rosters."""

import hashlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

from pokejax.engine.reset import Teams
from pokejax.engine.state import SLEEP, TEAM_SLOTS
from pokejax.teams import load_npz


class Particles(NamedTuple):
	sets: np.ndarray
	disguise: np.ndarray


class Catalogue:
	def __init__(self, sets, weight, rosters, dex):
		self.sets = sets
		self.weight = weight
		self.species = np.unique(sets.species)
		self.by_species = {int(s): np.flatnonzero(sets.species == s) for s in self.species}
		self.ids = dex.species.index
		self.move = dex.moves.index
		self.zoroark = self.ids["zoroark"]
		self.base_form = {
			self.ids[form]: self.ids[base]
			for base, form in (
				("meloetta", "meloettapirouette"),
				("castform", "castformrainy"),
				("castform", "castformsunny"),
				("castform", "castformsnowy"),
				("cherrim", "cherrimsunshine"),
			)
		}
		self.trace = dex.abilities.index["trace"]
		self.regenerator = dex.abilities.index["regenerator"]
		self.pressure = dex.abilities.index["pressure"]
		self.ghost = dex.types.index["ghost"] + 1
		targets = np.asarray(dex.moves.target_names)[dex.moves.target]
		self.pressure_target = (
			np.isin(
				targets,
				(
					"normal",
					"any",
					"randomNormal",
					"allAdjacent",
					"allAdjacentFoes",
					"all",
					"scripted",
				),
			)
			| dex.moves.flags[:, dex.moves.flag_names.index("mustpressure")]
		)
		self.weather_moves = {self.move[m] for m in ("sandstorm", "raindance", "sunnyday", "hail")}
		self.callers = (self.move["sleeptalk"], self.move["naturepower"])
		self.sleep_moves = {
			self.move[name]
			for name in (
				"spore",
				"sleeppowder",
				"sing",
				"hypnosis",
				"lovelykiss",
				"darkvoid",
				"grasswhistle",
			)
		}
		self.mutable_abilities = {dex.abilities.index[name] for name in ("mummy", "trace")}
		self.transfer_moves = {self.move[m] for m in ("trick", "switcheroo", "thief", "covet")}
		self.ability_moves = {
			self.move[m]
			for m in ("entrainment", "gastroacid", "simplebeam", "skillswap", "worryseed")
		}
		# Smoothed empirical conditional species prior. Start from intact generator rows.
		member = np.zeros((len(rosters), len(dex.species.ids)), np.float32)
		member[np.arange(len(rosters))[:, None], rosters] = 1
		count = member.sum(0)
		pair = member.T @ member[:, self.species]
		self.prior = count[self.species] / count.sum()
		self.relative = (pair + 40 * self.prior) / (count[:, None] + 40) / self.prior
		self.species_number = np.asarray(dex.species.num)[self.species]
		self.number_by_id = dict(zip(self.species, self.species_number, strict=True))
		width = max(map(len, self.by_species.values()))
		self.set_rows = np.zeros((len(self.species), width), np.int32)
		self.set_cdf = np.ones((len(self.species), width))
		for i, s in enumerate(self.species):
			rows = self.by_species[s]
			self.set_rows[i, : len(rows)] = rows
			self.set_cdf[i, : len(rows)] = np.minimum(self.prob(rows).cumsum(), 1)
			self.set_cdf[i, len(rows) - 1] = 1
		self.illusion_rate = float(count[self.zoroark] / len(rosters))

	def prob(self, rows):
		w = self.weight[rows].astype(np.float64)
		return w / w.sum()

	@classmethod
	def load(cls, pool_path, dex, *, extra=None, cache=None):
		paths = [Path(pool_path), *([Path(extra)] if extra is not None else [])]
		digest = hashlib.sha256()
		for path in paths:
			with path.open("rb") as source:
				digest.update(hashlib.file_digest(source, "sha256").digest())
		fingerprint = digest.hexdigest()
		if cache is not None:
			cache = Path(cache)
		if cache is not None and cache.exists():
			with np.load(cache, allow_pickle=False) as data:
				if data["fingerprint"].item() == fingerprint:
					return cls(
						Teams(*(data[name] for name in Teams._fields)),
						data["weight"],
						data["rosters"],
						dex,
					)
		pools = [load_npz(path) for path in paths]
		if len({pool.showdown_commit for pool in pools}) != 1:
			raise ValueError("Catalogue inputs must come from the same pinned generator")
		columns = [
			np.concatenate([getattr(pool, name) for pool in pools]) for name in Teams._fields
		]
		rosters = columns[0]
		moves = columns[Teams._fields.index("moves")]
		moves[:] = -np.sort(-moves, axis=-1)
		flat = [column.reshape(len(rosters) * TEAM_SLOTS, -1) for column in columns]
		_, first, weight = np.unique(
			np.concatenate(flat, axis=1), axis=0, return_index=True, return_counts=True
		)
		sets = Teams(
			*(
				col[first].reshape(-1, *src.shape[2:])
				for col, src in zip(flat, columns, strict=True)
			)
		)
		if cache is not None:
			cache.parent.mkdir(parents=True, exist_ok=True)
			np.savez_compressed(
				cache, **sets._asdict(), weight=weight, rosters=rosters, fingerprint=fingerprint
			)
		return cls(sets, weight, rosters, dex)

	def candidates(self, species, level, moves, item, ability):
		rows = self.by_species.get(self.base_form.get(species, species), np.empty(0, np.int64))
		sets = self.sets
		if level:
			rows = rows[sets.level[rows] == level]
		for move in moves:
			if move:
				rows = rows[np.any(sets.moves[rows] == move, axis=-1)]
		if item:
			rows = rows[sets.item[rows] == item]
		if ability:
			rows = rows[(sets.ability[rows] == ability) | (sets.ability[rows] == self.trace)]
		return rows


class History(NamedTuple):
	spent_moves: np.ndarray
	spent_pp: np.ndarray
	last_move: np.ndarray
	active_turns: np.ndarray
	status_age: np.ndarray
	sleep_from_foe: np.ndarray
	sleep_unknown: np.ndarray
	last_faint: bool
	queued_ours: bool
	transform_source: int
	transformed_ranks: np.ndarray
	weather_permanent: bool
	hazard_order: np.ndarray


class Belief:
	"""Keep a bank at a fixed support size, redrawing when public set evidence changes."""

	def __init__(self, catalogue, rng, size=128):
		self.catalogue = catalogue
		self.rng = rng
		self.size = size
		self.signature = None
		self.bank = None
		self.previous = None
		self.active = 0
		self.turn = 0
		self.transform_source = 0
		self.transform_species = 0
		self.transformed_ranks = np.zeros(TEAM_SLOTS, bool)
		self.items_changed = False
		self.abilities_changed = False
		self.spent = [{} for _ in range(TEAM_SLOTS)]
		self.copied_spent = {}
		self.last_move = np.zeros(TEAM_SLOTS, np.int16)
		self.active_turns = np.zeros(TEAM_SLOTS, np.uint8)
		self.status_age = np.zeros(TEAM_SLOTS, np.int8)
		# -1 means the public history did not establish the source.
		self.sleep_origin = np.full(TEAM_SLOTS, -1, np.int8)
		self.fainted = 0
		self.last_faint = False
		self.took_turn = False
		self.weather_permanent = False
		self.hazard_order = np.zeros((2, 2), np.int8)

	def update(self, obs, refused=False, propose=True, sleep_origin=None):
		foe = obs.opponent
		count = int(np.count_nonzero(~foe.unseen.astype(bool)))
		active = int(np.argmax(foe.is_active))
		turn = round(float(obs.turn) * 1000)
		c = self.catalogue
		advanced = self.previous is not None and not refused
		event_move = int(obs.events.move[1]) if advanced else 0
		move = event_move
		if advanced:
			events = set(map(int, obs.events.move))
			self.items_changed |= bool(events & c.transfer_moves)
			self.abilities_changed |= bool(events & c.ability_moves)
			self.abilities_changed |= any(int(a) in c.mutable_abilities for a in foe.ability)
			# Item changes can be called by Sleep Talk or lost in the compressed event
			# record of a pivot. An announced removal by taking is ambiguous.
			self.items_changed |= bool(np.any(foe.item_knocked_off))
			if turn != self.turn:
				faints = int(foe.fainted.sum())
				self.last_faint = faints > self.fainted
				self.fainted = faints
			if obs.events.entered[1]:
				self.last_move[self.active] = 0
				self.active_turns[active] = 0
				self.copied_spent.clear()
			if move and obs.events.order[1] and not obs.events.entered[1]:
				# The public revealed slot list names the original caller. The event
				# can name the called move. When either was already known, PP is ambiguous.
				row = foe.moves[active]
				previous = self.previous.opponent.moves[active]
				new = row[(row > 0) & ~np.isin(row, previous)]
				called = move
				if move not in row:
					move = int(new[0]) if len(new) == 1 else 0
				elif len(new) == 1 and new[0] in c.callers:
					move = int(new[0])
				elif (
					np.any(np.isin(row, c.callers))
					and self.previous.opponent.status[active] == SLEEP
				):
					move = 0
				self.last_move[active] = move
				if move and not self.previous.opponent.charging:
					spent = self.copied_spent if foe.transformed else self.spent[active]
					pressured = self._pressure_cost(obs, active, move, called)
					spent[move] = min(255, spent.get(move, 0) + 1 + pressured)
			if turn != self.turn:
				self.active_turns[active] = min(int(self.active_turns[active]) + 1, 255)
				same = (foe.status == self.previous.opponent.status) & (foe.status > 0)
				self.status_age = np.where(same, np.minimum(self.status_age + 1, 20), 0).astype(
					np.int8
				)
		elif self.previous is None:
			self.active_turns[active] = 1
		# The event is a compressed public record; only an observed Rest establishes
		# self-inflicted sleep without relying on the hidden engine state. A directly
		# observed sleep move also identifies our side as the source.
		if advanced:
			new_sleep = (foe.status == SLEEP) & (self.previous.opponent.status != SLEEP)
			self.sleep_origin[new_sleep] = -1
			if event_move == c.move["rest"] and obs.events.status[1] == SLEEP:
				# A phazing move may have replaced the Rest user before this request.
				rest_user = self.active if obs.events.entered[1] else active
				self.sleep_origin[rest_user] = 0
			elif (
				obs.events.status[1] == SLEEP
				and (
					int(obs.events.move[0]) in c.sleep_moves
					or (self.previous.opponent.yawn and active == self.active)
				)
				and not obs.events.entered[1]
				and foe.status[active] == SLEEP
			):
				self.sleep_origin[active] = 1
		self.sleep_origin[foe.status != SLEEP] = -1
		if sleep_origin is not None:
			known = (foe.status == SLEEP) & (np.asarray(sleep_origin) >= 0)
			self.sleep_origin[known] = np.asarray(sleep_origin)[known]
		if foe.transformed and (
			self.previous is None or not self.previous.opponent.transformed or active != self.active
		):
			self.transform_species = (
				int(self.previous.opponent.species[active])
				if self.previous is not None
				and not obs.events.entered[1]
				and event_move == c.move["transform"]
				else c.ids["ditto"]
			)
			self.transform_source = int(np.argmax(obs.own.is_active))
		if foe.transformed:
			self.transformed_ranks[active] = True
		if obs.weather and (
			self.previous is None
			or obs.weather != self.previous.weather
			or obs.weather_age < self.previous.weather_age
		):
			self.weather_permanent = not bool(c.weather_moves & set(map(int, obs.events.move)))
		self.hazard_order[obs.hazards == 0] = 0
		for side, hazard in zip(
			*np.nonzero((obs.hazards > 0) & (self.hazard_order == 0)), strict=True
		):
			self.hazard_order[side, hazard] = self.hazard_order[side].max() + 1
		self.previous = obs
		self.active, self.turn = active, turn
		evidence = tuple(
			(
				self.transform_species if active == i and foe.transformed else int(foe.species[i]),
				round(float(foe.level[i]) * 100),
				() if active == i and foe.transformed else tuple(map(int, foe.moves[i])),
				0 if self.items_changed else int(foe.item[i]),
				0 if self.abilities_changed or self.transformed_ranks[i] else int(foe.ability[i]),
			)
			for i in range(count)
		)
		# Rejuvenate by drawing from the conditional proposal instead of rejecting
		# independently sampled whole teams. A retained bank does not count an event twice.
		if propose and evidence != self.signature:
			self.signature = evidence
			self.bank = self._propose(evidence)

		physical = self.spent[active]
		if foe.transformed:
			physical = self.copied_spent
		spent = [*self.spent]
		spent[active] = physical
		spent_pp = np.array(
			[
				[max(table.get(int(m), 1), 1) if m else 0 for m in row]
				for table, row in zip(spent, foe.moves, strict=True)
			],
			np.uint8,
		)
		queued = bool(
			obs.mid_turn
			and obs.request[0]
			and not obs.events.order[0]
			and not obs.events.entered[0]
		)
		if obs.mid_turn and not obs.request[1] and not obs.request[2] and obs.events.order[0]:
			self.took_turn = True
		elif not obs.mid_turn:
			self.took_turn = False
		return History(
			foe.moves,
			spent_pp,
			self.last_move.copy(),
			self.active_turns.copy(),
			self.status_age.copy(),
			self.sleep_origin != 0,
			(foe.status == SLEEP) & (self.sleep_origin < 0),
			self.last_faint,
			queued and not self.took_turn,
			self.transform_source,
			self.transformed_ranks.copy(),
			self.weather_permanent,
			self.hazard_order.copy(),
		)

	def _pressure_cost(self, obs, active, move, called):
		c = self.catalogue
		own = self.previous.own
		if obs.events.entered[0] and called != c.move["pursuit"]:
			own = obs.own
		target = np.argmax(own.is_active)
		if own.ability[target] != c.pressure or own.fainted[target]:
			return 0

		def charges(move):
			curse_self = move == c.move["curse"] and c.ghost not in obs.opponent.types[active]
			return bool(c.pressure_target[move] and not curse_self)

		return int(charges(move)) + int(move != called and move in c.callers and charges(called))

	def _hypotheses(self, evidence):
		c = self.catalogue
		normal = [c.candidates(*row) for row in evidence]
		count = len(evidence)
		shown = [c.base_form.get(row[0], row[0]) for row in evidence]
		for disguised in [-1, *range(count)]:
			if disguised >= 0 and (c.zoroark not in c.by_species or c.zoroark in shown):
				continue
			identity = shown.copy()
			rows = normal.copy()
			if disguised >= 0:
				_, _, moves, item, ability = evidence[disguised]
				identity[disguised] = c.zoroark
				rows[disguised] = c.candidates(c.zoroark, 0, moves, item, ability)
			numbers = [c.number_by_id.get(s, -1) for s in identity]
			if len(set(numbers)) != count or any(not len(r) for r in rows):
				continue
			target = -1
			if disguised >= 0:
				appearance = shown[disguised]
				target = identity.index(appearance) if appearance in identity else count
				if target >= TEAM_SLOTS or appearance not in c.by_species:
					continue
				if target == count and not len(
					c.candidates(appearance, evidence[disguised][1], (), 0, 0)
				):
					continue
			rate = c.illusion_rate
			weight = (1 - rate) if disguised < 0 else rate / TEAM_SLOTS
			for s, candidates in zip(identity, rows, strict=True):
				weight *= c.weight[candidates].sum() / c.weight[c.by_species[s]].sum()
			yield disguised, target, identity, rows, weight

	def _propose(self, evidence):
		c = self.catalogue
		hypotheses = list(self._hypotheses(evidence))
		if not hypotheses:
			return None
		weights = np.array([h[-1] for h in hypotheses])
		weights /= weights.sum()
		# Stratify the bank over identity hypotheses, retaining a rare feasible disguise.
		alloc = np.ones(len(hypotheses), np.int32)
		alloc += self.rng.multinomial(self.size - len(hypotheses), weights)
		teams = np.zeros((self.size, TEAM_SLOTS), np.int32)
		disguise = np.full((self.size, TEAM_SLOTS), -1, np.int8)
		particle_weights = np.empty(self.size)
		start = 0
		for (holder, target, identity, rows, _), n, weight in zip(
			hypotheses, alloc, weights, strict=True
		):
			end = start + n
			team = teams[start:end]
			particle_weights[start:end] = weight / n
			for slot, candidates in enumerate(rows):
				team[:, slot] = self.rng.choice(candidates, size=n, p=c.prob(candidates))
			pinned = list(identity)
			if holder >= 0:
				disguise[start:end, holder] = target
				if target == len(pinned):
					species, level, *_ = evidence[holder]
					species = c.base_form.get(species, species)
					candidates = c.candidates(species, level, (), 0, 0)
					team[:, target] = self.rng.choice(candidates, size=n, p=c.prob(candidates))
					pinned.append(species)
			prior = c.prior.copy()
			if pinned:
				prior *= np.prod(c.relative[pinned] ** (1 / len(pinned)), axis=0)
				prior[np.isin(c.species_number, [c.number_by_id[s] for s in pinned])] = 0
			for slot in range(len(pinned), TEAM_SLOTS):
				prob = np.broadcast_to(prior, (n, len(prior))).copy()
				if slot > len(pinned):
					used = c.sets.species[team[:, len(pinned) : slot]]
					idx = np.searchsorted(c.species, used)
					prob *= ~(
						c.species_number[None, :, None] == c.species_number[idx][:, None, :]
					).any(-1)
				cdf = np.cumsum(prob, axis=1)
				if np.any(cdf[:, -1] == 0):
					return None
				selected = (cdf <= self.rng.random(n)[:, None] * cdf[:, -1:]).sum(1)
				ranks = (c.set_cdf[selected] <= self.rng.random(n)[:, None]).sum(1)
				team[:, slot] = c.set_rows[selected, ranks]
			start = end
		return Particles(teams, disguise), particle_weights

	@staticmethod
	def empty(worlds):
		return (
			Particles(
				np.zeros((worlds, TEAM_SLOTS), np.int32), np.full((worlds, TEAM_SLOTS), -1, np.int8)
			),
			0,
			0.0,
		)

	def sample(self, worlds):
		if self.bank is None:
			return self.empty(worlds)
		particles, weights = self.bank
		indices = self.rng.choice(self.size, size=worlds, p=weights)
		return (
			Particles(particles.sets[indices], particles.disguise[indices]),
			self.size,
			float(1 / (weights @ weights)),
		)
