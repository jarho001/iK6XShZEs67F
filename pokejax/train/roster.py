"""The roster networks of Phase 6: network.arch "control" and "pairwise" (see network.py).

Both read only the player-view observation and the static dex columns of the species and moves
(power, accuracy, always-hits, priority, hit bounds, category, type and base stats). Nothing computed
from the type chart (effectiveness, STAB, immunities), no damage or matchup estimate, and no column of
pokejax/train/features.py enters: dex_columns is built from the move and species tables alone.

For one decision, with width d (network.width):

1. Descriptors projected to d with SiLU. A member (twelve: six own, six revealed foe) is 202 wide:
   species, types, item, ability, status, the mean of its four move descriptors, scalar state, own or
   zero stats, base stats, boosts, and for the active only its volatiles and move IDs. A move (28: the
   own roster's 24 and the foe active's four revealed moves) is 41 wide: move ID, type, dex columns and
   PP, and for the own active's moves legal, no-op and single-legal flags. The field (31) and each
   side's last-step events (43) have their own projections.
2. Pairwise only: each of the 48 directed pairs (own move against the foe active, and each move of the
   foe active against each own member) concatenates attacker, move and defender vectors with two
   learned type interactions, and a shared pair MLP (3d + 16 -> p -> p, RMSNorm) maps it to p. The
   interactions multiply a bias-free projection of the move type (one matrix) by a bias-free projection
   of each defending type (a second, shared, matrix). Invalid pairs are zero and every aggregate leaves
   them out.
3. Context: the own and foe active vectors, the masked means of the two rosters (the control also the
   masked max of each), the field, both events and, pairwise, the mean of the valid pairs. A layer
   (SiLU) and RMSNorm feed a GRU of network.core_width.
4. Heads: the GRU output is RMS-normalised. Each move slot is scored from its move vector (and its pair,
   pairwise) and each bench action from its member vector (and its mean outgoing and incoming pairs,
   pairwise), by a shared two-layer scorer per kind. The value is a two-layer head.
5. Training-only auxiliary heads (auxiliary.py) read the foe active's vector and the normalised GRU
   output. They are separate outputs: nothing they compute enters the policy, the value or the carry.

encode() returns one flat vector per decision: the GRU's projected input, the per-move and per-switch
features, then the foe active vector (the auxiliary heads' input). step_full() reads it back. parts()
exposes the intermediate vectors for the type probe (pokejax/train/probe.py).
"""

import functools
from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from pokejax.engine.actions import bench_slots
from pokejax.tables import Tables, load_tables
from pokejax.train.config import AuxiliarySection, NetworkSection
from pokejax.train.network import (
	ACTIVATIONS,
	ILLEGAL_LOGIT,
	MISSING_MEMBER,
	OWN_ACTIVE_FLAGS,
	OWN_ACTIVE_MOVES,
	ActionScorer,
	GateCore,
	Vocabulary,
	padded_embed,
)

MOVE_SLOTS = 4
TEAM = 6
BENCH = 5
ACTION_FLAGS = 3  # legal, no-op, single legal (the last columns of a move descriptor)
EVENT_SCALARS = ("missed", "failed", "critical", "hits", "hp_change", "fainted")
HIDDEN_POWER_STAND_IN = 70.0  # the power the generation 5 request does not show (as in features.py)
SPECIES_STAT_SCALE = 255.0
MOVE_POWER_SCALE = 100.0
MOVE_PRIORITY_SCALE = 6.0
MOVE_HITS_SCALE = 5.0
MOVE_ACCURACY_SCALE = 100.0
AGE_CAP = 10.0  # ages of the field are bounded by this many turns
SCORER_WIDTH = 64
VALUE_WIDTH = 64
AUX_STEM_WIDTH = 64
RMS_EPS = 1e-6


class Dex(NamedTuple):
	"""Static columns of the moves and species, as numpy arrays (row zero is no move, no species)."""

	move_type: (
		np.ndarray
	)  # int32 [M]: vocabulary index of the type (engine index + 1), zero for none
	move_power: np.ndarray  # float32 [M]: base power (Hidden Power at its stand-in)
	move_accuracy: np.ndarray  # float32 [M]: 0 to 100
	move_always_hits: np.ndarray  # float32 [M]
	move_priority: np.ndarray  # float32 [M]
	move_hits_min: np.ndarray  # float32 [M]
	move_hits_max: np.ndarray  # float32 [M]
	move_category: np.ndarray  # float32 [M, 3]: physical, special, status (one-hot)
	species_base: np.ndarray  # float32 [S, 6]: hp, atk, def, spa, spd, spe


def build_dex(tables: Tables) -> Dex:
	"""The dex columns of the roster networks, from the move and species tables only."""
	moves = tables.moves
	engine_types = len(tables.types.ids)
	move_type = np.asarray(moves.type, np.int64)
	move_type = np.where((move_type >= 0) & (move_type < engine_types), move_type + 1, 0)
	power = np.asarray(moves.base_power, np.float32).copy()
	hidden = moves.index.get("hiddenpower")
	if hidden is not None:
		power[hidden] = HIDDEN_POWER_STAND_IN
	category = np.eye(3, dtype=np.float32)[np.asarray(moves.category, np.int64)]
	columns = {
		"move_type": move_type.astype(np.int32),
		"move_power": power,
		"move_accuracy": np.asarray(moves.accuracy, np.float32).copy(),
		"move_always_hits": np.asarray(moves.always_hits, np.float32).copy(),
		"move_priority": np.asarray(moves.priority, np.float32).copy(),
		"move_hits_min": np.asarray(moves.hit_min, np.float32).copy(),
		"move_hits_max": np.asarray(moves.hit_max, np.float32).copy(),
		"move_category": category,
		"species_base": np.asarray(tables.species.base_stats, np.float32).copy(),
	}
	for name, column in columns.items():
		if name != "species_base":
			column[0] = 0
	columns["species_base"][0] = 0
	return Dex(**columns)


@functools.cache
def dex_columns() -> Dex:
	"""The dex columns of the default data file (built once per process)."""
	return build_dex(load_tables())


class RMSNorm(nn.Module):
	"""Root-mean-square normalisation over the last axis, with a learned scale."""

	@nn.compact
	def __call__(self, x):
		scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
		return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + RMS_EPS) * scale


def _pick(x, weights):
	"""Sum of x [..., N, *rest] over N with float weights [..., N]: [..., *rest]."""
	axis = weights.ndim - 1
	expanded = weights.reshape(weights.shape + (1,) * (x.ndim - weights.ndim))
	return jnp.sum(x * expanded, axis=axis)


def _rounded(x):
	"""Integer IDs carried as floats (a one-hot pick of them) as int32."""
	return jnp.round(x).astype(jnp.int32)


def _bounded(x):
	return jnp.minimum(x, AGE_CAP) / AGE_CAP


def _move_mean(block, ids):
	"""Mean over a member's valid moves of the 38-wide descriptors: [..., 6, 38]."""
	valid = (ids > 0).astype(jnp.float32)
	count = jnp.maximum(valid.sum(-1), 1.0)
	return (block[..., :-ACTION_FLAGS] * valid[..., None]).sum(-2) / count[..., None]


def _summary(vectors, present):
	"""Masked mean and max over the members (axis -2) of vectors [..., 6, d]: two [..., d]."""
	mask = present.astype(jnp.float32)[..., None]
	mean = (vectors * mask).sum(-2) / jnp.maximum(mask.sum(-2), 1.0)
	largest = jnp.max(jnp.where(mask > 0, vectors, MISSING_MEMBER), axis=-2)
	largest = jnp.where(present.any(-1)[..., None], largest, 0.0)
	return mean, largest


class Parts(NamedTuple):
	"""The intermediate vectors of a batch of decisions (leading axes [...])."""

	own_vec: jax.Array  # [..., 6, d]
	foe_vec: jax.Array  # [..., 6, d]
	own_active: jax.Array  # [..., d]
	foe_active: jax.Array  # [..., d]
	move_vecs: jax.Array  # [..., 4, d]: the own active's move slots
	foe_move_vecs: jax.Array  # [..., 4, d]: the foe active's revealed moves
	pairs_own: (
		jax.Array
	)  # [..., 6, 4, p]: own move against the foe active (zero if invalid; pairwise)
	pairs_foe: jax.Array  # [..., 4, 6, p]: foe active's move against each own member (pairwise)
	move_pairs: jax.Array  # [..., 4, p]: the own active's pairs (pairwise)
	own_valid: jax.Array  # [..., 6, 4] bool
	foe_valid: jax.Array  # [..., 4, 6] bool
	field: jax.Array  # [..., d]
	events_own: jax.Array  # [..., d]
	events_foe: jax.Array  # [..., d]
	own_present: jax.Array  # [..., 6] bool
	foe_present: jax.Array  # [..., 6] bool
	move_valid: jax.Array  # [..., 4] bool: the own active's slot holds a move


class RosterNetwork(nn.Module):
	"""The parameters of a roster network, and its feed-forward and recurrent parts."""

	vocab: Vocabulary
	network: NetworkSection
	auxiliary_targets: AuxiliarySection

	def setup(self):
		n, v = self.network, self.vocab
		width = n.width
		self.species = nn.Embed(v.species, n.species_embedding)
		self.moves = nn.Embed(v.moves, n.move_embedding)
		self.items = nn.Embed(v.items, n.item_embedding)
		self.abilities = nn.Embed(v.abilities, n.ability_embedding)
		self.types = nn.Embed(v.types, n.type_embedding)
		self.statuses = nn.Embed(v.statuses, n.code_embedding)
		self.weathers = nn.Embed(v.weathers, n.code_embedding)
		self.event_orders = nn.Embed(v.orders, n.code_embedding)
		self.event_cants = nn.Embed(v.cants, n.code_embedding)
		self.event_effects = nn.Embed(v.effects, n.code_embedding)
		self.event_entries = nn.Embed(v.entries, n.code_embedding)
		self.member_proj = nn.Dense(width, name="member")
		self.move_proj = nn.Dense(width, name="move")
		self.field_proj = nn.Dense(width, name="field")
		self.event_proj = nn.Dense(width, name="event")
		self.pairwise = n.arch == "pairwise"
		if self.pairwise:
			self.move_type_proj = nn.Dense(n.interaction_width, use_bias=False, name="move_type")
			self.defender_type_proj = nn.Dense(
				n.interaction_width, use_bias=False, name="defender_type"
			)
			self.pair_hidden = nn.Dense(n.pair_width, name="pair_hidden")
			self.pair_out = nn.Dense(n.pair_width, name="pair_out")
			self.pair_norm = RMSNorm(name="pair_norm")
		self.context_proj = nn.Dense(n.context_width, name="context")
		self.context_norm = RMSNorm(name="context_norm")
		self.core = GateCore("gru", n.core_width)
		self.out_norm = RMSNorm(name="out_norm")
		self.move_scorer = ActionScorer(SCORER_WIDTH, n.activation, name="move_scorer")
		self.switch_scorer = ActionScorer(SCORER_WIDTH, n.activation, name="switch_scorer")
		self.value_hidden = nn.Dense(VALUE_WIDTH, name="value_hidden")
		self.value_out = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0), name="value")
		aux = self.auxiliary_targets
		self.aux_stem = nn.Dense(AUX_STEM_WIDTH, name="aux_stem") if aux.enabled else None
		# Move or switch (2), move identities (the vocabulary) and switch ranks (six seen, one unseen).
		self.aux_next = nn.Dense(2 + v.moves + 7, name="aux_next") if aux.next_action > 0 else None
		self.aux_item = nn.Dense(v.items, name="aux_item") if aux.item > 0 else None
		self.aux_ability = nn.Dense(v.abilities, name="aux_ability") if aux.ability > 0 else None
		self.aux_moves = nn.Dense(v.moves, name="aux_moves") if aux.moves > 0 else None

	# Descriptors.

	def _move_block(self, ids, pp, hidden_type, return_power, flags):
		"""Move descriptors [..., K, 41] and the vocabulary type index of each move [..., K]."""
		dex = dex_columns()
		f32 = jnp.float32
		ids = ids.astype(jnp.int32)
		valid = (ids > 0).astype(f32)
		hidden_type = hidden_type.astype(jnp.int32)
		typed = jnp.where(hidden_type > 0, hidden_type, jnp.asarray(dex.move_type)[ids])
		power = jnp.where(
			return_power > 0, return_power.astype(f32), jnp.asarray(dex.move_power)[ids]
		)
		columns = jnp.concatenate(
			[
				padded_embed(self.moves, ids),
				padded_embed(self.types, typed),
				(power / MOVE_POWER_SCALE)[..., None],
				(jnp.asarray(dex.move_accuracy)[ids] / MOVE_ACCURACY_SCALE)[..., None],
				jnp.asarray(dex.move_always_hits)[ids][..., None],
				(jnp.asarray(dex.move_priority)[ids] / MOVE_PRIORITY_SCALE)[..., None],
				(jnp.asarray(dex.move_hits_min)[ids] / MOVE_HITS_SCALE)[..., None],
				(jnp.asarray(dex.move_hits_max)[ids] / MOVE_HITS_SCALE)[..., None],
				jnp.asarray(dex.move_category)[ids],
				pp.astype(f32)[..., None],
			],
			axis=-1,
		)
		block = columns * valid[..., None]
		return jnp.concatenate([block, flags], axis=-1), typed

	def _member_block(self, side, lead, own, move_mean, is_active):
		"""Member descriptors [..., 6, 202] of one side; the active block is on the active only."""
		dex = dex_columns()
		f32 = jnp.float32
		six = side.species.shape
		zeros = jnp.zeros(six, f32)
		ones = jnp.ones(six, f32)
		scalars = jnp.stack(
			[
				side.level,
				side.hp,
				side.fainted,
				side.is_active,
				zeros if own else side.unseen,
				ones if own else side.item_revealed,
				side.item_consumed.astype(f32),
				side.item_knocked_off.astype(f32),
				zeros if own else ones,
			],
			axis=-1,
		)
		stats = side.stats.astype(f32) if own else jnp.zeros(six + (6,), f32)
		base = jnp.asarray(dex.species_base)[side.species.astype(jnp.int32)] / SPECIES_STAT_SCALE
		boosts = side.boosts.astype(f32)[..., None, :] * is_active[..., :, None]
		flags = [
			getattr(side, name).astype(f32) if hasattr(side, name) else jnp.zeros(lead, f32)
			for name in OWN_ACTIVE_FLAGS
		]
		active = [jnp.stack(flags, axis=-1)]
		for name in OWN_ACTIVE_MOVES:
			if hasattr(side, name):
				active.append(padded_embed(self.moves, getattr(side, name).astype(jnp.int32)))
			else:
				active.append(jnp.zeros(lead + (self.network.move_embedding,), f32))
		active = jnp.concatenate(active, axis=-1)
		active = active[..., None, :] * is_active[..., :, None]
		descriptor = jnp.concatenate(
			[
				padded_embed(self.species, side.species),
				padded_embed(self.types, side.types).reshape(six + (-1,)),
				padded_embed(self.items, side.item),
				padded_embed(self.abilities, side.ability),
				padded_embed(self.statuses, side.status),
				move_mean,
				scalars,
				stats,
				base,
				boosts,
				active,
			],
			axis=-1,
		)
		return descriptor

	def _events(self, events):
		"""Each side's last-step event descriptor [..., 2, 43], the own side first."""
		f32 = jnp.float32
		side_bit = jnp.broadcast_to(jnp.array([[0.0], [1.0]], f32), events.order.shape + (1,))
		return jnp.concatenate(
			[
				padded_embed(self.event_orders, events.order),
				padded_embed(self.moves, events.move),
				padded_embed(self.event_cants, events.cant),
				padded_embed(self.event_entries, events.entered),
				padded_embed(self.statuses, events.status),
				padded_embed(self.event_effects, events.effectiveness),
				jnp.stack([getattr(events, name) for name in EVENT_SCALARS], axis=-1),
				side_bit,
			],
			axis=-1,
		)

	def _field(self, obs, lead):
		f32 = jnp.float32
		return jnp.concatenate(
			[
				obs.turn[..., None],
				obs.request,
				obs.mid_turn[..., None],
				obs.legal_actions.astype(f32),
				padded_embed(self.weathers, obs.weather),
				_bounded(obs.weather_age)[..., None],
				obs.hazards.reshape(lead + (4,)),
				obs.trick_room[..., None],
				_bounded(obs.trick_room_age)[..., None],
				obs.wish,
				_bounded(obs.wish_age),
				obs.healing_wish,
			],
			axis=-1,
		)

	# Vectors and pairs.

	def parts(self, obs) -> Parts:
		"""The vectors of each decision in obs (leading axes [...])."""
		n = self.network
		act = ACTIVATIONS[n.activation]
		f32 = jnp.float32
		lead = obs.turn.shape
		own, foe = obs.own, obs.opponent
		own_active = own.is_active.astype(f32)
		foe_active = foe.is_active.astype(f32)
		own_present = own.species > 0
		foe_present = foe.unseen < 0.5
		foe_live = foe_active.sum(-1) > 0
		# The own active's move flags: legal, the no-op of a request with none, a single legal move.
		legal = obs.legal_actions[..., :MOVE_SLOTS]
		request_none = obs.request[..., 0] > 0
		request_move = obs.request[..., 1] > 0
		single = (legal.sum(-1) == 1)[..., None] & legal & request_move[..., None]
		noop = (jnp.arange(MOVE_SLOTS) == 0) & request_none[..., None]
		flags = jnp.stack([legal, noop, single], axis=-1).astype(f32)  # [..., 4, 3]
		own_flags = flags[..., None, :, :] * own_active[..., :, None, None]  # [..., 6, 4, 3]

		own_block, own_type = self._move_block(
			own.moves, own.pp, own.hidden_type, own.return_power, own_flags
		)
		foe_zero_ids = jnp.zeros(foe.moves.shape, jnp.int32)
		foe_block, foe_type = self._move_block(
			foe.moves,
			jnp.zeros(foe.moves.shape, f32),
			foe_zero_ids,
			jnp.zeros(foe.moves.shape, f32),
			jnp.zeros(foe.moves.shape + (ACTION_FLAGS,), f32),
		)
		own_desc = self._member_block(own, lead, True, _move_mean(own_block, own.moves), own_active)
		foe_desc = self._member_block(
			foe, lead, False, _move_mean(foe_block, foe.moves), foe_active
		)
		own_vec = act(self.member_proj(own_desc))
		foe_vec = act(self.member_proj(foe_desc))
		own_act = _pick(own_vec, own_active)
		foe_act = _pick(foe_vec, foe_active)

		own_moves = act(self.move_proj(own_block))  # [..., 6, 4, d]
		move_vecs = _pick(own_moves, own_active)  # [..., 4, d]
		foe_active_block = _pick(foe_block, foe_active)  # [..., 4, 41]
		foe_move_vecs = act(self.move_proj(foe_active_block))  # [..., 4, d]
		foe_move_ids = _rounded(_pick(foe.moves.astype(f32), foe_active))  # [..., 4]
		foe_move_type = _rounded(_pick(foe_type.astype(f32), foe_active))  # [..., 4]
		foe_types = _rounded(_pick(foe.types.astype(f32), foe_active))  # [..., 2]
		move_valid = _rounded(_pick(own.moves.astype(f32), own_active)) > 0
		live = foe_live[..., None, None]
		own_valid = (own.moves > 0) & own_present[..., None] & live  # [..., 6, 4]
		foe_valid = (
			(foe_move_ids > 0)[..., :, None] & own_present[..., None, :] & live
		)  # [..., 4, 6]

		d = n.width
		zero_pairs = jnp.zeros(lead + (TEAM, MOVE_SLOTS, 0), f32)
		pairs_own, pairs_foe, move_pairs = (
			zero_pairs,
			jnp.zeros(lead + (MOVE_SLOTS, TEAM, 0), f32),
			jnp.zeros(lead + (MOVE_SLOTS, 0), f32),
		)
		if self.pairwise:
			move_inter = self.move_type_proj(padded_embed(self.types, own_type))  # [..., 6, 4, k]
			defend_inter = self.defender_type_proj(
				padded_embed(self.types, foe_types)
			)  # [..., 2, k]
			interaction = move_inter[..., :, :, None, :] * defend_inter[..., None, None, :, :]
			interaction = interaction.reshape(lead + (TEAM, MOVE_SLOTS, -1))
			own_attack = jnp.broadcast_to(own_vec[..., :, None, :], own_moves.shape)
			own_defender = jnp.broadcast_to(foe_act[..., None, None, :], own_moves.shape)
			pair_in = jnp.concatenate([own_attack, own_moves, own_defender, interaction], axis=-1)
			pairs_own = self.pair_norm(self.pair_out(act(self.pair_hidden(pair_in))))
			pairs_own = pairs_own * own_valid[..., None]

			foe_shape = lead + (MOVE_SLOTS, TEAM, d)
			foe_attack = jnp.broadcast_to(foe_act[..., None, None, :], foe_shape)
			foe_move = jnp.broadcast_to(foe_move_vecs[..., :, None, :], foe_shape)
			foe_defender = jnp.broadcast_to(own_vec[..., None, :, :], foe_shape)
			foe_move_inter = self.move_type_proj(padded_embed(self.types, foe_move_type))
			own_defend = self.defender_type_proj(
				padded_embed(self.types, own.types)
			)  # [..., 6, 2, k]
			foe_interaction = foe_move_inter[..., :, None, None, :] * own_defend[..., None, :, :, :]
			foe_interaction = foe_interaction.reshape(lead + (MOVE_SLOTS, TEAM, -1))
			pair_in = jnp.concatenate(
				[foe_attack, foe_move, foe_defender, foe_interaction], axis=-1
			)
			pairs_foe = self.pair_norm(self.pair_out(act(self.pair_hidden(pair_in))))
			pairs_foe = pairs_foe * foe_valid[..., None]
			move_pairs = _pick(pairs_own, own_active)

		field = act(self.field_proj(self._field(obs, lead)))
		events = act(self.event_proj(self._events(obs.events)))
		return Parts(
			own_vec=own_vec,
			foe_vec=foe_vec,
			own_active=own_act,
			foe_active=foe_act,
			move_vecs=move_vecs,
			foe_move_vecs=foe_move_vecs,
			pairs_own=pairs_own,
			pairs_foe=pairs_foe,
			move_pairs=move_pairs,
			own_valid=own_valid,
			foe_valid=foe_valid,
			field=field,
			events_own=events[..., 0, :],
			events_foe=events[..., 1, :],
			own_present=own_present,
			foe_present=foe_present,
			move_valid=move_valid,
		)

	@property
	def move_width(self) -> int:
		"""Width of one move's features: its vector, and its pair (pairwise)."""
		return self.network.width + (self.network.pair_width if self.pairwise else 0)

	@property
	def switch_width(self) -> int:
		"""Width of one switch's features: its member vector, and its mean outgoing and incoming pairs."""
		return self.network.width + (2 * self.network.pair_width if self.pairwise else 0)

	# Encoding and the recurrent step.

	def encode(self, obs):
		"""The flat vector of each decision: projected core input, move and switch features, foe active."""
		n = self.network
		act = ACTIVATIONS[n.activation]
		p = self.parts(obs)
		lead = obs.turn.shape
		own_mean, own_max = _summary(p.own_vec, p.own_present)
		foe_mean, foe_max = _summary(p.foe_vec, p.foe_present)
		if self.pairwise:
			pair_total = p.pairs_own.sum(axis=(-3, -2)) + p.pairs_foe.sum(axis=(-3, -2))
			pair_count = p.own_valid.sum(axis=(-2, -1)) + p.foe_valid.sum(axis=(-2, -1))
			pair_mean = pair_total / jnp.maximum(pair_count, 1.0)[..., None]
			context = [
				p.own_active,
				p.foe_active,
				own_mean,
				foe_mean,
				p.field,
				p.events_own,
				p.events_foe,
				pair_mean,
			]
		else:
			context = [
				p.own_active,
				p.foe_active,
				own_mean,
				own_max,
				foe_mean,
				foe_max,
				p.field,
				p.events_own,
				p.events_foe,
			]
		context = self.context_norm(act(self.context_proj(jnp.concatenate(context, axis=-1))))
		projected = self.core.project(context)

		moves = p.move_vecs
		if self.pairwise:
			moves = jnp.concatenate([moves, p.move_pairs], axis=-1)
		# Bench actions: the bench slots of the active, in action order.
		active = jnp.argmax(obs.own.is_active, axis=-1)
		slots = bench_slots(active)  # [..., 5]
		seats = (slots[..., None] == jnp.arange(TEAM)).astype(jnp.float32)  # [..., 5, 6]
		member = jnp.einsum("...km,...md->...kd", seats, p.own_vec)
		switches = [member]
		if self.pairwise:
			out_count = jnp.maximum(p.own_valid.sum(-1), 1.0)  # [..., 6]
			out_mean = p.pairs_own.sum(-2) / out_count[..., None]  # [..., 6, p]
			in_count = jnp.maximum(p.foe_valid.sum(-2), 1.0)  # [..., 6]
			in_mean = p.pairs_foe.sum(-3) / in_count[..., None]  # [..., 6, p]
			switches += [
				jnp.einsum("...km,...mp->...kp", seats, out_mean),
				jnp.einsum("...km,...mp->...kp", seats, in_mean),
			]
		switch_features = jnp.concatenate(switches, axis=-1)  # [..., 5, switch width]
		move_count = MOVE_SLOTS * self.move_width
		switch_count = BENCH * self.switch_width
		return jnp.concatenate(
			[
				projected,
				moves.reshape(lead + (move_count,)),
				switch_features.reshape(lead + (switch_count,)),
				p.foe_active,
			],
			axis=-1,
		)

	def step_full(self, carry, projected, legal):
		"""The core on one step's projected input, then the heads: (carry, logits, value, out)."""
		n = self.network
		act = ACTIVATIONS[n.activation]
		gates = 3 * n.core_width
		move_end = gates + MOVE_SLOTS * self.move_width
		switch_end = move_end + BENCH * self.switch_width
		carry, hidden = self.core.recur(carry, projected[..., :gates])
		out = self.out_norm(hidden)
		lead = projected.shape[:-1]
		move_f = projected[..., gates:move_end].reshape(lead + (MOVE_SLOTS, self.move_width))
		switch_f = projected[..., move_end:switch_end].reshape(lead + (BENCH, self.switch_width))
		logits = jnp.concatenate(
			[self.move_scorer(out, move_f), self.switch_scorer(out, switch_f)], axis=-1
		)
		logits = jnp.where(legal, logits, ILLEGAL_LOGIT)
		value = self.value_out(act(self.value_hidden(out)))[..., 0]
		return carry, logits, value, out

	def auxiliary(self, out, foe):
		"""The training-only heads over the normalised core output and the foe active vector."""
		act = ACTIVATIONS[self.network.activation]
		hidden = act(self.aux_stem(jnp.concatenate([foe, out], axis=-1)))
		heads = {}
		if self.aux_next is not None:
			heads["next_action"] = self.aux_next(hidden)
		if self.aux_item is not None:
			heads["item"] = self.aux_item(hidden)
		if self.aux_ability is not None:
			heads["ability"] = self.aux_ability(hidden)
		if self.aux_moves is not None:
			heads["moves"] = self.aux_moves(hidden)
		return heads

	def probe_features(self, obs):
		"""Per own move slot [..., 4, F] for the type probe: the pair (pairwise), or the move and
		the foe active vectors (control)."""
		p = self.parts(obs)
		if self.pairwise:
			return p.move_pairs
		foe = jnp.broadcast_to(p.foe_active[..., None, :], p.move_vecs.shape)
		return jnp.concatenate([p.move_vecs, foe], axis=-1)
