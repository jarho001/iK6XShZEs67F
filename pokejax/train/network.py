"""The policy and value network: embeddings, team encoders, trunk, recurrent core and heads.

The input is an Observation of pokejax.engine.observation for one learner (its own side and the
opponent as it sees them), with any number of leading batch axes. In order:

1. Embeddings shared by both sides: species, moves, items, abilities, types (engine IDs + 1) and
   small tables for the enumerated codes (status, weather, and the event order, cant, effect and
   entry codes). ID 0 is padding or "none" and embeds to zeros.
2. A MemberEncoder per side (a two-layer MLP shared by that side's six members), fed each member's
   embeddings and scalar features. Each side's team is pooled over its present members (mean and
   max) and the active member's encoding is kept apart: three vectors per side.
3. A trunk (two dense layers) over both teams, the field (turn, request, legal mask, weather,
   hazards, Trick Room, Wish and Healing Wish), each side's active volatiles and boosts, and the
   last step's events of both sides. With network.features the trunk also takes the public damage
   and matchup features of pokejax/train/features.py (each move slot's power, type effectiveness,
   STAB and the max-damage score; each switch's matchup and threat; the threat to the active).
   network.feature_set picks the columns that enter (features.py): "dex" leaves out the scores and
   the matchup estimates.
   Steps 1 to 3 are feed-forward: encode() runs them over any number of steps at once, and the
   input projections of the core are the last part of it.
4. A recurrent core (GRU or LSTM, from the config; GateCore), which is the only part a scan over
   time has to run, then the heads: policy logits over the nine actions with illegal actions set
   to ILLEGAL_LOGIT, and a scalar value. The dense head (network.action_head = "dense") is one
   linear layer over the core output; the factored head ("factored") scores each move slot and each
   switch from the core output and that action's own features (ActionScorer, shared by the moves
   and by the switches). step() is the core and the heads.
"""

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from pokejax.engine.actions import NUM_ACTIONS
from pokejax.engine.events import CANT_FOCUS_PUNCH, EFFECT_SUPER, ENTRY_REPLACEMENT, ORDER_SECOND
from pokejax.engine.state import NUM_WEATHERS, STATUS_NAMES, TEAM_SLOTS
from pokejax.tables import Tables
from pokejax.train.config import NetworkSection
from pokejax.train.features import (
	BENCH_COUNT,
	MOVE_COUNT,
	action_features,
	feature_widths,
	public_tables,
)

# The logit of an illegal action; its exponential is exactly zero in float32.
ILLEGAL_LOGIT = -1e9
# Stand-in for a missing member in a max over the team (never the largest value).
MISSING_MEMBER = -1e9
ACTIVATIONS = {"relu": nn.relu, "tanh": nn.tanh, "gelu": nn.gelu}

# The observation fields that feed the network, by group (observation.py names them).
OWN_MEMBER_SCALARS = ("level", "hp", "fainted", "is_active")
FOE_MEMBER_SCALARS = ("level", "hp", "fainted", "is_active", "unseen", "item_revealed")
MEMBER_ITEM_FLAGS = ("item_consumed", "item_knocked_off")
OWN_ACTIVE_FLAGS = (
	"confused",
	"charging",
	"recharging",
	"locked",
	"trapped",
	"substitute",
	"encore",
	"taunt",
	"disable",
	"leech_seed",
	"yawn",
	"curse",
	"destiny_bond",
	"transformed",
	"flash_fire",
	"slow_start",
	"autotomize",
	"protect",
	"roost",
	"choice_locked",
)
OWN_ACTIVE_MOVES = ("encore_move", "disabled_move", "choice_move")
FOE_ACTIVE_FLAGS = (
	"confused",
	"charging",
	"recharging",
	"trapped",
	"substitute",
	"encore",
	"taunt",
	"disable",
	"leech_seed",
	"yawn",
	"curse",
	"destiny_bond",
	"transformed",
	"flash_fire",
	"slow_start",
	"autotomize",
	"protect",
	"roost",
)
FOE_ACTIVE_MOVES = ("encore_move", "disabled_move")
EVENT_SCALARS = ("missed", "failed", "critical", "hits", "hp_change", "fainted")


class Vocabulary(NamedTuple):
	"""The number of rows of each embedding table (padding and "none" rows included)."""

	species: int
	moves: int
	items: int
	abilities: int
	types: int
	statuses: int
	weathers: int
	orders: int
	cants: int
	effects: int
	entries: int


def vocabulary(tables: Tables) -> Vocabulary:
	return Vocabulary(
		species=len(tables.species.ids),
		moves=len(tables.moves.ids),
		items=len(tables.items.ids),
		abilities=len(tables.abilities.ids),
		types=len(tables.types.ids) + 1,  # observation type IDs are engine IDs + 1
		statuses=len(STATUS_NAMES),
		weathers=NUM_WEATHERS,
		orders=ORDER_SECOND + 1,
		cants=CANT_FOCUS_PUNCH + 1,
		effects=EFFECT_SUPER + 1,
		entries=ENTRY_REPLACEMENT + 1,
	)


def initial_carry(network: NetworkSection, shape=()):
	"""Zero recurrent state for leading batch shape `shape`: one array (GRU) or (c, h) (LSTM)."""
	zeros = jnp.zeros(tuple(shape) + (network.core_width,), jnp.float32)
	return zeros if network.core == "gru" else (zeros, zeros)


def reset_carry(carry, ended):
	"""The carry with zeros where `ended` [...] is true: a fresh battle starts there."""
	return jax.tree.map(lambda x: jnp.where(ended[..., None], 0.0, x), carry)


def padded_embed(table: nn.Embed, ids):
	"""Embedding of `ids` [...] with ID 0 (padding, or no value) mapped to zeros."""
	ids = ids.astype(jnp.int32)
	return jnp.where((ids > 0)[..., None], table(ids), 0.0)


def column(x):
	"""A feature [...] as a float column [..., 1]."""
	return x.astype(jnp.float32)[..., None]


def vector(x, lead):
	"""A feature flattened to float [lead..., -1], trailing axes in order."""
	return x.astype(jnp.float32).reshape(lead + (-1,))


def pool_team(encoded, present, active):
	"""Mean and max over the present members, and the active member's encoding: [..., 3W].

	encoded is [..., 6, W]; present (bool) and active (float or bool) are [..., 6]. With no
	present member the mean and the max are zero.
	"""
	mask = present.astype(jnp.float32)[..., None]
	count = jnp.maximum(mask.sum(axis=-2), 1.0)
	mean = (encoded * mask).sum(axis=-2) / count
	largest = jnp.max(jnp.where(mask > 0, encoded, MISSING_MEMBER), axis=-2)
	largest = jnp.where(present.any(axis=-1)[..., None], largest, 0.0)
	chosen = (encoded * active.astype(jnp.float32)[..., None]).sum(axis=-2)
	return jnp.concatenate([mean, largest, chosen], axis=-1)


class MemberEncoder(nn.Module):
	"""A two-layer MLP applied to every member of one side (its six members share it)."""

	width: int
	activation: str

	@nn.compact
	def __call__(self, x):
		act = ACTIVATIONS[self.activation]
		x = act(nn.Dense(self.width, name="hidden")(x))
		return act(nn.Dense(self.width, name="output")(x))


class GateCore(nn.Module):
	"""A GRU or an LSTM whose input projections are taken out of the time loop.

	project(x) maps the inputs of all steps at once (one matrix product over every step); recur
	advances the carry by one step from the projected inputs, with a single product with the
	hidden state. The gates are those of the usual GRU (r, z, n with the hidden bias inside the
	reset product) and LSTM (i, f, g, o) equations.
	"""

	kind: str
	features: int

	def setup(self):
		gates = 3 if self.kind == "gru" else 4
		self.input_proj = nn.Dense(gates * self.features, name="input")
		self.hidden_proj = nn.Dense(
			gates * self.features, kernel_init=nn.initializers.orthogonal(), name="hidden"
		)

	def project(self, x):
		return self.input_proj(x)

	def recur(self, carry, projected):
		width = self.features
		if self.kind == "gru":
			hidden = carry
			hidden_gates = self.hidden_proj(hidden)
			reset = jax.nn.sigmoid(projected[..., :width] + hidden_gates[..., :width])
			update = jax.nn.sigmoid(
				projected[..., width : 2 * width] + hidden_gates[..., width : 2 * width]
			)
			candidate = jnp.tanh(
				projected[..., 2 * width :] + reset * hidden_gates[..., 2 * width :]
			)
			hidden = (1 - update) * candidate + update * hidden
			return hidden, hidden
		cell, hidden = carry
		gates = projected + self.hidden_proj(hidden)
		i, f, g, o = jnp.split(gates, 4, axis=-1)
		cell = jax.nn.sigmoid(f) * cell + jax.nn.sigmoid(i) * jnp.tanh(g)
		hidden = jax.nn.sigmoid(o) * jnp.tanh(cell)
		return (cell, hidden), hidden


class ActionScorer(nn.Module):
	"""One logit per action of a kind, from the core output and each action's own features.

	h [..., W] and features [..., A, F] give logits [..., A]: a two-layer MLP over the core output
	(repeated for each action) and that action's features, shared by the actions of the kind.
	"""

	width: int
	activation: str

	@nn.compact
	def __call__(self, h, features):
		act = ACTIVATIONS[self.activation]
		shared = jnp.broadcast_to(h[..., None, :], features.shape[:-1] + h.shape[-1:])
		x = act(nn.Dense(self.width, name="hidden")(jnp.concatenate([shared, features], -1)))
		return nn.Dense(1, kernel_init=nn.initializers.orthogonal(0.01), name="output")(x)[..., 0]


class PokePolicy(nn.Module):
	"""(observation, recurrent carry) to (next carry, masked policy logits, value).

	The network is split so that the replay of a training sequence can encode all its steps at
	once: encode(obs) is the feed-forward part (embeddings, team encoders, trunk) and maps any
	leading batch axes; step(carry, features, legal) is the recurrent core and the heads, the only
	part a scan over time must run. __call__ is both. The carry has the same leading axes as the
	observation (a pair of such arrays for an LSTM).
	"""

	vocab: Vocabulary
	network: NetworkSection

	def setup(self):
		n, v = self.network, self.vocab
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
		self.own_member = MemberEncoder(n.member_width, n.activation)
		self.foe_member = MemberEncoder(n.member_width, n.activation)
		self.trunk_0 = nn.Dense(n.trunk_width)
		self.trunk_1 = nn.Dense(n.trunk_width)
		self.core = GateCore(n.core, n.core_width)
		if n.action_head == "factored":
			self.move_scorer = ActionScorer(n.action_width, n.activation, name="move_scorer")
			self.switch_scorer = ActionScorer(n.action_width, n.activation, name="switch_scorer")
		else:
			self.policy = nn.Dense(NUM_ACTIONS, kernel_init=nn.initializers.orthogonal(0.01))
		self.value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))

	def _members(self, side, lead, scalars, extra=()):
		"""Per member [..., 6, F]: embeddings, then the listed scalars and extra arrays."""
		parts = [
			padded_embed(self.species, side.species),
			padded_embed(self.types, side.types).reshape(lead + (TEAM_SLOTS, -1)),
			padded_embed(self.moves, side.moves).reshape(lead + (TEAM_SLOTS, -1)),
			padded_embed(self.abilities, side.ability),
			padded_embed(self.items, side.item),
			padded_embed(self.statuses, side.status),
			*(column(getattr(side, name)) for name in scalars),
			*extra,
		]
		return jnp.concatenate(parts, axis=-1)

	def encode(self, obs):
		"""The recurrent core's projected inputs for each observation: [..., gates x core_width].

		Feed-forward throughout, so a replay runs it over all of a chunk's steps at once.
		"""
		n = self.network
		act = ACTIVATIONS[n.activation]
		lead = obs.turn.shape
		own, foe = obs.own, obs.opponent
		own_members = self._members(
			own,
			lead,
			OWN_MEMBER_SCALARS + MEMBER_ITEM_FLAGS,
			extra=(own.pp.astype(jnp.float32), own.stats.astype(jnp.float32)),
		)
		foe_members = self._members(foe, lead, FOE_MEMBER_SCALARS + MEMBER_ITEM_FLAGS)
		own_encoded = self.own_member(own_members)
		foe_encoded = self.foe_member(foe_members)
		own_team = pool_team(own_encoded, own.species > 0, own.is_active)
		foe_team = pool_team(foe_encoded, foe.unseen < 0.5, foe.is_active)

		field = jnp.concatenate(
			[
				vector(obs.turn, lead),
				vector(obs.mid_turn, lead),
				vector(obs.request, lead),
				vector(obs.legal_actions, lead),
				padded_embed(self.weathers, obs.weather),
				vector(obs.weather_age, lead),
				vector(obs.hazards, lead),
				vector(obs.trick_room, lead),
				vector(obs.trick_room_age, lead),
				vector(obs.wish, lead),
				vector(obs.wish_age, lead),
				vector(obs.healing_wish, lead),
			],
			axis=-1,
		)
		own_active = _active(own, OWN_ACTIVE_FLAGS, OWN_ACTIVE_MOVES, self.moves, lead)
		foe_active = _active(foe, FOE_ACTIVE_FLAGS, FOE_ACTIVE_MOVES, self.moves, lead)

		events = obs.events
		event_parts = [
			padded_embed(self.event_orders, events.order),
			padded_embed(self.moves, events.move),
			padded_embed(self.event_cants, events.cant),
			padded_embed(self.event_effects, events.effectiveness),
			padded_embed(self.event_entries, events.entered),
			padded_embed(self.statuses, events.status),
			*(column(getattr(events, name)) for name in EVENT_SCALARS),
		]
		event_features = jnp.concatenate(event_parts, axis=-1).reshape(lead + (-1,))

		parts = [own_team, foe_team, field, own_active, foe_active, event_features]
		action_inputs = ()
		if n.features:
			move_f, switch_f, threat = action_features(obs, public_tables(), n.feature_set)
			move_flat = move_f.reshape(lead + (-1,))
			switch_flat = switch_f.reshape(lead + (-1,))
			parts += [move_flat, switch_flat]
			if threat is not None:
				parts.append(threat[..., None])
			action_inputs = (move_flat, switch_flat)
		x = jnp.concatenate(parts, axis=-1)
		x = act(self.trunk_0(x))
		projected = self.core.project(act(self.trunk_1(x)))
		if n.action_head == "factored":
			projected = jnp.concatenate([projected, *action_inputs], axis=-1)
		return projected

	def step(self, carry, projected, legal):
		"""The recurrent core on one step's projected inputs, then the heads: (carry, logits, value).

		With the factored head, projected also carries each action's features (encode) after the
		core's inputs; the policy logits are the move scorer's (4) and the switch scorer's (5).
		"""
		n = self.network
		if n.action_head == "factored":
			move_width, switch_width, _ = feature_widths(n.feature_set)
			width = (3 if n.core == "gru" else 4) * n.core_width
			carry, out = self.core.recur(carry, projected[..., :width])
			features = projected[..., width:]
			lead = features.shape[:-1]
			move_f = features[..., : MOVE_COUNT * move_width]
			switch_f = features[..., MOVE_COUNT * move_width :]
			move_f = move_f.reshape(lead + (MOVE_COUNT, move_width))
			switch_f = switch_f.reshape(lead + (BENCH_COUNT, switch_width))
			logits = jnp.concatenate(
				[self.move_scorer(out, move_f), self.switch_scorer(out, switch_f)], axis=-1
			)
		else:
			carry, out = self.core.recur(carry, projected)
			logits = self.policy(out)
		logits = jnp.where(legal, logits, ILLEGAL_LOGIT)
		return carry, logits, self.value(out)[..., 0]

	def __call__(self, obs, carry):
		return self.step(carry, self.encode(obs), obs.legal_actions)


def _active(side, flags, move_fields, moves, lead):
	"""A side's active volatiles, the move IDs it names and its boosts, as one vector."""
	parts = [column(getattr(side, name)) for name in flags]
	parts += [padded_embed(moves, getattr(side, name)) for name in move_fields]
	parts.append(side.boosts.astype(jnp.float32))
	return jnp.concatenate(parts, axis=-1).reshape(lead + (-1,))


def count_parameters(params) -> int:
	return sum(int(x.size) for x in jax.tree.leaves(params))
