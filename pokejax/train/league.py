"""Self-play, pool and baseline rows: the groups of a batch, the lanes training reads, and their steps.

The batch of B battle rows is split into contiguous groups, in this order (empty groups are left
out): selfplay, pool, then one group per baseline in POLICY_NAMES order.

- A selfplay row: the current network plays both sides.
- A pool row: the current network plays one side and a pool member the other. The pool group is
  cut into K equal slices, each played by one member, with one forward pass over the stacked
  parameters (vmap over the members).
- A baseline row: the current network plays one side and a baseline the other.

In every row but selfplay the learner's side is drawn at random at each reset; the other side is
the opponent. Rows keep their group across resets, and the environment's own rules (auto-reset,
truncation, refused switches) are unchanged.

Training reads lanes, one per trajectory of learner-controlled steps. A selfplay row gives two
lanes, one per seat: lanes [0, S) hold seat 0 of the selfplay rows and lanes [S, 2S) seat 1, and
each has its own recurrent state, perspective and reward. Every other row gives one lane, its
learner's side, after the selfplay lanes. Opponent sides give no lanes: they sample actions and
carry their own recurrent state where the policy has one, and they get no gradient.

A held action: when a side's switch is refused by a hidden trap, the battle does not advance and
the other side's action of that step is kept and played at the next step (pokejax.env). The held
side sees the same observation twice, so the step that plays the held action does not advance its
recurrent state (its carry stays as it was), for lanes and pool rows alike, and the sample it draws
there is discarded: that lane has weight zero and advance false. The sample of the step before, which
is played, keeps its weight and advances the carry.

Weights (Transition.weight and Transition.actor): a refused action is a decision of its own (the
retry that follows it is a transition), so it counts; a substituted illegal action does not count.
The policy's losses and their normalisation use only the decisions with a real choice (actor): not a
no-op or a single legal action, whose log-probability is zero.

Pool rows keep the parameters they started a battle with. Each pool member has its current and its
previous parameters (PoolParams); a pool row reads the current ones from the start of a battle until
the insertion that replaces its member, and the previous ones after it (pool_fresh is false), so an
insertion never changes a battle in progress. The config keeps pool.size * pool.every *
rollout.rollout_length at least env.max_steps, so no battle outlives two insertions into its member.
"""

import functools
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import POLICY_NAMES, RandomPolicy
from pokejax.engine.actions import MOVE_ACTIONS, bench_slots, block_switches
from pokejax.engine.observation import Observation, observe
from pokejax.engine.state import ILLUSION_SLOT, MOVE, NUM_SIDES, SWITCH, TRANSFORMED
from pokejax.env import Env, EnvState
from pokejax.tables import load_tables
from pokejax.train.config import GROUP_ORDER, NetworkSection
from pokejax.train.network import PokePolicy, initial_carry, reset_carry

# The rank an unseen opposing member has in the switch targets (six seen ranks come first).
UNSEEN_RANK = 6


def take(tree, index):
	"""Every leaf indexed along its leading axis by a slice or an index array."""
	return jax.tree.map(lambda x: x[index], tree)


class Layout:
	"""The rows of each group of a batch and the lanes training reads, from the row counts.

	rows maps group names to counts; the batch is their sum. Rows are numbered from 0 in
	GROUP_ORDER, so selfplay rows come first, then pool rows, then each baseline's rows.
	"""

	def __init__(self, rows: Mapping[str, int]):
		unknown = [name for name in rows if name not in GROUP_ORDER]
		if unknown:
			raise ValueError(f"unknown group(s): {', '.join(unknown)}")
		names = [name for name in GROUP_ORDER if rows.get(name, 0) > 0]
		self.sizes = {name: int(rows[name]) for name in names}
		self.batch = sum(self.sizes.values())
		if self.batch == 0:
			raise ValueError("a layout needs at least one row")
		self.spans: dict[str, tuple[int, int]] = {}
		start = 0
		for name in names:
			self.spans[name] = (start, start + self.sizes[name])
			start += self.sizes[name]
		self.selfplay = self.sizes.get("selfplay", 0)
		self.pool = self.sizes.get("pool", 0)
		self.baseline_names = tuple(name for name in names if name in POLICY_NAMES)
		self.baselines = tuple((name, *self.spans[name]) for name in self.baseline_names)
		s = self.selfplay
		self.lanes = 2 * s + self.batch - s
		self.lane_rows = np.concatenate(
			[np.arange(s), np.arange(s), np.arange(s, self.batch)]
		).astype(np.int32)
		self.selfplay_seats = np.concatenate([np.zeros(s), np.ones(s)]).astype(np.int32)
		self.pool_rows = np.arange(s, s + self.pool, dtype=np.int32)
		self.opponent_rows = np.arange(s, self.batch, dtype=np.int32)
		self.is_selfplay_row = np.arange(self.batch) < s
		self.learner_redraw = ~self.is_selfplay_row

	@property
	def identity_lanes(self) -> bool:
		"""True when lane i is row i (no selfplay rows), so no gather is needed."""
		return self.selfplay == 0

	def lane_seats(self, learner):
		"""The seat of each lane [L]: fixed for selfplay lanes, the learner's side for the rest."""
		if self.identity_lanes:
			return learner
		return jnp.concatenate([jnp.asarray(self.selfplay_seats), learner[self.selfplay :]])

	def lane_state(self, state):
		"""Battle states [B, ...] as lane states [L, ...]."""
		return state if self.identity_lanes else take(state, self.lane_rows)

	def lanes_of(self, rows):
		"""A per-row array [B, ...] as per-lane [L, ...]."""
		return rows if self.identity_lanes else rows[self.lane_rows]

	def lanes_in_time(self, rows):
		"""A per-row array [T, B] as per-lane [T, L]."""
		return rows if self.identity_lanes else rows[:, self.lane_rows]


class PoolParams(NamedTuple):
	"""The pool's parameters, stacked on a leading member axis [K, ...]: current and previous."""

	current: Any
	previous: Any


def select_lanes(mask, new, old):
	"""Per lane (or row): new where mask [N] is true, old elsewhere, for every leaf [N, ...]."""
	return jax.tree.map(
		lambda a, b: jnp.where(mask.reshape(mask.shape + (1,) * (a.ndim - 1)), a, b), new, old
	)


class LeagueState(NamedTuple):
	"""What the rollout carries between steps; battle leaves have the batch axis."""

	env: EnvState  # [B, ...]
	learner: jax.Array  # int32 [B]: the learner's side in each row (unused in selfplay rows)
	obs: Observation  # [L, ...]: what each lane sees now
	hidden: Any  # [L, W]: each lane's recurrent carry (a pair for an LSTM)
	pool_hidden: Any  # [P, W]: the carry of each pool row's member
	pool_fresh: jax.Array  # bool [P]: the row's battle began after its member's last insertion
	key: jax.Array


class AuxLabels(NamedTuple):
	"""Training targets of one lane's decision, read from the battle before the step.

	They never enter the observation, the policy, the value or the carry (auxiliary.py uses them).
	next_kind: int8 [L], the opposing side's action kind (0 move, 1 switch), -1 when no label applies
	(no decision with a real choice, or a held, refused or illegal decision on either side).
	next_move: int16 [L], the move ID of a move action (0 for a switch).
	next_rank: int8 [L], the public rank of a switch's target (0 to 5 seen, 6 unseen).
	identity: bool [L], identity targets may be used (no Illusion holder on the opposing roster).
	private: bool [L], the opposing active is live, undisguised and untransformed.
	item and ability: int16 [L], the opposing active's current held item and ability.
	moves: int16 [L, 4], the opposing active's four move IDs (its hidden ones are the targets).
	"""

	next_kind: jax.Array
	next_move: jax.Array
	next_rank: jax.Array
	identity: jax.Array
	private: jax.Array
	item: jax.Array
	ability: jax.Array
	moves: jax.Array


@functools.cache
def _illusion_ability() -> int:
	"""The ability ID of Illusion in the engine's ability table (the host table's index)."""
	return load_tables().abilities.index["illusion"]


class Transition(NamedTuple):
	"""One step: lane fields are [L, ...], row fields [B, ...] and seat fields [B, 2, ...]."""

	obs: Observation  # what each lane acted on
	hidden: Any  # each lane's carry before the step
	action: jax.Array  # int32 [L]
	seat: jax.Array  # int32 [L]: the seat each lane played (0 or 1)
	log_prob: jax.Array  # float32 [L]: of the action, under the policy that sampled it
	value: jax.Array  # float32 [L]
	after: Any  # each lane's carry after its observation (kept, not advanced, where held)
	final_obs: Observation  # each lane's observation of the battle before an automatic reset
	reward: jax.Array  # float32 [L]: the reward of the lane's seat
	advance: jax.Array  # bool [L]: False where the lane's observation is a repeat (its action held)
	weight: jax.Array  # bool [L]: a decision of the lane counts (not held, not substituted)
	actor: jax.Array  # bool [L]: a counted decision with a real choice (two or more legal actions)
	terminated: jax.Array  # bool [B]
	truncated: jax.Array  # bool [B]
	won: jax.Array  # bool [B]: the battle that ended was won from its perspective
	steps: jax.Array  # int32 [B]: environment steps of the battle that ended, else 0
	turns: jax.Array  # int32 [B]
	illegal: jax.Array  # bool [B, 2]
	rejected: jax.Array  # bool [B, 2]
	aux: Any = None  # AuxLabels of the lanes (None unless the league builds labels)


class League:
	"""The rows of one batch on one environment, and the step that plays them.

	env has batch_size equal to the layout's batch. policies maps each baseline of the layout to
	its policy (pokejax.baselines). model is the learner's PokePolicy; pool_size is K, used when
	the layout has pool rows.
	"""

	def __init__(
		self,
		env: Env,
		layout: Layout,
		model: PokePolicy,
		network: NetworkSection,
		pool_size: int,
		policies: Mapping[str, Callable],
		labels: bool = False,
	):
		if env.config.batch_size != layout.batch:
			raise ValueError(
				f"the layout has {layout.batch} rows, the environment {env.config.batch_size}"
			)
		if layout.pool and layout.pool % pool_size:
			raise ValueError("the pool rows must split into pool_size equal slices")
		missing = [name for name in layout.baseline_names if name not in policies]
		if missing:
			raise ValueError(f"no policy for {', '.join(missing)}")
		self.env = env
		self.layout = layout
		self.model = model
		self.network = network
		self.pool_size = pool_size
		self.policies = dict(policies)
		self.tables = env.tables
		self.labels = labels

	# Observations.

	def observe_lanes(self, state, learner) -> Observation:
		"""The observation of each lane [L]: its seat's view of its row's battle."""
		seats = self.layout.lane_seats(learner)
		lane_state = self.layout.lane_state(state)
		return jax.vmap(observe, in_axes=(0, 0, None))(lane_state, seats, self.tables)

	def observe_pool(self, state, learner) -> Observation:
		"""The observation of the opponent seat of each pool row [P]."""
		rows = self.layout.pool_rows
		return jax.vmap(observe, in_axes=(0, 0, None))(
			take(state, rows), 1 - learner[rows], self.tables
		)

	# Start.

	def reset(self, key, learner=None) -> LeagueState:
		"""Fresh battles in every row; the learner sides are drawn at random unless given [B]."""
		layout = self.layout
		key, reset_key, side_key = jax.random.split(key, 3)
		env_state = self.env.init(self.env.new_battles(reset_key), key)
		if learner is None:
			learner = jax.random.bernoulli(side_key, shape=(layout.batch,)).astype(jnp.int32)
		learner = jnp.asarray(learner, jnp.int32)
		return LeagueState(
			env=env_state,
			learner=learner,
			obs=self.observe_lanes(env_state.state, learner),
			hidden=initial_carry(self.network, (layout.lanes,)),
			pool_hidden=initial_carry(self.network, (layout.pool,)),
			pool_fresh=jnp.ones((layout.pool,), jnp.bool_),
			key=key,
		)

	# Opponents and actions.

	def pool_forward(self, pool: PoolParams, obs, hidden, fresh):
		"""Each pool row's member on its row: the carries after and the logits [P].

		A row reads its member's current parameters while fresh (its battle began after the member's
		last insertion) and the previous ones otherwise, so both sets run on every pool row.
		"""
		members = self.pool_size
		rows = self.layout.pool
		per_member = rows // members

		def cut(x):
			return x.reshape((members, per_member) + x.shape[1:])

		def join(x):
			return x.reshape((rows,) + x.shape[2:])

		def apply(params, member_obs, member_hidden):
			after, logits, _ = self.model.apply({"params": params}, member_obs, member_hidden)
			return after, logits

		def forward(stacked):
			after, logits = jax.vmap(apply)(
				stacked, jax.tree.map(cut, obs), jax.tree.map(cut, hidden)
			)
			return jax.tree.map(join, after), join(logits)

		after_now, logits_now = forward(pool.current)
		after_before, logits_before = forward(pool.previous)
		return (
			select_lanes(fresh, after_now, after_before),
			jnp.where(fresh[:, None], logits_now, logits_before),
		)

	def opponent_actions(self, pool: PoolParams, state: LeagueState, key):
		"""The opponent seats' actions, in row order [B - S], and the pool's carries after."""
		layout = self.layout
		pool_key, baseline_key = jax.random.split(key)
		parts = []
		pool_after = None
		if layout.pool:
			obs = self.observe_pool(state.env.state, state.learner)
			after, logits = self.pool_forward(pool, obs, state.pool_hidden, state.pool_fresh)
			parts.append(jax.random.categorical(pool_key, logits).astype(jnp.int32))
			pool_after = after
		if layout.baselines:
			keys = jax.random.split(baseline_key, layout.batch)
			for name, start, stop in layout.baselines:
				group = take(state.env.state, slice(start, stop))
				sides = 1 - state.learner[start:stop]
				action = jax.vmap(self.policies[name])(group, sides, keys[start:stop])
				parts.append(action.astype(jnp.int32))
		if not parts:
			return jnp.zeros((0,), jnp.int32), pool_after
		return jnp.concatenate(parts), pool_after

	def assemble(self, lane_actions, opponent_actions, learner):
		"""The actions of both seats of every row [B, 2]."""
		layout = self.layout
		actions = jnp.zeros((layout.batch, NUM_SIDES), jnp.int32)
		actions = actions.at[layout.lane_rows, layout.lane_seats(learner)].set(lane_actions)
		if layout.opponent_rows.size:
			seats = 1 - learner[layout.selfplay :]
			actions = actions.at[layout.opponent_rows, seats].set(opponent_actions)
		return actions

	# Training labels.

	def aux_labels(self, state: LeagueState, actions) -> AuxLabels:
		"""The auxiliary targets of each learner lane, read from the battle before the step.

		The opposing side's action comes from the assembled actions of this step; its kind is labelled
		only when it is a decision with a real choice (two or more engine-legal actions). Identity
		targets are dropped on rosters with an Illusion holder, where the protocol's public ranks may not
		match the true members. The item, ability and moves are the opposing active's, on a live
		active that is neither disguised nor transformed.
		"""
		layout = self.layout
		battles = layout.lane_state(state.env.state)
		lanes = jnp.arange(layout.lanes)
		seats = layout.lane_seats(state.learner)
		foe = 1 - seats
		action = actions[layout.lane_rows, foe]
		request = battles.request[lanes, foe]
		active = battles.active[lanes, foe].astype(jnp.int32)
		legal = block_switches(layout.lanes_of(state.env.base), layout.lanes_of(state.env.blocked))
		real_choice = legal[lanes, foe].sum(-1) >= 2
		decision = ((request == MOVE) | (request == SWITCH)) & real_choice
		is_move = action < MOVE_ACTIONS
		kind = jnp.where(decision, jnp.where(is_move, 0, 1), -1)
		move_ids = battles.moves[lanes, foe, active].astype(jnp.int32)
		move = move_ids[lanes, jnp.clip(action, 0, MOVE_ACTIONS - 1)]
		slots = bench_slots(active)
		slot = slots[lanes, jnp.clip(action - MOVE_ACTIONS, 0, slots.shape[-1] - 1)]
		seen = battles.seen_order[lanes, foe, slot].astype(jnp.int32)
		rank = jnp.where(seen > 0, seen - 1, UNSEEN_RANK)
		illusion = jnp.any(battles.ability_base[lanes, foe] == _illusion_ability(), axis=-1)
		disguised = battles.volatiles[lanes, foe, ILLUSION_SLOT] > 0
		transformed = battles.volatiles[lanes, foe, TRANSFORMED] > 0
		alive = battles.hp[lanes, foe, active] > 0
		return AuxLabels(
			next_kind=kind.astype(jnp.int8),
			next_move=jnp.where(kind == 0, move, 0).astype(jnp.int16),
			next_rank=jnp.where(kind == 1, rank, 0).astype(jnp.int8),
			identity=~illusion,
			private=alive & ~disguised & ~transformed & ~illusion,
			item=battles.item[lanes, foe, active].astype(jnp.int16),
			ability=battles.ability[lanes, foe, active].astype(jnp.int16),
			moves=move_ids.astype(jnp.int16),
		)

	def finish_labels(self, labels: AuxLabels, state: LeagueState, outcome) -> AuxLabels:
		"""Drop the next-action label where either side's action was held, refused or illegal, and the
		private targets where the lane's own action was held (its observation repeats)."""
		layout = self.layout
		touched = (
			layout.lanes_of(state.env.has_held)
			| layout.lanes_of(outcome["illegal"])
			| layout.lanes_of(outcome["rejected"])
		).any(axis=-1)
		seats = layout.lane_seats(state.learner)
		own_held = state.env.has_held[layout.lane_rows, seats]
		return labels._replace(
			next_kind=jnp.where(touched, -1, labels.next_kind).astype(jnp.int8),
			private=labels.private & ~own_held,
		)

	# Steps.

	def _advance(self, state: LeagueState, actions, after, pool_after):
		"""The environment step, the next learner sides and observations, and the step's outcome.

		after is each lane's carry after its observation and pool_after each pool row's, both already
		kept (not advanced) where the action was held. A held lane's and a held pool row's carry is
		kept here as well, so the carries of this step are the ones the next step starts from.
		"""
		layout = self.layout
		key, side_key = jax.random.split(state.key)
		env_state, advance = self.env.advance(state.env, actions)
		info = advance.info
		ended = info.ended
		redraw = jax.random.bernoulli(side_key, shape=(layout.batch,)).astype(jnp.int32)
		learner = jnp.where(ended & layout.learner_redraw, redraw, state.learner)
		seats = layout.lane_seats(state.learner)
		held = state.env.has_held[layout.lane_rows, seats]
		hidden = reset_carry(after, layout.lanes_of(ended))
		pool_held = state.env.has_held[layout.pool_rows, 1 - state.learner[layout.pool_rows]]
		pool_fresh = state.pool_fresh | ended[layout.pool_rows]
		if pool_after is None:
			pool_hidden = state.pool_hidden
		else:
			kept = select_lanes(pool_held, state.pool_hidden, pool_after)
			pool_hidden = reset_carry(kept, ended[layout.pool_rows])
		next_state = LeagueState(
			env=env_state,
			learner=learner,
			obs=self.observe_lanes(env_state.state, learner),
			hidden=hidden,
			pool_hidden=pool_hidden,
			pool_fresh=pool_fresh,
			key=key,
		)
		perspective = jnp.where(layout.is_selfplay_row, 0, state.learner)
		counted = ~(info.illegal[layout.lane_rows, seats] | held)
		real_choice = jnp.sum(state.obs.legal_actions, axis=-1) >= 2
		outcome = {
			"final_obs": self.observe_lanes(advance.final_state, state.learner),
			"reward": advance.reward[layout.lane_rows, seats],
			"advance": ~held,
			"weight": counted,
			"actor": counted & real_choice,
			"terminated": advance.terminated,
			"truncated": advance.truncated,
			"won": advance.terminated & (info.winner == perspective),
			"steps": info.steps,
			"turns": info.turns,
			"illegal": info.illegal,
			"rejected": info.rejected,
		}
		return next_state, outcome

	def play(self, params, pool: PoolParams | None, state: LeagueState, greedy=False):
		"""One step: the learner's lanes sample (or take the argmax when greedy), the opponents act.

		Returns the next state and the step's Transition. pool holds the pool's current and previous
		parameters (None when the layout has no pool rows). greedy may be a traced boolean.
		"""
		key, lane_key, opponent_key = jax.random.split(state.key, 3)
		state = state._replace(key=key)
		after, logits, value = self.model.apply({"params": params}, state.obs, state.hidden)
		sampled = jax.random.categorical(lane_key, logits)
		action = jnp.where(greedy, jnp.argmax(logits, axis=-1), sampled).astype(jnp.int32)
		log_prob = jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=-1)[:, 0]
		opponents, pool_after = self.opponent_actions(pool, state, opponent_key)
		actions = self.assemble(action, opponents, state.learner)
		seats = self.layout.lane_seats(state.learner)
		held = state.env.has_held[self.layout.lane_rows, seats]
		after = select_lanes(held, state.hidden, after)
		labels = self.aux_labels(state, actions) if self.labels else None
		next_state, outcome = self._advance(state, actions, after, pool_after)
		if labels is not None:
			labels = self.finish_labels(labels, state, outcome)
		transition = Transition(
			obs=state.obs,
			hidden=state.hidden,
			action=action,
			seat=seats,
			log_prob=log_prob,
			value=value,
			after=after,
			aux=labels,
			**outcome,
		)
		return next_state, transition

	def random_play(self, state: LeagueState) -> LeagueState:
		"""One step with uniform random legal actions on every side (the environment alone)."""
		layout = self.layout
		key, lane_key, opponent_key = jax.random.split(state.key, 3)
		state = state._replace(key=key)
		random = RandomPolicy(self.tables)
		lane_actions = jax.vmap(random)(
			layout.lane_state(state.env.state),
			layout.lane_seats(state.learner),
			jax.random.split(lane_key, layout.lanes),
		)
		opponents = jnp.zeros((0,), jnp.int32)
		if layout.opponent_rows.size:
			rows = slice(layout.selfplay, None)
			opponents = jax.vmap(random)(
				take(state.env.state, rows),
				1 - state.learner[rows],
				jax.random.split(opponent_key, layout.batch - layout.selfplay),
			)
		actions = self.assemble(lane_actions, opponents, state.learner)
		pool_after = state.pool_hidden if layout.pool else None
		next_state, _ = self._advance(state, actions, state.hidden, pool_after)
		return next_state
