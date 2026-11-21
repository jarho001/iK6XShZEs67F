"""Batched, functional environments around the engine, for self-play and for a fixed opponent.

`Env` runs B battles at once. `reset(key)` and `step(env_state, actions)` are pure functions of
their arguments (plus the host tables built once in the constructor), so they jit, vmap and scan.
A battle that ends is replaced inside the same step by a fresh one with teams from the sampler
(pokejax.sampler), and the observation returned for it is the new battle's first one.

Actions are int [B, 2] in pokejax.engine.actions encoding, one per side. Every step is one decision
point: a battle waits for both sides (a side with nothing to choose plays the no-op action 0).

Termination and truncation:
- terminated: the engine ended the battle (a win, a loss or a tie), or Showdown's turn limit
  (sim/battle.ts: more than 1000 turns is a tie) did. The engine itself has no turn limit.
- truncated: the battle took `max_steps` environment steps without ending.
- Rewards, per side [B, 2], are zero-sum: win 1, loss -1, tie 0, when the battle terminates.
  A truncated battle gets no terminal reward.
- Optional shaping (config.shaping): r += coef * (gamma * phi(s') - phi(s)) with phi the side's
  summed remaining HP fraction minus the foe's, over 6. phi is zero at reset and for the state
  after an ended battle (the fresh one), so the shaped return of an episode adds nothing.

Illegal and refused actions:
- A trapped switch (Shadow Tag, Arena Trap, Magnet Pull) that the request did not announce is
  refused as Showdown refuses it (sim/side.ts chooseSwitch): the battle does not advance, the
  refused side's flag BattleState.trap_known is set, so its observed mask has no switches from
  now on, and the other side's action of this step is kept and used at the next step, whatever
  that step is called with (its choice is already locked in). Showdown asks the refused side
  again with trapped: true; here that is the next step. info.rejected marks the refused sides.
- Any other illegal action is a caller bug: it is counted in info.illegal and the first legal
  action is played instead. That includes a switch from a side that was already refused: its
  observed mask no longer offers one.

Randomness: `step` splits env_state.key into (key, step_key, reset_key). The engine key of battle b
is split(step_key, B)[b], and the fresh battles come from new_battles(reset_key). The same
key and actions therefore always give the same trajectory.

`OpponentEnv` plays one learner side per battle against a policy function (state, side, key)
-> action that runs inside step; its time steps have batch shape [B]. Its step splits the key
into (key, opponent_key, side_key): the opponent's key for battle b is split(opponent_key, B)[b],
and side_key draws the learner sides after a reset. pokejax.baselines has four standard opponents
(random, random-move, max-damage, heuristics) with that signature.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.baselines import RandomPolicy
from pokejax.engine.actions import (
	NUM_ACTIONS,
	action_masks,
	block_switches,
)
from pokejax.engine.data import build_engine_tables
from pokejax.engine.observation import observe
from pokejax.engine.reset import (
	Teams,
	check_teams,
	prepare_teams,
	reset,
)
from pokejax.engine.state import (
	MOVE,
	NUM_SIDES,
	ONGOING,
	SIDE0,
	SIDE1,
	TEAM_SLOTS,
	TIE,
	BattleState,
)
from pokejax.engine.step import step
from pokejax.sampler import sample_team_indices, supported_team_indices
from pokejax.tables import Tables, load_tables
from pokejax.teams import TeamPool, load_npz

DEFAULT_POOL = Path("work/teams/gen5-teams.npz")
# sim/battle.ts maybeTriggerEndlessBattleClause: after turn 1000 the battle is a tie.
TURN_LIMIT = 1000
OBSERVE_MODES = ("both", "none")


@dataclass(frozen=True)
class EnvConfig:
	batch_size: int = 256
	max_steps: int = 1000  # environment steps of one battle before it is truncated
	shaping: bool = False
	shaping_coef: float = 0.1
	gamma: float = 0.99  # discount of the shaping term
	observe: str = "both"  # "both" or "none"; "none" skips observing, for baselines that read state
	keep_final: bool = False  # also return the observation before the reset of battles that ended

	def __post_init__(self):
		if min(self.batch_size, self.max_steps) < 1:
			raise ValueError("batch_size and max_steps must be positive")
		if self.observe not in OBSERVE_MODES:
			raise ValueError(f"observe must be one of {OBSERVE_MODES}")
		if self.keep_final and self.observe == "none":
			raise ValueError("keep_final needs observations")


class EnvState(NamedTuple):
	"""Everything step needs; leaves have a leading batch axis except key."""

	state: BattleState  # [B, ...]
	key: jax.Array
	steps: jax.Array  # int32 [B], environment steps of the current battle
	held: jax.Array  # int32 [B, 2], an action kept from a refused step
	has_held: jax.Array  # bool [B, 2], held is used at the next step
	potential: jax.Array  # float32 [B], phi from side 0's view (zero unless shaping)
	# What the engine allows now, without the hidden traps (see actions.action_masks), and which
	# sides a hidden trap forbids switches. Computed once per step for the next one.
	base: jax.Array  # bool [B, 2, 9]
	blocked: jax.Array  # bool [B, 2]


class Info(NamedTuple):
	ended: jax.Array  # bool [B], the battle ended in this step (terminated or truncated)
	winner: jax.Array  # int8 [B], SIDE0, SIDE1 or TIE when terminated, else ONGOING (-1)
	steps: jax.Array  # int32 [B], environment steps of the battle that ended, else 0
	turns: jax.Array  # int32 [B], Showdown's turn counter when it ended, else 0
	illegal: jax.Array  # bool [B, 2], the side's action was illegal and was replaced
	rejected: jax.Array  # bool [B, 2], the side's switch was refused by a hidden trap


class TimeStep(NamedTuple):
	obs: object  # Observation [B, 2, ...] (side axis 1), None when config.observe is "none"
	reward: jax.Array  # float32 [B, 2]
	terminated: jax.Array  # bool [B]
	truncated: jax.Array  # bool [B]
	legal: jax.Array  # bool [B, 2, 9], the observed legal mask of the returned battle
	info: Info
	final_obs: object  # observation before the reset (meaningful where info.ended), or None


class Advance(NamedTuple):
	"""What a step computed besides the new state; unused fields cost nothing under jit."""

	reward: jax.Array
	terminated: jax.Array
	truncated: jax.Array
	info: Info
	final_state: BattleState  # the battles after this step, before any reset


def select(mask, new, old):
	"""Per battle: new where mask [B], else old; every leaf has a leading batch axis."""
	return jax.tree.map(
		lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (o.ndim - 1)), n, o), new, old
	)


def hp_potential(state: BattleState) -> jax.Array:
	"""phi [B] from side 0's view: (own remaining HP fractions - the foe's) / 6; side 1's is -phi."""
	fraction = state.hp / jnp.maximum(state.stats[..., 0], 1)
	return (fraction[:, 0].sum(-1) - fraction[:, 1].sum(-1)) / TEAM_SLOTS


def side_reward(winner):
	"""+1 for side 0's win, -1 for its loss, 0 for a tie or a battle in progress."""
	return jnp.where(winner == SIDE0, 1.0, jnp.where(winner == SIDE1, -1.0, 0.0))


@jax.tree_util.register_pytree_node_class
class Env:
	"""Host setup once; reset and step are pure and take no host data.

	The arrays (team pool, prepared teams, tables, supported team indices) are the pytree's
	leaves, so `jax.jit(lambda env, s, a: env.step(s, a))` takes them as arguments; a jitted
	`env.step` closes over them as constants instead.
	"""

	def __init__(
		self,
		config: EnvConfig = EnvConfig(),
		pool_path=DEFAULT_POOL,
		*,
		team_pool: TeamPool | None = None,
		tables: Tables | None = None,
	):
		tables = load_tables() if tables is None else tables
		host_pool = load_npz(pool_path) if team_pool is None else team_pool
		if not len(host_pool.team_size) or np.any(host_pool.team_size != TEAM_SLOTS):
			raise ValueError("The pool must contain only complete six-member teams")
		check_teams(
			Teams(
				*(getattr(host_pool, n)[:, None].repeat(NUM_SIDES, axis=1) for n in Teams._fields)
			),
			tables,
		)
		self.config = config
		self.pool = Teams(*(jnp.asarray(getattr(host_pool, n)) for n in Teams._fields))
		self.tables = build_engine_tables(tables)
		# Only fully supported teams are drawn; the index array is computed once here.
		self.team_indices = jnp.asarray(supported_team_indices(host_pool, tables))
		if not self.team_indices.shape[0]:
			raise ValueError("The pool has no fully supported team")
		self.prepared = prepare_teams(self.pool, self.tables)

	def tree_flatten(self):
		return (self.pool, self.prepared, self.tables, self.team_indices), self.config

	@classmethod
	def tree_unflatten(cls, config, children):
		env = object.__new__(cls)
		env.config = config
		env.pool, env.prepared, env.tables, env.team_indices = children
		return env

	def replace(self, **changes) -> "Env":
		"""The same host data under a changed config (for example another batch size)."""
		children, config = self.tree_flatten()
		return Env.tree_unflatten(dataclasses.replace(config, **changes), children)

	def with_team_indices(self, team_indices) -> "Env":
		"""The same environment drawing teams only from these pool indices."""
		children, config = self.tree_flatten()
		return Env.tree_unflatten(config, (*children[:3], jnp.asarray(team_indices, jnp.int32)))

	# Fresh battles.

	def new_battles(self, key) -> BattleState:
		"""Reset a batch on team pairs drawn from the sampler's supported teams."""
		index_key, reset_key = jax.random.split(key)
		batch = self.config.batch_size
		indices = sample_team_indices(index_key, self.team_indices, batch)
		teams = jax.tree.map(lambda column: column[indices], self.pool)
		prepared = jax.tree.map(lambda column: column[indices], self.prepared)
		keys = jax.random.split(reset_key, batch)
		return jax.vmap(lambda k, t, p: reset(k, t, self.tables, None, p))(keys, teams, prepared)

	def init(self, state: BattleState, key) -> EnvState:
		"""An environment state around battle states [B, ...] that the caller made."""
		batch = self.config.batch_size
		if state.winner.shape != (batch,):
			raise ValueError(f"Expected {batch} battles, got {state.winner.shape}")
		base, blocked = jax.vmap(action_masks, in_axes=(0, None))(state, self.tables)
		potential = hp_potential(state) if self.config.shaping else jnp.zeros(batch)
		return EnvState(
			state=state,
			key=key,
			steps=jnp.zeros(batch, jnp.int32),
			held=jnp.zeros((batch, NUM_SIDES), jnp.int32),
			has_held=jnp.zeros((batch, NUM_SIDES), jnp.bool_),
			potential=potential.astype(jnp.float32),
			base=base,
			blocked=blocked,
		)

	# Masks and observations.

	def engine_mask(self, es: EnvState) -> jax.Array:
		"""The engine's legal mask [B, 2, 9], hidden traps included (what a policy may pick from)."""
		return block_switches(es.base, es.blocked)

	def observed_mask(self, es: EnvState) -> jax.Array:
		"""The legal mask a player's request shows [B, 2, 9] (what the observation carries)."""
		state = es.state
		moving = (state.request == MOVE) & (state.winner == ONGOING)[:, None]
		return block_switches(es.base, state.trap_known & moving)

	def observe_both(self, state: BattleState):
		"""Observations of both sides, leaves [B, 2, ...]."""
		both = jax.vmap(observe, in_axes=(None, 0, None))
		return jax.vmap(lambda s: both(s, jnp.arange(NUM_SIDES), self.tables))(state)

	def _observation(self, state):
		return self.observe_both(state) if self.config.observe == "both" else None

	# Transitions.

	def reset(self, key) -> tuple[EnvState, TimeStep]:
		key, reset_key = jax.random.split(key)
		es = self.init(self.new_battles(reset_key), key)
		obs = self._observation(es.state)
		batch = self.config.batch_size
		timestep = TimeStep(
			obs=obs,
			reward=jnp.zeros((batch, NUM_SIDES), jnp.float32),
			terminated=jnp.zeros(batch, jnp.bool_),
			truncated=jnp.zeros(batch, jnp.bool_),
			legal=self.observed_mask(es),
			info=Info(
				ended=jnp.zeros(batch, jnp.bool_),
				winner=jnp.full(batch, ONGOING, jnp.int8),
				steps=jnp.zeros(batch, jnp.int32),
				turns=jnp.zeros(batch, jnp.int32),
				illegal=jnp.zeros((batch, NUM_SIDES), jnp.bool_),
				rejected=jnp.zeros((batch, NUM_SIDES), jnp.bool_),
			),
			final_obs=obs if self.config.keep_final else None,
		)
		return es, timestep

	def step(self, es: EnvState, actions) -> tuple[EnvState, TimeStep]:
		"""Advance every battle by one decision point; see the module docstring."""
		es, advance = self.advance(es, actions)
		final_obs = None
		if self.config.keep_final:
			final_obs = self.observe_both(advance.final_state)
		timestep = TimeStep(
			obs=self._observation(es.state),
			reward=advance.reward,
			terminated=advance.terminated,
			truncated=advance.truncated,
			legal=self.observed_mask(es),
			info=advance.info,
			final_obs=final_obs,
		)
		return es, timestep

	def advance(self, es: EnvState, actions) -> tuple[EnvState, Advance]:
		"""The transition without observations; step and the wrappers build on it."""
		c, t = self.config, self.tables
		batch = c.batch_size
		actions = jnp.asarray(actions, jnp.int32)
		if actions.shape != (batch, NUM_SIDES):
			raise ValueError(f"Expected actions of shape {(batch, NUM_SIDES)}, got {actions.shape}")
		state = es.state
		key, step_key, reset_key = jax.random.split(es.key, 3)

		# A side whose foe was refused last step plays the action it had locked in.
		actions = jnp.where(es.has_held, es.held, actions)
		legal = block_switches(es.base, es.blocked)
		in_range = (actions >= 0) & (actions < NUM_ACTIONS)
		index = jnp.clip(actions, 0, NUM_ACTIONS - 1)[..., None]
		chosen_legal = in_range & jnp.take_along_axis(legal, index, axis=-1)[..., 0]
		chosen_base = in_range & jnp.take_along_axis(es.base, index, axis=-1)[..., 0]
		# Only a hidden trap separates the two masks, and only on the switch actions.
		# A side that already knows about the trap was told so: its switch is an illegal action.
		rejected = ~chosen_legal & chosen_base & es.blocked & ~state.trap_known
		illegal = ~chosen_legal & ~rejected
		actions = jnp.where(illegal, jnp.argmax(legal, axis=-1), actions)
		refused = jnp.any(rejected, axis=-1)

		keys = jax.random.split(step_key, batch)
		stepped = jax.vmap(step, in_axes=(0, 0, 0, None, None))(state, actions, keys, None, t)
		# A refused battle does not advance; its refused sides now know about the trap.
		kept = select(refused, state, stepped)
		kept = kept._replace(
			trap_known=jnp.where(refused[:, None], state.trap_known | rejected, kept.trap_known)
		)
		steps = es.steps + 1
		over = (kept.turn > TURN_LIMIT) & (kept.winner == ONGOING)
		kept = kept._replace(winner=jnp.where(over, TIE, kept.winner).astype(kept.winner.dtype))

		terminated = kept.winner != ONGOING
		truncated = ~terminated & (steps >= c.max_steps)
		ended = terminated | truncated
		terminal = side_reward(kept.winner)
		potential = jnp.zeros(batch, jnp.float32)
		if c.shaping:
			potential = jnp.where(ended, 0.0, hp_potential(kept)).astype(jnp.float32)
			terminal += c.shaping_coef * (c.gamma * potential - es.potential)
		reward = jnp.stack((terminal, -terminal), axis=1).astype(jnp.float32)

		fresh = self.new_battles(reset_key)
		state_next = select(ended, fresh, kept)
		base, blocked = jax.vmap(action_masks, in_axes=(0, None))(state_next, t)
		keep_held = refused[:, None] & ~rejected & ~ended[:, None]
		es_next = EnvState(
			state=state_next,
			key=key,
			steps=jnp.where(ended, 0, steps),
			held=jnp.where(keep_held, actions, 0),
			has_held=keep_held,
			potential=potential,
			base=base,
			blocked=blocked,
		)
		info = Info(
			ended=ended,
			winner=jnp.where(terminated, kept.winner, ONGOING).astype(jnp.int8),
			steps=jnp.where(ended, steps, 0),
			turns=jnp.where(ended, kept.turn, 0),
			illegal=illegal,
			rejected=rejected,
		)
		return es_next, Advance(reward, terminated, truncated, info, kept)


class OpponentState(NamedTuple):
	env: EnvState
	learner: jax.Array  # int32 [B], the learner's side in each battle


class LearnerInfo(NamedTuple):
	ended: jax.Array  # bool [B]
	winner: jax.Array  # int8 [B], SIDE0, SIDE1 or TIE when terminated, else ONGOING
	steps: jax.Array  # int32 [B], steps of the battle that ended, else 0
	turns: jax.Array  # int32 [B], turns of the battle that ended, else 0
	learner: jax.Array  # int32 [B], the learner's side in the battle that ended (else current)
	illegal: jax.Array  # bool [B], the learner's action was illegal and was replaced
	opponent_illegal: jax.Array  # bool [B], the opponent policy's action was
	rejected: jax.Array  # bool [B], the learner's switch was refused by a hidden trap


@jax.tree_util.register_pytree_node_class
class OpponentEnv:
	"""One learner side per battle against a fixed policy; every array has batch shape [B].

	The learner's side is drawn at random at each reset and each auto-reset (state.learner).
	opponent(state, side, key) -> action takes one battle's BattleState and runs inside step
	(vmapped); the default is RandomPolicy over the engine's legal mask. It should choose
	from the engine's legal mask: a switch that only a hidden trap forbids is refused like the
	learner's own (the battle does not advance and the learner's action is kept).
	Observations are the learner's, so only one side is observed; config.observe "none" skips it.
	"""

	def __init__(self, env: Env, opponent=None):
		self.env = env
		self.opponent = RandomPolicy(env.tables) if opponent is None else opponent

	def tree_flatten(self):
		if jax.tree_util.treedef_is_leaf(jax.tree.structure(self.opponent)):
			return (self.env,), self.opponent  # a plain function is static
		return (self.env, self.opponent), None

	@classmethod
	def tree_unflatten(cls, static, children):
		wrapper = object.__new__(cls)
		wrapper.env = children[0]
		wrapper.opponent = static if static is not None else children[1]
		return wrapper

	@property
	def config(self) -> EnvConfig:
		return self.env.config

	def replace(self, **changes) -> "OpponentEnv":
		return OpponentEnv(self.env.replace(**changes), self.opponent)

	def _learner_observation(self, state, learner):
		if self.config.observe == "none":
			return None
		return jax.vmap(observe, in_axes=(0, 0, None))(state, learner, self.env.tables)

	def _learner_mask(self, es, learner):
		mask = self.env.observed_mask(es)
		return jnp.take_along_axis(mask, learner[:, None, None], axis=1)[:, 0]

	def reset(self, key) -> tuple[OpponentState, TimeStep]:
		batch = self.config.batch_size
		key, side_key, reset_key = jax.random.split(key, 3)
		es = self.env.init(self.env.new_battles(reset_key), key)
		learner = jax.random.bernoulli(side_key, shape=(batch,)).astype(jnp.int32)
		obs = self._learner_observation(es.state, learner)
		zero = jnp.zeros(batch, jnp.bool_)
		timestep = TimeStep(
			obs=obs,
			reward=jnp.zeros(batch, jnp.float32),
			terminated=zero,
			truncated=zero,
			legal=self._learner_mask(es, learner),
			info=LearnerInfo(
				ended=zero,
				winner=jnp.full(batch, ONGOING, jnp.int8),
				steps=jnp.zeros(batch, jnp.int32),
				turns=jnp.zeros(batch, jnp.int32),
				learner=learner,
				illegal=zero,
				opponent_illegal=zero,
				rejected=zero,
			),
			final_obs=obs if self.config.keep_final else None,
		)
		return OpponentState(es, learner), timestep

	def step(self, ws: OpponentState, action) -> tuple[OpponentState, TimeStep]:
		"""Advance with the learner's action [B]; the opponent's is computed here."""
		batch = self.config.batch_size
		rows = jnp.arange(batch)
		key, opponent_key, side_key = jax.random.split(ws.env.key, 3)
		es = ws.env._replace(key=key)
		opponent_side = 1 - ws.learner
		opponent_action = jax.vmap(self.opponent)(
			es.state, opponent_side, jax.random.split(opponent_key, batch)
		)
		action = jnp.asarray(action, jnp.int32)
		if action.shape != (batch,):
			raise ValueError(f"Expected actions of shape {(batch,)}, got {action.shape}")
		learner_acts = jnp.arange(NUM_SIDES)[None, :] == ws.learner[:, None]
		actions = jnp.where(learner_acts, action[:, None], opponent_action[:, None])
		es, advance = self.env.advance(es, actions)
		info = advance.info
		fresh = jax.random.bernoulli(side_key, shape=(batch,)).astype(jnp.int32)
		learner = jnp.where(info.ended, fresh, ws.learner)
		final_obs = None
		if self.config.keep_final:
			final_obs = self._learner_observation(advance.final_state, ws.learner)
		timestep = TimeStep(
			obs=self._learner_observation(es.state, learner),
			reward=advance.reward[rows, ws.learner],
			terminated=advance.terminated,
			truncated=advance.truncated,
			legal=self._learner_mask(es, learner),
			info=LearnerInfo(
				ended=info.ended,
				winner=info.winner,
				steps=info.steps,
				turns=info.turns,
				learner=ws.learner,
				illegal=info.illegal[rows, ws.learner],
				opponent_illegal=info.illegal[rows, opponent_side],
				rejected=info.rejected[rows, ws.learner],
			),
			final_obs=final_obs,
		)
		return OpponentState(es, learner), timestep
