"""The environment: auto-reset, rewards, termination, the trapped-switch rule and the wrapper."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.baselines import POLICY_NAMES, make_policy
from pokejax.engine.actions import legal_actions, observed_actions
from pokejax.engine.observation import observe
from pokejax.engine.state import MOVE, ONGOING, SIDE0, SIDE1, TIE
from pokejax.engine.step import step
from pokejax.env import (
	TURN_LIMIT,
	Env,
	EnvConfig,
	LearnerInfo,
	OpponentEnv,
	OpponentState,
	RandomPolicy,
	hp_potential,
)
from pokejax.sampler import supported_team_indices
from pokejax.tables import load_tables
from pokejax.teams import load_team_pool

BATCH = 16
GAMMA = 0.9
COEF = 0.5
KEY = jax.random.key(11)


@pytest.fixture(scope="module")
def tables():
	return load_tables()


@pytest.fixture(scope="module")
def pool(tables, tmp_path_factory):
	fixture = json.loads(Path("tests/fixtures/reset.json").read_text())
	path = tmp_path_factory.mktemp("env") / "teams.json"
	path.write_text(
		json.dumps(
			{
				"format": "gen5randombattle",
				"count": len(fixture["teamSets"]),
				"seed": 0,
				"showdownCommit": fixture["showdownCommit"],
				"teams": fixture["teamSets"],
			}
		)
	)
	return load_team_pool(path, tables)


@pytest.fixture(scope="module")
def env(pool, tables):
	"""No observations: baselines read the state, and the compiled step is the smallest."""
	return Env(EnvConfig(BATCH, observe="none"), team_pool=pool, tables=tables)


@pytest.fixture(scope="module")
def full_env(env):
	"""Both observations, the final observation and shaping: one more compiled variant."""
	return env.replace(
		observe="both", keep_final=True, shaping=True, shaping_coef=COEF, gamma=GAMMA
	)


@pytest.fixture(scope="module")
def wrapper(env):
	return OpponentEnv(env.replace(observe="both"))


# Pure functions with the environment as an argument, compiled once per config.
RESET = jax.jit(lambda env, key: env.reset(key))
STEP = jax.jit(lambda env, es, actions: env.step(es, actions))
NEW_BATTLES = jax.jit(lambda env, key: env.new_battles(key))
ENGINE_STEP = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None, None)))
ENGINE_MASK = jax.jit(lambda env, es: env.engine_mask(es))
OBSERVE_SIDE = jax.jit(jax.vmap(observe, in_axes=(0, None, None)), static_argnums=1)
RANDOM_ACTIONS = jax.jit(
	lambda env, es, key: jax.random.categorical(
		key, jnp.where(env.engine_mask(es), 0.0, -jnp.inf), axis=-1
	)
)
OBSERVE_LEARNER = jax.jit(jax.vmap(observe, in_axes=(0, 0, None)))
WRAPPER_RESET = jax.jit(lambda w, key: w.reset(key))
WRAPPER_STEP = jax.jit(lambda w, ws, action: w.step(ws, action))
ZEROS = jnp.zeros((BATCH, 2), jnp.int32)


def as_array(x):
	return jax.random.key_data(x) if jnp.issubdtype(x.dtype, jax.dtypes.prng_key) else x


def assert_equal(a, b):
	for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
		np.testing.assert_array_equal(as_array(x), as_array(y))


def select(mask, new, old):
	return jax.tree.map(
		lambda n, o: jnp.where(mask.reshape(mask.shape + (1,) * (o.ndim - 1)), n, o), new, old
	)


def reference(env, es, actions):
	"""The engine's own step and the fresh battles, from the key layout the module documents."""
	_, step_key, reset_key = jax.random.split(es.key, 3)
	keys = jax.random.split(step_key, env.config.batch_size)
	return ENGINE_STEP(es.state, actions, keys, None, env.tables), NEW_BATTLES(env, reset_key)


def with_winner(es, winners):
	return es._replace(state=es.state._replace(winner=jnp.asarray(winners, jnp.int8)))


def set_active(state, name, side, value, *slot):
	"""Set a member field of one side's active member, in every battle (slot: further indices)."""
	rows = jnp.arange(state.winner.shape[0])
	field = getattr(state, name)
	return state._replace(**{name: field.at[rows, side, state.active[:, side], *slot].set(value)})


def trapped_state(env, tables, name="shadowtag"):
	"""Every battle: side 1's active has the trapping ability, side 0's does not, and both only
	know Splash, so nobody faints, leaves or changes while the test plays turns."""
	state = NEW_BATTLES(env, KEY)
	for side in (0, 1):
		state = set_active(state, "moves", side, tables.moves.index["splash"], 0)
		state = set_active(state, "pp", side, 20, 0)
	state = set_active(state, "ability", 1, tables.abilities.index[name])
	return set_active(state, "ability", 0, tables.abilities.index["pressure"])


def test_reset_gives_the_first_decision_and_teams_from_the_sampler(env, full_env, pool, tables):
	es, timestep = RESET(full_env, KEY)
	assert es.state.winner.shape == (BATCH,)
	assert np.all(np.asarray(es.state.winner) == ONGOING)
	assert np.all(np.asarray(es.state.request) == MOVE)
	assert np.all(np.asarray(es.steps) == 0) and not np.any(np.asarray(es.has_held))
	assert timestep.reward.shape == (BATCH, 2) and not np.any(np.asarray(timestep.reward))
	assert not np.any(np.asarray(timestep.terminated | timestep.truncated))
	assert not np.any(np.asarray(timestep.info.ended))
	# The observation of each side is the engine's, for the same battle.
	for side in (0, 1):
		got = jax.tree.map(lambda x, side=side: x[:, side], timestep.obs)
		assert_equal(OBSERVE_SIDE(es.state, side, full_env.tables), got)
	np.testing.assert_array_equal(timestep.legal, timestep.obs.legal_actions)
	np.testing.assert_array_equal(timestep.legal, full_env.observed_mask(es))
	# Observations are skipped when the config asks for none.
	assert RESET(env, KEY)[1].obs is None
	# Both sides' teams are rows of the pool, drawn by the sampler from the supported teams.
	allowed = set(np.asarray(env.team_indices).tolist())
	assert allowed == set(supported_team_indices(pool, tables).tolist())
	species = np.asarray(es.state.species_base)  # a lead may have changed into its foe (Imposter)
	rows = {i: np.asarray(env.pool.species[i]) for i in allowed}
	for b in range(BATCH):
		for side in (0, 1):
			assert any(np.array_equal(row, species[b, side]) for row in rows.values()), (b, side)


def test_auto_reset_draws_from_the_sampler_only(env):
	only = env.with_team_indices(jnp.full_like(env.team_indices, 3))
	state = NEW_BATTLES(only, KEY)
	expected = np.asarray(env.pool.species[3])
	for side in (0, 1):
		for b in range(BATCH):
			np.testing.assert_array_equal(np.asarray(state.species_base[b, side]), expected)
	es = with_winner(only.init(state, KEY), np.full(BATCH, SIDE0))
	es, timestep = STEP(only, es, ZEROS)
	assert np.all(np.asarray(timestep.info.ended))
	for side in (0, 1):
		for b in range(BATCH):
			np.testing.assert_array_equal(np.asarray(es.state.species_base[b, side]), expected)


def test_only_finished_battles_change_and_the_rest_are_untouched(env):
	es, _ = RESET(env, KEY)
	key = jax.random.key(3)
	mixed = False
	finished_total = 0
	for _ in range(110):
		key, action_key = jax.random.split(key)
		actions = RANDOM_ACTIONS(env, es, action_key)
		stepped, fresh = reference(env, es, actions)
		es_next, timestep = STEP(env, es, actions)
		ended = np.asarray(timestep.info.ended)
		assert np.array_equal(ended, np.asarray(timestep.terminated))  # no truncation here
		assert np.array_equal(ended, np.asarray(stepped.winner) != ONGOING)
		# Bit for bit: a battle that went on is the engine's step, a finished one is fresh.
		assert_equal(es_next.state, select(jnp.asarray(ended), fresh, stepped))
		mixed |= 0 < ended.sum() < BATCH
		finished_total += int(ended.sum())
		steps = np.asarray(es.steps) + 1
		np.testing.assert_array_equal(es_next.steps, np.where(ended, 0, steps))
		np.testing.assert_array_equal(timestep.info.steps, np.where(ended, steps, 0))
		np.testing.assert_array_equal(
			timestep.info.turns, np.where(ended, np.asarray(stepped.turn), 0)
		)
		assert not np.asarray(timestep.info.illegal).any()
		es = es_next
	assert mixed and finished_total >= BATCH


def test_rewards_are_zero_sum_with_win_loss_and_tie(env):
	es, _ = RESET(env, KEY)
	winners = np.full(BATCH, ONGOING)
	winners[:4] = [SIDE0, SIDE1, TIE, SIDE0]
	es, timestep = STEP(env, with_winner(es, winners), ZEROS)
	reward = np.asarray(timestep.reward)
	np.testing.assert_array_equal(reward[:4], [[1, -1], [-1, 1], [0, 0], [1, -1]])
	assert not reward[4:].any()
	np.testing.assert_array_equal(reward.sum(axis=1), 0)
	ended = np.arange(BATCH) < 4
	np.testing.assert_array_equal(timestep.info.ended, ended)
	np.testing.assert_array_equal(timestep.terminated, ended)
	assert not np.asarray(timestep.truncated).any()
	np.testing.assert_array_equal(timestep.info.winner, np.where(ended, winners, ONGOING))
	# The finished battles are fresh ones that start over.
	assert np.all(np.asarray(es.state.winner) == ONGOING)
	np.testing.assert_array_equal(np.asarray(es.steps)[:4], 0)
	np.testing.assert_array_equal(np.asarray(es.steps)[4:], 1)


def test_shaping_is_off_by_default_and_only_the_end_pays(env):
	assert EnvConfig().shaping is False
	es, _ = RESET(env, KEY)
	key = jax.random.key(5)
	for _ in range(60):
		key, action_key = jax.random.split(key)
		es, timestep = STEP(env, es, RANDOM_ACTIONS(env, es, action_key))
		reward = np.asarray(timestep.reward)
		assert not reward[~np.asarray(timestep.info.ended)].any()
		assert set(np.unique(reward)) <= {-1.0, 0.0, 1.0}


def test_potential_based_shaping(full_env):
	"""r += coef * (gamma * phi(s') - phi(s)) with phi zero at a fresh battle: an episode's
	discounted shaping sums to zero."""
	es, _ = RESET(full_env, KEY)
	key = jax.random.key(6)
	discounted = np.zeros(BATCH)
	scale = np.ones(BATCH)
	episodes = 0
	for _ in range(120):
		key, action_key = jax.random.split(key)
		actions = RANDOM_ACTIONS(full_env, es, action_key)
		before = np.asarray(hp_potential(es.state))
		np.testing.assert_allclose(np.asarray(es.potential), before, atol=1e-6)
		stepped, _ = reference(full_env, es, actions)
		es, timestep = STEP(full_env, es, actions)
		ended = np.asarray(timestep.info.ended)
		winner = np.asarray(stepped.winner)
		after = np.where(ended, 0.0, np.asarray(hp_potential(stepped)))
		shaping = COEF * (GAMMA * after - before)
		terminal = np.where(winner == SIDE0, 1.0, np.where(winner == SIDE1, -1.0, 0.0))
		reward = np.asarray(timestep.reward)
		np.testing.assert_allclose(reward[:, 0], terminal + shaping, atol=1e-5)
		np.testing.assert_allclose(reward[:, 1], -(terminal + shaping), atol=1e-5)
		discounted += scale * shaping
		scale *= GAMMA
		for b in np.flatnonzero(ended):
			np.testing.assert_allclose(discounted[b], 0.0, atol=1e-5)
			episodes += 1
		discounted[ended] = 0
		scale[ended] = 1
	assert episodes > 0
	# Both teams are whole at the start of a battle, so phi is zero there.
	assert not np.asarray(hp_potential(RESET(full_env, KEY)[0].state)).any()


def test_truncation_at_the_step_cap_is_separate_from_termination(env):
	es, _ = RESET(env, KEY)
	cap = env.config.max_steps
	steps = np.zeros(BATCH, np.int32)
	steps[[0, 1, 2, 3]] = cap - 1  # the next step is the cap's
	steps[4] = cap - 2  # one more step to go
	winners = np.full(BATCH, ONGOING)
	winners[2] = SIDE1  # the engine ends it in the very step of the cap: termination wins
	es = with_winner(es, winners)._replace(steps=jnp.asarray(steps))
	es, timestep = STEP(env, es, ZEROS)
	terminated, truncated = np.asarray(timestep.terminated), np.asarray(timestep.truncated)
	np.testing.assert_array_equal(truncated[:5], [True, True, False, True, False])
	np.testing.assert_array_equal(terminated[:5], [False, False, True, False, False])
	assert not terminated[5:].any() and not truncated[5:].any()
	reward = np.asarray(timestep.reward)
	assert not reward[[0, 1, 3]].any()  # no terminal reward for a truncated battle
	np.testing.assert_array_equal(reward[2], [-1, 1])
	info = timestep.info
	np.testing.assert_array_equal(np.asarray(info.ended)[:5], [True] * 4 + [False])
	np.testing.assert_array_equal(np.asarray(info.winner)[:4], [ONGOING, ONGOING, SIDE1, ONGOING])
	np.testing.assert_array_equal(np.asarray(info.steps)[:5], [cap, cap, cap, cap, 0])
	# The truncated slots are fresh battles; the one that is not finished keeps counting.
	assert int(es.steps[4]) == cap - 1
	np.testing.assert_array_equal(np.asarray(es.steps)[:4], 0)
	assert np.all(np.asarray(es.state.winner) == ONGOING)


def test_showdowns_turn_limit_is_a_tie(env):
	es, _ = RESET(env, KEY)
	turn = jnp.where(jnp.arange(BATCH) < 2, TURN_LIMIT + 1, es.state.turn)
	es = es._replace(state=es.state._replace(turn=turn))
	es, timestep = STEP(env, es, ZEROS)
	np.testing.assert_array_equal(np.asarray(timestep.terminated)[:3], [True, True, False])
	np.testing.assert_array_equal(np.asarray(timestep.info.winner)[:3], [TIE, TIE, ONGOING])
	assert not np.asarray(timestep.reward).any() and not np.asarray(timestep.truncated).any()
	assert int(timestep.info.turns[0]) > TURN_LIMIT


def test_keep_final_returns_the_observation_before_the_reset(env, full_env):
	assert STEP(env, RESET(env, KEY)[0], ZEROS)[1].final_obs is None
	assert RESET(env, KEY)[1].final_obs is None
	es, _ = RESET(full_env, KEY)
	winners = np.full(BATCH, ONGOING)
	winners[1] = SIDE0
	winners[2] = TIE
	crafted = with_winner(es, winners)
	stepped, _ = reference(full_env, crafted, ZEROS)
	after, timestep = STEP(full_env, crafted, ZEROS)
	for side in (0, 1):
		final = jax.tree.map(lambda x, side=side: x[:, side], timestep.final_obs)
		new = jax.tree.map(lambda x, side=side: x[:, side], timestep.obs)
		assert_equal(final, OBSERVE_SIDE(stepped, side, full_env.tables))
		assert_equal(new, OBSERVE_SIDE(after.state, side, full_env.tables))
		# A finished battle's final observation has no request: nothing to choose. The one
		# returned with it is the new battle's first decision.
		np.testing.assert_array_equal(np.asarray(final.request)[[1, 2]], [[1, 0, 0]] * 2)
		np.testing.assert_array_equal(np.asarray(new.request)[[1, 2]], [[0, 1, 0]] * 2)
		np.testing.assert_array_equal(np.asarray(final.legal_actions)[1], [True] + [False] * 8)
		# Battles that did not end are the same in both.
		for a, b in zip(jax.tree.leaves(final), jax.tree.leaves(new), strict=True):
			np.testing.assert_array_equal(np.asarray(a)[3:], np.asarray(b)[3:])
	np.testing.assert_array_equal(timestep.legal, timestep.obs.legal_actions)


def test_the_same_key_and_actions_give_the_same_trajectory(env):
	def run(seed):
		es, _ = RESET(env, jax.random.key(seed))
		key = jax.random.key(9)
		out = []
		for _ in range(70):
			key, action_key = jax.random.split(key)
			es, timestep = STEP(env, es, RANDOM_ACTIONS(env, es, action_key))
			out.append((es, timestep))
		return out

	first, second, other = run(1), run(1), run(2)
	for a, b in zip(first, second, strict=True):
		assert_equal(a, b)
	assert not np.array_equal(first[-1][0].state.species, other[-1][0].state.species)


def test_the_engine_mask_and_the_observed_mask(env, tables):
	es, timestep = RESET(env, KEY)
	np.testing.assert_array_equal(
		np.asarray(ENGINE_MASK(env, es)),
		np.asarray(jax.vmap(legal_actions, in_axes=(0, None))(es.state, env.tables)),
	)
	np.testing.assert_array_equal(
		np.asarray(timestep.legal),
		np.asarray(jax.vmap(observed_actions, in_axes=(0, None))(es.state, env.tables)),
	)
	# A hidden trap: the engine's mask has no switch, the observed one still offers it.
	es = env.init(trapped_state(env, tables), KEY)
	engine, seen = np.asarray(ENGINE_MASK(env, es)), np.asarray(env.observed_mask(es))
	assert not engine[:, 0, 4:].any() and seen[:, 0, 4:].any(axis=1).all()
	np.testing.assert_array_equal(engine[:, 1], seen[:, 1])


def test_a_hidden_trap_refuses_the_switch_and_keeps_the_other_action(env, tables):
	state = trapped_state(env, tables)
	es = env.init(state, KEY)
	refused, timestep = STEP(env, es, jnp.tile(jnp.array([[4, 0]]), (BATCH, 1)))
	info = timestep.info
	# The battle did not advance; only the refused side's flag changed.
	np.testing.assert_array_equal(np.asarray(info.rejected), [[True, False]] * BATCH)
	assert not np.asarray(info.illegal).any() and not np.asarray(info.ended).any()
	assert not np.asarray(timestep.reward).any()
	assert not np.asarray(timestep.terminated | timestep.truncated).any()
	assert_equal(
		refused.state, state._replace(trap_known=jnp.tile(jnp.array([True, False]), (BATCH, 1)))
	)
	np.testing.assert_array_equal(np.asarray(refused.steps), 1)
	# That side's observed mask has no switches now; the other side's mask is as before.
	seen = np.asarray(timestep.legal)
	assert not seen[:, 0, 4:].any() and seen[:, 0, :4].any(axis=1).all()
	np.testing.assert_array_equal(seen[:, 1], np.asarray(env.observed_mask(es))[:, 1])
	# The other side's action is locked in: it is played at the next step whatever is passed.
	np.testing.assert_array_equal(np.asarray(refused.has_held), [[False, True]] * BATCH)
	stepped, _ = reference(env, refused, ZEROS)
	after, timestep = STEP(env, refused, jnp.tile(jnp.array([[0, 4]]), (BATCH, 1)))
	assert not np.asarray(timestep.info.illegal).any() and not np.asarray(timestep.info.ended).any()
	assert_equal(after.state, stepped)
	assert not np.asarray(after.has_held).any()
	np.testing.assert_array_equal(np.asarray(after.steps), 2)


def test_a_known_trap_stays_until_either_active_changes(env, tables):
	state = trapped_state(env, tables)
	refused, _ = STEP(env, env.init(state, KEY), jnp.tile(jnp.array([[4, 0]]), (BATCH, 1)))
	known = np.tile([True, False], (BATCH, 1))
	# Both sides play moves; nobody leaves: the trap is still known, the mask has no switches.
	es, timestep = STEP(env, refused, ZEROS)
	np.testing.assert_array_equal(np.asarray(es.state.trap_known), known)
	assert not np.asarray(timestep.legal)[:, 0, 4:].any()
	assert not np.asarray(ENGINE_MASK(env, es))[:, 0, 4:].any()
	# Another switch is now an illegal action: counted, and the first legal action is played.
	stepped, _ = reference(env, es, ZEROS)
	after, timestep = STEP(env, es, jnp.tile(jnp.array([[5, 0]]), (BATCH, 1)))
	np.testing.assert_array_equal(np.asarray(timestep.info.illegal), [[True, False]] * BATCH)
	assert not np.asarray(timestep.info.rejected).any()
	assert_equal(after.state, stepped)
	# The trapper switches out: the knowledge ends, and the observed mask offers the switch.
	es, timestep = STEP(env, es, jnp.tile(jnp.array([[0, 4]]), (BATCH, 1)))
	assert not np.asarray(es.state.trap_known).any()
	np.testing.assert_array_equal(np.asarray(es.state.active)[:, 1], 1)
	assert np.asarray(timestep.legal)[:, 0, 4:].any(axis=1).all()


def test_both_sides_can_be_refused_at_once(env, tables):
	state = NEW_BATTLES(env, KEY)
	state = set_active(state, "ability", 1, tables.abilities.index["magnetpull"])
	state = set_active(state, "ability", 0, tables.abilities.index["arenatrap"])
	steel, normal = tables.types.index["steel"], tables.types.index["normal"]
	state = set_active(state, "types", 0, jnp.array([steel, -1], jnp.int8))
	state = set_active(state, "types", 1, jnp.array([normal, -1], jnp.int8))
	es = env.init(state, KEY)
	assert not np.asarray(ENGINE_MASK(env, es))[:, :, 4:].any()
	refused, timestep = STEP(env, es, jnp.full((BATCH, 2), 4))
	np.testing.assert_array_equal(np.asarray(timestep.info.rejected), True)
	np.testing.assert_array_equal(np.asarray(refused.state.trap_known), True)
	assert not np.asarray(refused.has_held).any()
	assert_equal(refused.state._replace(trap_known=state.trap_known), state)


def test_a_switch_to_a_fainted_member_is_an_illegal_action_not_a_refusal(env, tables):
	# Side 0's first bench member is fainted: switching to it fails whatever the trap says.
	state = trapped_state(env, tables)
	state = state._replace(hp=state.hp.at[:, 0, 1].set(0))
	es = env.init(state, KEY)
	stepped, _ = reference(env, es, ZEROS)
	after, timestep = STEP(env, es, jnp.tile(jnp.array([[4, 0]]), (BATCH, 1)))
	np.testing.assert_array_equal(np.asarray(timestep.info.illegal), [[True, False]] * BATCH)
	assert not np.asarray(timestep.info.rejected).any()
	assert_equal(after.state, stepped)
	assert not np.asarray(after.state.trap_known).any()


def test_illegal_actions_are_counted_and_replaced_by_the_first_legal_one(env, tables):
	state = NEW_BATTLES(env, KEY)
	# Battles 1..3: the active has no fourth move, so action 3 is illegal there.
	for side in (0, 1):
		rows = jnp.arange(1, 4)
		state = state._replace(moves=state.moves.at[rows, side, state.active[rows, side], 3].set(0))
	es = env.init(state, KEY)
	actions = np.zeros((BATCH, 2), np.int32)
	actions[0] = [99, -3]
	actions[1:4, 0] = 3
	actions[4] = [8, 8]  # the 5th bench member: a legal switch at the start
	mask = np.asarray(ENGINE_MASK(env, es))
	clipped = np.clip(actions, 0, 8)
	rows = np.arange(BATCH)[:, None]
	legal = (actions >= 0) & (actions < 9) & mask[rows, np.arange(2)[None, :], clipped]
	played = np.where(legal, actions, np.argmax(mask, axis=-1))
	assert (~legal).sum() >= 5 and legal[4].all()
	stepped, fresh = reference(env, es, jnp.asarray(played))
	after, timestep = STEP(env, es, jnp.asarray(actions))
	np.testing.assert_array_equal(np.asarray(timestep.info.illegal), ~legal)
	assert not np.asarray(timestep.info.rejected).any()
	ended = jnp.asarray(timestep.info.ended)
	assert_equal(after.state, select(ended, fresh, stepped))


def test_invalid_arguments_are_refused(env):
	with pytest.raises(ValueError, match="keep_final"):
		EnvConfig(observe="none", keep_final=True)
	with pytest.raises(ValueError, match="observe"):
		EnvConfig(observe="own")
	with pytest.raises(ValueError, match="positive"):
		EnvConfig(max_steps=0)
	es, _ = RESET(env, KEY)
	with pytest.raises(ValueError, match="shape"):
		env.step(es, jnp.zeros((BATCH,), jnp.int32))


def test_the_environment_jits_with_closed_over_constants_and_vmaps(env):
	es, _ = RESET(env, KEY)
	actions = RANDOM_ACTIONS(env, es, jax.random.key(4))
	# vmap over a leading axis of environments, jitted with the environment closed over as
	# constants: each row is what the single call (the environment as an argument) gives.
	stacked = jax.jit(jax.vmap(lambda key: env.reset(key)[0]))(jax.random.split(KEY, 2))
	rows = [jax.tree.map(lambda x, i=i: x[i], stacked) for i in range(2)]
	actions = jnp.stack(
		[RANDOM_ACTIONS(env, row, jax.random.key(20 + i)) for i, row in enumerate(rows)]
	)
	batched = jax.jit(jax.vmap(env.step))(stacked, actions)
	for i, row in enumerate(rows):
		assert_equal(jax.tree.map(lambda x, i=i: x[i], batched), STEP(env, row, actions[i]))


def test_the_wrapper_assigns_a_random_side_and_shows_the_learner_perspective(wrapper):
	ws, timestep = WRAPPER_RESET(wrapper, KEY)
	learner = np.asarray(ws.learner)
	assert learner.shape == (BATCH,) and set(learner.tolist()) == {0, 1}
	state = ws.env.state
	assert timestep.reward.shape == (BATCH,) and timestep.terminated.shape == (BATCH,)
	assert timestep.legal.shape == (BATCH, 9) and timestep.info.winner.shape == (BATCH,)
	expected = OBSERVE_LEARNER(state, ws.learner, wrapper.env.tables)
	assert_equal(timestep.obs, expected)
	np.testing.assert_array_equal(
		np.asarray(timestep.obs.own.species)[np.arange(BATCH)],
		np.asarray(state.species)[np.arange(BATCH), learner],
	)
	np.testing.assert_array_equal(
		np.asarray(timestep.legal), np.asarray(timestep.obs.legal_actions)
	)
	# Play: the learner picks from its mask, the default opponent is always legal.
	key = jax.random.key(1)
	ended_total = 0
	for _ in range(110):
		key, action_key = jax.random.split(key)
		action = jax.random.categorical(action_key, jnp.where(timestep.legal, 0.0, -jnp.inf))
		ws, timestep = WRAPPER_STEP(wrapper, ws, action)
		info = timestep.info
		assert isinstance(info, LearnerInfo)
		assert not np.asarray(info.illegal).any() and not np.asarray(info.opponent_illegal).any()
		# The reward is the learner's side of the zero-sum pair; info names that side.
		ended = np.asarray(info.ended)
		winner, side = np.asarray(info.winner), np.asarray(info.learner)
		sign = np.where(winner == TIE, 0, np.where(winner == side, 1, -1))
		np.testing.assert_array_equal(np.asarray(timestep.reward)[ended], sign[ended])
		assert not np.asarray(timestep.reward)[~ended].any()
		expected = OBSERVE_LEARNER(ws.env.state, ws.learner, wrapper.env.tables)
		assert_equal(timestep.obs, expected)
		ended_total += int(ended.sum())
	assert ended_total > 0


def test_the_wrapper_redraws_the_side_at_each_reset_and_plays_the_opponent_policy(
	wrapper, env, tables
):
	state = trapped_state(env, tables, "pressure")  # nothing traps: both abilities are Pressure
	ws, _ = WRAPPER_RESET(wrapper, KEY)
	ws = OpponentState(wrapper.env.init(state, KEY), ws.learner)
	learner = np.asarray(ws.learner)
	# The learner switches (4); the opponent plays what its policy picks with its own key.
	key, opponent_key, _ = jax.random.split(ws.env.key, 3)
	opponent = jax.vmap(wrapper.opponent)(
		ws.env.state, 1 - ws.learner, jax.random.split(opponent_key, BATCH)
	)
	played = jnp.where(jnp.arange(2)[None, :] == ws.learner[:, None], 4, opponent[:, None])
	stepped, _ = reference(wrapper.env, ws.env._replace(key=key), played)
	moved, timestep = WRAPPER_STEP(wrapper, ws, jnp.full(BATCH, 4))
	assert not np.asarray(timestep.info.ended).any()
	assert_equal(moved.env.state, stepped)
	assert not np.asarray(timestep.info.illegal).any()
	# A finished battle is replaced and its learner side is drawn again.
	crafted = OpponentState(with_winner(ws.env, np.full(BATCH, SIDE0)), ws.learner)
	after, timestep = WRAPPER_STEP(wrapper, crafted, jnp.zeros(BATCH, jnp.int32))
	assert np.asarray(timestep.info.ended).all()
	np.testing.assert_array_equal(np.asarray(timestep.reward), np.where(learner == 0, 1, -1))
	np.testing.assert_array_equal(np.asarray(timestep.info.learner), learner)
	redrawn = np.asarray(after.learner)
	assert (redrawn != learner).any() and set(redrawn.tolist()) == {0, 1}


def test_the_default_opponent_is_random_over_the_engine_mask(env, tables):
	state = jax.tree.map(lambda x: x[0], trapped_state(env, tables))
	pick = jax.jit(jax.vmap(RandomPolicy(env.tables), in_axes=(None, None, 0)))
	picks = np.asarray(pick(state, 0, jax.random.split(KEY, 64)))
	# Side 0 is trapped by a hidden Shadow Tag: the engine's mask has no switch to pick.
	assert set(picks.tolist()) <= {0, 1, 2, 3} and len(set(picks.tolist())) > 1
	free = np.asarray(pick(state, 1, jax.random.split(KEY, 64)))
	assert (free >= 4).any() and (free < 4).any()


@pytest.mark.parametrize("name", POLICY_NAMES)
def test_a_baseline_opponent_only_picks_what_the_engine_mask_allows(env, tables, name):
	"""Side 0 is trapped by a hidden Shadow Tag: no baseline picks a switch the engine refuses."""
	state = jax.tree.map(lambda x: x[0], trapped_state(env, tables))
	mask = np.asarray(legal_actions(state, env.tables))
	assert not mask[0][4:].any() and mask[1][4:].any()
	pick = jax.jit(jax.vmap(make_policy(name, env.tables, tables), in_axes=(None, None, 0)))
	for side in (0, 1):
		picks = np.asarray(pick(state, side, jax.random.split(KEY, 64)))
		assert mask[side][picks].all(), (name, side)


def test_the_wrapper_plays_a_baseline_opponent_without_illegal_actions(env, tables):
	wrapper = OpponentEnv(env, make_policy("heuristics", env.tables, tables))
	ws, timestep = WRAPPER_RESET(wrapper, KEY)
	key = jax.random.key(3)
	ended_total = 0
	for _ in range(100):
		key, action_key = jax.random.split(key)
		action = jax.random.categorical(action_key, jnp.where(timestep.legal, 0.0, -jnp.inf))
		ws, timestep = WRAPPER_STEP(wrapper, ws, action)
		assert not np.asarray(timestep.info.illegal).any()
		assert not np.asarray(timestep.info.opponent_illegal).any()
		ended_total += int(np.asarray(timestep.info.ended).sum())
	assert ended_total > 0


def test_a_trapped_learner_is_refused_and_its_opponent_action_is_kept(wrapper, env, tables):
	state = trapped_state(env, tables)
	ws, _ = WRAPPER_RESET(wrapper, KEY)
	ws = OpponentState(wrapper.env.init(state, KEY), jnp.zeros(BATCH, jnp.int32))  # side 0, trapped
	ws, timestep = WRAPPER_STEP(wrapper, ws, jnp.full(BATCH, 4))
	np.testing.assert_array_equal(np.asarray(timestep.info.rejected), True)
	assert not np.asarray(timestep.info.illegal).any() and not np.asarray(timestep.info.ended).any()
	assert not np.asarray(timestep.legal)[:, 4:].any() and not np.asarray(timestep.reward).any()
	np.testing.assert_array_equal(np.asarray(ws.env.state.trap_known), [[True, False]] * BATCH)
	np.testing.assert_array_equal(np.asarray(ws.env.has_held), [[False, True]] * BATCH)
	# The learner moves on; the opponent's action from the refused step is the one played.
	played = jnp.stack((jnp.zeros(BATCH, jnp.int32), ws.env.held[:, 1]), axis=1)
	advance_key = jax.random.split(ws.env.key, 3)[0]
	stepped, _ = reference(wrapper.env, ws.env._replace(key=advance_key), played)
	after, timestep = WRAPPER_STEP(wrapper, ws, jnp.zeros(BATCH, jnp.int32))
	assert not np.asarray(timestep.info.ended).any()
	assert_equal(after.env.state, stepped)
