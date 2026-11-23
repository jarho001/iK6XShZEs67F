"""The league: the row groups and lanes, lane observations, opponents and the steps that count.

The zero-sum and weight checks read one rollout of the shared trainer (conftest.league_trainer),
in which battles end.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.baselines import make_policy
from pokejax.engine.actions import action_masks, legal_actions
from pokejax.engine.observation import observe
from pokejax.train.league import Layout, PoolParams, take

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


@pytest.fixture(scope="module")
def rollout(league_trainer, cpu_only):
	"""One rollout of the shared trainer from fresh battles, on the host."""
	learner = league_trainer.init()
	_, traj = league_trainer.rollout(learner.params, learner.pool, learner.league)
	return jax.device_get(traj)


def test_layout_orders_the_groups_and_lays_out_the_lanes():
	layout = Layout({"random": 2, "max-damage": 2, "pool": 4, "selfplay": 4})
	assert layout.spans == {
		"selfplay": (0, 4),
		"pool": (4, 8),
		"random": (8, 10),
		"max-damage": (10, 12),
	}
	assert layout.batch == 12 and layout.selfplay == 4 and layout.pool == 4
	assert layout.baselines == (("random", 8, 10), ("max-damage", 10, 12))
	# Lanes 0 to 3 are seat 0 of the selfplay rows, 4 to 7 their seat 1, then one per other row.
	assert layout.lanes == 16
	assert layout.lane_rows.tolist() == [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
	assert layout.selfplay_seats.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
	assert layout.pool_rows.tolist() == [4, 5, 6, 7]
	assert not layout.identity_lanes
	learner = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0], np.int32)
	seats = np.asarray(layout.lane_seats(learner)).tolist()
	assert seats[:8] == [0, 0, 0, 0, 1, 1, 1, 1]
	assert seats[8:] == learner[4:].tolist()


def test_a_layout_without_selfplay_has_one_lane_per_row():
	layout = Layout({"heuristics": 4, "random": 4})
	assert layout.identity_lanes and layout.lanes == 8
	assert layout.lane_rows.tolist() == list(range(8))
	assert np.asarray(layout.lane_seats(np.arange(8) % 2)).tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
	assert layout.pool_rows.size == 0


def test_unknown_groups_and_empty_layouts_are_errors():
	with pytest.raises(ValueError, match="unknown group"):
		Layout({"oracle": 2})
	with pytest.raises(ValueError, match="at least one row"):
		Layout({"random": 0})


def test_lane_observations_are_the_views_of_their_seats(league_trainer):
	state = league_trainer.init().league
	layout = league_trainer.layout
	seats = np.asarray(layout.lane_seats(state.learner))
	for lane in range(layout.lanes):
		battle = take(state.env.state, int(layout.lane_rows[lane]))
		expected = observe(battle, int(seats[lane]), league_trainer.env.tables)
		for want, got in zip(jax.tree.leaves(expected), jax.tree.leaves(take(state.obs, lane))):
			np.testing.assert_array_equal(np.asarray(want), np.asarray(got))


def test_the_two_seats_of_a_selfplay_row_are_opposite_and_zero_sum(league_trainer, rollout):
	selfplay = league_trainer.layout.selfplay
	terminal = rollout.terminated[:, :selfplay]
	assert terminal.sum() > 0, "the rollout must end some battles for this check to mean anything"
	first = rollout.reward[:, :selfplay]
	second = rollout.reward[:, selfplay : 2 * selfplay]
	np.testing.assert_array_equal(first[terminal], -second[terminal])
	assert np.abs(first[terminal]).max() == 1.0


def test_the_two_seats_of_a_selfplay_row_keep_separate_recurrent_states(league_trainer, rollout):
	selfplay = league_trainer.layout.selfplay
	# After one step each seat has seen its own view, so the carries of a row differ.
	own = rollout.hidden[1, :selfplay]
	other = rollout.hidden[1, selfplay : 2 * selfplay]
	assert (np.abs(own - other).max(axis=-1) > 1e-6).all()


def test_held_and_substituted_steps_get_no_weight_and_refused_ones_count(league_trainer, rollout):
	rows = league_trainer.layout.lane_rows
	seats = rollout.seat[..., None]  # [T, L, 1]
	refused_by_seat = rollout.rejected[:, rows]  # [T, L, 2]
	refused = np.take_along_axis(refused_by_seat, seats, axis=-1)[..., 0]
	illegal_by_seat = rollout.illegal[:, rows]
	illegal = np.take_along_axis(illegal_by_seat, seats, axis=-1)[..., 0]
	# A step is held when the other side was refused at the step before and this one was not.
	before = np.concatenate([np.zeros_like(rollout.rejected[:1]), rollout.rejected[:-1]])
	ended = rollout.terminated | rollout.truncated
	ended_before = np.concatenate([np.zeros_like(ended[:1]), ended[:-1]])
	before_lane = before[:, rows]
	other_refused = (
		before_lane.any(axis=-1) & ~np.take_along_axis(before_lane, seats, axis=-1)[..., 0]
	)
	held = other_refused & ~ended_before[:, rows]
	assert (refused | held).any(), "the rollout must refuse or hold some steps"
	# A refused action is a decision of its own; a held step's sample and a substitute do not count.
	np.testing.assert_array_equal(rollout.weight, ~(held | illegal))
	assert rollout.weight[refused].all()
	np.testing.assert_array_equal(rollout.advance, ~held)
	real_choice = np.asarray(rollout.obs.legal_actions).sum(axis=-1) >= 2
	np.testing.assert_array_equal(rollout.actor, rollout.weight & real_choice)


def test_baseline_rows_play_their_policies_and_pool_slices_their_members(league_trainer):
	layout, league = league_trainer.layout, league_trainer.league
	learner = league_trainer.init()
	# Make the members differ, so that a slice given to the wrong member is caught.
	scale = 1.0 + 0.5 * np.arange(league_trainer.config.pool.size, dtype=np.float32)
	scaled = jax.tree.map(
		lambda x: x * scale.reshape((-1,) + (1,) * (x.ndim - 1)), learner.pool.current
	)
	pool = PoolParams(current=scaled, previous=scaled)
	state = learner.league
	key = jax.random.key(3)
	opponents, _ = league.opponent_actions(pool, state, key)
	opponents = np.asarray(opponents)

	start, stop = layout.spans["max-damage"]
	policy = make_policy("max-damage", league_trainer.env.tables, league_trainer.tables)
	battles = take(state.env.state, slice(start, stop))
	expected = jax.vmap(policy)(
		battles, 1 - state.learner[start:stop], jax.random.split(key, 8)[start:stop]
	)
	offset = start - layout.selfplay
	np.testing.assert_array_equal(opponents[offset : offset + stop - start], np.asarray(expected))

	start, stop = layout.spans["random"]
	offset = start - layout.selfplay
	for row in range(start, stop):
		battle = take(state.env.state, row)
		side = int(1 - state.learner[row])
		legal = np.asarray(legal_actions(battle, league_trainer.env.tables))[side]
		assert legal[opponents[offset + row - start]]

	observed = league.observe_pool(state.env.state, state.learner)
	_, logits = league.pool_forward(pool, observed, state.pool_hidden, state.pool_fresh)
	per_member = layout.pool // league_trainer.config.pool.size
	for member in range(league_trainer.config.pool.size):
		rows = slice(member * per_member, (member + 1) * per_member)
		params = take(pool.current, member)
		_, direct, _ = league_trainer.model.apply(
			{"params": params}, take(observed, rows), take(state.pool_hidden, rows)
		)
		np.testing.assert_allclose(np.asarray(logits[rows]), np.asarray(direct), atol=1e-5)


def test_reset_takes_the_given_learner_sides(league_trainer):
	learner = np.arange(8, dtype=np.int32) % 2
	state = league_trainer.league.reset(jax.random.key(2), learner)
	assert np.asarray(state.learner).tolist() == learner.tolist()
	# One observation per lane, whatever the learner sides are.
	assert state.obs.own.species.shape[0] == league_trainer.layout.lanes


# The held lane and the refused seat, and the pool generations (the review's findings 8 and 9).

KEY = jax.random.key(29)


def test_a_held_lane_and_a_held_pool_row_keep_their_carry_and_do_not_count(league_trainer):
	"""The held seat of row 0 (lane 4) and the held pool row's opponent do not advance: their carries
	stay, and their decisions get no weight; the other lanes advance and count."""
	league = league_trainer.league
	layout = league.layout
	state = league.reset(KEY, jnp.zeros((layout.batch,), jnp.int32))
	env = state.env._replace(
		has_held=state.env.has_held.at[0, 1].set(True).at[layout.pool_rows[0], 1].set(True),
		held=state.env.held.at[0, 1].set(0).at[layout.pool_rows[0], 1].set(0),
	)
	state = state._replace(env=env)
	learner = league_trainer.init()
	next_state, transition = league.play(learner.params, learner.pool, state)
	lane = 4  # row 0's seat 1: the selfplay lanes of seat 1 follow the seat 0 lanes
	assert int(layout.lane_rows[lane]) == 0 and int(layout.lane_seats(state.learner)[lane]) == 1
	np.testing.assert_array_equal(
		np.asarray(next_state.hidden[lane]), np.asarray(state.hidden[lane])
	)
	assert not np.asarray(transition.advance)[lane]
	assert not np.asarray(transition.weight)[lane] and not np.asarray(transition.actor)[lane]
	np.testing.assert_array_equal(
		np.asarray(next_state.pool_hidden[0]), np.asarray(state.pool_hidden[0])
	)
	# The lane that was not held advanced, and its decision counts.
	assert np.asarray(transition.advance)[0] and np.asarray(transition.weight)[0]
	assert not np.array_equal(np.asarray(next_state.hidden[0]), np.asarray(state.hidden[0]))
	assert np.asarray(next_state.pool_fresh).shape == (layout.pool,)


def arena_trapped(league, state, tables):
	"""Row 0's side 1 active is an Arena Trap user; side 0's active is a grounded Normal type."""
	battles = state.env.state
	normal = tables.types.index["normal"]
	battles = battles._replace(
		ability=battles.ability.at[0, 1, battles.active[0, 1]]
		.set(tables.abilities.index["arenatrap"])
		.at[0, 0, battles.active[0, 0]]
		.set(tables.abilities.index["noability"]),
		types=battles.types.at[0, 0, battles.active[0, 0]].set(
			jnp.array([normal, -1], battles.types.dtype)
		),
	)
	base, blocked = jax.vmap(action_masks, in_axes=(0, None))(battles, league.tables)
	env = state.env._replace(state=battles, base=base, blocked=blocked)
	return state._replace(env=env, obs=league.observe_lanes(battles, state.learner))


def test_a_refused_switch_counts_and_its_retry_is_a_decision_of_its_own(league_trainer, tables):
	"""Side 0 of row 0 switches under Arena Trap and is refused: its refused action counts, the
	battle does not advance, and the other side's action is held. At the retry the refused seat's
	decision counts again, and the held seat's decision does not (its action was played before)."""
	league = league_trainer.league
	layout = league.layout
	state = arena_trapped(league, league.reset(KEY, jnp.zeros((layout.batch,), jnp.int32)), tables)
	lanes = np.asarray(layout.lane_rows)
	zeros = jnp.zeros((layout.batch, 2), jnp.int32)
	refused_step = zeros.at[0].set(jnp.array([4, 0]))
	after = state.hidden + 1.0
	pool_after = state.pool_hidden + 1.0
	next1, out1 = league._advance(state, refused_step, after, pool_after)
	assert np.asarray(out1["rejected"])[0].tolist() == [True, False]
	assert np.asarray(next1.env.has_held)[0].tolist() == [False, True]
	assert np.asarray(out1["advance"])[0] and np.asarray(out1["weight"])[0]
	assert np.asarray(out1["advance"])[4] and np.asarray(out1["weight"])[4]
	np.testing.assert_allclose(np.asarray(next1.hidden[0]), np.asarray(after[0]))
	# The retry: side 0 moves; side 1's new sample is replaced by its held action.
	retry = zeros.at[0].set(jnp.array([0, 1]))
	kept = next1.hidden + 2.0
	kept = kept.at[4].set(next1.hidden[4])  # the held lane's carry is kept by the caller (play)
	next2, out2 = league._advance(next1, retry, kept, next1.pool_hidden + 2.0)
	assert np.asarray(out2["advance"])[0] and np.asarray(out2["weight"])[0]
	assert not np.asarray(out2["advance"])[4] and not np.asarray(out2["weight"])[4]
	assert not np.asarray(out2["actor"])[4]
	np.testing.assert_array_equal(np.asarray(next2.hidden[4]), np.asarray(next1.hidden[4]))
	assert not np.asarray(out2["rejected"]).any()
	assert lanes[0] == 0


def test_an_insertion_leaves_battles_in_progress_on_their_parameters(league_trainer):
	trainer = league_trainer
	league = trainer.league
	learner = trainer.init()
	pool = learner.pool
	layout = league.layout
	members = trainer.config.pool.size
	rows = layout.pool
	per_member = rows // members
	new = jax.tree.map(lambda x: x + 1.0, learner.params)
	fresh = jnp.ones((rows,), jnp.bool_)
	inserted, fresh_after, count = trainer._insert(
		pool, fresh, jnp.int32(0), new, trainer.config.pool.every
	)
	assert int(count) == 1
	# Member 0 keeps its old parameters as previous, and takes the new ones as current.
	for old, previous in zip(jax.tree.leaves(pool.current), jax.tree.leaves(inserted.previous)):
		np.testing.assert_array_equal(np.asarray(old)[0], np.asarray(previous)[0])
		np.testing.assert_array_equal(np.asarray(old)[1:], np.asarray(previous)[1:])
	for value, current in zip(jax.tree.leaves(new), jax.tree.leaves(inserted.current)):
		np.testing.assert_array_equal(np.asarray(value), np.asarray(current)[0])
	# Its rows began on the old parameters: they now read previous; member 1's rows are untouched.
	np.testing.assert_array_equal(np.asarray(fresh_after), np.arange(rows) >= per_member)
	state = league.reset(KEY)
	obs = league.observe_pool(state.env.state, state.learner)
	hidden = state.pool_hidden
	mixed = league.pool_forward(inserted, obs, hidden, fresh_after)[1]
	now = league.pool_forward(inserted, obs, hidden, jnp.ones((rows,), jnp.bool_))[1]
	before = league.pool_forward(inserted, obs, hidden, jnp.zeros((rows,), jnp.bool_))[1]
	np.testing.assert_allclose(np.asarray(mixed)[:per_member], np.asarray(before)[:per_member])
	np.testing.assert_allclose(np.asarray(mixed)[per_member:], np.asarray(now)[per_member:])
	assert not np.allclose(np.asarray(now)[:per_member], np.asarray(before)[:per_member])


def test_evaluation_refuses_a_count_that_is_not_whole_batches(league_trainer):
	learner = league_trainer.init()
	with pytest.raises(ValueError, match="multiple of the evaluation batch"):
		league_trainer.evaluate(learner.params, "random", 5)
