"""The league: the row groups and lanes, lane observations, opponents and the steps that count.

The zero-sum and weight checks read one rollout of the shared trainer (conftest.league_trainer),
in which battles end.
"""

import jax
import numpy as np
import pytest

from pokejax.baselines import make_policy
from pokejax.engine.actions import legal_actions
from pokejax.engine.observation import observe
from pokejax.train.league import Layout, take

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


def test_refused_and_held_steps_get_no_weight(league_trainer, rollout):
	rows = league_trainer.layout.lane_rows
	seats = rollout.seat[..., None]  # [T, L, 1]
	refused_by_seat = rollout.rejected[:, rows]  # [T, L, 2]
	refused = np.take_along_axis(refused_by_seat, seats, axis=-1)[..., 0]
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
	np.testing.assert_array_equal(rollout.weight, ~(refused | held))


def test_baseline_rows_play_their_policies_and_pool_slices_their_members(league_trainer):
	layout, league = league_trainer.layout, league_trainer.league
	learner = league_trainer.init()
	# Make the members differ, so that a slice given to the wrong member is caught.
	scale = 1.0 + 0.5 * np.arange(league_trainer.config.pool.size, dtype=np.float32)
	pool = jax.tree.map(lambda x: x * scale.reshape((-1,) + (1,) * (x.ndim - 1)), learner.pool)
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
	_, logits = league.pool_forward(pool, observed, state.pool_hidden)
	per_member = layout.pool // league_trainer.config.pool.size
	for member in range(league_trainer.config.pool.size):
		rows = slice(member * per_member, (member + 1) * per_member)
		params = take(pool, member)
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
