"""Self-play with the pool: the pool's rotation, the learner's steps, the groups' statistics and evaluation."""

import jax
import numpy as np
import pytest

from pokejax.train.league import take

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


def same(first, second):
	return all(
		np.array_equal(np.asarray(a), np.asarray(b))
		for a, b in zip(jax.tree.leaves(first), jax.tree.leaves(second))
	)


def test_the_pool_starts_as_copies_and_each_insertion_replaces_the_oldest(league_trainer):
	learner = league_trainer.init()
	assert same(take(learner.pool, 0), learner.params)
	assert same(take(learner.pool, 1), learner.params)
	assert int(learner.pool_inserts) == 0
	parameters = []
	# With pool.every = 1 and two members, the insertions go to slots 0, 1, 0.
	for slot in (0, 1, 0):
		before = learner.pool
		learner, _ = league_trainer.update(learner)
		assert same(take(learner.pool, slot), learner.params)
		assert same(take(learner.pool, 1 - slot), take(before, 1 - slot))
		parameters.append(learner.params)
	assert int(learner.pool_inserts) == 3 and int(learner.update) == 3
	# The first insertion was replaced: slot 0 holds the third update's parameters.
	assert same(take(learner.pool, 0), parameters[2])
	assert not same(take(learner.pool, 0), parameters[0])


def test_the_learners_steps_do_not_depend_on_the_opponents_outputs(league_trainer):
	"""Changing only the pool (the opponents' parameters) leaves every learner-lane output alone.

	The first step of a rollout is compared: later steps may differ, because the opponents' moves
	change the battles.
	"""
	learner = league_trainer.init()
	changed = jax.tree.map(lambda x: x * 4.0 + 0.5, learner.pool)
	observed = league_trainer.league.observe_pool(learner.league.env.state, learner.league.learner)
	_, logits = league_trainer.league.pool_forward(
		learner.pool, observed, learner.league.pool_hidden
	)
	_, changed_logits = league_trainer.league.pool_forward(
		changed, observed, learner.league.pool_hidden
	)
	assert not np.allclose(np.asarray(logits), np.asarray(changed_logits))

	_, before = league_trainer.rollout(learner.params, learner.pool, learner.league)
	_, after = league_trainer.rollout(learner.params, changed, learner.league)
	for name in ("obs", "hidden", "after"):
		for want, got in zip(
			jax.tree.leaves(take(getattr(before, name), 0)),
			jax.tree.leaves(take(getattr(after, name), 0)),
		):
			np.testing.assert_array_equal(np.asarray(want), np.asarray(got))
	for name in ("action", "log_prob", "value"):
		np.testing.assert_array_equal(
			np.asarray(getattr(before, name)[0]), np.asarray(getattr(after, name)[0])
		)


def test_the_update_reports_every_group_and_its_battles(league_trainer):
	_, metrics = league_trainer.update(league_trainer.init())
	metrics = {name: float(value) for name, value in jax.device_get(metrics).items()}
	assert all(np.isfinite(value) for value in metrics.values())
	groups = league_trainer.config.groups
	assert groups == {"selfplay": 4, "pool": 2, "random": 1, "max-damage": 1}
	for name in groups:
		assert 0.0 <= metrics[f"win_rate_{name}"] <= 1.0
	completed = sum(metrics[f"completed_{name}"] for name in groups)
	assert completed == metrics["completed_battles"]


def test_evaluation_repeats_exactly_for_fixed_seeds(league_trainer):
	learner = league_trainer.init()
	first = league_trainer.evaluate_all(learner.params, 4)
	assert first == league_trainer.evaluate_all(learner.params, 4)
	assert [(r["opponent"], r["policy"]) for r in first] == [
		("max-damage", "sampled"),
		("max-damage", "greedy"),
	]
	for result in first:
		assert result["battles"] == 4 and 0 <= result["wins"] <= 4
		low, high = result["ci95"]
		assert 0.0 <= low <= result["win_rate"] <= high <= 1.0
