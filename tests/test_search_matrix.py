"""Method 2 equilibrium, request masks and player-view isolation."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_search_belief import TinyValue, assert_tree

from pokejax.engine.state import NONE
from pokejax.env import Env, EnvConfig
from pokejax.search.belief import Catalogue
from pokejax.search.method2 import MatrixConfig, Method2, _columns, _first_lanes, solve_matrix
from pokejax.search.rebuild import views
from pokejax.search.transition import init
from pokejax.teams import save_npz, team_pool_from_data

KEY = jax.random.key(79)


def solve(matrix, p0, q0, foe=0.1, weights=None):
	matrix = jnp.asarray(matrix, dtype=jnp.float32)
	if weights is None:
		weights = jnp.ones(matrix.shape[0]) / matrix.shape[0]
	return jax.jit(solve_matrix)(
		matrix, jnp.asarray(weights), jnp.log(jnp.asarray(p0)), jnp.log(jnp.asarray(q0)), 0.1, foe
	)


def test_matching_pennies_hedges_even_under_a_skewed_prior():
	m = [[[1, -1], [-1, 1]]]
	solution = solve(m, [0.9, 0.1], [[0.8, 0.2]])
	np.testing.assert_allclose(solution.p, [0.54, 0.46], atol=0.03)
	assert float(solution.gap) < 1e-5


def test_large_foe_penalty_converges_to_fixed_prior_backup_and_ranking():
	matrix = np.asarray([[[0.7, -0.8], [0.4, 0.3]], [[0.3, -0.3], [0.1, 0.6]]])
	weights = np.asarray([0.7, 0.3])
	q0 = np.asarray([[0.1, 0.9], [0.8, 0.2]])
	expected = np.einsum("w,wab,wb->a", weights, matrix, q0)
	result = solve(matrix, [0.5, 0.5], q0, foe=100_000.0, weights=weights)
	np.testing.assert_allclose(result.row_values, expected, atol=5e-6)
	np.testing.assert_allclose(result.q, q0, atol=5e-6)
	assert int(result.p.argmax()) == int(expected.argmax())
	assert float(result.gap) < 1e-6


def test_dominated_and_illegal_rows_receive_negligible_or_zero_mass():
	matrix = [[[1, 1, -100], [-1, -1, -100], [100, 100, 100]]]
	result = solve(matrix, [0.5, 0.5, 0], [[0.5, 0.5, 0]])
	assert result.p[1] < 1e-7 and result.p[2] == 0
	assert result.q[0, 2] == 0
	assert float(result.gap) < 1e-6


def test_single_row_and_column_apply_their_masks_with_zero_weight_worlds():
	result = solve(
		[[[900, -1], [900, 0.37]], [[-100, 200], [-300, 400]]],
		[0, 1],
		[[0, 1], [1, 0]],
		weights=[1, 0],
	)
	np.testing.assert_array_equal(result.p, [0, 1])
	np.testing.assert_array_equal(result.q, [[0, 1], [1, 0]])
	np.testing.assert_allclose(result.row_values, [0, 0.37])
	assert result.gap == 0


@pytest.fixture(scope="module")
def game(tables, tmp_path_factory):
	recording = json.loads(Path("tests/fixtures/reset.json").read_text())
	teams = recording["teamSets"]
	pool = team_pool_from_data(
		{
			"format": "gen5randombattle",
			"seed": 0,
			"count": len(teams),
			"showdownCommit": recording["showdownCommit"],
			"teams": teams,
		},
		tables,
	)
	path = tmp_path_factory.mktemp("matrix") / "pool.npz"
	save_npz(pool, path)
	env = Env(EnvConfig(1), path, tables=tables)
	catalogue = Catalogue.load(path, tables)
	state = jax.jit(env.new_battles)(KEY)
	position = jax.jit(jax.vmap(init, in_axes=(0, None)))(state, env.tables)
	return env.tables, catalogue, position


def test_matrix_lanes_cover_small_requests_and_keep_held_columns(game, tables):
	_, _, batched = game
	position = jax.tree.map(lambda a: a[0], batched)
	foe = 1
	head = jnp.zeros(len(tables.moves.ids) + 9)
	both = position._replace(
		base=position.base.at[foe].set(jnp.isin(jnp.arange(9), jnp.array([2, 6])))
	)
	columns, prior = _columns(both, foe, head, KEY, 2)
	assert set(np.asarray(columns)) == {2, 6}
	np.testing.assert_allclose(jnp.exp(prior).sum(), 1)

	wide = position._replace(base=position.base.at[foe].set(jnp.ones(9, bool)))
	columns, prior = _columns(wide, foe, head, KEY, 3)
	assert len(set(np.asarray(columns))) == 3
	assert set(np.asarray(columns[:2])) == {0, 1}
	assert jnp.isfinite(prior).all()

	waiting = both._replace(state=both.state._replace(request=both.state.request.at[foe].set(NONE)))
	columns, prior = _columns(waiting, foe, head, KEY, 2)
	assert int(columns[0]) == 0 and prior[0] == 0 and prior[1] == -jnp.inf

	held = both._replace(held=both.held.at[foe].set(6), has_held=both.has_held.at[foe].set(True))
	columns, prior = _columns(held, foe, head, KEY, 2)
	assert int(columns[0]) == 6 and prior[0] == 0 and prior[1] == -jnp.inf
	pairs = jnp.zeros((1, 9, 1, 2), bool).at[0, 2, 0, 0].set(True)
	lanes, ids = _first_lanes(
		2,
		2,
		jax.tree.map(lambda a: a[None, None], held),
		pairs,
		jnp.array([0]),
		jnp.ones((1, 2)),
		columns[None, None],
		jax.random.split(KEY, 2)[None, None],
	)
	np.testing.assert_array_equal(lanes.actions, [[2, 6], [2, 6]])
	np.testing.assert_array_equal(lanes.positions.held[:, 1], [6, 6])
	assert len(set(np.asarray(ids))) == 2


def test_search_ignores_real_hidden_foe_set_and_held_choice(game, tables):
	engine_tables, catalogue, position = game
	state = position.state
	twin = position._replace(
		state=state._replace(
			item=state.item.at[0, 1, 0].set(catalogue.sets.item[0]),
			moves=state.moves.at[0, 1, 1].set(catalogue.sets.moves[0]),
			turn_moves=state.turn_moves.at[0, 1].set(catalogue.sets.moves[0, 0]),
			turn_actions=state.turn_actions.at[0, 1].set(2),
		),
	)
	method = Method2(
		TinyValue(len(tables.moves.ids)), engine_tables, catalogue, MatrixConfig(worlds=4)
	)

	def query(position, held):
		position = position._replace(
			held=position.held.at[0, 1].set(held), has_held=position.has_held.at[0, 1].set(True)
		)
		view = views(position, jnp.zeros(1, jnp.int32), engine_tables, jnp.zeros(1, bool))
		result = method.request(
			{},
			view,
			method.new_beliefs(87, 1),
			jnp.zeros((1, 2)),
			KEY[None],
			jnp.zeros((1, 9)),
			jnp.zeros(1, jnp.int32),
		)
		return view, result, method.last_solution

	first, a, solution = query(position, 0)
	second, b, comparison = query(twin, 3)
	assert_tree(first, second)
	assert_tree(a, b)
	assert_tree(solution, comparison)
	assert a[0].searched[0]
	assert first.obs.legal_actions[0, a[0].action[0]]
	assert np.isfinite(np.asarray(solution.gap)).all()
