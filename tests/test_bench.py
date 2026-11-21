import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.bench import _new_states, benchmark, rollout
from pokejax.engine.data import build_engine_tables
from pokejax.engine.reset import Teams
from pokejax.engine.state import ONGOING
from pokejax.teams import save_npz


def test_benchmark_and_finished_battles(tmp_path, synthetic_pool, tables):
	path = tmp_path / "teams.npz"
	save_npz(synthetic_pool, path)
	result = benchmark(path, batch_size=8, steps=4, repetitions=1, seed=1)
	assert result["compile_seconds"] > 0
	assert result["steps_per_second"] > 0
	assert result["completed_battles"] >= 0
	assert result["mean_battle_length"] is None or result["mean_battle_length"] > 0

	pool = Teams(*(jnp.asarray(getattr(synthetic_pool, name)) for name in Teams._fields))
	engine_tables = build_engine_tables(tables)
	key, init_key = jax.random.split(jax.random.key(2))
	state = _new_states(init_key, pool, engine_tables, 8)
	# Side 1 has one HP left and no bench; force the only legal move to hit.
	hp = state.hp.at[:, 0, 1:].set(0).at[:, 1, :].set(0).at[:, 1, 0].set(1)
	state = state._replace(hp=hp, pp=state.pp.at[:, :, :, 1:].set(0))
	engine_tables = engine_tables._replace(
		move_always_hits=jnp.ones_like(engine_tables.move_always_hits),
		move_base_power=jnp.full_like(engine_tables.move_base_power, 100),
		type_immune=jnp.zeros_like(engine_tables.type_immune),
	)
	carry, (count, total_length) = jax.jit(rollout, static_argnums=(3, 4))(
		(state, key, jnp.full(8, 7, dtype=jnp.int32)), pool, engine_tables, 4, False
	)
	assert int(count) == 8
	assert int(total_length) == 64
	assert np.all(np.asarray(carry[0].winner) == ONGOING)
	assert np.all(np.asarray(carry[2]) == 3)


@pytest.mark.parametrize("with_observation", [False, True])
def test_environment_benchmark(tmp_path, synthetic_pool, with_observation):
	path = tmp_path / "teams.npz"
	save_npz(synthetic_pool, path)
	result = benchmark(
		path,
		batch_size=8,
		steps=4,
		repetitions=1,
		seed=1,
		with_observation=with_observation,
		environment=True,
	)
	assert result["mode"] == "environment"
	assert result["steps_per_second"] > 0 and result["compile_seconds"] > 0
	assert result["completed_battles"] >= 0
	assert result["mean_battle_length"] is None or result["mean_battle_length"] > 0
	assert (result["observation_checksum"] is not None) == with_observation
