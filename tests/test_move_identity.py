"""Move identity before the wide row load, and the prepared team inputs of the pool sampler."""

import jax
import jax.numpy as jnp
import numpy as np

from pokejax.bench import _new_states
from pokejax.engine.data import build_engine_tables
from pokejax.engine.randomness import ORDER, UNFORCED
from pokejax.engine.reset import Teams, prepare_teams, reset


def test_sleep_talk_and_nature_power_flags_are_disjoint(tables):
	"""The call resolution takes one static-call row for both calls, so no move may carry both."""
	engine_tables = build_engine_tables(tables)
	sleep_talk = np.asarray(engine_tables.move_table.column("move_sleep_talk"))
	static_call = np.asarray(engine_tables.move_table.column("move_call"))
	assert sleep_talk.any()
	assert (static_call > 0).any()
	assert not ((sleep_talk != 0) & (static_call > 0)).any()


def test_prepared_team_inputs_reset_identically(synthetic_pool, tables):
	engine_tables = build_engine_tables(tables)
	pool = Teams(*(jnp.asarray(getattr(synthetic_pool, name)) for name in Teams._fields))
	prepared = prepare_teams(pool, engine_tables)
	key = jax.random.key(4)
	plain = _new_states(key, pool, engine_tables, 16)
	fast = _new_states(key, pool, engine_tables, 16, prepared=prepared)
	for a, b in zip(jax.tree.leaves(plain), jax.tree.leaves(fast), strict=True):
		np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_unforced_none_and_all_unforced_array_reset_identically(synthetic_pool, tables):
	engine_tables = build_engine_tables(tables)
	teams = Teams(*(jnp.asarray(getattr(synthetic_pool, name)[:2]) for name in Teams._fields))
	key = jax.random.key(5)
	by_none = reset(key, teams, engine_tables, None)
	by_array = reset(key, teams, engine_tables, UNFORCED)
	forced = UNFORCED.at[ORDER, 0, 0].set(1)
	for a, b in zip(jax.tree.leaves(by_none), jax.tree.leaves(by_array), strict=True):
		np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
	assert reset(key, teams, engine_tables, forced).turn == by_none.turn
