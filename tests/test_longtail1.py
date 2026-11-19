"""Focused checks for task 13 rules that random battles rarely reach.

The states come from the first battle of the long-tail fixture; the engine functions
run eagerly on them, so no compiled step is needed.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.diff import load_recording
from pokejax.engine import state as S
from pokejax.engine.abilities import active_abilities
from pokejax.engine.actions import SIDES
from pokejax.engine.data import build_engine_tables
from pokejax.engine.delayed import (
	apply_trick_room,
	apply_wish,
	healing_wish_switch_in,
	wish_residual,
)
from pokejax.engine.item_moves import apply_item_swap
from pokejax.engine.move_data import read_move
from pokejax.engine.randomness import UNFORCED, random_bits
from pokejax.engine.reset import reset, teams_from_pool
from pokejax.engine.residual import _leech_seed, residual
from pokejax.engine.restrictions import restrict_legal_moves
from pokejax.engine.step import _resolve_action_winner
from pokejax.engine.substitute import apply_substitute, split_hits
from pokejax.tables import load_tables
from pokejax.teams import team_pool_from_data

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/diff-longtail1.jsonl.gz"


@pytest.fixture(scope="module")
def world():
	header, battles = load_recording(FIXTURE)
	tables = load_tables()
	pool = team_pool_from_data(dict(header, teams=battles[0]["teams"], count=2), tables)
	engine = build_engine_tables(tables)
	state = reset(jax.random.key(0), teams_from_pool(pool, 0, 1), engine, UNFORCED)
	return tables, engine, state


def _set(state, **fields):
	return state._replace(
		**{name: jnp.asarray(value, getattr(state, name).dtype) for name, value in fields.items()}
	)


def test_split_hits_breaking_substitute_passes_later_hits_through():
	# 30 + 30 + 30 against 40 HP: the second hit breaks it, the third reaches the target.
	absorbed, dealt = split_hits(
		jnp.asarray(40), jnp.asarray([30, 30, 30, 0, 0]), jnp.asarray([True] * 3 + [False] * 2)
	)
	assert absorbed.tolist() == [True, True, False, False, False]
	assert dealt.tolist() == [30, 10, 0, 0, 0]


def test_substitute_costs_a_quarter_and_needs_more_than_a_quarter(world):
	_, _engine, state = world
	maximum = int(state.stats[0, state.active[0], 0])
	after = apply_substitute(state, 0, jnp.asarray(True))
	assert int(after.volatiles[0, S.SUBSTITUTE]) == maximum // 4
	assert int(after.hp[0, state.active[0]]) == int(state.hp[0, state.active[0]]) - max(
		maximum // 4, 1
	)
	low = _set(state, hp=state.hp.at[0, state.active[0]].set(maximum // 4))
	refused = apply_substitute(low, 0, jnp.asarray(True))
	assert int(refused.volatiles[0, S.SUBSTITUTE]) == 0


def test_taunt_removes_status_moves_from_the_legal_mask(world):
	_, engine, state = world
	moves = state.moves[SIDES, state.active].astype(jnp.int32)
	usable = jnp.ones_like(moves, dtype=bool)
	taunted = _set(state, volatiles=state.volatiles.at[0, S.TAUNT].set(3))
	mask = np.asarray(restrict_legal_moves(taunted, moves, usable, engine))
	status = np.asarray(engine.move_category[moves[0]]) == 2
	assert not np.any(mask[0] & status)
	assert np.all(mask[1] == np.asarray(usable[1]))


def test_trick_room_second_use_ends_it(world):
	_, _, state = world
	on = apply_trick_room(state, jnp.asarray(True))
	assert int(on.trick_room) == 5
	off = apply_trick_room(on, jnp.asarray(True))
	assert int(off.trick_room) == 0


def test_wish_resolves_on_the_second_residual_and_heals_the_slot(world):
	_, engine, state = world
	maximum = int(state.stats[0, state.active[0], 0])
	hurt = _set(state, hp=state.hp.at[0, state.active[0]].set(maximum // 4))
	set_wish = apply_wish(hurt, 0, jnp.asarray(True))
	assert int(set_wish.wish_turns[0]) == 2
	assert int(set_wish.wish_hp[0]) == maximum // 2
	healed = wish_residual(
		set_wish._replace(wish_turns=set_wish.wish_turns.at[0].set(1)),
		jnp.asarray(True),
		None,
		UNFORCED,
		engine,
		None,
	)
	assert int(healed.hp[0, state.active[0]]) == min(maximum, maximum // 4 + maximum // 2)
	assert int(healed.wish_turns[0]) == 0


def test_healing_wish_heals_and_cures_the_entrant(world):
	_, _, state = world
	slots = jnp.arange(6)
	bench = int(np.argmax(np.asarray((state.hp[1] > 0) & (slots != state.active[1]))))
	hurt = _set(
		state, healing_wish=jnp.asarray([False, True]), status=state.status.at[1, bench].set(S.BURN)
	)
	targets = jnp.asarray([int(state.active[0]), bench], jnp.int32)
	healed = healing_wish_switch_in(hurt, jnp.asarray([False, True]), targets)
	assert not bool(healed.healing_wish[1])
	assert int(healed.status[1, bench]) == S.HEALTHY
	assert int(healed.hp[1, bench]) == int(state.stats[1, bench, 0])


def test_trick_swaps_items_and_clears_choice_locks(world):
	tables, engine, state = world
	choice = tables.items.index["choicescarf"]
	lifeorb = tables.items.index["lifeorb"]
	set_items = _set(
		state,
		item=state.item.at[0, state.active[0]].set(choice).at[1, state.active[1]].set(lifeorb),
	)
	locked = _set(set_items, volatiles=set_items.volatiles.at[0, S.CHOICE_MOVE].set(5))
	rows = active_abilities(locked, engine)
	swapped, ok = apply_item_swap(locked, 0, jnp.asarray(True), engine, rows)
	assert bool(ok)
	assert int(swapped.item[0, swapped.active[0]]) == lifeorb
	assert int(swapped.item[1, swapped.active[1]]) == choice
	assert int(swapped.volatiles[0, S.CHOICE_MOVE]) == 0


def test_leech_seed_heal_survives_the_seeded_faint(world):
	# Battle 23 step 92 and battle 131 step 53: the seeded Pokemon faints from the damage,
	# and the seeder still takes the heal in the same callback (Showdown onResidual).
	tables, engine, state = world
	seeded = state.active[0]
	seeder = state.active[1]
	st = _set(
		state,
		hp=state.hp.at[0, seeded].set(1).at[1, seeder].set(state.stats[1, seeder, 0] - 5),
		volatiles=state.volatiles.at[0, S.LEECH_SEED].set(1),
		ability=state.ability.at[0, seeded].set(tables.abilities.index["static"]),
	)
	bits = random_bits(jax.random.key(0))
	rows = active_abilities(st, engine)
	after = _leech_seed(
		st,
		jnp.asarray(True),
		jnp.asarray(0, jnp.int32),
		jnp.zeros(2, bool),
		bits,
		UNFORCED,
		engine,
		rows,
	)
	assert int(after.hp[0, seeded]) == 0
	assert int(after.hp[1, seeder]) == int(st.stats[1, seeder, 0]) - 4


def test_poison_heal_is_not_revealed_while_the_residual_waits(world):
	# Battle 166 step 24: a U-turn pauses the turn before its residual, so Poison Heal stays hidden.
	tables, engine, state = world
	slot = state.active[0]
	poisoned = _set(
		state,
		ability=state.ability.at[0, slot].set(tables.abilities.index["poisonheal"]),
		status=state.status.at[0, slot].set(S.POISON),
		hp=state.hp.at[0, slot].set(state.stats[0, slot, 0] // 2),
	)
	bits = random_bits(jax.random.key(0))
	rows = active_abilities(poisoned, engine)
	paused = residual(poisoned, jnp.asarray(False), bits, UNFORCED, engine, rows)
	assert int(paused.ability_public[0, slot]) == 0
	ran = residual(poisoned, jnp.asarray(True), bits, UNFORCED, engine, rows)
	assert int(ran.ability_public[0, slot]) == tables.abilities.index["poisonheal"]


def test_aftermath_kill_of_the_attacker_gives_the_win_to_the_defender(world):
	# Battle 287 step 88: Aftermath is instafaint, so the attacker is processed first and
	# the defender, processed last, decides checkWin.
	tables, engine, state = world
	row = read_move(engine, int(tables.moves.index["tackle"]))
	both_out = jnp.asarray([False, False])
	plain = _resolve_action_winner(state, None, both_out, 1, row, False)
	aftermath = _resolve_action_winner(state, None, both_out, 1, row, True)
	assert int(plain.winner) == 1
	assert int(aftermath.winner) == 0
