"""Trace: the holder copies the foe's ability when it switches in and at each Update.

Showdown: onStart seeks unless the adjacent foe has No Ability (the seek then stays off for the
stay); onUpdate copies a foe ability that is not notrace and not No Ability (sampling one foe
in singles, which draws from the PRNG with one candidate). The copied ability then runs its
own switch-in effects (setAbility runs Start), so Intimidate and the weather setters fire.
"""

import jax.numpy as jnp

from .abilities import NOTRACE, TRACE, has_hook, reveal_abilities, start_slow_start
from .actions import SIDES
from .state import TEAM_SLOTS, TRACE_SEEK


def trace_enter(state, entering, tables, view):
	"""onStart: a Trace holder seeks unless the foe's ability is No Ability, then Update copies
	a copyable foe at once. `view` is the foe as seen at this switch-in (see forms.py)."""
	active = state.active
	rows = tables.ability_hooks[state.ability[SIDES, active].astype(jnp.int32)]
	# A holder that a hazard already fainted never starts (fieldEvent skips fainted holders).
	holder = entering & has_hook(rows, TRACE) & (state.hp[SIDES, active] > 0)
	seek = holder & (view.ability != tables.noability_id)
	foe_rows = tables.ability_hooks[view.ability.astype(jnp.int32)]
	copyable = (
		(view.ability != 0)
		& (view.ability != tables.noability_id)
		& ~has_hook(foe_rows, NOTRACE)
		& (view.hp > 0)
	)
	copied = seek & copyable
	slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	member = slots & copied[:, None]
	ability = jnp.where(member, view.ability[:, None].astype(state.ability.dtype), state.ability)
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(
		entering[:, None] & (columns[None, :] == TRACE_SEEK),
		(seek & ~copied)[:, None].astype(state.volatiles.dtype),
		state.volatiles,
	)
	state = state._replace(ability=ability, volatiles=volatiles)
	return reveal_abilities(start_slow_start(state, copied, tables, foe_rows), copied)


def trace_copy(state, tables):
	"""onUpdate: a seeking Trace holder copies a copyable foe ability. Returns (state, copied)."""
	active = state.active
	rows = tables.ability_hooks[state.ability[SIDES, active].astype(jnp.int32)]
	seeking = (state.volatiles[:, TRACE_SEEK] > 0) & has_hook(rows, TRACE)
	seeking &= state.hp[SIDES, active] > 0
	foe = 1 - SIDES
	foe_ability = state.ability[foe, active[foe]]
	foe_rows = tables.ability_hooks[foe_ability.astype(jnp.int32)]
	copyable = (
		(foe_ability != 0)
		& (foe_ability != tables.noability_id)
		& ~has_hook(foe_rows, NOTRACE)
		& (state.hp[foe, active[foe]] > 0)
	)
	copied = seeking & copyable
	slots = jnp.arange(TEAM_SLOTS)[None, :] == active[:, None]
	member = slots & copied[:, None]
	ability = jnp.where(member, foe_ability[:, None].astype(state.ability.dtype), state.ability)
	columns = jnp.arange(state.volatiles.shape[1])
	volatiles = jnp.where(
		copied[:, None] & (columns[None, :] == TRACE_SEEK), 0, state.volatiles
	).astype(state.volatiles.dtype)
	state = state._replace(ability=ability, volatiles=volatiles)
	return reveal_abilities(start_slow_start(state, copied, tables, foe_rows), copied), copied
