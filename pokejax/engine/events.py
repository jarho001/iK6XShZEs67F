"""The last step's visible events: one small fixed record per side, written by step.

A step covers what happened between two decision points (part of a turn when a pivot or a
faint pauses it). step clears the record first, so it describes that step only. Each column
is what the protocol shows for the side, named where the write happens:

- order: 1 if this side's move attempt was the first of the step to run, 2 if the other side's
  attempt ran first, 0 if the side made no move attempt (a switch, a waiting side, a finished
  battle). Protocol: the sequence of |move| and |cant| lines of the two sides (a Magic Bounce
  reflection's |move| is the holder's, not an attempt of its own).
- move: the last move ID this side used in the step, as its |move| line names it (a called
  move such as Sleep Talk's or Nature Power's is the called move; 0 when none). Protocol: |move|.
- cant: why this side's move attempt did not start (0 when it did or made none). Protocol:
  |cant| lines, and -activate|confusion for a self-hit.
- outcome: bit flags of this side's last move. MISS: -miss. FAILED: the move had no effect
  on its target, as the protocol shows it (-fail, -immune, -nothing, -block, a Protect or
  Magic Bounce block, a refused status, a Substitute or Wish or Encore that is refused,
  Recover at full HP, Sleep Talk while awake, Belly Drum at half HP, a move at a fainted
  target, Trick with an empty hand, Roar with no bench member). CRIT: -crit. Protocol: those
  lines; for a reflected move, the holder's own attempt.
- effect: the effectiveness class of this side's last move: 1 immune (-immune), 2 resisted
  (-resisted), 3 neutral (a damaging hit with no effectiveness line), 4 super-effective
  (-supereffective). Protocol: those lines; 0 when the move hit nothing.
- hits: the hit count of a multi-hit move (-hitcount), 0 otherwise.
- entry: how this side's Pokemon entered the field in the step: 1 a chosen switch, 2 a drag
  (|drag|), 3 a switch after its own pivot (U-turn, Volt Switch), 4 a switch after a faint.
  Protocol: |switch| and |drag|; the request kind tells choice from replacement.
- hp: the change of the protocol percentage (observation.py rounding) of the Pokemon that was
  active at the start of the step, from the start to the end of the step, counted only while
  it stays on the field (a switch-out heal such as Regenerator prints nothing). Protocol:
  -damage, -heal and faint lines.
- status: the last major status written on this side's Pokemon in the step (1..6), 0 when
  none. Protocol: -status (including one a berry cures in the same step).
- fainted: 1 when a Pokemon of this side fainted in the step. Protocol: faint.

Columns of BattleState.event are int8 [2, EVENT_FIELDS]; BattleState.event_move is int16 [2].
"""

import jax.numpy as jnp

from .state import HP, MOVE

# actions.py also defines this; events.py must not import actions (items.py imports events).
SIDES = jnp.arange(2)

(
	EV_ORDER,
	EV_CANT,
	EV_OUTCOME,
	EV_EFFECT,
	EV_HITS,
	EV_ENTRY,
	EV_HP,
	EV_STATUS,
	EV_FAINTED,
) = range(9)
EVENT_FIELDS = 9

ORDER_NONE, ORDER_FIRST, ORDER_SECOND = range(3)

CANT_NONE = 0
(
	CANT_PARALYSIS,
	CANT_SLEEP,
	CANT_FREEZE,
	CANT_FLINCH,
	CANT_CONFUSION,
	CANT_RECHARGE,
	CANT_TRUANT,
	CANT_DISABLE,
	CANT_TAUNT,
	CANT_FOCUS_PUNCH,
) = range(1, 11)

OUTCOME_MISS, OUTCOME_FAILED, OUTCOME_CRIT = 1, 2, 4

EFFECT_NONE, EFFECT_IMMUNE, EFFECT_RESISTED, EFFECT_NEUTRAL, EFFECT_SUPER = range(5)

ENTRY_NONE, ENTRY_CHOSEN, ENTRY_DRAGGED, ENTRY_PIVOT, ENTRY_REPLACEMENT = range(5)


def protocol_percent(hp, maximum):
	"""observation.py's HP percentage: ceil(100 * hp / max), at most 99 while damaged, 0 at 0."""
	hp = hp.astype(jnp.int32)
	maximum = jnp.maximum(maximum.astype(jnp.int32), 1)
	percent = (100 * hp + maximum - 1) // maximum
	return jnp.where(hp < maximum, jnp.minimum(percent, 99), percent)


def first_cut(reason, before, after, code):
	"""A prevention that stops a move this check (before -> after) sets the reason, if none is set.

	The checks run in Showdown's BeforeMove order, so the first one to stop the move names it.
	"""
	return jnp.where(before & ~after & (reason == CANT_NONE), code, reason)


def clear_events(state, enabled):
	"""Start of a live step: the record describes only the step that is about to run.

	A finished battle's step is a no-op, so its record keeps the step that ended the battle.
	"""
	return state._replace(
		event=jnp.where(enabled, jnp.zeros_like(state.event), state.event),
		event_move=jnp.where(enabled, jnp.zeros_like(state.event_move), state.event_move),
	)


def write_columns(state, sides, values, columns):
	"""Write the listed columns of a {column: value} dict for the sides selected by a [2] mask.

	Each value is a scalar or [2]. The record is rebuilt with elementwise selects only, so the
	whole write fuses into one kernel.
	"""
	indices = jnp.arange(EVENT_FIELDS)
	dtype = state.event.dtype
	patch = jnp.zeros((2, EVENT_FIELDS), dtype)
	for column in columns:
		value = jnp.broadcast_to(jnp.asarray(values[column], dtype), (2,))
		patch = jnp.where(indices[None, :] == column, value[:, None], patch)
	touch = jnp.asarray([column in columns for column in range(EVENT_FIELDS)])
	mask = sides[:, None] & touch[None, :]
	return state._replace(event=jnp.where(mask, patch, state.event))


def write_event(state, sides, column, value):
	"""Write one event column for the sides selected by a [2] mask, with a scalar or [2] value."""
	return write_columns(state, sides, {column: value}, (column,))


def write_attempt(state, sides, move, cant, outcome, effect, hits, order=False):
	"""One side's move attempt: the move it used or why it could not, and the move's outcome.

	Overwrites the side's earlier attempt in this step, so the record describes the last one.
	With order, the attempt also takes its place in the step's order: first if the other side has
	not attempted a move yet, second otherwise. The columns are written in one masked write.
	"""
	values = {EV_CANT: cant, EV_OUTCOME: outcome, EV_EFFECT: effect, EV_HITS: hits}
	columns = (EV_CANT, EV_OUTCOME, EV_EFFECT, EV_HITS)
	if order:
		foe_moved = state.event[::-1, EV_ORDER] > ORDER_NONE
		values[EV_ORDER] = jnp.where(foe_moved, ORDER_SECOND, ORDER_FIRST)
		columns = columns + (EV_ORDER,)
	state = write_columns(state, sides, values, columns)
	move = jnp.broadcast_to(jnp.asarray(move, state.event_move.dtype), (2,))
	return state._replace(event_move=jnp.where(sides, move, state.event_move))


def record_step_bounds(state):
	"""Start-of-step facts for the end-of-step record: each side's active slot and its HP."""
	active = state.active
	hp = state.hp[SIDES, active]
	return active, hp


def finish_events(state, start, enabled):
	"""End of a live step: HP change of the start-of-step active, and faints of each side.

	Only the active Pokemon loses HP in a step. A faint is therefore the HP of the active
	reaching zero: the start-of-step active's while it is still on the field, or the
	switch-in's otherwise (a switch-in can be hurt by hazards). The start-of-step active
	is still the active when its replacement is awaited, so the faint is seen there.
	"""
	start_active, start_hp = start
	maximum = state.stats[SIDES, state.active, HP]
	end_hp = state.hp[SIDES, state.active]
	same_slot = state.active == start_active
	# The HP change counts only while the start-of-step Pokemon stays on the field: a heal on
	# switch-out (Regenerator) prints nothing in the protocol. A Pokemon dragged back into its
	# own slot after a switch in the same step counts as entered, like any switch-in.
	entered = state.event[:, EV_ENTRY] != ENTRY_NONE
	change = protocol_percent(end_hp, maximum) - protocol_percent(start_hp, maximum)
	change = jnp.where(same_slot & ~entered, change, 0)
	fainted = (end_hp == 0) & (~same_slot | (start_hp > 0))
	sides = jnp.broadcast_to(enabled, (2,))
	return write_columns(state, sides, {EV_HP: change, EV_FAINTED: fainted}, (EV_HP, EV_FAINTED))


def entry_kinds(state, requests, bounds):
	"""Why each side's Pokemon enters in this step: a voluntary switch, or a replacement.

	A replacement after the active fainted is requested only when the active is at zero HP;
	a switch request with a living active comes from that Pokemon's own pivot move. bounds is
	record_step_bounds of the same state, so its HP is the start-of-step active's HP.
	"""
	alive = bounds[1] > 0
	return jnp.where(
		requests == MOVE,
		ENTRY_CHOSEN,
		jnp.where(alive, ENTRY_PIVOT, ENTRY_REPLACEMENT),
	).astype(state.event.dtype)


def write_entry(state, sides, kind):
	"""The switch-in of the sides selected by a [2] mask, with its kind (scalar or [2])."""
	return write_event(state, sides, EV_ENTRY, kind)
