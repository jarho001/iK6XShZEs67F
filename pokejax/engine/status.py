"""Major status effects, move prevention and status move primitives."""

import jax.numpy as jnp

from .abilities import (
	OWN_TEMPO,
	SLEEP_EARLY_BIRD,
	SLEEP_HOOK,
	STATUS_EVENT,
	STATUS_EVENT_SYNCHRONIZE,
	active_abilities,
	effective_weather,
	has_hook,
	reveal_abilities,
	status_immunity,
)
from .abilities import (
	TRUANT as TRUANT_ABILITY,
)
from .actions import SIDES
from .events import (
	CANT_DISABLE,
	CANT_FREEZE,
	CANT_NONE,
	CANT_PARALYSIS,
	CANT_RECHARGE,
	CANT_SLEEP,
	CANT_TAUNT,
	CANT_TRUANT,
	EV_STATUS,
	first_cut,
	write_event,
)
from .field import effective_move_type
from .move_data import read_move
from .randomness import FULL_PARALYSIS, SLEEP_DURATION, THAW, chance, integer
from .restrictions import taunt_blocks
from .state import (
	BURN,
	DESTINY_BOND,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	FREEZE,
	HEALTHY,
	MUST_RECHARGE,
	PARALYSIS,
	POISON,
	SLEEP,
	TEAM_SLOTS,
	TOXIC,
	TWO_TURN_MOVE,
)
from .state import (
	TRUANT as TRUANT_VOLATILE,
)
from .turn_moves import set_active_volatile
from .volatiles import before_move as volatile_before_move

REST, PARTY_CURE, TRANSFER, REFRESH = range(1, 5)


def switch_in(state, side, slot, enabled):
	status = state.status[side, slot]
	counter = state.status_counter[side, slot]
	counter = jnp.where(status == TOXIC, 0, counter)
	counter = jnp.where(status == SLEEP, state.sleep_start[side, slot], counter)
	return state._replace(
		status_counter=state.status_counter.at[side, slot].set(
			jnp.where(enabled, counter, state.status_counter[side, slot])
		)
	)


def before_move(state, side, move, enabled, key, forced, tables, ability_rows, recharge=False):
	"""BeforeMove handlers in Showdown's priority order.

	Sleep (10), flinch (8), Disable (7), Taunt (5), confusion (3), paralysis (1),
	then Destiny Bond (-1), which only clears when no earlier handler stopped the move.
	Returns (state, can_act, cant): cant is the events.py reason of the first check that stopped it.
	"""
	row = read_move(tables, move)
	started = enabled
	was_charging = state.volatiles[side, TWO_TURN_MOVE] > 0
	recharge_now = enabled & recharge
	state = set_active_volatile(
		state,
		side,
		MUST_RECHARGE,
		jnp.where(recharge_now, 0, state.volatiles[side, MUST_RECHARGE]),
	)
	reason = CANT_NONE
	before = enabled
	enabled &= ~recharge_now
	reason = first_cut(reason, before, enabled, CANT_RECHARGE)
	slot = state.active[side]
	status = state.status[side, slot]
	counter = state.status_counter[side, slot]
	sleeping = enabled & (status == SLEEP)
	early_bird = ability_rows[side, SLEEP_HOOK] == SLEEP_EARLY_BIRD
	counter = jnp.where(sleeping, counter - 1 - early_bird.astype(counter.dtype), counter)
	wake = sleeping & (counter <= 0)
	can_defrost = row.move_defrost
	thaw = enabled & (status == FREEZE) & ~can_defrost & chance(key, forced, THAW, side, 0, 1, 5)
	frozen = enabled & (status == FREEZE) & ~thaw & ~can_defrost
	clear = wake | thaw
	member_mask = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == slot)
	state = state._replace(
		status=jnp.where(member_mask & clear, HEALTHY, state.status),
		status_counter=jnp.where(member_mask, jnp.where(clear, 0, counter), state.status_counter),
	)
	# Sleep Talk and the other sleep-usable moves may be used while asleep.
	before = enabled
	enabled &= ~(sleeping & ~wake & ~row.move_sleep_usable)
	reason = first_cut(reason, before, enabled, CANT_SLEEP)
	before = enabled
	enabled &= ~frozen
	reason = first_cut(reason, before, enabled, CANT_FREEZE)
	truant_ability = has_hook(ability_rows[side], TRUANT_ABILITY)
	truant_active = truant_ability & (state.volatiles[side, TRUANT_VOLATILE] > 0)
	truant_block = enabled & ~recharge_now & truant_active
	truant_value = jnp.where(
		enabled & ~recharge_now & truant_ability,
		jnp.where(truant_active, 0, 1),
		state.volatiles[side, TRUANT_VOLATILE],
	)
	truant_value = jnp.where(recharge_now, 0, truant_value)
	state = set_active_volatile(state, side, TRUANT_VOLATILE, truant_value)
	before = enabled
	enabled &= ~truant_block
	reason = first_cut(reason, before, enabled, CANT_TRUANT)
	state = reveal_abilities(state, (SIDES == side) & truant_block)
	disabled = (state.volatiles[side, DISABLE_REMAINING] > 0) & (
		row.id == state.volatiles[side, DISABLE_MOVE]
	)
	before = enabled
	enabled &= ~disabled
	reason = first_cut(reason, before, enabled, CANT_DISABLE)
	before = enabled
	enabled &= ~taunt_blocks(state, side, row)
	reason = first_cut(reason, before, enabled, CANT_TAUNT)
	state, enabled, volatile_reason = volatile_before_move(
		state, side, enabled, key, forced, tables, ability_rows[side]
	)
	reason = jnp.where(reason == CANT_NONE, volatile_reason, reason)
	before = enabled
	paralyzed = enabled & (status == PARALYSIS) & chance(key, forced, FULL_PARALYSIS, side, 0, 1, 4)
	can_act = enabled & ~paralyzed
	reason = first_cut(reason, before, can_act, CANT_PARALYSIS)
	# Destiny Bond ends when its user attempts any other move: its BeforeMove handler (priority
	# -1) still runs after a sleep, flinch or paralysis check has stopped the move.
	destiny_clear = started & (row.move_self_volatile != 4)
	state = set_active_volatile(
		state,
		side,
		DESTINY_BOND,
		jnp.where(destiny_clear, 0, state.volatiles[side, DESTINY_BOND]),
	)
	state = set_active_volatile(
		state,
		side,
		TWO_TURN_MOVE,
		jnp.where(was_charging & started & ~can_act, 0, state.volatiles[side, TWO_TURN_MOVE]),
	)
	return state, can_act, reason


def inflict_status(
	state,
	side,
	inflicted,
	enabled,
	key,
	forced,
	tables,
	ability_rows=None,
	*,
	reveal_immunity=False,
):
	"""Pokemon.trySetStatus: HP, current status, type immunity, then Sleep Clause."""
	slot = state.active[side]
	types = state.types[side, slot]
	type_immune = jnp.any((types >= 0) & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	rows = active_abilities(state, tables) if ability_rows is None else ability_rows
	weather = effective_weather(state, rows)
	weather_immune = tables.weather_status_immune[weather, inflicted]
	ability_immune = status_immunity(rows[side], inflicted, weather)
	immune = type_immune | weather_immune | ability_immune
	clause = jnp.any(
		(state.status[side] == SLEEP) & (state.sleep_from_foe[side] != 0) & (state.hp[side] > 0)
	)
	attempt = enabled & (state.hp[side, slot] > 0) & (inflicted != HEALTHY)
	attempt &= (state.status[side, slot] == HEALTHY) & ~((inflicted == SLEEP) & clause)
	apply = attempt & ~immune
	duration = (integer(key, forced, SLEEP_DURATION, side, 0, 3) + 2).astype(jnp.int8)
	member = (SIDES == side)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == slot)
	status = jnp.where(member & apply, jnp.asarray(inflicted, state.status.dtype), state.status)
	counter = jnp.where(
		member & apply,
		jnp.where(inflicted == SLEEP, duration, 0),
		state.status_counter,
	)
	start = jnp.where(member & apply & (inflicted == SLEEP), duration, state.sleep_start)
	source = jnp.where(member & apply, inflicted == SLEEP, state.sleep_from_foe)
	state = state._replace(
		status=status, status_counter=counter, sleep_start=start, sleep_from_foe=source
	)
	state = write_event(state, (SIDES == side) & apply, EV_STATUS, inflicted)
	# Synchronize supplies a status-like pseudo effect, which Showdown treats
	# like a named move status in the immunity protocol.
	if reveal_immunity:
		reveal = attempt & ~type_immune & ~weather_immune & ability_immune
		state = reveal_abilities(state, (SIDES == side) & reveal)
	return state


def synchronize_status(
	state, target_side, source_side, inflicted, enabled, key, forced, tables, ability_rows=None
):
	"""Copy a reflected major status once, without allowing Synchronize to chain.

	Both holders reveal Synchronize: the target when the status reaches it, and the source
	when the copy lands on it (Showdown runs onAfterSetStatus for each status that is set).
	"""
	rows = active_abilities(state, tables) if ability_rows is None else ability_rows
	statusable = (inflicted == BURN) | (inflicted == POISON) | (inflicted == TOXIC)
	statusable |= inflicted == PARALYSIS
	trigger = enabled & statusable & (target_side != source_side)
	trigger &= rows[target_side, STATUS_EVENT] == STATUS_EVENT_SYNCHRONIZE
	source_slot = state.active[source_side]
	before = state.status[source_side, source_slot]
	state = inflict_status(
		state, source_side, inflicted, trigger, key, forced, tables, rows, reveal_immunity=True
	)
	copied = trigger & (state.status[source_side, source_slot] != before)
	copied &= rows[source_side, STATUS_EVENT] == STATUS_EVENT_SYNCHRONIZE
	revealed = ((SIDES == target_side) & trigger) | ((SIDES == source_side) & copied)
	return reveal_abilities(state, revealed)


def apply_move(
	state, side, move, enabled, hit, key, forced, tables, abilities, weather, reach=None
):
	"""Apply move status effects with dense team masks and one field write.

	reach is the move reaching its target (accuracy passed, nothing blocked it); a status the
	target refuses (immunity, a status it already has) fails with a reach even when the move's
	effect is not applied (Showdown's TryHit and trySetStatus). Defaults to hit.
	"""
	reach = hit if reach is None else reach
	row = read_move(tables, move)
	foe = 1 - side
	user, target = state.active[side], state.active[foe]
	effect = row.move_effect
	user_status = state.status[side, user]
	target_status = state.status[foe, target]
	team_slots = jnp.arange(TEAM_SLOTS)
	user_mask = (SIDES == side)[:, None] & (team_slots == user)[None, :]
	target_mask = (SIDES == foe)[:, None] & (team_slots == target)[None, :]
	# ModifyMove cures a defrost move's user; DamagingHit thaws its target later,
	# AFTER secondaries (BattleActions.spreadMoveHit, Gen 5+ event order).
	self_thaw = enabled & (user_status == FREEZE) & row.move_defrost
	status = jnp.where(user_mask & self_thaw, HEALTHY, state.status)
	user_status = jnp.where(self_thaw, HEALTHY, user_status)
	transfer = effect == TRANSFER
	inflicted = jnp.where(transfer, user_status, row.move_status)
	types = state.types[foe, target]
	immune = jnp.any((types >= 0) & tables.status_immune[inflicted, jnp.maximum(types, 0)])
	immune |= tables.weather_status_immune[weather, inflicted]
	ability_immune = status_immunity(abilities[foe], inflicted, weather)
	clause = jnp.any(
		(state.status[foe] == SLEEP) & (state.sleep_from_foe[foe] != 0) & (state.hp[foe] > 0)
	)
	inflict = enabled & hit & ~row.move_status_self
	inflict &= (state.hp[foe, target] > 0) & (inflicted != HEALTHY)
	inflict &= (target_status == HEALTHY) & ~immune & ~((inflicted == SLEEP) & clause)
	# The target's ability handler runs before Sleep Clause, so the immunity still shows.
	blocked_by_ability = (
		enabled
		& hit
		& ~row.move_status_self
		& (state.hp[foe, target] > 0)
		& (inflicted != HEALTHY)
		& (target_status == HEALTHY)
		& ~immune
		& ability_immune
	)
	reveal_target = blocked_by_ability & ~transfer
	own_tempo = has_hook(abilities[foe], OWN_TEMPO) & (state.hp[foe, target] > 0)
	reveal_target |= enabled & hit & row.move_confusion & own_tempo
	inflict &= ~ability_immune
	duration = (integer(key, forced, SLEEP_DURATION, foe, 0, 3) + 2).astype(jnp.int8)
	target_status_after = jnp.where(inflict, inflicted, target_status)
	status = jnp.where(target_mask & inflict, inflicted, status)
	counter = jnp.where(
		target_mask & inflict,
		jnp.where(inflicted == SLEEP, duration, 0),
		state.status_counter,
	)
	start = jnp.where(target_mask & inflict & (inflicted == SLEEP), duration, state.sleep_start)
	source = jnp.where(target_mask & inflict, inflicted == SLEEP, state.sleep_from_foe)
	clear_transfer = transfer & (target_status == HEALTHY) & (target_status_after != HEALTHY)
	refresh = enabled & (effect == REFRESH) & (user_status != SLEEP) & (user_status != FREEZE)
	party = enabled & (effect == PARTY_CURE)
	clear_team = party & (state.hp[side] > 0)
	clear_team_mask = (SIDES == side)[:, None] & clear_team[None, :]
	status = jnp.where(clear_team_mask, HEALTHY, status)
	counter = jnp.where(clear_team_mask, 0, counter)
	clear_user = clear_transfer | refresh
	status = jnp.where(user_mask & clear_user, HEALTHY, status)
	counter = jnp.where(user_mask & clear_user, 0, counter)
	rest = (
		enabled
		& (effect == REST)
		& (user_status != SLEEP)
		& (state.hp[side, user] < state.stats[side, user, 0])
	)
	sleep_immunity = status_immunity(abilities[side], SLEEP, weather)
	permanent_sleep_immunity = status_immunity(abilities[side], SLEEP, 0)
	revealed = ((SIDES == foe) & reveal_target) | (
		(SIDES == side) & rest & permanent_sleep_immunity
	)
	rest &= ~sleep_immunity
	status = jnp.where(user_mask & rest, SLEEP, status)
	counter = jnp.where(user_mask & rest, 3, counter)
	start = jnp.where(user_mask & rest, 3, start)
	source = jnp.where(user_mask & rest, False, source)
	hp = jnp.where(user_mask & rest, state.stats[..., 0], state.hp)
	state = state._replace(
		status=status,
		status_counter=counter,
		sleep_start=start,
		sleep_from_foe=source,
		hp=hp,
	)
	state = write_event(
		state,
		((SIDES == foe) & inflict) | ((SIDES == side) & rest),
		EV_STATUS,
		jnp.where(SIDES == side, SLEEP, inflicted),
	)
	state = synchronize_status(state, foe, side, inflicted, inflict, key, forced, tables, abilities)
	# The status is refused by the target's type, the weather or its ability (-immune).
	# Showdown's trySetStatus refuses an already-statused target (-fail) before its immunity check.
	status_immune = (
		enabled
		& reach
		& ~row.move_status_self
		& (state.hp[foe, target] > 0)
		& (inflicted != HEALTHY)
		& (target_status == HEALTHY)
		& (immune | ability_immune)
	)
	# Synchronize copies the status back to the user, which can refuse it with -immune.
	copy_src_types = state.types[side, user]
	copy_immune = jnp.any(
		(copy_src_types >= 0) & tables.status_immune[inflicted, jnp.maximum(copy_src_types, 0)]
	)
	copy_immune |= tables.weather_status_immune[weather, inflicted] | status_immunity(
		abilities[side], inflicted, weather
	)
	statusable = (inflicted == BURN) | (inflicted == POISON) | (inflicted == TOXIC)
	statusable |= inflicted == PARALYSIS
	synchronize_holder = (
		inflict & statusable & (abilities[foe, STATUS_EVENT] == STATUS_EVENT_SYNCHRONIZE)
	)
	synchronize_refused = synchronize_holder & (user_status == HEALTHY) & copy_immune
	status_immune = status_immune | synchronize_refused
	# A copy onto a user that already has a status fails (-fail), with no immunity shown.
	synchronize_busy = synchronize_holder & (user_status != HEALTHY)
	# When the user copies the status back (it is Synchronize too), the holder already has it:
	# Showdown prints that chained -fail on the holder during the user's move.
	synchronize_chain = (
		synchronize_holder
		& (user_status == HEALTHY)
		& ~copy_immune
		& (abilities[side, STATUS_EVENT] == STATUS_EVENT_SYNCHRONIZE)
	)
	# A status move that does not inflict its status fails (already statused, a refusal, Sleep
	# Clause); Rest fails at full HP, while asleep, or against an immune sleeper (onTry).
	# Sleep Clause refuses a second sleeper with a message and hint only, not a -fail.
	clause_refusal = (inflicted == SLEEP) & clause & (target_status == HEALTHY)
	# Psycho Shift transfers the user's status; with none to transfer it fails.
	transfer_fail = enabled & hit & (effect == TRANSFER) & (user_status == HEALTHY)
	# Refresh fails unless it cures a burn, poison, toxic or paralysis of its user (onHit).
	refresh_fail = (
		enabled
		& (effect == REFRESH)
		& ((user_status == HEALTHY) | (user_status == SLEEP) | (user_status == FREEZE))
	)
	status_failed = (
		(
			enabled
			& reach
			& ~row.move_status_self
			& (state.hp[foe, target] > 0)
			& (inflicted != HEALTHY)
			& ~inflict
			& ~clause_refusal
		)
		| (enabled & (effect == REST) & ~rest)
		| refresh_fail
		| transfer_fail
		| synchronize_busy
		| synchronize_chain
	)
	return state, revealed, status_immune, status_failed


def after_hit(state, side, move, enabled, tables, weather=None):
	row = read_move(tables, move)
	foe = 1 - side
	slot = state.active[foe]
	status = state.status[foe, slot]
	thaw = enabled & (state.hp[foe, slot] > 0) & (status == FREEZE)
	thaw &= (
		effective_move_type(state, side, row, tables, weather) == tables.fire_type
	) | row.move_thaws_target
	target_mask = (SIDES == foe)[:, None] & (jnp.arange(TEAM_SLOTS)[None, :] == slot)
	return state._replace(status=jnp.where(target_mask & thaw, HEALTHY, state.status))
