"""Build simulated requests using a one-sided engine view and a sampled opposing team."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pokejax.engine import state as s
from pokejax.engine.abilities import NOTRACE, has_hook
from pokejax.engine.actions import action_masks
from pokejax.engine.events import OUTCOME_CRIT, OUTCOME_FAILED, OUTCOME_MISS
from pokejax.engine.observation import Observation, observe
from pokejax.engine.reset import PreparedTeams, Teams
from pokejax.search.transition import SearchState

# Only index these leaves on our side. Global fields come from Observation.
OWN_FIELDS = tuple(
	name
	for name in s.BattleState._fields
	if name
	not in {
		"weather",
		"weather_turns",
		"trick_room",
		"turn",
		"winner",
		"last_success",
		"weather_age",
		"trick_room_age",
		"turn_stage",
		"turn_first",
		"turn_actions",
		"turn_moves",
		"turn_pending",
		"hazards",
		"hazard_order",
		"request",
		"event",
		"event_move",
		"wish_age",
		"healing_wish",
	}
)


class PlayerView(NamedTuple):
	obs: Observation
	own: tuple
	side: jax.Array
	held: jax.Array
	has_held: jax.Array
	refused: jax.Array
	result: jax.Array


def player_view(position, side, tables, refused=False):
	"""Engine adapter. The lookup of hidden rosters ends at the observation boundary."""
	state = position.state
	obs = observe(state, side, tables)
	own = {name: getattr(state, name)[side] for name in OWN_FIELDS}
	# The raw partial-trap source is an opposing private roster slot.
	own["volatiles"] = own["volatiles"].at[s.PARTIAL_TRAP_SOURCE].set(0)
	return PlayerView(
		obs,
		tuple(own.values()),
		side,
		position.held[side],
		position.has_held[side],
		refused,
		state.winner,
	)


views = jax.jit(jax.vmap(player_view, in_axes=(0, 0, None, 0)))


class SetData(NamedTuple):
	teams: Teams
	prepared: PreparedTeams
	trace: jax.Array
	regenerator: jax.Array


def _hp_from_percent(fraction, maximum):
	percent = jnp.rint(fraction * 100).astype(jnp.int32)
	# Choose the largest compatible integer HP; 99 also represents rounded-up 100.
	return jnp.where(
		percent == 100,
		maximum,
		jnp.where(percent == 99, maximum - 1, percent * maximum // 100),
	).astype(jnp.int16)


def _same_leaf(expected, actual, protocol):
	if protocol and expected.dtype == jnp.float32:
		# JSON uses a double-precision quotient, while XLA may use a reciprocal.
		return jnp.all((expected == actual) | (jnp.nextafter(expected, actual) == actual))
	return jnp.all(expected == actual)


def rebuild(
	view,
	roster,
	disguise,
	history,
	own_previous_action,
	queue_action,
	data,
	tables,
	protocol=False,
):
	"""A canonical opposing roster (public ranks followed by indistinguishable unseen slots)."""
	obs, side = view.obs, view.side
	foe = obs.opponent
	our = dict(zip(OWN_FIELDS, view.own, strict=True))
	ours = our["active"]
	theirs = jnp.argmax(foe.is_active).astype(jnp.int8)
	slots = jnp.arange(s.TEAM_SLOTS)
	seen = foe.unseen == 0
	active = slots == theirs
	team = jax.tree.map(lambda a: a[roster], data.teams)
	prep = jax.tree.map(lambda a: a[roster], data.prepared)
	disguised = disguise[theirs] >= 0
	hp_seen = jnp.where(seen, _hp_from_percent(foe.hp, prep.stats[:, s.HP]), prep.stats[:, s.HP])
	public_types = foe.types.astype(jnp.int8) - 1
	types = jnp.where((active & ~disguised)[:, None], public_types, prep.types)
	roost_types = types
	flying = types == tables.flying_type
	types = jnp.where((active & bool_array(foe.roost))[:, None] & flying, s.NO_TYPE, types)
	types = jnp.where(
		(active & bool_array(foe.roost))[:, None] & jnp.all(types == s.NO_TYPE, axis=1)[:, None],
		jnp.asarray([tables.normal_type, s.NO_TYPE], jnp.int8),
		types,
	)
	transformed = bool_array(foe.transformed)
	source = history.transform_source
	species = jnp.where(active & ~disguised, foe.species, team.species)
	normal = team.moves
	copy_moves = our["moves"][source]
	moves = jnp.where((active & transformed)[:, None], copy_moves, normal)
	copy_pp = jnp.minimum(tables.move_base_pp[copy_moves], 5).astype(jnp.uint8)
	max_pp = jnp.where((active & transformed)[:, None], copy_pp, prep.max_pp)
	matched = (moves[..., None] == history.spent_moves[:, None, :]) & (moves[..., None] != 0)
	spent = (matched * history.spent_pp[:, None, :]).sum(-1)
	pp = jnp.maximum(max_pp.astype(jnp.int16) - spent, 0).astype(jnp.uint8)
	revealed = matched.any(-1).astype(jnp.uint8)
	removed = bool_array(foe.item_consumed) | bool_array(foe.item_knocked_off)
	item = jnp.where(removed, 0, jnp.where(foe.item_revealed, foe.item, team.item))
	ability = jnp.where(foe.ability > 0, foe.ability, team.ability)
	# A public copied ability supersedes the base only during the affected stay.
	trace = ability[theirs] == data.trace
	copyable = (our["ability"][ours] != tables.noability_id) & ~has_hook(
		tables.ability_hooks[our["ability"][ours]], NOTRACE
	)
	ability = jnp.where(active & transformed, our["ability"][source], ability)
	ability = jnp.where(active & trace & copyable, our["ability"][ours], ability)
	restored = ~active & (history.transformed_ranks | (team.ability == data.trace))
	ability = jnp.where(restored, team.ability, ability)
	regen = data.regenerator
	hp = jnp.where(
		seen & ~active & (ability == regen) & (hp_seen > 0),
		jnp.minimum(prep.stats[:, s.HP], hp_seen + prep.stats[:, s.HP] // 3),
		hp_seen,
	)
	stats = prep.stats
	changed = (species == tables.meloetta_forms[1]) & (team.species == tables.meloetta_forms[0])
	stats = stats.at[:, 1:].set(jnp.where(changed[:, None], prep.pirouette_stats, stats[:, 1:]))
	stats = stats.at[theirs, 1:].set(
		jnp.where(transformed, our["stats"][source, 1:], stats[theirs, 1:])
	)
	last = history.last_move[theirs]
	charge_matches = tables.move_charge[moves[theirs]]
	charge = jnp.where(jnp.any(charge_matches), moves[theirs, jnp.argmax(charge_matches)], last)
	charge = jnp.where(charge > 0, charge, moves[theirs, 0])
	v = jnp.zeros(s.NUM_VOLATILES, jnp.int16)
	for index, value in {
		s.CONFUSION: 2 * foe.confused,
		s.TWO_TURN_MOVE: charge * foe.charging,
		s.MUST_RECHARGE: foe.recharging,
		s.PARTIALLY_TRAPPED: 3 * foe.trapped,
		s.PARTIAL_TRAP_SOURCE: (ours + 1) * foe.trapped,
		s.SUBSTITUTE: jnp.maximum(1, stats[theirs, s.HP] // 4) * foe.substitute,
		s.ENCORE_MOVE: foe.encore_move,
		s.ENCORE_REMAINING: 2 * foe.encore,
		s.TAUNT: 2 * foe.taunt,
		s.DISABLE_MOVE: foe.disabled_move,
		s.DISABLE_REMAINING: 2 * foe.disable,
		s.LEECH_SEED: foe.leech_seed,
		s.YAWN: foe.yawn,
		s.CURSE: foe.curse,
		s.DESTINY_BOND: foe.destiny_bond,
		s.TRANSFORMED: transformed,
		s.FLASH_FIRE: foe.flash_fire,
		s.SLOW_START: 3 * foe.slow_start,
		s.AUTOTOMIZE: foe.autotomize,
		s.PROTECT: foe.protect,
		s.ROOST: foe.roost,
		s.ILLUSION_SLOT: disguise[theirs] + 1,
		s.CHOICE_MOVE: last,
		s.TRACE_SEEK: trace & ~copyable,
	}.items():
		v = v.at[index].set(jnp.asarray(value, jnp.int16))

	# Preserve own private battle state. Reindex a trap's source from private
	# opponent slot to its public rank. An expired source has no battle effect.
	our["volatiles"] = (
		our["volatiles"].at[s.PARTIAL_TRAP_SOURCE].set(jnp.where(obs.own.trapped, theirs + 1, 0))
	)

	def pair(our_value, their_value):
		their_value = jnp.asarray(their_value, dtype=our_value.dtype)
		return jnp.stack(
			(
				jnp.where(side == 0, our_value, their_value),
				jnp.where(side == 0, their_value, our_value),
			)
		)

	party = slots.at[0].set(theirs).at[theirs].set(0)
	target = disguise[theirs]
	target_party_index = jnp.argmax(party == target)
	last_living = jnp.argmax(jnp.where(hp[party] > 0, slots, -1))
	last_member = party[last_living]
	party = jnp.where(
		disguised, party.at[target_party_index].set(last_member).at[last_living].set(target), party
	)
	item_gone = jnp.where(
		foe.item_consumed, s.ITEM_USED, jnp.where(foe.item_knocked_off, s.ITEM_TAKEN, 0)
	)
	slept = foe.status == s.SLEEP
	status_counter = jnp.where(
		slept,
		jnp.maximum(1, 3 - history.status_age),
		jnp.where(foe.status == s.TOXIC, history.status_age, 0),
	)
	patch = {
		**team._asdict(),
		"species": species,
		"species_base": team.species,
		"types": types,
		"base_types": prep.types,
		"roost_types": roost_types,
		"stats": stats,
		"hp": hp,
		"moves": moves,
		"pp": pp,
		"max_pp": max_pp,
		"ability": ability,
		"ability_base": team.ability,
		"ability_public": foe.ability,
		"item": item,
		"item_consumed": bool_array(foe.item_consumed),
		"item_knocked_off": bool_array(foe.item_knocked_off),
		"item_revealed": bool_array(foe.item_revealed),
		"item_public": foe.item,
		"last_item": jnp.where(foe.item_consumed, foe.item, 0),
		"status": foe.status,
		"status_counter": status_counter,
		"sleep_start": jnp.where(slept, 3, 0),
		"sleep_from_foe": slept & history.sleep_from_foe,
		"seen_order": jnp.where(seen, slots + 1, 0),
		"revealed_moves": revealed,
		"active": theirs,
		"active_turns": history.active_turns,
		"boosts": jnp.rint(foe.boosts * 6),
		"volatiles": v,
		"fainted_last": history.last_faint,
		"alive_mark": hp > 0,
		"last_move": history.last_move,
		"wish_turns": jnp.where(obs.wish[1], jnp.maximum(1, 2 - obs.wish_age[1]), 0),
		"wish_hp": stats[theirs, s.HP] // 2,
		"saved_stats": prep.stats[theirs, 1:],
		"saved_moves": normal[theirs],
		"saved_pp": jnp.maximum(prep.max_pp[theirs].astype(jnp.int16) - 1, 0),
		"saved_max_pp": prep.max_pp[theirs],
		"pirouette_stats": prep.pirouette_stats,
		"shown_species": foe.species,
		"shown_level": jnp.rint(100 * foe.level),
		"party": party,
		"hp_seen": hp_seen,
		"item_gone": item_gone,
	}
	own_fields = {
		name: pair(value, patch.get(name, jnp.zeros_like(value))) for name, value in our.items()
	}

	own_request = jnp.argmax(obs.request).astype(jnp.int8)
	mid = bool_array(obs.mid_turn)
	dead = foe.fainted[theirs] > 0
	our_dead = obs.own.fainted[ours] > 0
	own_pivot = mid & (own_request == s.SWITCH) & ~our_dead
	foe_switch = ~own_pivot & (dead | (own_request == s.NONE))
	foe_request = jnp.where(mid, jnp.where(foe_switch, s.SWITCH, s.NONE), s.MOVE).astype(jnp.int8)
	foe_pivot = mid & (foe_request == s.SWITCH) & ~dead
	foe_queued = (
		own_pivot & ~dead & ~bool_array(obs.events.order[1]) & ~bool_array(obs.events.entered[1])
	)
	own_queued = foe_pivot & ~our_dead & history.queued_ours & (own_previous_action < s.MOVE_SLOTS)
	our_move = jnp.where(
		our["volatiles"][s.TWO_TURN_MOVE] > 0,
		our["volatiles"][s.TWO_TURN_MOVE],
		jnp.where(
			our["volatiles"][s.LOCKED_MOVE] > 0,
			our["volatiles"][s.LOCKED_MOVE],
			our["moves"][ours, jnp.minimum(own_previous_action, s.MOVE_SLOTS - 1)],
		),
	)
	foe_move = jnp.where(foe.charging, charge, moves[theirs, queue_action])
	stage = jnp.where(
		~mid,
		s.TURN_IDLE,
		jnp.where(
			own_pivot | foe_pivot,
			jnp.where(own_queued | foe_queued, s.TURN_MOVES, s.TURN_RESIDUAL),
			s.TURN_REPLACEMENT,
		),
	)
	e = obs.events
	outcome = OUTCOME_MISS * e.missed + OUTCOME_FAILED * e.failed + OUTCOME_CRIT * e.critical
	events = jnp.stack(
		(
			e.order,
			e.cant,
			outcome,
			e.effectiveness,
			e.hits,
			e.entered,
			jnp.rint(100 * e.hp_change),
			e.status,
			e.fainted,
		),
		axis=1,
	).astype(jnp.int8)
	last_success = jnp.where(e.order[0] > e.order[1], e.move[0], e.move[1])
	hazard_order = history.hazard_order
	full = s.BattleState(
		**own_fields,
		weather=obs.weather.astype(jnp.int8),
		weather_turns=jnp.where(
			(obs.weather > 0) & ~history.weather_permanent, jnp.maximum(1, 5 - obs.weather_age), 0
		).astype(jnp.int8),
		weather_age=obs.weather_age.astype(jnp.int8),
		trick_room=jnp.where(obs.trick_room, jnp.maximum(1, 5 - obs.trick_room_age), 0).astype(
			jnp.int8
		),
		trick_room_age=obs.trick_room_age.astype(jnp.int8),
		turn=jnp.rint(1000 * obs.turn).astype(jnp.int32),
		winner=view.result,
		last_success=jnp.where(mid, last_success, 0).astype(jnp.int16),
		request=pair(own_request, foe_request),
		turn_stage=stage.astype(jnp.int8),
		turn_first=jnp.where(foe_pivot, 1 - side, side).astype(jnp.int8),
		turn_actions=pair(
			jnp.minimum(own_previous_action, s.MOVE_SLOTS - 1).astype(jnp.int8), queue_action
		),
		turn_moves=pair(our_move, foe_move),
		turn_pending=pair(own_queued, foe_queued),
		hazards=pair(*obs.hazards.astype(jnp.int8)),
		hazard_order=pair(*hazard_order.astype(jnp.int8)),
		event=pair(*events),
		event_move=pair(*e.move),
		wish_age=pair(*obs.wish_age.astype(jnp.int8)),
		healing_wish=pair(*bool_array(obs.healing_wish)),
	)
	base, blocked = action_masks(full, tables)
	reconstructed = observe(full, side, tables)
	valid = jnp.all(
		jnp.array(
			jax.tree.leaves(
				jax.tree.map(
					lambda expected, actual: _same_leaf(expected, actual, protocol),
					obs,
					reconstructed,
				)
			)
		)
	)
	valid &= ~our["trap_known"] | blocked[side] | (own_request != s.MOVE)
	# A waiting opponent's submitted choice is private. The head will draw it
	# when preparing lanes exactly as for an ordinary simultaneous request.
	return SearchState(full, pair(view.held, 0), pair(view.has_held, False), base, blocked), valid


def bool_array(value):
	return jnp.asarray(value, jnp.bool_)
