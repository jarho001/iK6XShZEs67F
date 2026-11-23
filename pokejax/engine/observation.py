"""Policy input at a decision point, for one battle and one perspective.

Every field is what that player can know at the decision point: its own request (team, stats,
moves, PP, item, ability, HP, status and the legal mask), the protocol lines it has received
(its own split lines included), and the rules that Showdown's protocol shows. Each field's
source is named where it is defined. Private durations and counts (sleep, confusion, Toxic,
Encore, Taunt, Disable, Yawn and lock counters, Substitute HP, Wish HP, Trick Room, weather
and Wish remaining turns, an Illusion disguise's slot) never reach this function; the public
facts they gate do, as flags (remaining > 0 is announced by the -start and -end lines).

Own members retain fixed team slots for actions.bench_slots. Opponent members
are packed in first-appearance order, followed by zero rows with unseen=1.
Revealed opponent moves are sorted by descending ID with trailing zeros;
even the owner's move-slot positions are private. No private slot indices escape.

IDs are int16, with 0 reserved for padding (type IDs are engine IDs + 1). Enumerated codes
(events.py) are int16 with 0 as "none". All other features are float32: level / 100, stored
stats / 1000, boosts / 6, turn / 1000; HP and PP are fractions, flags and requests are 0/1,
counts are raw.

Pinned Showdown config/formats.ts:4609-4613 enables HP Percentage Mod for
gen5randombattle; data/rulesets.ts:1336-1340 sets reportPercentages.
sim/pokemon.ts:2096-2102 uses ceil(100*hp/maxhp), capped at 99 if damaged;
:2082 reports zero on faint. We expose that integer percentage divided by 100.

Types follow the protocol: a species' types, changed by -start typechange (Color Change),
by -transform (Transform, Imposter), and by forme changes that name the species. Roost's
Flying removal is not announced (only -singleturn move: Roost is), so it is not shown.

reference/player-view.cjs builds this same observation from one player's requests and protocol
lines in Node, and pokejax/obs_diff.py compares the two at every decision point of recorded
battles. What that check settled, beyond the fields above:

- A member off the field shows the HP it had when it left (hp_seen): Regenerator's heal prints
  nothing. A fainted member shows no status ("0 fnt").
- How an item left is announced: -enditem with [eat], a used Gem or a Sash is use (item_consumed),
  Knock Off, Trick, Switcheroo and a stolen berry are taken (item_knocked_off); getting an item
  again clears both (item_gone).
- The legal-action mask leaves out Shadow Tag, Arena Trap and Magnet Pull: the request says
  maybeTrapped whenever the foe's species could have one, so the player cannot know. A switch
  choice that such a trap forbids is refused and the side is asked again with trapped: true;
  from then on its mask has no switches (BattleState.trap_known, set by the environment and
  cleared when either active changes). The foe's flag is never read.
- Own moves: the request's move IDs name Hidden Power's type (generation 5 shows no power) and
  Return's power, which OwnObservation carries as hidden_type and return_power, zero for other moves.
- A Transformed member keeps its own stats and base ability in its request (the copied ones are
  the foe's, and show only where the protocol names an ability); a foe's lock (Outrage) is never
  announced and is not observed; the own lock is the move request's single move without PP.
- An Illusion holder is shown at the level its switch-in line announced: the level of the member
  it copies while disguised, and the real level once the disguise breaks (the replace line). A
  silent switch-out keeps the announced level (BattleState.shown_level). Which roster entry a later
  switch-in belongs to is ambiguous while a holder is in play (the protocol names it only when
  the disguise breaks), so obs_diff leaves the opponent's roster arrays out from the first time
  a holder is on the field. The engine keeps one roster row per team member, while the protocol
  reader merges entries by name: a holder that enters under the name of a member already seen
  gives the engine one more row than the player sees (a remaining difference: it needs per-row
  storage of the revealed details, and obs_diff leaves those roster leaves out).
- At the last decision point (no request follows) HP of members that left the field in the last
  step and the stats after a forme change are what the last request said; obs_diff skips them.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from pokejax.mechanics.hidden_power import hidden_power

from .actions import move_availability, observed_actions
from .events import (
	EV_CANT,
	EV_EFFECT,
	EV_ENTRY,
	EV_FAINTED,
	EV_HITS,
	EV_HP,
	EV_ORDER,
	EV_OUTCOME,
	EV_STATUS,
	OUTCOME_CRIT,
	OUTCOME_FAILED,
	OUTCOME_MISS,
	protocol_percent,
)
from .move_data import read_move
from .state import (
	AUTOTOMIZE,
	CHOICE_MOVE,
	CONFUSION,
	CURSE,
	DESTINY_BOND,
	DISABLE_MOVE,
	DISABLE_REMAINING,
	ENCORE_MOVE,
	ENCORE_REMAINING,
	FLASH_FIRE,
	ILLUSION_SLOT,
	ITEM_TAKEN,
	ITEM_USED,
	LEECH_SEED,
	LOCKED_MOVE,
	MOVE,
	MUST_RECHARGE,
	NONE,
	ONGOING,
	PARTIAL_TRAP_SOURCE,
	PARTIALLY_TRAPPED,
	PROTECT,
	ROOST,
	SLOW_START,
	SUBSTITUTE,
	TAUNT,
	TEAM_SLOTS,
	TRANSFORMED,
	TURN_IDLE,
	TWO_TURN_MOVE,
	YAWN,
)
from .variable_moves import BP_RETURN, TYPE_HIDDEN_POWER


class OwnObservation(NamedTuple):
	# Own request: species, team, stats, moves, PP, item, ability, HP and status.
	species: jax.Array  # int16 [6]
	types: jax.Array  # int16 [6, 2], current types (Roost's Flying removal not shown)
	moves: jax.Array  # int16 [6, 4]
	# The request's move IDs carry the details the owner sees: Hidden Power's type (gen 5 shows no
	# power) and Return's power. Zero for every other move.
	hidden_type: jax.Array  # int16 [6, 4], Hidden Power's type ID plus one
	return_power: jax.Array  # int16 [6, 4], Return's power from happiness
	level: jax.Array  # float32 [6]
	pp: jax.Array  # float32 [6, 4]
	hp: jax.Array  # float32 [6]
	status: jax.Array  # int16 [6], major status ID (0 means healthy)
	fainted: jax.Array  # float32 [6]
	is_active: jax.Array  # float32 [6]
	stats: jax.Array  # float32 [6, 6], hp/atk/def/spa/spd/spe
	boosts: jax.Array  # float32 [7], atk/def/spa/spd/spe/accuracy/evasion
	ability: jax.Array  # int16 [6]
	item: jax.Array  # int16 [6], current held item IDs
	item_consumed: jax.Array  # bool [6], gone by use (eaten, a Gem, a Sash)
	item_knocked_off: jax.Array  # bool [6], gone by Knock Off, Trick, Thief or a stolen berry
	# Active member's volatiles, from its own -start/-end/-activate lines. Durations are private.
	confused: jax.Array  # float32 scalar: -start confusion
	charging: jax.Array  # float32 scalar: -prepare (two-turn move), no duration
	recharging: jax.Array  # float32 scalar: cant recharge is owed
	locked: jax.Array  # float32 scalar: the move request offers only the locked move (Outrage)
	trapped: jax.Array  # float32 scalar: -activate partial trap, source still active
	substitute: jax.Array  # float32 scalar: -start Substitute (its HP is private)
	encore: jax.Array  # float32 scalar: -start Encore
	encore_move: jax.Array  # int16 scalar: the encored move (its last move at the -start)
	taunt: jax.Array  # float32 scalar: -start move: Taunt
	disable: jax.Array  # float32 scalar: -start Disable
	disabled_move: jax.Array  # int16 scalar: the move named by -start Disable
	leech_seed: jax.Array  # float32 scalar: -start move: Leech Seed
	yawn: jax.Array  # float32 scalar: -start move: Yawn
	curse: jax.Array  # float32 scalar: -start Curse (Ghost user)
	destiny_bond: jax.Array  # float32 scalar: -singlemove Destiny Bond, until the next move
	transformed: jax.Array  # float32 scalar: -transform
	flash_fire: jax.Array  # float32 scalar: -start ability: Flash Fire
	slow_start: jax.Array  # float32 scalar: -start ability: Slow Start, until -end
	autotomize: jax.Array  # float32 scalar: count of -start Autotomize lines (weight stages)
	protect: jax.Array  # float32 scalar: -singleturn move: Protect (this turn)
	roost: jax.Array  # float32 scalar: -singleturn move: Roost (this turn)
	# Own only: the request shows a Choice lock (other moves are disabled), so it is visible.
	choice_locked: jax.Array  # float32 scalar
	choice_move: jax.Array  # int16 scalar: the locked move, zero when not locked


class OpponentObservation(NamedTuple):
	# Revealed by the protocol (switch, -damage, -status, faint, move and item lines).
	species: jax.Array  # int16 [6], shown species (Illusion keeps its disguise)
	types: jax.Array  # int16 [6, 2], shown species types, the active's current types
	moves: jax.Array  # int16 [6, 4], revealed IDs descending, then zeros
	level: jax.Array  # float32 [6]
	hp: jax.Array  # float32 [6], protocol percentage / 100
	status: jax.Array  # int16 [6], zero for unseen members
	fainted: jax.Array  # float32 [6]
	is_active: jax.Array  # float32 [6]
	unseen: jax.Array  # float32 [6]
	boosts: jax.Array  # float32 [7], active only
	ability: jax.Array  # int16 [6]
	item: jax.Array  # int16 [6], revealed opponent item IDs; zero while hidden
	item_revealed: jax.Array  # float32 [6], protocol has named the item
	item_consumed: jax.Array  # float32 [6], visible after announced consumption
	item_knocked_off: jax.Array  # float32 [6], visible after announced removal
	# Active member's public volatiles, as for the own side (no choice lock: not announced).
	confused: jax.Array  # float32 scalar
	charging: jax.Array  # float32 scalar
	recharging: jax.Array  # float32 scalar
	trapped: jax.Array  # float32 scalar
	substitute: jax.Array  # float32 scalar
	encore: jax.Array  # float32 scalar
	encore_move: jax.Array  # int16 scalar
	taunt: jax.Array  # float32 scalar
	disable: jax.Array  # float32 scalar
	disabled_move: jax.Array  # int16 scalar
	leech_seed: jax.Array  # float32 scalar
	yawn: jax.Array  # float32 scalar
	curse: jax.Array  # float32 scalar
	destiny_bond: jax.Array  # float32 scalar
	transformed: jax.Array  # float32 scalar
	flash_fire: jax.Array  # float32 scalar
	slow_start: jax.Array  # float32 scalar
	autotomize: jax.Array  # float32 scalar
	protect: jax.Array  # float32 scalar
	roost: jax.Array  # float32 scalar


class EventObservation(NamedTuple):
	"""The last step's visible events (events.py), perspective-ordered: [own, opponent]."""

	order: jax.Array  # int16 [2]: 0 no move attempt, 1 first, 2 second (|move| / |cant| order)
	move: jax.Array  # int16 [2]: the last move used (|move|, called moves as shown), 0 none
	cant: jax.Array  # int16 [2]: why the move attempt did not start (|cant|, -activate)
	missed: jax.Array  # float32 [2]: -miss on this side's move
	failed: jax.Array  # float32 [2]: the move had no effect (events.py OUTCOME_FAILED)
	critical: jax.Array  # float32 [2]: -crit
	effectiveness: jax.Array  # int16 [2]: 0 none, 1 immune, 2 resisted, 3 neutral, 4 super
	hits: jax.Array  # float32 [2]: -hitcount of a multi-hit move
	entered: jax.Array  # int16 [2]: 0 none, 1 chosen, 2 dragged, 3 pivot, 4 replacement
	hp_change: jax.Array  # float32 [2]: change of the start-of-step active's percent / 100
	status: jax.Array  # int16 [2]: last major status inflicted on this side (-status)
	fainted: jax.Array  # float32 [2]: a Pokemon of this side fainted (faint)


class Observation(NamedTuple):
	own: OwnObservation
	opponent: OpponentObservation
	turn: jax.Array  # float32 scalar
	request: jax.Array  # float32 [3], NONE/MOVE/SWITCH
	mid_turn: jax.Array  # float32 scalar; saved turn work remains after this choice
	legal_actions: jax.Array  # bool [9], actions.observed_actions (hidden traps once discovered)
	weather: jax.Array  # int16 scalar; public WEATHER_NAMES ID (-weather lines)
	weather_age: jax.Array  # float32 scalar; turns since the weather started, from upkeep lines
	hazards: jax.Array  # float32 [own/opponent, Spikes/Toxic Spikes] layer counts (-sidestart)
	trick_room: jax.Array  # float32 scalar; Trick Room stands (-fieldstart / -fieldend)
	trick_room_age: jax.Array  # float32 scalar; turns since Trick Room started
	wish: jax.Array  # float32 [own/opponent]; a Wish is pending (its |move| line, until -heal)
	wish_age: jax.Array  # float32 [own/opponent]; turns since that Wish was used
	healing_wish: jax.Array  # float32 [own/opponent]; a Healing Wish waits for a switch-in
	events: EventObservation


def observe(state, side, engine_tables) -> Observation:
	"""Pure, branch-free single-perspective observation; jit/vmap externally."""
	foe = 1 - side
	order = jnp.stack((side, foe))

	def f(x):
		return x.astype(jnp.float32)

	def public_turn_state(member_side):
		source_side = 1 - member_side
		source_slot = state.volatiles[member_side, PARTIAL_TRAP_SOURCE].astype(jnp.int32) - 1
		source_active = source_slot == state.active[source_side]
		source_alive = state.hp[source_side, jnp.clip(source_slot, 0, TEAM_SLOTS - 1)] > 0
		trapped = (
			(state.volatiles[member_side, PARTIALLY_TRAPPED] > 0) & source_active & source_alive
		)
		return (
			f(state.volatiles[member_side, TWO_TURN_MOVE] > 0),
			f(state.volatiles[member_side, MUST_RECHARGE] > 0),
			f(state.volatiles[member_side, LOCKED_MOVE] > 0),
			f(trapped),
		)

	def active_volatiles(member_side):
		"""The active member's public volatiles, in the order of the observation fields."""
		v = state.volatiles[member_side]
		encore = v[ENCORE_REMAINING] > 0
		disable = v[DISABLE_REMAINING] > 0
		return (
			f(v[SUBSTITUTE] > 0),
			f(encore),
			jnp.where(encore, v[ENCORE_MOVE], 0).astype(jnp.int16),
			f(v[TAUNT] > 0),
			f(disable),
			jnp.where(disable, v[DISABLE_MOVE], 0).astype(jnp.int16),
			f(v[LEECH_SEED] > 0),
			f(v[YAWN] > 0),
			f(v[CURSE] > 0),
			f(v[DESTINY_BOND] > 0),
			f(v[TRANSFORMED] > 0),
			f(v[FLASH_FIRE] > 0),
			f(v[SLOW_START] > 0),
			f(v[AUTOTOMIZE]),
			f(v[PROTECT] > 0),
			f(v[ROOST] > 0),
		)

	request = jnp.where(state.winner == ONGOING, state.request[side], NONE)
	own_charging, own_recharging, own_locked, own_trapped = public_turn_state(side)
	# A hard lock shows in a move request (a single move without PP), and in no other.
	own_locked = own_locked * f(request == MOVE)
	# A foe's lock (Outrage) is never announced, only the moves that start it: not observed.
	foe_charging, foe_recharging, _foe_locked, foe_trapped = public_turn_state(foe)
	own_active = active_volatiles(side)
	foe_active = active_volatiles(foe)

	# Own types: the current types, with Roost's Flying removal undone (not announced).
	own_roosting = (state.volatiles[side, ROOST] > 0) & (
		jnp.arange(TEAM_SLOTS) == state.active[side]
	)
	own_types = jnp.where(own_roosting[:, None], state.roost_types[side], state.types[side])
	choice_active = move_availability(state, engine_tables)[2][side]
	# A Transformed member keeps its own stats in the request (baseStoredStats) and its own
	# ability is the copied one, which the protocol does not name: the request shows its base.
	active_member = jnp.arange(TEAM_SLOTS) == state.active[side]
	reverted = (state.volatiles[side, TRANSFORMED] > 0) & active_member
	own_saved = jnp.concatenate(
		(state.stats[side, :, :1], jnp.broadcast_to(state.saved_stats[side], (TEAM_SLOTS, 5))),
		axis=1,
	)
	own_stats = jnp.where(reverted[:, None], own_saved, state.stats[side])
	public_ability = state.ability_public[side]
	own_ability = jnp.where(
		reverted,
		jnp.where(public_ability != 0, public_ability, state.ability_base[side]),
		state.ability[side],
	)
	# Fainted members show no status: the request says '0 fnt'.
	own_status = jnp.where(state.hp[side] == 0, 0, state.status[side])

	own_moves = read_move(engine_tables, state.moves[side])
	hidden_kind, _ = hidden_power(state.ivs[side].astype(jnp.int32))
	member_hidden_type = engine_tables.hidden_power_type[hidden_kind]
	own_hidden_type = jnp.where(
		own_moves.move_type_rule == TYPE_HIDDEN_POWER, member_hidden_type[:, None] + 1, 0
	)
	returned = jnp.maximum(state.happiness[side].astype(jnp.int32) * 10 // 25, 1)
	own_return_power = jnp.where(own_moves.move_bp_rule == BP_RETURN, returned[:, None], 0)

	own = OwnObservation(
		state.species[side],
		own_types.astype(jnp.int16) + 1,
		state.moves[side],
		own_hidden_type.astype(jnp.int16),
		own_return_power.astype(jnp.int16),
		f(state.level[side]) / 100,
		f(state.pp[side]) / jnp.maximum(f(state.max_pp[side]), 1),
		f(state.hp[side]) / jnp.maximum(f(state.stats[side, :, 0]), 1),
		own_status.astype(jnp.int16),
		f(state.hp[side] == 0),
		f(active_member),
		f(own_stats) / 1000,
		f(state.boosts[side]) / 6,
		own_ability,
		state.item[side],
		state.item_gone[side] == ITEM_USED,
		state.item_gone[side] == ITEM_TAKEN,
		f(state.volatiles[side, CONFUSION] > 0),
		own_charging,
		own_recharging,
		own_locked,
		own_trapped,
		*own_active,
		f(choice_active),
		jnp.where(choice_active, state.volatiles[side, CHOICE_MOVE], 0).astype(jnp.int16),
	)
	# A public rank selects one private slot without exposing the slot itself.
	ranks = jnp.arange(1, TEAM_SLOTS + 1)
	match = state.seen_order[foe, :, None] == ranks[None, :]
	slots = jnp.argmax(match, axis=0)
	seen = jnp.any(match, axis=0)
	# The opponent sees the species it was last shown: a disguise until Illusion breaks.
	species = jnp.where(seen, state.shown_species[foe, slots], 0)
	# A member off the field shows the HP it had when it left (Regenerator's heal is silent).
	on_field = slots == state.active[foe]
	public_hp = jnp.where(on_field, state.hp[foe, slots], state.hp_seen[foe, slots])
	percent = protocol_percent(public_hp, state.stats[foe, slots, 0])
	item_revealed = seen & state.item_revealed[foe, slots]
	item_gone = state.item_gone[foe, slots]
	moves = jnp.where(state.revealed_moves[foe, slots] != 0, state.moves[foe, slots], 0)
	# Sorting by -id puts padding last; reverse sign restores descending IDs.
	moves = -jnp.sort(-moves, axis=-1)
	# The active's types follow the protocol (Color Change, Transform); a disguised holder shows
	# its disguise species' types. Roost's removal is not announced, so it is undone.
	foe_roosting = (state.volatiles[foe, ROOST] > 0) & (jnp.arange(TEAM_SLOTS) == state.active[foe])
	foe_live = jnp.where(foe_roosting[:, None], state.roost_types[foe], state.types[foe])
	disguise = state.volatiles[foe, ILLUSION_SLOT].astype(jnp.int32) - 1
	disguised = disguise >= 0
	live = on_field & ~disguised
	# Illusion Level Mod: a disguised holder is shown at the level of the member it copies, and the
	# level stays the announced one after a silent switch-out.
	shown_level = state.shown_level[foe, slots]
	shown_types = engine_tables.species_types[species]
	types = jnp.where(live[:, None], foe_live[slots], shown_types)
	opponent = OpponentObservation(
		species,
		jnp.where(seen[:, None], types.astype(jnp.int16) + 1, 0),
		jnp.where(seen[:, None], moves, 0),
		jnp.where(seen, f(shown_level) / 100, 0),
		jnp.where(seen, f(percent) / 100, 0),
		jnp.where(seen & (public_hp > 0), state.status[foe, slots], 0).astype(jnp.int16),
		f(seen & (public_hp == 0)),
		f(seen & on_field),
		f(~seen),
		f(state.boosts[foe]) / 6,
		jnp.where(seen, state.ability_public[foe, slots], 0),
		jnp.where(item_revealed, state.item_public[foe, slots], 0).astype(jnp.int16),
		f(item_revealed),
		f(item_revealed & (item_gone == ITEM_USED)),
		f(item_revealed & (item_gone == ITEM_TAKEN)),
		f(state.volatiles[foe, CONFUSION] > 0),
		foe_charging,
		foe_recharging,
		foe_trapped,
		*foe_active,
	)
	events = state.event[order]
	outcome = events[:, EV_OUTCOME]
	events_obs = EventObservation(
		order=events[:, EV_ORDER].astype(jnp.int16),
		move=state.event_move[order],
		cant=events[:, EV_CANT].astype(jnp.int16),
		missed=f((outcome & OUTCOME_MISS) != 0),
		failed=f((outcome & OUTCOME_FAILED) != 0),
		critical=f((outcome & OUTCOME_CRIT) != 0),
		effectiveness=events[:, EV_EFFECT].astype(jnp.int16),
		hits=f(events[:, EV_HITS]),
		entered=events[:, EV_ENTRY].astype(jnp.int16),
		hp_change=f(events[:, EV_HP]) / 100,
		status=events[:, EV_STATUS].astype(jnp.int16),
		fainted=f(events[:, EV_FAINTED]),
	)
	return Observation(
		own,
		opponent,
		f(state.turn) / 1000,
		f(jnp.arange(3) == request),
		f(state.turn_stage > TURN_IDLE),
		observed_actions(state, engine_tables)[side],
		state.weather.astype(jnp.int16),
		f(state.weather_age),
		f(state.hazards[order]),
		f(state.trick_room > 0),
		f(state.trick_room_age),
		f(state.wish_turns[order] > 0),
		f(state.wish_age[order]),
		f(state.healing_wish[order]),
		events_obs,
	)
