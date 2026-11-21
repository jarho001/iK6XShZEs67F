"""Focused checks for the engine defects that fresh recordings found after Phase 2.

Each scenario is a short battle built from explicit members and played through one compiled
step with forced outcomes (forced_from_records addresses, as in the recorder). The expected
values were checked against Showdown at the pinned commit.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pokejax.engine import state as S
from pokejax.engine.abilities import active_abilities
from pokejax.engine.data import build_engine_tables
from pokejax.engine.forms import sync_weather_formes
from pokejax.engine.items import showdown_item_consumed
from pokejax.engine.randomness import forced_from_records
from pokejax.engine.reset import Teams, reset
from pokejax.engine.step import step
from pokejax.tables import load_tables

KEY = jax.random.key(23)
STEP = jax.jit(step)
FILLER = {"species": "mew", "moves": ["tackle"]}


@pytest.fixture(scope="module")
def world():
	tables = load_tables()
	return tables, build_engine_tables(tables)


def make_teams(tables, side0, side1):
	"""Six members per side; unlisted slots are plain Mew. Members: species, ability, item, moves."""
	species = np.zeros((2, 6), np.int16)
	ability = np.zeros((2, 6), np.int16)
	item = np.zeros((2, 6), np.int16)
	moves = np.zeros((2, 6, 4), np.int16)
	for s, side in enumerate((side0, side1)):
		members = list(side) + [FILLER] * (6 - len(side))
		for slot, member in enumerate(members):
			species[s, slot] = tables.species.index[member["species"]]
			ability[s, slot] = tables.abilities.index[member.get("ability", "noability")]
			item[s, slot] = tables.items.index.get(member.get("item", ""), 0)
			for m, move in enumerate(member["moves"]):
				moves[s, slot, m] = tables.moves.index[move]
	full = (2, 6)
	return Teams(
		species=species,
		level=np.full(full, 80, np.uint8),
		moves=moves,
		ability=ability,
		item=item,
		nature=np.full(full, tables.natures.index["serious"], np.int8),
		gender=np.full(full, 1, np.int8),
		evs=np.full((*full, 6), 85, np.uint8),
		ivs=np.full((*full, 6), 31, np.uint8),
		happiness=np.full(full, 255, np.uint8),
	)


def start(world, side0, side1):
	tables, engine = world
	return reset(KEY, make_teams(tables, side0, side1), engine)


def turn(world, state, actions, records=()):
	_, engine = world
	forced = forced_from_records(list(records))
	return STEP(state, jnp.asarray(actions, jnp.int32), KEY, forced, engine)


def hit(side, value=1, occurrence=0):
	return {"kind": "hit", "side": side, "occurrence": occurrence, "value": value}


REGIGIGAS = {
	"species": "regigigas",
	"ability": "slowstart",
	"item": "leftovers",
	"moves": ["tackle"],
}


def slow_start(state):
	return [int(state.volatiles[side, S.SLOW_START]) for side in range(2)]


def test_trace_copying_slow_start_starts_its_counter(world):
	# Showdown: Trace's setAbility runs Slow Start's onStart, so the copy is halved for 5 turns.
	tables, _ = world
	state = start(
		world,
		[REGIGIGAS],
		[FILLER, {"species": "gardevoir", "ability": "trace", "moves": ["tackle"]}],
	)
	assert slow_start(state) == [5, 0]
	state = turn(world, state, [0, 4], [hit(0)])
	assert int(state.ability[1, 1]) == tables.abilities.index["slowstart"]
	# The lead has used up one turn, the entrant has not.
	assert slow_start(state) == [4, 5]
	state = turn(world, state, [0, 0], [hit(0), hit(1)])
	assert slow_start(state) == [3, 4]


def test_imposter_copying_slow_start_starts_its_counter(world):
	tables, _ = world
	state = start(
		world,
		[FILLER, {"species": "ditto", "ability": "imposter", "moves": ["transform"]}],
		[REGIGIGAS],
	)
	state = turn(world, state, [4, 0], [hit(1)])
	assert int(state.species[0, 1]) == tables.species.index["regigigas"]
	assert int(state.ability[0, 1]) == tables.abilities.index["slowstart"]
	assert slow_start(state) == [5, 4]


def test_transform_copying_slow_start_starts_its_counter(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "ditto", "ability": "limber", "moves": ["transform"]}],
		[REGIGIGAS],
	)
	assert slow_start(state) == [0, 5]
	state = turn(world, state, [0, 0], [hit(1)])
	assert int(state.ability[0, 0]) == tables.abilities.index["slowstart"]
	assert slow_start(state) == [4, 4]


def test_trace_of_another_ability_leaves_no_slow_start(world):
	state = start(
		world,
		[{"species": "gyarados", "ability": "intimidate", "moves": ["tackle"]}],
		[FILLER, {"species": "gardevoir", "ability": "trace", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 4], [hit(0)])
	assert slow_start(state) == [0, 0]


KNOCKER = {"species": "mew", "moves": ["knockoff"]}


def test_knock_off_after_a_focus_sash_save_leaves_the_sash_consumed(world):
	# Showdown: the sash is used in the Damage event, so Knock Off's takeItem finds no item and
	# the holder reads as having consumed it (lastItem), not as knocked off.
	state = start(
		world,
		[KNOCKER],
		[
			{
				"species": "shedinja",
				"ability": "wonderguard",
				"item": "focussash",
				"moves": ["tackle"],
			}
		],
	)
	assert int(state.hp[1, 0]) == 1
	state = turn(world, state, [0, 0], [hit(0)])
	assert int(state.hp[1, 0]) == 1
	assert int(state.item[1, 0]) == 0
	assert bool(state.item_consumed[1, 0])
	assert bool(showdown_item_consumed(state)[1, 0])


def test_knock_off_removes_an_unused_focus_sash(world):
	# A sash that did not trigger (the holder is below full HP) is knocked off, not consumed.
	state = start(
		world,
		[KNOCKER],
		[{"species": "mew", "item": "focussash", "moves": ["tackle"]}],
	)
	state = state._replace(hp=state.hp.at[1, 0].set(100))
	state = turn(world, state, [0, 0], [hit(0)])
	assert 0 < int(state.hp[1, 0]) < 100
	assert int(state.item[1, 0]) == 0
	assert not bool(state.item_consumed[1, 0])
	assert not bool(showdown_item_consumed(state)[1, 0])


CHOICE_USER = {
	"species": "galvantula",
	"ability": "compoundeyes",
	"item": "choicespecs",
	"moves": ["bugbuzz", "transform"],
}
PARASECT = {"species": "parasect", "ability": "dryskin", "item": "leftovers", "moves": ["spore"]}


def test_transform_ends_the_choice_lock_of_a_move_the_user_no_longer_has(world):
	# Showdown: choicelock's DisableMove removes the volatile once hasMove(locked move) fails.
	tables, _ = world
	# The foe switches in first and takes no later move, so nothing else would clear the lock.
	state = start(world, [CHOICE_USER], [FILLER, PARASECT])
	state = turn(world, state, [1, 4])
	assert int(state.species[0, 0]) == tables.species.index["parasect"]
	assert int(state.volatiles[0, S.CHOICE_MOVE]) == 0
	assert int(state.item[0, 0]) == tables.items.index["choicespecs"]


def test_a_choice_lock_on_a_move_the_user_keeps_stays(world):
	tables, _ = world
	state = start(world, [CHOICE_USER], [PARASECT])
	state = turn(world, state, [0, 0], [hit(0), hit(1, value=0)])
	assert int(state.volatiles[0, S.CHOICE_MOVE]) == tables.moves.index["bugbuzz"]


def rec(kind, side, value=1, occurrence=0):
	return {"kind": kind, "side": side, "occurrence": occurrence, "value": value}


def roll(side, value=8, occurrence=0):
	return [rec("crit", side, 0, occurrence), rec("damage_roll", side, value, occurrence)]


def set_hp(state, side, slot, hp):
	return state._replace(hp=state.hp.at[side, slot].set(hp))


def set_volatile(state, side, column, value):
	return state._replace(volatiles=state.volatiles.at[side, column].set(value))


# Healing Wish


HEALING_USER = {"species": "gardevoir", "ability": "trace", "moves": ["healingwish", "tackle"]}


def test_healing_wish_fails_without_a_living_replacement(world):
	# Showdown onTryHit: with no ally to switch in, Healing Wish prints -fail and the user stays.
	state = start(world, [HEALING_USER], [FILLER])
	state = state._replace(hp=state.hp.at[0, 1:].set(0))
	state = turn(world, state, [0, 0], [hit(1)])
	assert int(state.hp[0, 0]) > 0
	assert not bool(state.healing_wish[0])
	assert int(state.request[0]) == S.MOVE


def test_healing_wish_with_a_replacement_still_faints_the_user(world):
	state = start(world, [HEALING_USER], [FILLER])
	state = turn(world, state, [0, 0], [hit(1)])
	assert int(state.hp[0, 0]) == 0
	assert bool(state.healing_wish[0])


# Substitute hits and secondary self effects


def test_a_self_boost_secondary_applies_when_the_hit_lands_on_a_substitute(world):
	# Showdown rolls secondaries with a null target for a substitute hit: the user's own
	# effects (Flame Charge's speed boost) still happen, the target's would not.
	state = start(
		world,
		[{"species": "reshiram", "moves": ["flamecharge"]}],
		[{"species": "kyurem", "moves": ["tackle"]}],
	)
	state = set_volatile(state, 1, S.SUBSTITUTE, 400)
	hp_before = int(state.hp[1, 0])
	state = turn(world, state, [0, 0], [hit(0), *roll(0), rec("secondary", 0, 1), hit(1)])
	assert int(state.hp[1, 0]) == hp_before
	assert int(state.volatiles[1, S.SUBSTITUTE]) < 400
	assert int(state.boosts[0, S.BOOST_SPE]) == 1


def test_a_target_secondary_does_not_apply_behind_a_substitute(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["thunderbolt"]}],
		[{"species": "kyurem", "moves": ["tackle"]}],
	)
	state = set_volatile(state, 1, S.SUBSTITUTE, 400)
	state = turn(world, state, [0, 0], [hit(0), *roll(0), rec("secondary", 0, 1), hit(1)])
	assert int(state.status[1, 0]) == S.HEALTHY


def test_rapid_spin_spins_off_a_substitute_hit(world):
	state = start(
		world,
		[{"species": "sandslash", "moves": ["rapidspin"]}],
		[{"species": "kyurem", "moves": ["tackle"]}],
	)
	state = set_volatile(state, 1, S.SUBSTITUTE, 400)
	state = set_volatile(state, 0, S.LEECH_SEED, 1)
	state = state._replace(hazards=state.hazards.at[0, S.SPIKES].set(2))
	state = turn(world, state, [0, 0], [hit(0), *roll(0), hit(1)])
	assert int(state.volatiles[1, S.SUBSTITUTE]) < 400
	assert int(state.hazards[0, S.SPIKES]) == 0
	assert int(state.volatiles[0, S.LEECH_SEED]) == 0


def test_counter_hits_a_substitute_in_front_of_its_foe(world):
	# Counter's target is scripted (the foe); the substitute takes the doubled damage.
	state = start(
		world,
		[{"species": "wobbuffet", "moves": ["counter"]}],
		[{"species": "mew", "moves": ["tackle"]}],
	)
	state = set_volatile(state, 1, S.SUBSTITUTE, 30)
	hp_before = int(state.hp[1, 0])
	state = turn(world, state, [0, 0], [hit(1), *roll(1), hit(0)])
	assert int(state.hp[1, 0]) == hp_before
	assert int(state.volatiles[1, S.SUBSTITUTE]) == 0


def test_an_absorbing_ability_takes_a_status_move_aimed_at_its_substitute(world):
	# TryHit (Sap Sipper) comes before the substitute stops the move.
	state = start(
		world,
		[{"species": "parasect", "moves": ["spore"]}],
		[{"species": "girafarig", "ability": "sapsipper", "moves": ["substitute"]}],
	)
	state = set_volatile(state, 1, S.SUBSTITUTE, 50)
	state = turn(world, state, [0, 0], [hit(0)])
	assert int(state.boosts[1, S.BOOST_ATK]) == 1
	assert int(state.status[1, 0]) == S.HEALTHY


# Magic Bounce


def test_magic_bounce_does_not_reflect_a_move_aimed_at_a_fainted_foe(world):
	# Espeon (4 HP, Life Orb) faints on its own Psychic; the later Toxic has no target.
	state = start(
		world,
		[{"species": "mew", "moves": ["toxic"]}],
		[
			{
				"species": "espeon",
				"ability": "magicbounce",
				"item": "lifeorb",
				"moves": ["psychic"],
			}
		],
	)
	state = set_hp(state, 1, 0, 4)
	records = [hit(1), *roll(1), rec("secondary", 1, 0), rec("bounce_hit", 1, 1)]
	state = turn(world, state, [0, 0], records)
	assert int(state.hp[1, 0]) == 0
	assert int(state.status[0, 0]) == S.HEALTHY
	assert int(state.status[1, 0]) == S.HEALTHY


# Replacement and switch-in order


def double_ko(world, state):
	"""Side 0's Explosion faints both leads (side 1's lead has 1 HP): both sides must replace."""
	state = set_hp(state, 1, 0, 1)
	state = turn(world, state, [0, 0], [rec("order", 0, 0), hit(0), *roll(0)])
	assert [int(r) for r in state.request] == [S.SWITCH, S.SWITCH]
	return state


EXPLODER = {"species": "mew", "moves": ["explosion"]}


def test_imposter_copies_the_new_foe_when_both_sides_replace_a_fainted_lead(world):
	# Both instaswitch actions run before any runSwitch, so the faster Ditto sees the new foe.
	tables, _ = world
	state = start(
		world,
		[EXPLODER, {"species": "shuckle", "moves": ["tackle"]}],
		[FILLER, {"species": "ditto", "ability": "imposter", "moves": ["transform"]}],
	)
	state = double_ko(world, state)
	state = turn(world, state, [4, 4])
	assert int(state.active[0]) == 1 and int(state.active[1]) == 1
	assert int(state.species[1, 1]) == tables.species.index["shuckle"]
	assert int(state.volatiles[1, S.TRANSFORMED]) == 1


def test_an_imposter_that_copies_after_its_foes_download_takes_the_boost(world):
	# Genesect's Download runs first, then Ditto's Imposter copies its boosts, then the copied
	# Download starts: Ditto ends at +2 Special Attack, Genesect at +1.
	state = start(
		world,
		[FILLER, {"species": "genesect", "ability": "download", "moves": ["tackle"]}],
		[FILLER, {"species": "ditto", "ability": "imposter", "moves": ["transform"]}],
	)
	state = turn(world, state, [4, 4], [rec("order", 0, 0)])
	assert [int(b) for b in state.boosts[:, S.BOOST_SPA]] == [1, 2]


def test_a_toxic_counter_is_not_reset_when_the_faster_entrants_hazard_ends_the_battle(world):
	# Showdown's fieldEvent stops at the faint that ends the battle, so the slower entrant's
	# status SwitchIn handler never runs: its Toxic counter stays at 3.
	state = start(
		world,
		[EXPLODER, {"species": "jolteon", "moves": ["tackle"]}],
		[FILLER, {"species": "shuckle", "moves": ["tackle"]}],
	)
	status = state.status.at[0, 1].set(S.TOXIC).at[1, 1].set(S.TOXIC)
	counter = state.status_counter.at[0, 1].set(3).at[1, 1].set(3)
	hp = state.hp.at[0, 2:].set(0).at[0, 1].set(18)
	state = state._replace(status=status, status_counter=counter, hp=hp)
	state = state._replace(hazards=state.hazards.at[0, S.SPIKES].set(3))
	state = double_ko(world, state)
	state = turn(world, state, [4, 4])
	assert int(state.winner) == S.SIDE1
	assert int(state.status_counter[1, 1]) == 3


def test_a_toxic_counter_restarts_at_zero_on_a_normal_switch_in(world):
	state = start(
		world,
		[EXPLODER, {"species": "jolteon", "moves": ["tackle"]}],
		[FILLER, {"species": "shuckle", "moves": ["tackle"]}],
	)
	status = state.status.at[0, 1].set(S.TOXIC).at[1, 1].set(S.TOXIC)
	counter = state.status_counter.at[0, 1].set(3).at[1, 1].set(3)
	state = state._replace(status=status, status_counter=counter)
	state = double_ko(world, state)
	state = turn(world, state, [4, 4])
	assert int(state.winner) == S.ONGOING
	assert int(state.status_counter[0, 1]) == 0
	assert int(state.status_counter[1, 1]) == 0


def test_a_leaving_flower_gift_holder_follows_the_faster_entrants_cloud_nine(world):
	tables, engine = world
	state = start(
		world,
		[{"species": "cherrim", "ability": "flowergift", "moves": ["tackle"]}, FILLER],
		[
			{"species": "ninjask", "moves": ["tackle"]},
			{"species": "golduck", "ability": "cloudnine", "moves": ["tackle"]},
		],
	)
	state = state._replace(
		weather=jnp.asarray(S.SUN, jnp.int8), weather_turns=jnp.asarray(5, jnp.int8)
	)
	state = sync_weather_formes(state, engine, active_abilities(state, engine))
	assert int(state.shown_species[0, 0]) == tables.species.index["cherrimsunshine"]
	state = turn(world, state, [4, 4])
	assert int(state.active[0]) == 1
	assert int(state.shown_species[0, 0]) == tables.species.index["cherrim"]


# Contact abilities, recoil and multi-hit moves


def test_a_contact_status_lands_before_recoil_faints_the_attacker(world):
	# Showdown: Flare Blitz damage, Static, Synchronize copying it back, and only then recoil.
	state = start(
		world,
		[{"species": "mew", "ability": "synchronize", "moves": ["flareblitz"]}],
		[{"species": "raichu", "ability": "static", "moves": ["tackle"]}],
	)
	state = set_hp(state, 0, 0, 1)
	records = [rec("order", 0, 0), hit(0), *roll(0), rec("secondary", 0, 0)]
	state = turn(world, state, [0, 0], [*records, rec("ability_chance", 1, 1)])
	assert int(state.hp[0, 0]) == 0
	assert int(state.status[1, 0]) == S.PARALYSIS


def test_a_multi_hit_move_stops_when_contact_damage_faints_the_attacker(world):
	# Tail Slap against Rough Skin and Rocky Helmet costs 84 HP a hit: a 100 HP user gets two.
	def damage_dealt(foe, attacker_hp):
		state = start(world, [{"species": "mew", "moves": ["tailslap"]}], [foe])
		state = set_hp(state, 0, 0, attacker_hp)
		records = [rec("order", 0, 0), hit(0), rec("hit_count", 0, 5)]
		for occurrence in range(5):
			records += roll(0, 8, occurrence)
		after = turn(world, state, [0, 0], records)
		return int(state.hp[1, 0]) - int(after.hp[1, 0])

	plain = {"species": "druddigon", "moves": ["tackle"]}
	spiky = {**plain, "ability": "roughskin", "item": "rockyhelmet"}
	assert damage_dealt(spiky, 100) * 5 == damage_dealt(plain, 291) * 2


# Destiny Bond


def test_destiny_bond_ends_when_its_own_move_is_stopped(world):
	state = start(world, [{"species": "mismagius", "moves": ["destinybond"]}], [FILLER])
	state = set_volatile(state, 0, S.DESTINY_BOND, 1)
	state = state._replace(status=state.status.at[0, 0].set(S.PARALYSIS))
	stopped = turn(world, state, [0, 0], [hit(1), rec("full_paralysis", 0, 1)])
	assert int(stopped.volatiles[0, S.DESTINY_BOND]) == 0
	used = turn(world, state, [0, 0], [hit(1), rec("full_paralysis", 0, 0)])
	assert int(used.volatiles[0, S.DESTINY_BOND]) == 1


def test_a_pursuit_ko_does_not_trigger_the_destiny_bond_of_the_leaving_pokemon(world):
	state = start(
		world,
		[{"species": "wobbuffet", "moves": ["destinybond"]}, FILLER],
		[{"species": "bisharp", "moves": ["pursuit"]}],
	)
	state = set_hp(state, 0, 0, 1)
	state = set_volatile(state, 0, S.DESTINY_BOND, 1)
	state = turn(world, state, [4, 0], [hit(1), *roll(1)])
	assert int(state.hp[0, 0]) == 0
	assert int(state.hp[1, 0]) > 0


def test_moxie_does_not_boost_or_show_when_destiny_bond_takes_the_attacker(world):
	state = start(
		world,
		[{"species": "honchkrow", "ability": "moxie", "moves": ["bravebird"]}],
		[{"species": "mismagius", "moves": ["destinybond"]}],
	)
	state = set_hp(state, 1, 0, 1)
	state = turn(world, state, [0, 0], [hit(0), *roll(0)])
	assert int(state.hp[0, 0]) == 0
	assert int(state.ability_public[0, 0]) == 0


# Abilities that only some holders reveal


def test_bad_dreams_is_not_shown_when_magic_guard_blocks_the_damage(world):
	tables, _ = world
	bad_dreams = {"species": "darkrai", "ability": "baddreams", "moves": ["nastyplot"]}
	for ability, shown in (("magicguard", False), ("noability", True)):
		state = start(
			world,
			[bad_dreams],
			[{"species": "reuniclus", "ability": ability, "moves": ["recover"]}],
		)
		state = state._replace(
			status=state.status.at[1, 0].set(S.SLEEP),
			status_counter=state.status_counter.at[1, 0].set(3),
		)
		hp_before = int(state.hp[1, 0])
		state = turn(world, state, [0, 0])
		revealed = int(state.ability_public[0, 0]) == tables.abilities.index["baddreams"]
		assert revealed == shown, ability
		assert (int(state.hp[1, 0]) == hp_before) == (not shown), ability


def test_poison_touch_poison_makes_the_receivers_synchronize_show(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "muk", "ability": "poisontouch", "moves": ["poisonjab"]}],
		[{"species": "umbreon", "ability": "synchronize", "moves": ["tackle"]}],
	)
	records = [hit(0), *roll(0), rec("secondary", 0, 0), rec("ability_chance", 0, 1, 1), hit(1)]
	state = turn(world, state, [0, 0], records)
	assert int(state.status[1, 0]) == S.POISON
	# Muk is a Poison type: the copy is refused, but the receiver's ability was used.
	assert int(state.status[0, 0]) == S.HEALTHY
	assert int(state.ability_public[1, 0]) == tables.abilities.index["synchronize"]


def test_yawn_against_insomnia_shows_the_ability_and_fails(world):
	tables, _ = world
	yawner = {"species": "uxie", "moves": ["yawn"]}
	state = start(
		world, [yawner], [{"species": "hypno", "ability": "insomnia", "moves": ["tackle"]}]
	)
	state = turn(world, state, [0, 0], [hit(1)])
	assert int(state.volatiles[1, S.YAWN]) == 0
	assert int(state.ability_public[1, 0]) == tables.abilities.index["insomnia"]
	state = start(world, [yawner], [FILLER])
	state = turn(world, state, [0, 0], [hit(1)])
	assert int(state.volatiles[1, S.YAWN]) > 0


def test_knock_off_against_sticky_hold_shows_the_ability_and_keeps_the_item(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "carnivine", "moves": ["knockoff"]}],
		[
			{
				"species": "accelgor",
				"ability": "stickyhold",
				"item": "leftovers",
				"moves": ["tackle"],
			}
		],
	)
	state = turn(world, state, [0, 0], [hit(0), *roll(0), hit(1)])
	assert int(state.item[1, 0]) == tables.items.index["leftovers"]
	assert int(state.ability_public[1, 0]) == tables.abilities.index["stickyhold"]


# Items and move choice


def test_a_phazed_pokemon_eats_its_berry_before_it_is_dragged_out(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["dragontail"]}],
		[{"species": "tropius", "item": "sitrusberry", "moves": ["tackle"]}, FILLER],
	)
	state = set_hp(state, 1, 0, 150)
	records = [hit(0), *roll(0), rec("drag_in", 1, 1), hit(1)]
	state = turn(world, state, [0, 0], records)
	assert int(state.active[1]) == 1
	assert int(state.item[1, 0]) == 0
	assert bool(state.item_consumed[1, 0])
	assert int(state.hp[1, 0]) > 150


def test_sheer_force_follows_an_encore_override_of_the_chosen_move(world):
	def damage_dealt(ability):
		state = start(
			world,
			[{"species": "mew", "ability": ability, "moves": ["earthquake", "rockslide"]}],
			[{"species": "mew", "moves": ["tackle"]}],
		)
		tables, _ = world
		state = set_volatile(state, 0, S.ENCORE_MOVE, tables.moves.index["rockslide"])
		state = set_volatile(state, 0, S.ENCORE_REMAINING, 3)
		records = [rec("order", 0, 0), hit(0), *roll(0), rec("secondary", 0, 0), hit(1)]
		after = turn(world, state, [0, 0], records)
		return int(state.hp[1, 0]) - int(after.hp[1, 0])

	assert damage_dealt("sheerforce") > damage_dealt("noability") * 1.2


# Residual order


def test_a_locked_move_still_locks_after_lum_berry_cures_the_yawn_sleep(world):
	# lockedmove has no residual order: it runs after Yawn's sleep, which Lum Berry cures at once.
	def play(item):
		state = start(
			world,
			[{"species": "salamence", "item": item, "moves": ["earthquake", "roost", "outrage"]}],
			[{"species": "relicanth", "moves": ["headsmash", "yawn", "waterfall"]}],
		)
		first = [rec("order", 0, 0), hit(0), *roll(0, 15), rec("lock_duration", 0, 1)]
		state = turn(world, state, [2, 1], first)
		assert int(state.volatiles[1 - 1, S.LOCKED_MOVE]) > 0
		second = [hit(0), *roll(0, 14), rec("sleep_duration", 0, 0)]
		return turn(world, state, [0, 2], second)

	state = play("lumberry")
	assert int(state.status[0, 0]) == S.HEALTHY
	assert int(state.volatiles[0, S.LOCKED_MOVE]) > 0
	state = play("")
	assert int(state.status[0, 0]) == S.SLEEP
	assert int(state.volatiles[0, S.LOCKED_MOVE]) == 0


# Second round: defects found by the seed 7 recordings


def test_a_trace_holder_fainted_by_a_hazard_does_not_copy(world):
	# fieldEvent skips the Start of a holder that Spikes already fainted.
	tables, _ = world
	state = start(
		world,
		[FILLER, {"species": "gardevoir", "ability": "trace", "moves": ["tackle"]}],
		[{"species": "gyarados", "ability": "intimidate", "moves": ["tackle"]}],
	)
	state = set_hp(state, 0, 1, 1)
	state = state._replace(hazards=state.hazards.at[0, S.SPIKES].set(3))
	state = turn(world, state, [4, 0], [hit(1)])
	assert int(state.hp[0, 1]) == 0
	assert int(state.ability[0, 1]) == tables.abilities.index["trace"]
	assert int(state.ability_public[0, 1]) == 0


def test_rough_skin_and_rocky_helmet_are_not_shown_against_a_magic_guard_attacker(world):
	tables, _ = world
	spiky = {
		"species": "druddigon",
		"ability": "roughskin",
		"item": "rockyhelmet",
		"moves": ["tackle"],
	}
	for ability, shown in (("magicguard", False), ("noability", True)):
		state = start(world, [{"species": "mew", "ability": ability, "moves": ["tackle"]}], [spiky])
		state = turn(world, state, [0, 0], [hit(0), *roll(0), hit(1)])
		assert (int(state.ability_public[1, 0]) == tables.abilities.index["roughskin"]) == shown
		assert bool(state.item_revealed[1, 0]) == shown


def test_light_ball_keeps_working_after_transform(world):
	# Showdown reads baseSpecies: a Transformed Pikachu is still a Pikachu for its Light Ball.
	def damage_dealt(item):
		state = start(
			world,
			[{"species": "pikachu", "item": item, "moves": ["transform", "gigadrain"]}],
			[{"species": "yanmega", "moves": ["tackle", "gigadrain"]}],
		)
		state = turn(world, state, [0, 0], [hit(1), *roll(1)])
		assert int(state.volatiles[0, S.TRANSFORMED]) == 1
		state = turn(world, state, [1, 0], [rec("order", 0, 0), hit(0), *roll(0), hit(1)])
		return int(state.hp[1, 0])

	assert damage_dealt("") > damage_dealt("lightball")


def test_weak_armor_triggers_on_each_hit_before_the_fatal_one(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "cinccino", "ability": "skilllink", "moves": ["tailslap"]}],
		[{"species": "vanilluxe", "ability": "weakarmor", "moves": ["tackle"]}],
	)
	state = set_hp(state, 1, 0, 120)
	records = [rec("order", 0, 0), hit(0), rec("hit_count", 0, 5)]
	for occurrence in range(5):
		records += roll(0, 8, occurrence)
	state = turn(world, state, [0, 0], records)
	assert int(state.hp[1, 0]) == 0
	assert int(state.ability_public[1, 0]) == tables.abilities.index["weakarmor"]


def test_a_phazing_move_still_drags_when_contact_damage_faints_its_user(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["circlethrow"]}],
		[
			{
				"species": "druddigon",
				"ability": "roughskin",
				"item": "rockyhelmet",
				"moves": ["tackle"],
			},
			FILLER,
		],
	)
	state = set_hp(state, 0, 0, 40)
	state = turn(world, state, [0, 0], [hit(0), *roll(0), rec("drag_in", 1, 1), hit(1)])
	assert int(state.hp[0, 0]) == 0
	assert int(state.active[1]) == 1


def test_leech_seed_damages_the_seeded_side_even_when_liquid_ooze_faints_the_seeder(world):
	state = start(
		world,
		[{"species": "tentacruel", "ability": "liquidooze", "moves": ["tackle"]}],
		[{"species": "tropius", "moves": ["tackle"]}],
	)
	state = set_volatile(state, 0, S.LEECH_SEED, 1)
	state = set_hp(state, 1, 0, 2)
	state = state._replace(hp=state.hp.at[1, 1:].set(0))
	hp_before = int(state.hp[0, 0])
	state = turn(world, state, [0, 0], [rec("order", 0, 0), hit(0, 0), hit(1, 0)])
	assert int(state.hp[1, 0]) == 0
	assert int(state.hp[0, 0]) < hp_before


def test_volt_absorb_takes_a_reflected_thunder_wave(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "lanturn", "ability": "voltabsorb", "moves": ["thunderwave"]}],
		[{"species": "espeon", "ability": "magicbounce", "moves": ["tackle"]}],
	)
	state = turn(world, state, [0, 0], [rec("bounce_hit", 1, 1), hit(1), *roll(1)])
	assert int(state.status[0, 0]) == S.HEALTHY
	assert int(state.ability_public[0, 0]) == tables.abilities.index["voltabsorb"]


def test_a_flame_body_burn_halves_the_later_hits_of_a_multi_hit_move(world):
	def damage_dealt(burn_on_hit):
		state = start(
			world,
			[{"species": "mew", "moves": ["tailslap"]}],
			[{"species": "magcargo", "ability": "flamebody", "moves": ["tackle"]}],
		)
		records = [rec("order", 0, 0), hit(0), rec("hit_count", 0, 5)]
		for occurrence in range(5):
			records += roll(0, 8, occurrence)
			chance_value = int(occurrence == burn_on_hit)
			records.append(rec("ability_chance", 1, chance_value, 2 * occurrence))
		after = turn(world, state, [0, 0], records)
		return int(state.hp[1, 0]) - int(after.hp[1, 0]), int(after.status[0, 0])

	plain, plain_status = damage_dealt(-1)
	burned, burned_status = damage_dealt(2)
	assert plain_status == S.HEALTHY and burned_status == S.BURN
	# Hits 3, 4 and 5 are halved (hit 3 itself lands before the burn).
	assert plain * 0.7 < burned < plain * 0.9


def test_a_dragged_in_imposter_takes_spikes_before_it_copies_a_flying_foe(world):
	state = start(
		world,
		[FILLER, {"species": "ditto", "ability": "imposter", "moves": ["transform"]}],
		[{"species": "skarmory", "moves": ["whirlwind"]}],
	)
	state = state._replace(hazards=state.hazards.at[0, S.SPIKES].set(1))
	before = int(state.hp[0, 1])
	state = turn(world, state, [0, 0], [hit(1), rec("drag_in", 0, 1)])
	assert int(state.active[0]) == 1
	assert int(state.hp[0, 1]) < before
	assert int(state.volatiles[0, S.TRANSFORMED]) == 1


# Third round: defects found by the seed 8 recordings


def test_a_berry_blocked_by_unnerve_goes_off_before_its_holder_leaves(world):
	# The Update after Aerodactyl's replacement lets the foe that has still to leave eat its berry.
	state = start(
		world,
		[{"species": "aerodactyl", "ability": "unnerve", "moves": ["tackle"]}, FILLER],
		[{"species": "linoone", "item": "sitrusberry", "moves": ["tackle"]}, FILLER],
	)
	state = set_hp(state, 1, 0, 100)
	state = turn(world, state, [4, 4])
	assert int(state.active[1]) == 1
	assert int(state.item[1, 0]) == 0
	assert bool(state.item_consumed[1, 0])
	assert int(state.hp[1, 0]) > 100


def test_lum_berry_cures_a_contact_status_before_the_fatigue_confusion_starts(world):
	state = start(
		world,
		[{"species": "mew", "item": "lumberry", "moves": ["outrage"]}],
		[{"species": "raichu", "ability": "static", "moves": ["tackle"]}],
	)
	state = set_volatile(state, 0, S.LOCKED_MOVE, world[0].moves.index["outrage"])
	state = set_volatile(state, 0, S.LOCKED_REMAINING, 1)
	state = set_volatile(state, 0, S.LOCKED_TRUE_DURATION, 1)
	records = [rec("order", 0, 0), hit(0), *roll(0), rec("ability_chance", 1, 1), hit(1)]
	state = turn(world, state, [0, 0], [*records, rec("confusion_duration", 0, 0)])
	assert int(state.status[0, 0]) == S.HEALTHY
	assert bool(state.item_consumed[0, 0])
	assert int(state.volatiles[0, S.CONFUSION]) > 0


def test_overcoat_does_not_stop_effect_spore_in_gen_5(world):
	state = start(
		world,
		[{"species": "mew", "ability": "overcoat", "moves": ["tackle"]}],
		[{"species": "vileplume", "ability": "effectspore", "moves": ["tackle"]}],
	)
	records = [rec("order", 0, 0), hit(0), *roll(0), rec("effect_spore", 1, 5, 0)]
	state = turn(world, state, [0, 0], [*records, rec("sleep_duration", 0, 1), hit(1)])
	assert int(state.status[0, 0]) == S.SLEEP


def test_rapid_spin_does_nothing_when_rocky_helmet_faints_its_user(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["rapidspin"]}],
		[{"species": "skarmory", "item": "rockyhelmet", "moves": ["tackle"]}],
	)
	state = set_hp(state, 0, 0, 5)
	state = state._replace(hazards=state.hazards.at[0, S.SPIKES].set(2))
	state = turn(world, state, [0, 0], [rec("order", 0, 0), hit(0), *roll(0), hit(1)])
	assert int(state.hp[0, 0]) == 0
	assert int(state.hazards[0, S.SPIKES]) == 2


def test_mummy_replaces_moxie_before_the_ko_can_boost(world):
	tables, _ = world
	state = start(
		world,
		[{"species": "salamence", "ability": "moxie", "moves": ["outrage"]}],
		[{"species": "cofagrigus", "ability": "mummy", "moves": ["tackle"]}],
	)
	state = set_hp(state, 1, 0, 1)
	state = turn(
		world, state, [0, 0], [rec("order", 0, 0), hit(0), *roll(0), rec("lock_duration", 0, 0)]
	)
	assert int(state.hp[1, 0]) == 0
	assert int(state.ability[0, 0]) == tables.abilities.index["mummy"]
	assert int(state.boosts[0, S.BOOST_ATK]) == 0


def test_color_change_ignores_struggle(world):
	state = start(
		world,
		[{"species": "mew", "moves": ["tackle"]}],
		[{"species": "kecleon", "ability": "colorchange", "moves": ["tackle"]}],
	)
	# Kecleon has already turned Grass; Struggle (typeless) must not turn it Normal again.
	grass = world[0].types.index["grass"]
	state = state._replace(types=state.types.at[1, 0].set(jnp.asarray([grass, -1], jnp.int8)))
	types = np.asarray(state.types[1, 0]).tolist()
	state = state._replace(pp=state.pp.at[0, 0].set(0))
	state = turn(world, state, [0, 0], [rec("order", 0, 0), *roll(0), hit(1)])
	assert np.asarray(state.types[1, 0]).tolist() == types


def test_an_imposter_that_copies_a_trace_holder_after_it_traced_takes_the_traced_ability(world):
	tables, _ = world
	state = start(
		world,
		[
			{"species": "ninjask", "moves": ["tackle"]},
			{"species": "porygon2", "ability": "trace", "moves": ["tackle"]},
		],
		[
			{"species": "leavanny", "ability": "swarm", "moves": ["tackle"]},
			{"species": "ditto", "ability": "imposter", "moves": ["transform"]},
		],
	)
	state = turn(world, state, [4, 4])
	swarm = tables.abilities.index["swarm"]
	assert int(state.ability[0, 1]) == swarm
	assert int(state.ability[1, 1]) == swarm
	assert int(state.ability_public[1, 1]) == tables.abilities.index["imposter"]


def test_a_sitrus_berry_heals_between_the_hits_of_a_multi_hit_move(world):
	# Showdown eats the berry in the Update after hit 1, so the later hits meet the healed HP:
	# a target that three hits would just faint survives with the berry.
	def play(item, hp):
		state = start(
			world,
			[{"species": "mew", "moves": ["tailslap"]}],
			[{"species": "tropius", "item": item, "moves": ["tackle"]}],
		)
		if hp is not None:
			state = set_hp(state, 1, 0, hp)
		records = [rec("order", 0, 0), hit(0), rec("hit_count", 0, 3)]
		for occurrence in range(3):
			records += roll(0, 8, occurrence)
		return state, turn(world, state, [0, 0], [*records, hit(1)])

	full, after = play("", None)
	total = int(full.hp[1, 0]) - int(after.hp[1, 0])
	assert total < int(full.stats[1, 0, 0]) // 2
	_, plain = play("", total - 1)
	assert int(plain.hp[1, 0]) == 0
	_, berry = play("sitrusberry", total - 1)
	assert int(berry.hp[1, 0]) > 0
	assert int(berry.item[1, 0]) == 0 and bool(berry.item_consumed[1, 0])


def test_harvest_restores_the_berry_it_ate_after_a_trick_took_another_item_away(world):
	# lastItem stays the Sitrus Berry although the Life Orb that Trick handed over was named later.
	tables, _ = world
	state = start(
		world,
		[{"species": "tropius", "ability": "harvest", "moves": ["tackle"]}],
		[{"species": "mew", "moves": ["tackle"]}],
	)
	sitrus, life_orb = tables.items.index["sitrusberry"], tables.items.index["lifeorb"]
	state = state._replace(
		item_consumed=state.item_consumed.at[0, 0].set(True),
		last_item=state.last_item.at[0, 0].set(sitrus),
		item_public=state.item_public.at[0, 0].set(life_orb),
		item_revealed=state.item_revealed.at[0, 0].set(True),
	)
	state = turn(
		world, state, [0, 0], [rec("order", 0, 0), hit(0, 0), hit(1, 0), rec("harvest", 0, 1)]
	)
	assert int(state.item[0, 0]) == sitrus


def test_an_imposter_that_copies_a_traced_slow_start_starts_its_counter(world):
	tables, _ = world
	state = start(
		world,
		[
			{"species": "ninjask", "moves": ["tackle"]},
			{"species": "gardevoir", "ability": "trace", "moves": ["tackle"]},
		],
		[REGIGIGAS, {"species": "ditto", "ability": "imposter", "moves": ["transform"]}],
	)
	state = turn(world, state, [4, 4])
	slowstart = tables.abilities.index["slowstart"]
	assert int(state.ability[0, 1]) == slowstart and int(state.ability[1, 1]) == slowstart
	assert slow_start(state) == [5, 5]
