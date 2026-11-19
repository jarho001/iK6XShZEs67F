"""Gen 5 singles ability rows, grouped by the events that consume them."""

import jax.numpy as jnp
import numpy as np

from .modifiers import chain, modify
from .move_data import read_move
from .state import (
	ATK,
	BURN,
	CONFUSION,
	DEF,
	EVASION,
	FLASH_FIRE,
	FREEZE,
	HAIL,
	HEALTHY,
	HP,
	PARALYSIS,
	POISON,
	RAIN,
	SAND,
	SLEEP,
	SMACKDOWN,
	SUN,
	TEAM_SLOTS,
	TOXIC,
)

SIDES = jnp.arange(2)

# Boolean hooks share columns; numeric hooks retain their event parameters.
(
	FLAGS,
	DEFENSE,
	ATTACK,
	SPEED,
	ACCURACY,
	BASE_POWER,
	FINAL_DAMAGE,
	STATUS,
	DROP,
	ABSORB_TYPE,
	ABSORB,
	WEATHER,
	WEATHER_IMMUNITY,
	START,
	RESIDUAL,
	CONTACT,
	DAMAGE_TAKEN,
	SWITCH_OUT,
	ORDER,
	TRAP,
	PP_COST,
	STATUS_EVENT,
	SLEEP_HOOK,
	ITEM_GUARD,
	BOOST_EVENT,
) = range(25)
(
	BREAKABLE,
	MOLD_BREAKER,
	LEVITATE,
	SCRAPPY,
	NO_GUARD,
	ADAPTABILITY,
	CRIT_BLOCK,
	CRIT_BONUS,
	IGNORE_BOOSTS,
	SHEER_FORCE,
	SKILL_LINK,
	SERENE_GRACE,
	SHIELD_DUST,
	OWN_TEMPO,
	INNER_FOCUS,
	OBLIVIOUS,
	CLOUD_NINE,
	UNNERVE,
	ROCK_HEAD,
	MAGIC_GUARD,
	TRUANT,
	MUMMY_IMMUNE,
	POWDER_IMMUNITY,
) = (1 << bit for bit in range(23))
ATTACK_DOUBLE, ATTACK_GUTS, ATTACK_HUSTLE = range(1, 4)
ATTACK_DEFEATIST, ATTACK_SLOW_START = range(4, 6)
ATTACK_BLAZE, ATTACK_OVERGROW, ATTACK_SWARM, ATTACK_TORRENT = range(6, 10)
DEFENSE_THICK_FAT, DEFENSE_MARVEL_SCALE = range(1, 3)
SPEED_QUICK_FEET = 5
SPEED_UNBURDEN, SPEED_SLOW_START = range(6, 8)
(
	ACCURACY_COMPOUND,
	ACCURACY_HUSTLE,
	ACCURACY_VICTORY,
	ACCURACY_TANGLED,
	ACCURACY_WONDER,
	ACCURACY_SAND,
	ACCURACY_HAIL,
) = range(1, 8)
(
	POWER_TECHNICIAN,
	POWER_PUNCH,
	POWER_RECOIL,
	POWER_ANALYTIC,
	POWER_TOXIC,
	POWER_DRY_SKIN,
) = range(1, 7)
DAMAGE_SNIPER, DAMAGE_TINTED, DAMAGE_FILTER, DAMAGE_MULTISCALE = range(1, 5)
ABSORB_HEAL, ABSORB_FLASH, ABSORB_WONDER, ABSORB_ATK, ABSORB_SPA, ABSORB_SPE = range(1, 7)
START_ANNOUNCE, START_INTIMIDATE, START_DOWNLOAD, START_FRISK, START_ANTICIPATION = range(1, 6)
(
	RESIDUAL_POISON_HEAL,
	RESIDUAL_RAIN_DISH,
	RESIDUAL_ICE_BODY,
	RESIDUAL_DRY_SKIN,
	RESIDUAL_HYDRATION,
	RESIDUAL_SHED_SKIN,
	RESIDUAL_SPEED_BOOST,
	RESIDUAL_BAD_DREAMS,
	RESIDUAL_SLOW_START,
	RESIDUAL_HARVEST,
) = range(1, 11)
(
	CONTACT_STATIC,
	CONTACT_FLAME_BODY,
	CONTACT_EFFECT_SPORE,
	CONTACT_ROUGH_SKIN,
	CONTACT_IRON_BARBS,
	CONTACT_POISON_TOUCH,
	CONTACT_MUMMY,
	CONTACT_CURSED_BODY,
	CONTACT_COLOR_CHANGE,
	CONTACT_WEAK_ARMOR,
	CONTACT_JUSTIFIED,
	CONTACT_AFTERMATH,
	CONTACT_LIQUID_OOZE,
) = range(1, 14)
DAMAGE_TAKEN_STURDY = 1
SWITCH_OUT_REGENERATOR, SWITCH_OUT_NATURAL_CURE = range(1, 3)
ORDER_PRANKSTER = 1
TRAP_ARENA, TRAP_SHADOW, TRAP_MAGNET = range(1, 4)
PP_COST_PRESSURE = 1
STATUS_EVENT_SYNCHRONIZE = 1
SLEEP_EARLY_BIRD = 1
ITEM_GUARD_STICKY_HOLD = 1
BOOST_EVENT_DEFIANT, BOOST_EVENT_MOXIE = range(1, 3)
DROP_ALL = 127
DOUBLE_STAGES = 128
REVERSE_STAGES = 256
STATUS_SUN = 1 << 7


def build_ability_hooks(tables):
	"""Each exported ability has a row. Unimplemented and singles-inert hooks are zero."""
	rows = np.zeros((len(tables.abilities.ids), BOOST_EVENT + 1), np.int32)
	rows[:, ABSORB_TYPE] = -1

	def set_hook(column, value, names):
		for name in names.split():
			rows[tables.abilities.index[name], column] |= value

	for bit, names in (
		(MOLD_BREAKER, "moldbreaker teravolt turboblaze"),
		(LEVITATE, "levitate"),
		(SCRAPPY, "scrappy"),
		(NO_GUARD, "noguard"),
		(ADAPTABILITY, "adaptability"),
		(CRIT_BLOCK, "battlearmor shellarmor"),
		(CRIT_BONUS, "superluck"),
		(IGNORE_BOOSTS, "unaware"),
		(SHEER_FORCE, "sheerforce"),
		(SKILL_LINK, "skilllink"),
		(SERENE_GRACE, "serenegrace"),
		(SHIELD_DUST, "shielddust"),
		(OWN_TEMPO, "owntempo"),
		(INNER_FOCUS, "innerfocus"),
		(OBLIVIOUS, "oblivious"),
		(CLOUD_NINE, "cloudnine airlock"),
		(UNNERVE, "unnerve"),
		(ROCK_HEAD, "rockhead"),
		(MAGIC_GUARD, "magicguard"),
		(TRUANT, "truant"),
		(MUMMY_IMMUNE, "multitype"),
		(POWDER_IMMUNITY, "overcoat"),
		(
			BREAKABLE,
			(
				"battlearmor bigpecks clearbody contrary dryskin filter flashfire hypercutter "
				"immunity innerfocus insomnia keeneye leafguard levitate lightningrod limber magmaarmor "
				"marvelscale motordrive multiscale oblivious owntempo sandveil sapsipper shellarmor "
				"shielddust simple snowcloak solidrock stormdrain tangledfeet thickfat unaware "
				"vitalspirit voltabsorb waterabsorb waterveil wonderguard wonderskin sturdy stickyhold"
			),
		),
	):
		set_hook(FLAGS, bit, names)
	for column, entries in (
		(
			ATTACK,
			(
				(ATTACK_DOUBLE, "hugepower purepower"),
				(ATTACK_GUTS, "guts"),
				(ATTACK_HUSTLE, "hustle"),
				(ATTACK_DEFEATIST, "defeatist"),
				(ATTACK_SLOW_START, "slowstart"),
				(ATTACK_BLAZE, "blaze"),
				(ATTACK_OVERGROW, "overgrow"),
				(ATTACK_SWARM, "swarm"),
				(ATTACK_TORRENT, "torrent"),
			),
		),
		(DEFENSE, ((DEFENSE_THICK_FAT, "thickfat"), (DEFENSE_MARVEL_SCALE, "marvelscale"))),
		(
			SPEED,
			(
				(RAIN, "swiftswim"),
				(SUN, "chlorophyll"),
				(SAND, "sandrush"),
				(SPEED_QUICK_FEET, "quickfeet"),
				(SPEED_UNBURDEN, "unburden"),
				(SPEED_SLOW_START, "slowstart"),
			),
		),
		(
			ACCURACY,
			(
				(ACCURACY_COMPOUND, "compoundeyes"),
				(ACCURACY_HUSTLE, "hustle"),
				(ACCURACY_VICTORY, "victorystar"),
				(ACCURACY_TANGLED, "tangledfeet"),
				(ACCURACY_WONDER, "wonderskin"),
				(ACCURACY_SAND, "sandveil"),
				(ACCURACY_HAIL, "snowcloak"),
			),
		),
		(
			BASE_POWER,
			(
				(POWER_TECHNICIAN, "technician"),
				(POWER_PUNCH, "ironfist"),
				(POWER_RECOIL, "reckless"),
				(POWER_ANALYTIC, "analytic"),
				(POWER_TOXIC, "toxicboost"),
				(POWER_DRY_SKIN, "dryskin"),
			),
		),
		(
			FINAL_DAMAGE,
			(
				(DAMAGE_SNIPER, "sniper"),
				(DAMAGE_TINTED, "tintedlens"),
				(DAMAGE_FILTER, "filter solidrock"),
				(DAMAGE_MULTISCALE, "multiscale"),
			),
		),
		(
			STATUS,
			(
				(1 << BURN, "waterveil"),
				((1 << POISON) | (1 << TOXIC), "immunity"),
				(1 << PARALYSIS, "limber"),
				(1 << SLEEP, "insomnia vitalspirit"),
				(1 << FREEZE, "magmaarmor"),
				(STATUS_SUN, "leafguard"),
			),
		),
		(
			DROP,
			(
				(DROP_ALL, "clearbody"),
				(1 << (ATK - 1), "hypercutter"),
				(1 << (DEF - 1), "bigpecks"),
				(1 << (EVASION - 1), "keeneye"),
				(DOUBLE_STAGES, "simple"),
				(REVERSE_STAGES, "contrary"),
			),
		),
		(
			WEATHER,
			((RAIN, "drizzle"), (SUN, "drought"), (SAND, "sandstream"), (HAIL, "snowwarning")),
		),
		(
			WEATHER_IMMUNITY,
			((1 << SAND, "sandrush sandveil overcoat"), (1 << HAIL, "snowcloak overcoat")),
		),
		(
			START,
			(
				(
					START_ANNOUNCE,
					"pressure moldbreaker teravolt turboblaze cloudnine airlock unnerve slowstart",
				),
				(START_INTIMIDATE, "intimidate"),
				(START_DOWNLOAD, "download"),
				(START_FRISK, "frisk"),
				(START_ANTICIPATION, "anticipation"),
			),
		),
	):
		for value, names in entries:
			set_hook(column, value, names)
	for column, entries in (
		(
			RESIDUAL,
			(
				(RESIDUAL_POISON_HEAL, "poisonheal"),
				(RESIDUAL_RAIN_DISH, "raindish"),
				(RESIDUAL_ICE_BODY, "icebody"),
				(RESIDUAL_DRY_SKIN, "dryskin"),
				(RESIDUAL_HYDRATION, "hydration"),
				(RESIDUAL_SHED_SKIN, "shedskin"),
				(RESIDUAL_SPEED_BOOST, "speedboost"),
				(RESIDUAL_BAD_DREAMS, "baddreams"),
				(RESIDUAL_SLOW_START, "slowstart"),
				(RESIDUAL_HARVEST, "harvest"),
			),
		),
		(
			CONTACT,
			(
				(CONTACT_STATIC, "static"),
				(CONTACT_FLAME_BODY, "flamebody"),
				(CONTACT_EFFECT_SPORE, "effectspore"),
				(CONTACT_ROUGH_SKIN, "roughskin"),
				(CONTACT_IRON_BARBS, "ironbarbs"),
				(CONTACT_POISON_TOUCH, "poisontouch"),
				(CONTACT_MUMMY, "mummy"),
				(CONTACT_CURSED_BODY, "cursedbody"),
				(CONTACT_COLOR_CHANGE, "colorchange"),
				(CONTACT_WEAK_ARMOR, "weakarmor"),
				(CONTACT_JUSTIFIED, "justified"),
				(CONTACT_AFTERMATH, "aftermath"),
				(CONTACT_LIQUID_OOZE, "liquidooze"),
			),
		),
		(DAMAGE_TAKEN, ((DAMAGE_TAKEN_STURDY, "sturdy"),)),
		(
			SWITCH_OUT,
			((SWITCH_OUT_REGENERATOR, "regenerator"), (SWITCH_OUT_NATURAL_CURE, "naturalcure")),
		),
		(ORDER, ((ORDER_PRANKSTER, "prankster"),)),
		(
			TRAP,
			((TRAP_ARENA, "arenatrap"), (TRAP_SHADOW, "shadowtag"), (TRAP_MAGNET, "magnetpull")),
		),
		(PP_COST, ((PP_COST_PRESSURE, "pressure"),)),
		(STATUS_EVENT, ((STATUS_EVENT_SYNCHRONIZE, "synchronize"),)),
		(SLEEP_HOOK, ((SLEEP_EARLY_BIRD, "earlybird"),)),
		(ITEM_GUARD, ((ITEM_GUARD_STICKY_HOLD, "stickyhold"),)),
		(BOOST_EVENT, ((BOOST_EVENT_DEFIANT, "defiant"), (BOOST_EVENT_MOXIE, "moxie"))),
	):
		for value, names in entries:
			set_hook(column, value, names)
	rows[tables.abilities.index["icebody"], WEATHER_IMMUNITY] |= 1 << HAIL
	for kind, type_name, names in (
		(ABSORB_HEAL, "water", "waterabsorb dryskin"),
		(ABSORB_HEAL, "electric", "voltabsorb"),
		(ABSORB_FLASH, "fire", "flashfire"),
		(ABSORB_ATK, "grass", "sapsipper"),
		(ABSORB_SPA, "electric", "lightningrod"),
		(ABSORB_SPA, "water", "stormdrain"),
		(ABSORB_SPE, "electric", "motordrive"),
		(ABSORB_WONDER, None, "wonderguard"),
	):
		set_hook(ABSORB, kind, names)
		if type_name:
			for name in names.split():
				rows[tables.abilities.index[name], ABSORB_TYPE] = tables.types.index[type_name]
	for name in ("noability", ""):
		if name in tables.abilities.index:
			rows[tables.abilities.index[name], ABSORB_TYPE] = 0
	return jnp.asarray(rows)


def active_abilities(state, tables):
	return tables.ability_hooks[state.ability[SIDES, state.active]]


def has_hook(rows, flag):
	return (rows[..., FLAGS] & flag) != 0


def move_abilities(rows, side):
	"""Mold Breaker suppresses only the opposing holder's breakable hooks."""
	ignore = has_hook(rows[side], MOLD_BREAKER) & has_hook(rows[1 - side], BREAKABLE)
	return jnp.where(((SIDES != side) & ignore)[:, None], 0, rows)


def suppresses_weather(rows, alive):
	return jnp.any(has_hook(rows, CLOUD_NINE) & alive)


def effective_weather(state, rows):
	suppressed = suppresses_weather(rows, state.hp[SIDES, state.active] > 0)
	return jnp.where(suppressed, 0, state.weather)


def reveal_abilities(state, sides):
	"""Remember the last ability publicly named for each fixed team member."""
	held = state.ability[SIDES, state.active]
	slots = jnp.arange(TEAM_SLOTS)[None, :] == state.active[:, None]
	return state._replace(
		ability_public=jnp.where(slots & sides[:, None], held[:, None], state.ability_public)
	)


def status_immunity(rows, status, weather):
	mask = rows[..., STATUS]
	return ((mask & (1 << status)) != 0) | (((mask & STATUS_SUN) != 0) & (weather == SUN))


def boost_delta(delta, rows, external):
	hook = rows[..., DROP]
	scale = jnp.where(hook == DOUBLE_STAGES, 2, jnp.where(hook == REVERSE_STAGES, -1, 1))
	delta = delta * scale[..., None]
	blocked = ((hook[..., None] & (1 << jnp.arange(delta.shape[-1]))) != 0) & (delta < 0)
	return jnp.where(blocked & jnp.asarray(external)[..., None], 0, delta)


def apply_boost_delta(boosts, delta, rows, external):
	delta = boost_delta(delta, rows, external)
	defiant = (rows[:, BOOST_EVENT] == BOOST_EVENT_DEFIANT) & jnp.asarray(external)
	defiant &= jnp.any(delta < 0, axis=-1)
	atk = jnp.arange(boosts.shape[-1]) == 0
	delta += (defiant[:, None] & atk[None, :]).astype(delta.dtype) * 2
	return jnp.clip(boosts + delta, -6, 6).astype(boosts.dtype)


def berry_allowed(state, rows):
	return ~(has_hook(rows[::-1], UNNERVE) & (state.hp[SIDES, state.active][::-1] > 0))


def ability_cures(state, rows):
	"""OnUpdate cures conditions imposed while the immunity was bypassed."""
	status = state.status[SIDES, state.active]
	live = state.hp[SIDES, state.active] > 0
	status_cure = live & (status != HEALTHY) & ((rows[:, STATUS] & (1 << status)) != 0)
	confusion_cure = live & has_hook(rows, OWN_TEMPO) & (state.volatiles[:, CONFUSION] > 0)
	return status_cure, confusion_cure


def absorption(state, side, ability, type_, exponent, immune, enabled, move, tables):
	"""TryHit runs after natural type immunity and protection, before accuracy."""
	row = read_move(tables, move)
	foe = 1 - side
	kind = ability[ABSORB]
	user_move = row.move_target_foe
	eligible = enabled & user_move & ~immune & (state.hp[foe, state.active[foe]] > 0)
	wonder = (kind == ABSORB_WONDER) & (row.move_category != 2)
	wonder &= (row.id != tables.struggle_id) & (exponent <= 0)
	match = (kind > 0) & (kind != ABSORB_WONDER) & (ability[ABSORB_TYPE] == type_)
	return eligible & (match | wonder)


def absorb_hit(state, foe, row, absorbed, hp):
	kind = row[ABSORB]
	maximum = state.stats[foe, state.active[foe], HP].astype(jnp.int32)
	heal = absorbed & (kind == ABSORB_HEAL)
	hp = jnp.where(heal, jnp.minimum(hp + jnp.maximum(maximum // 4, 1), maximum), hp)
	boost_stat = jnp.where(kind == ABSORB_ATK, 0, jnp.where(kind == ABSORB_SPA, 2, 4))
	boost = absorbed & (kind >= ABSORB_ATK)
	delta = ((SIDES == foe)[:, None] & (jnp.arange(7) == boost_stat) & boost).astype(jnp.int8)
	boosts = jnp.clip(state.boosts + delta, -6, 6)
	flash = absorbed & (kind == ABSORB_FLASH)
	flash_mask = (SIDES == foe)[:, None] & (jnp.arange(state.volatiles.shape[1]) == FLASH_FIRE)
	volatiles = jnp.where(flash & flash_mask, 1, state.volatiles)
	state = state._replace(boosts=boosts, volatiles=volatiles)
	return state, hp


def grounded_ability(state, rows):
	return ~has_hook(rows, LEVITATE) | (state.volatiles[:, SMACKDOWN] > 0)


def attack_modifiers(
	rows,
	side,
	attack,
	physical,
	status,
	fire_or_ice,
	flash_fire,
	low_hp_half,
	low_hp_third,
	move_type,
	tables,
	slow_start,
):
	"""Showdown's ModifyAtk/ModifySpA ability hooks and chained modifiers."""
	hook = rows[side, ATTACK]
	guts = (hook == ATTACK_GUTS) & (status != HEALTHY)
	hustle = physical & (hook == ATTACK_HUSTLE)
	attack = modify(attack, jnp.where(hustle, 6144, 4096))
	threshold = (hook == ATTACK_BLAZE) & (move_type == tables.fire_type)
	threshold |= (hook == ATTACK_OVERGROW) & (move_type == tables.grass_type)
	threshold |= (hook == ATTACK_SWARM) & (move_type == tables.bug_type)
	threshold |= (hook == ATTACK_TORRENT) & (move_type == tables.water_type)
	thick_fat = (rows[1 - side, DEFENSE] == DEFENSE_THICK_FAT) & fire_or_ice
	defeatist = (hook == ATTACK_DEFEATIST) & low_hp_half
	slow = (hook == ATTACK_SLOW_START) & slow_start & physical
	modifier = jnp.where(
		physical & (hook == ATTACK_DOUBLE),
		8192,
		jnp.where((physical & guts) | flash_fire | (threshold & low_hp_third), 6144, 4096),
	)
	modifier = chain(modifier, jnp.where(defeatist | slow, 2048, 4096))
	return attack, chain(jnp.where(thick_fat, 2048, 4096), modifier)


def accuracy_modifier(rows, side, physical, target_confused, weather):
	source, target = rows[side, ACCURACY], rows[1 - side, ACCURACY]
	attacker = jnp.where(
		source == ACCURACY_COMPOUND,
		5325,
		jnp.where(
			source == ACCURACY_VICTORY,
			4506,
			jnp.where((source == ACCURACY_HUSTLE) & physical, 3277, 4096),
		),
	)
	evade = (target == ACCURACY_SAND) & (weather == SAND)
	evade |= (target == ACCURACY_HAIL) & (weather == HAIL)
	defender = jnp.where(
		(target == ACCURACY_TANGLED) & target_confused,
		2048,
		jnp.where(evade, 3277, 4096),
	)
	return chain(attacker, defender)


def base_power_modifier(
	rows,
	side,
	move,
	power,
	modifier,
	physical,
	status,
	fire,
	sheer,
	analytic,
	tables,
):
	row = read_move(tables, move)
	source = rows[side, BASE_POWER]
	dry_skin = (rows[1 - side, BASE_POWER] == POWER_DRY_SKIN) & fire
	reckless = (source == POWER_RECOIL) & (
		((row.move_recoil[0] > 0) & (row.id != tables.struggle_id)) | row.move_crash_damage
	)
	fist = (source == POWER_PUNCH) & row.move_punch
	toxic = (source == POWER_TOXIC) & physical & ((status == POISON) | (status == TOXIC))
	analytic &= source == POWER_ANALYTIC
	# Higher-priority ability hooks come before Dry Skin, then move/item hooks.
	boost = jnp.where(
		sheer | analytic, 5325, jnp.where(fist | reckless, 4915, jnp.where(toxic, 6144, 4096))
	)
	technician = (source == POWER_TECHNICIAN) & (power <= 60)
	boost = jnp.where(technician, 6144, boost)
	return chain(chain(boost, jnp.where(dry_skin, 5120, 4096)), modifier)


def final_damage_modifier(rows, side, critical, exponent, full_hp, item_modifier):
	source, target = rows[side, FINAL_DAMAGE], rows[1 - side, FINAL_DAMAGE]
	attack = jnp.where(
		(source == DAMAGE_SNIPER) & critical,
		6144,
		jnp.where((source == DAMAGE_TINTED) & (exponent < 0), 8192, 4096),
	)
	defense = jnp.where(
		(target == DAMAGE_MULTISCALE) & full_hp,
		2048,
		jnp.where((target == DAMAGE_FILTER) & (exponent > 0), 3072, 4096),
	)
	return chain(chain(attack, defense), item_modifier)


def skip_burn(row):
	return row[ATTACK] == ATTACK_GUTS
