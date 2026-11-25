'use strict';

// Incremental engine inputs inferred solely from the same PlayerView as the policy.
const { Dex } = require('./vendor/pokemon-showdown/dist/sim/dex');

const STATS = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];
const REFS = /^p([12])[a-z]?: /;
const GENDER = { M: 1, F: 2 };
const STALL_GUARDS = new Set(['Protect', 'Endure', 'Quick Guard', 'Wide Guard']);
const id = value => (value || '').toLowerCase().replace(/[^a-z0-9]/g, '');
const pad = (values, size = 4) => Array.from({ length: size }, (_, i) => values[i] ?? 0);
const flags = (n, value = 0) => Array(n).fill(value);
const pokemonName = entry => entry.ident.split(': ')[1];

function stat(base, level, iv, ev, hp = false) {
	return Math.floor((2 * base + iv + Math.floor(ev / 4) + (hp ? 100 : 0)) * level / 100)
		+ (hp ? 10 : 5);
}

function generatorSpread(species, member, dex) {
	const hidden = member.hiddenTypes.find(Boolean);
	const type = [...dex.types].find(([, index]) => index + 1 === hidden)?.[0];
	const hpType = species.id === 'ditto' ? 'ice' : type;
	const hpIvs = hpType ? Dex.mod('gen5').types.get(hpType).HPivs : {};
	const ivs = STATS.map(name => hpIvs[name] ?? 31);
	const evs = flags(6, 85);
	for (const index of [1, 5]) {
		const name = STATS[index];
		if (stat(species.baseStats[name], member.level, ivs[index], 85) !== member.stats[name]) {
			evs[index] = 0;
			ivs[index] = hidden && species.id !== 'ditto' ? ivs[index] - 28 : 0;
		}
	}
	if (species.id !== 'shedinja') {
		while (evs[0] > 1 && stat(species.baseStats.hp, member.level, ivs[0], evs[0], true) > member.maxhp) {
			evs[0] -= 4;
		}
	}
	return { ivs, evs };
}

class OwnState {
	constructor(view) {
		this.view = view;
		view.ownTracker = this;
		this.members = new Map();
		this.faintPending = false;
		this.faintedLast = false;
		this.damage = [0, 0, 0];
		this.hit = false;
		this.wishHp = 0;
	}

	get side() {
		return this.view.sides[this.view.me];
	}

	initialize(request) {
		for (const [i, member] of this.side.members.entries()) {
			const entry = request.side.pokemon[i];
			const species = Dex.mod('gen5').species.get(member.baseSpecies);
			const ditto = species.id === 'ditto';
			const spread = generatorSpread(species, member, this.view.tables);
			const pirouette = Dex.mod('gen5').species.get('Meloetta-Pirouette').baseStats;
			this.members.set(member, {
				species: this.view.speciesIndex(species.id),
				types: this.types(this.view.speciesTypes(species.id)),
				spread,
				gender: GENDER[entry.details.split(', ').find(part => /^[MF]$/.test(part))] || 0,
				happiness: member.returnPowers.includes(1) ? 0 : 255,
				stats: [member.maxhp, ...STATS.slice(1).map(name => member.stats[name])],
				pirouette: STATS.slice(1).map((name, index) =>
					stat(pirouette[name], member.level, spread.ivs[index + 1], spread.evs[index + 1])),
				hpSeen: member.hp, seen: 0, shown: 0, shownLevel: 0, turns: 0,
				statusStart: 0, statusAge: 0, sleepFromFoe: false, alive: true,
				lastItem: 0, lastMove: 0, abilityPublic: 0, revealed: new Set(),
				baseMoves: ditto ? ['transform'] : [...member.moves],
			});
		}
	}

	request(request) {
		this.party = request.side.pokemon.map(entry =>
			this.side.members.findIndex(member => member.name === pokemonName(entry)));
		const active = this.side.active;
		if (this.side.vol.transformed && active.savedMoves?.join() === active.moves.join()) {
			// A Transform can happen before the first delivered request (Imposter).
			const data = this.members.get(active);
			active.savedMoves = data.baseMoves;
			active.savedPp = data.baseMoves.map(move => this.view.maxPp(move));
			active.savedMaxpp = [...active.savedPp];
		}
	}

	types(values) {
		return [values[0] ?? -1, values[1] ?? -1];
	}

	before(parts) {
		this.previous = this.side.active;
		this.hpBefore = this.previous?.hp;
		const own = Number(REFS.exec(parts[2])?.[1]) - 1 === this.view.me;
		if (own && ['switch', 'drag'].includes(parts[1]) && this.previous) {
			this.members.get(this.previous).lastMove = 0;
			this.timers = {};
			this.substitute = 0;
			this.stall = 0;
			this.lastProtect = -2;
			this.fakeOut = false;
			this.guard = null;
			this.savedRevealed = null;
			this.copied = null;
		}
	}

	protocol(parts) {
		const [, command, target, detail] = parts;
		const sideNumber = Number(REFS.exec(target)?.[1]) - 1;
		const own = sideNumber === this.view.me;
		const member = this.side.active;
		const data = this.members.get(member);
		if (command === 'upkeep') this.guard = null;
		if (command === 'turn') {
			this.damage = [0, 0, 0];
			this.hit = false;
			this.faintedLast = this.faintPending;
			this.faintPending = false;
			for (const [pokemon, value] of this.members) value.alive = pokemon.hp > 0;
			if (data) data.turns = Math.min(255, data.turns + 1);
		}
		if (!data) return;
		if (command === '-ability' && own) data.abilityPublic = this.view.index('abilities', member.ability);
		const source = parts.find(part => part.startsWith('[of] '))?.slice(5);
		const holder = ['-damage', '-status', '-start'].includes(command) && source ? source : target || source;
		const namedAbility = parts.slice(3).map(part => /^(?:\[from\]\s*)?ability: (.+)$/.exec(part))
			.find(Boolean)?.[1];
		if (Number(REFS.exec(holder)?.[1]) - 1 === this.view.me && namedAbility) {
			data.abilityPublic = this.view.index('abilities', id(namedAbility));
		}
		if (!own) return;
		const move = name => this.view.index('moves', id(name).replace(/^hiddenpower.*/, 'hiddenpower'));
		const name = detail?.replace(/^(move|ability): /, '');
		const from = parts.find(part => part.startsWith('[from]'));
		if (command === 'switch' || command === 'drag' || command === 'replace') {
			if (!data.seen) data.seen = [...this.members.values()].filter(value => value.seen).length + 1;
			data.shown = this.view.speciesIndex(id(detail.split(', ')[0]));
			data.shownLevel = Number(/, L(\d+)/.exec(detail)?.[1] || 100);
			if (command !== 'replace') {
				data.turns = 0;
				if (member.status === 'tox' || member.status === 'slp') data.statusAge = 0;
				this.timers = {};
				this.previous && (this.members.get(this.previous).hpSeen = this.hpBefore);
			}
		}
		if (command === 'move' || command === 'cant' || (command === '-activate' && name === 'confusion')) {
			this.fakeOut = true;
		}
		if (command === 'move' && (!from || from === '[from] move: Pursuit' || from === '[from] lockedmove')) {
			data.lastMove = move(detail);
		}
		if (command === 'move' && (!from || from === '[from] move: Pursuit')) {
			data.revealed.add(data.lastMove);
			if (name === 'Wish' && this.view.wish[this.view.me].usedTurn === this.view.turn) {
				this.wishHp = Math.floor(member.maxhp / 2);
			}
		}
		if (command === '-singleturn' && STALL_GUARDS.has(name)) {
			this.stall = this.lastProtect === this.view.turn - 1 ? Math.min(256, this.stall * 2) : 2;
			this.lastProtect = this.view.turn;
			this.guard = name;
		}
		if (command === '-transform') {
			const foe = this.view.sides[1 - this.view.me].active;
			const species = Dex.mod('gen5').species.get(this.view.currentSpecies(foe));
			this.copied = {
				stats: STATS.slice(1).map(name => stat(species.baseStats[name], foe.level, 31, 85)),
				ability: this.view.index('abilities', foe.ability),
			};
			this.savedRevealed = new Set(data.revealed);
			data.revealed.clear();
		}
		if (command === '-status') {
			data.statusAge = 0;
			data.statusStart = detail === 'slp' ? (from === '[from] move: Rest' ? 3 : 2) : 0;
			data.sleepFromFoe = detail === 'slp' && from !== '[from] move: Rest';
		}
		if (command === 'cant' && detail === 'slp') data.statusAge++;
		if (command === '-damage' && from === '[from] psn' && member.status === 'tox') data.statusAge++;
		if (command === '-damage' && !from && this.hpBefore > member.hp) {
			const current = this.view.step.currentMove;
			const row = current && this.view.moveRow(current);
			const amount = this.hpBefore - member.hp;
			this.damage = [amount, row?.category === 'Physical' ? amount : 0,
				row?.category === 'Special' ? amount : 0];
			this.hit = true;
		}
		if (command === '-enditem' && member.consumed) data.lastItem = this.view.index('items', id(detail));
		if (command === '-start' || command === '-prepare' || command === '-activate') {
			this.timers ||= {};
			if (command !== '-activate' || !(name in this.timers)) this.timers[name] = this.view.turn;
			if (command === '-start' && name === 'Substitute') this.substitute = Math.max(1, Math.floor(member.maxhp / 4));
			if (command === '-prepare') this.charging = move(name);
			if (command === '-activate' && this.side.vol.trapped) this.timers.partiallytrapped ??= this.view.turn;
		}
		if (command === 'faint') {
			this.faintPending = true;
			this.lastProtect = -2;
			this.fakeOut = false;
			this.guard = null;
		}
	}

	state(obs) {
		const view = this.view;
		const side = this.side;
		const { own } = obs;
		const active = side.members.indexOf(side.active);
		const map = fn => side.members.map(member => fn(member, this.members.get(member)));
		const row = this.members.get(side.active);
		const remaining = (flag, name, duration) => flag
			? Math.max(1, duration - (view.turn - (this.timers?.[name] ?? view.turn))) : 0;
		const stats = map(member => [
			member.maxhp,
			...(own.transformed && member === side.active && this.copied
				? this.copied.stats : STATS.slice(1).map(name => member.stats[name])),
		]);
		const ability = [...own.ability];
		if (own.transformed && this.copied?.ability && side.active.ability === side.active.baseAbility) {
			ability[active] = this.copied.ability;
		}
		const roostTypes = own.types.map(types => types.map(type => type - 1));
		const types = roostTypes.map(row => [...row]);
		if (own.roost) {
			const flying = view.tables.types.get('flying');
			types[active] = types[active].map(type => type === flying ? -1 : type);
			if (types[active].every(type => type === -1)) types[active][0] = view.tables.types.get('normal');
		}
		const itemGone = map(member => member.consumed ? 1 : member.knockedOff ? 2 : 0);
		const encodeMoves = moves => pad((moves || []).map(move => view.index('moves', move)));
		const savedMoves = side.active.savedMoves;
		const locked = own.locked ? view.index('moves', id(view.request.active[0].moves[0].id)) : 0;
		const disguise = row.shown && row.shown !== row.species && !own.transformed
			&& side.active.ability === 'illusion'
			? side.members.findIndex(member => this.members.get(member).species === row.shown) + 1 : 0;
		return {
			species: own.species, level: map(member => member.level), types, base_types: map((_, data) => data.types),
			roost_types: roostTypes, stats, hp: map(member => member.hp), moves: own.moves,
			pp: map(member => pad(member.pp)), max_pp: map(member => pad(member.maxpp)),
			ability, ability_base: map(member => view.index('abilities', member.baseAbility)),
			ability_public: map((_, data) => data.abilityPublic), item: own.item,
			item_consumed: own.item_consumed, item_knocked_off: own.item_knocked_off,
			item_revealed: map(member => member.itemRevealed),
			item_public: map(member => view.index('items', member.itemPublic)), last_item: map((_, data) => data.lastItem),
			nature: map(() => view.index('natures', 'serious')), gender: map((_, data) => data.gender),
			evs: map((_, data) => data.spread.evs), ivs: map((_, data) => data.spread.ivs),
			happiness: map((_, data) => data.happiness), status: own.status,
			status_counter: map((member, data) => member.status === 'slp'
				? Math.max(1, data.statusStart - data.statusAge) : member.status === 'tox' ? Math.min(15, data.statusAge) : 0),
			sleep_start: map((member, data) => member.status === 'slp' ? data.statusStart : 0),
			sleep_from_foe: map((member, data) => member.status === 'slp' && data.sleepFromFoe),
			seen_order: map((_, data) => data.seen),
			revealed_moves: map((member, data) => pad(member.moves.map(move => Number(data.revealed.has(view.index('moves', move)))))),
			active, active_turns: map((_, data) => data.turns), boosts: [...side.boosts],
			volatiles: {
				CONFUSION: remaining(own.confused, 'confusion', 3),
				ROOST: own.roost, LEECH_SEED: own.leech_seed,
				PARTIALLY_TRAPPED: remaining(own.trapped, 'partiallytrapped', 5),
				PROTECT: own.protect, STALL: this.lastProtect >= view.turn - 1 ? this.stall || 0 : 0,
				STALL_USED: Number(this.lastProtect === view.turn),
				ENDURE: Number(this.guard === 'Endure'), QUICK_GUARD: Number(this.guard === 'Quick Guard'),
				WIDE_GUARD: Number(this.guard === 'Wide Guard'),
				TWO_TURN_MOVE: own.charging ? this.charging || side.lastMove : 0,
				LOCKED_MOVE: locked, LOCKED_REMAINING: Number(Boolean(locked)),
				LOCKED_TRUE_DURATION: Number(Boolean(locked)),
				MUST_RECHARGE: own.recharging, FAKE_OUT_USED: Number(Boolean(this.fakeOut)),
				CHOICE_MOVE: side.choiceMove, FLASH_FIRE: own.flash_fire,
				DISABLE_MOVE: own.disabled_move, DISABLE_REMAINING: remaining(own.disable, 'Disable', 4),
				SLOW_START: remaining(own.slow_start, 'Slow Start', 5),
				CURSE: own.curse, AUTOTOMIZE: own.autotomize,
				SUBSTITUTE: own.substitute ? this.substitute || Math.max(1, Math.floor(side.active.maxhp / 4)) : 0,
				ENCORE_MOVE: own.encore_move, ENCORE_REMAINING: remaining(own.encore, 'Encore', 3),
				TAUNT: remaining(own.taunt, 'Taunt', 3), YAWN: own.yawn,
				DESTINY_BOND: own.destiny_bond, TRANSFORMED: own.transformed, ILLUSION_SLOT: disguise,
			},
			turn_damage: this.damage, turn_hit: this.hit, focus_lost: this.hit,
			fainted_last: this.faintedLast, alive_mark: map((_, data) => data.alive),
			last_move: map((_, data) => data.lastMove),
			wish_turns: obs.wish[0] ? Math.max(1, 2 - obs.wish_age[0]) : 0, wish_hp: obs.wish[0] ? this.wishHp : 0,
			species_base: map((_, data) => data.species), saved_stats: own.transformed ? row.stats.slice(1) : flags(5),
			saved_moves: own.transformed ? encodeMoves(savedMoves) : flags(4),
			saved_pp: own.transformed ? pad(side.active.savedPp || []) : flags(4),
			saved_max_pp: own.transformed ? pad(side.active.savedMaxpp || []) : flags(4),
			saved_revealed: own.transformed ? encodeMoves(savedMoves).map(move => Number(this.savedRevealed?.has(move) || false)) : flags(4),
			pirouette_stats: map((_, data) => data.pirouette), shown_species: map((_, data) => data.shown),
			shown_level: map((_, data) => data.shownLevel), party: this.party,
			hp_seen: map((_, data) => data.hpSeen), item_gone: itemGone, trap_known: view.trapHolds(),
		};
	}
}

module.exports = { OwnState };
