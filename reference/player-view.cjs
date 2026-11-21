'use strict';

/* One player's view of a Gen 5 singles battle, built only from what that player's client receives.
 *
 * Input: the protocol lines and |request| JSON of one player's stream, exactly as
 * getPlayerStreams delivers them (sim/battle-stream.ts): the side's own split lines and the public
 * lines otherwise, and the sideupdate request. No access to the Battle object. Output: the
 * observation of pokejax/engine/observation.py as plain JSON (same names, nesting, shapes and
 * values; IDs as pokejax/tables.py assigns them from data/gen5-data-v1.json).
 *
 *   const { PlayerView } = require('./player-view.cjs');
 *   const view = new PlayerView();      // the side comes from the first |request|
 *   view.receive(chunk);                // any text: protocol lines and |request|{...} lines
 *   const observation = view.observation();   // the decision point after the last request
 *
 * A step is the protocol between two requests. The event record describes the step that ended
 * with the latest request (all zero before the first request has been followed by another).
 */

const fs = require('node:fs');
const path = require('node:path');

const DATA_PATH = path.join(__dirname, '..', 'data', 'gen5-data-v1.json');
const TEAM_SLOTS = 6;
const MOVE_SLOTS = 4;
const STATUS_NAMES = ['', 'brn', 'psn', 'tox', 'par', 'slp', 'frz'];
const WEATHER_NAMES = ['', 'raindance', 'sunnyday', 'sandstorm', 'hail'];
const BOOST_NAMES = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const STAT_NAMES = ['hp', 'atk', 'def', 'spa', 'spd', 'spe'];

// Event record codes (pokejax/engine/events.py).
const CANT_CODES = {
	par: 1, slp: 2, frz: 3, flinch: 4, confusion: 5, recharge: 6, 'ability: Truant': 7,
	Disable: 8, 'move: Taunt': 9, 'Focus Punch': 10,
};
const OUTCOME_MISS = 1;
const OUTCOME_FAILED = 2;
const OUTCOME_CRIT = 4;
const EFFECT_IMMUNE = 1;
const EFFECT_RESISTED = 2;
const EFFECT_NEUTRAL = 3;
const EFFECT_SUPER = 4;
const ENTRY_CHOSEN = 1;
const ENTRY_DRAGGED = 2;
const ENTRY_PIVOT = 3;
const ENTRY_REPLACEMENT = 4;
const FAIL_LINES = new Set(['-fail', '-immune', '-nothing', '-block']);
const GUARD_ACTIVATES = new Set(['move: Protect', 'move: Wide Guard', 'move: Quick Guard']);

function toID(text) {
	return String(text ?? '').toLowerCase().replace(/[^a-z0-9]/g, '');
}

function idCompare(a, b) {
	return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

/* The tables of pokejax/tables.py: rows sorted by Showdown id, entity indices reserve 0 for none,
 * type indices follow the export's order. */
function loadTables(dataPath = DATA_PATH) {
	const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
	const entity = rows => {
		const list = [...rows].sort(idCompare);
		return { rows: list, index: new Map(list.map((row, i) => [row.id, i + 1])) };
	};
	const types = new Map(data.types.map((row, i) => [toID(row.name), i]));
	const species = entity(data.species);
	const cosmetic = new Map();
	for (const row of species.rows) {
		for (const form of row.cosmeticFormes || []) {
			if (!species.index.has(toID(form))) cosmetic.set(toID(form), row.id);
		}
	}
	return {
		types,
		species: { ...species, cosmetic },
		moves: entity(data.moves),
		abilities: entity(data.abilities),
		items: entity(data.items),
	};
}

function percentOf(hp, maximum) {
	// pokejax.engine.events.protocol_percent: ceil, at most 99 while damaged, 0 at 0.
	if (hp <= 0) return 0;
	maximum = Math.max(maximum, 1);
	const percent = Math.floor((100 * hp + maximum - 1) / maximum);
	return hp < maximum ? Math.min(percent, 99) : percent;
}

function parseRef(text) {
	// 'p1a: Name' (a position) or 'p1: Name' (a team member).
	const match = /^(p[1-4])([a-z]?): (.*)$/.exec(text || '');
	if (!match) return null;
	return { side: Number(match[1][1]) - 1, position: match[2], name: match[3] };
}

function parseDetails(text) {
	const [species, ...flags] = String(text).split(', ');
	let level = 100;
	for (const flag of flags) {
		const match = /^L(\d+)$/.exec(flag);
		if (match) level = Number(match[1]);
	}
	return { species, level };
}

function parseHealth(text) {
	// '252/252', '37/100 par', '0 fnt'.
	const [condition, status = ''] = String(text).split(' ');
	if (status === 'fnt') return { hp: 0, maxhp: null, status: '' };
	const [hp, maxhp] = condition.split('/').map(Number);
	return { hp, maxhp: maxhp ?? null, status };
}

function tagValue(parts, name) {
	// A [name] value tag: '[from] ability: X' or '[from]ability: X'; '[of] p1a: Y'.
	const prefix = '[' + name + ']';
	for (const part of parts) {
		if (part.startsWith(prefix)) return part.slice(prefix.length).trim();
	}
	return null;
}

function hasTag(parts, name) {
	return parts.includes('[' + name + ']');
}

function sideIndex(value) {
	if (value === 0 || value === 1) return value;
	const match = /^p([12])$/.exec(String(value));
	if (!match) throw new Error('Unknown side: ' + value);
	return Number(match[1]) - 1;
}

function freshVolatiles() {
	return {
		confused: false, charging: false, recharging: false, locked: false, trapped: false,
		substitute: false, encore: false, encoreMove: 0, taunt: false, disable: false,
		disabledMove: 0, leechSeed: false, yawn: false, curse: false, destinyBond: false,
		transformed: false, flashFire: false, slowStart: false, autotomize: 0, protect: false,
		roost: false,
	};
}

function newSide(index) {
	return {
		index, members: [], active: null, boosts: Array(7).fill(0), vol: freshVolatiles(),
		pct: 100, entries: 0, hazards: [0, 0], lastMove: 0, choiceMove: 0, name: '',
	};
}

function newMember(name) {
	return {
		name, species: '', baseSpecies: '', level: 100, hp: 0, maxhp: 0, status: '', stats: null,
		moves: [], pp: [], maxpp: [], baseAbility: '', ability: '', item: '', consumed: false,
		knockedOff: false, itemRevealed: false, itemPublic: '', revealed: new Set(),
		savedRevealed: null, types: null, seen: 0, fainted: false, formeSpecies: '',
	};
}

function newRecord() {
	return { order: 0, move: 0, cant: 0, outcome: 0, effect: 0, hits: 0, entry: 0, status: 0, fainted: 0 };
}

function newWish() {
	return { pending: false, age: 0, usedTurn: 0 };
}

class PlayerView {
	constructor(options = {}) {
		this.tables = options.tables || loadTables(options.dataPath);
		this.fixedSide = options.side === undefined || options.side === null ? null : sideIndex(options.side);
		this.reset();
	}

	reset() {
		this.me = null;
		this.sides = [newSide(0), newSide(1)];
		this.buffered = [];
		this.turn = 0;
		this.ended = false;
		this.weather = 0;
		this.weatherAge = 0;
		this.trickRoom = false;
		this.trickRoomAge = 0;
		this.wish = [newWish(), newWish()];
		this.healingWish = [false, false];
		this.requestKind = 'none';
		this.request = null;
		this.trapKnown = null;
		this.midTurn = false;
		this.stepLines = 0;
		this.decisions = 0;
		this.events = [newRecord(), newRecord()];
		this.step = null;
		this.beginStep();
	}

	// Input.

	receive(chunk) {
		for (const line of String(chunk).split('\n')) this.receiveLine(line);
	}

	receiveLine(line) {
		if (!line.startsWith('|')) return;
		const cut = line.indexOf('|', 1);
		const command = cut < 0 ? line.slice(1) : line.slice(1, cut);
		if (command === 'request') return this.receiveRequest(line.slice(cut + 1));
		if (command === 'error') return;
		const parts = line.split('|');
		if (this.me === null) this.buffered.push(parts);
		else this.protocol(parts);
	}

	receiveRequest(text) {
		const request = JSON.parse(text);
		if (request === null) return;
		// A move request that gains trapped: true with no protocol line since the last one answers
		// a refused switch choice (sim/side.ts chooseSwitch): the hidden trap (Shadow Tag, Arena
		// Trap, Magnet Pull) is now known, and the next turn's request will only say maybeTrapped.
		const refused = this.request !== null && this.stepLines === 0 && this.requestKind === 'move'
			&& !(this.request.active && this.request.active[0] && this.request.active[0].trapped)
			&& Boolean(request.active && request.active[0] && request.active[0].trapped === true);
		const me = sideIndex(request.side.id);
		if (this.fixedSide !== null && this.fixedSide !== me) {
			throw new Error('Request for ' + request.side.id + ' on the view of p' + (this.fixedSide + 1));
		}
		if (this.me === null) {
			// The first request names the player and its team; the setup lines refer to both.
			this.me = me;
			this.createOwnMembers(request);
			this.applyRequest(request, 'move');
			for (const parts of this.buffered) this.protocol(parts);
			this.buffered = [];
		} else if (this.me !== me) {
			throw new Error('Request for ' + request.side.id + ' on the view of p' + (this.me + 1));
		}
		const kind = request.wait ? 'none' : request.forceSwitch ? 'switch' : request.active ? 'move' : 'none';
		this.endStep(kind);
		this.applyRequest(request, kind);
		// The entry counts name the two actives: a switch line for either side ends the knowledge.
		if (refused) this.trapKnown = { own: this.sides[this.me].entries, foe: this.sides[1 - this.me].entries };
	}

	// The own side was refused a switch and neither active has changed since.
	trapHolds() {
		const known = this.trapKnown;
		return known !== null && known.own === this.sides[this.me].entries
			&& known.foe === this.sides[1 - this.me].entries;
	}

	// Steps. A step is the protocol between two requests; endStep closes it at the next request.

	beginStep() {
		this.step = {
			startPct: this.sides.map(side => side.pct),
			entries: this.sides.map(side => side.entries),
			records: [newRecord(), newRecord()],
			current: null,
			currentMove: null,
			syncMode: null,
			syncChain: false,
			statusFromAbility: false,
			attempted: [false, false],
			skipOrder: [false, false],
			damageOnFoe: false,
			turnLine: false,
			// The step began at a turn start when the player's request was a move request.
			turnStart: this.requestKind === 'move',
		};
	}

	endStep(kind) {
		if (this.stepLines > 0) {
			this.finishRecords();
			this.events = this.decisions === 0 ? [newRecord(), newRecord()] : this.step.records;
			this.midTurn = !this.step.turnLine && !this.ended;
			this.stepLines = 0;
			this.decisions++;
			this.requestKind = kind;
			this.beginStep();
		}
		this.requestKind = kind;
	}

	finishRecords() {
		const step = this.step;
		this.closeMove();
		for (const side of this.sides) {
			const index = side.index;
			const still = side.entries === step.entries[index];
			step.records[index].hp = still ? side.pct - step.startPct[index] : 0;
		}
	}

	// The event record (events.py), from the protocol lines of the step.

	moveRow(id) {
		return this.tables.moves.rows[(this.tables.moves.index.get(id) || 0) - 1];
	}

	closeMove() {
		const step = this.step;
		if (step.current === null || !step.currentMove || !step.damageOnFoe) return;
		const record = step.records[step.current];
		const row = this.moveRow(step.currentMove);
		if (record.effect === 0 && row && row.category !== 'Status') record.effect = EFFECT_NEUTRAL;
	}

	attempt(side, move, cant) {
		const step = this.step;
		if (!step.attempted[side] && !step.skipOrder[side]) {
			step.attempted[side] = true;
			step.records[side].order = step.attempted[1 - side] ? 2 : 1;
		}
		step.skipOrder[side] = false;
		Object.assign(step.records[side], { move, cant, outcome: 0, effect: 0, hits: 0 });
	}

	record(parts) {
		const step = this.step;
		const records = step.records;
		const command = parts[1];
		const ref = parseRef(parts[2]);
		const side = ref ? ref.side : null;
		// Synchronize copies a status back to the Pokemon that gave it. What the move's record
		// takes from the next -fail or -immune depends on whose Synchronize it is: the target's
		// copy onto the user is the move's outcome unless an ability (Effect Spore, Poison Touch)
		// gave the status, and the user's own Synchronize counts only as the chained copy back.
		const skipSync = step.syncMode === 'skip';
		const fromAbility = step.statusFromAbility;
		step.syncMode = null;
		step.statusFromAbility = command === '-status' && /^ability: /.test(tagValue(parts, 'from') || '');
		if (command === '-activate' && parts[3] === 'ability: Synchronize') {
			if (side !== step.current) {
				step.syncMode = fromAbility ? 'skip' : 'count';
				step.syncChain = step.syncMode === 'count';
			} else {
				step.syncMode = step.syncChain ? 'count' : 'skip';
				step.syncChain = false;
			}
		}
		if (command === 'move') {
			this.closeMove();
			step.current = side;
			step.currentMove = moveId(parts[3]);
			step.damageOnFoe = false;
			if (tagValue(parts, 'from') === 'ability: Magic Bounce') step.skipOrder[side] = true;
			this.attempt(side, this.tables.moves.index.get(step.currentMove) || 0, 0);
			return;
		}
		if (command === 'cant') {
			this.closeMove();
			step.current = side;
			step.currentMove = null;
			step.damageOnFoe = false;
			this.attempt(side, 0, CANT_CODES[parts[3]] || 0);
			return;
		}
		if (command === '-activate' && parts[3] === 'confusion') {
			this.closeMove();
			step.current = side;
			step.currentMove = null;
			step.damageOnFoe = false;
			this.attempt(side, 0, CANT_CODES.confusion);
			return;
		}
		if (command === 'switch' || command === 'drag') {
			this.closeMove();
			step.current = null;
			step.currentMove = null;
			step.damageOnFoe = false;
			const pivot = tagValue(parts, 'from') !== null;
			records[side].entry = command === 'drag' ? ENTRY_DRAGGED : pivot ? ENTRY_PIVOT
				: step.turnStart ? ENTRY_CHOSEN : ENTRY_REPLACEMENT;
		} else if (command === 'faint') {
			records[side].fainted = 1;
		} else if (command === '-status') {
			records[side].status = Math.max(STATUS_NAMES.indexOf(parts[3]), 0);
		} else if (command === '-miss') {
			records[side].outcome |= OUTCOME_MISS;
		} else if (FAIL_LINES.has(command) || (command === '-activate' && GUARD_ACTIVATES.has(parts[3]))) {
			// A stat drop that an ability blocks is named by the blocker (-fail ... unboost): the
			// move itself may still have done something (Memento's user faints).
			if (step.current !== null && parts[3] !== 'unboost' && !skipSync) {
				records[step.current].outcome |= OUTCOME_FAILED;
			}
		}
		if (step.current === null) return;
		const record = records[step.current];
		if (command === '-immune') {
			if (!skipSync) record.effect = EFFECT_IMMUNE;
		} else if (command === '-crit') record.outcome |= OUTCOME_CRIT;
		else if (command === '-supereffective') record.effect = EFFECT_SUPER;
		else if (command === '-resisted') record.effect = EFFECT_RESISTED;
		else if (command === '-hitcount') record.hits = Number(parts[3]);
		else if (command === '-damage' && parts.length === 4 && side !== step.current) step.damageOnFoe = true;
	}

	// Members and sides.

	activeOf(sideIndex) {
		return this.sides[sideIndex].active;
	}

	memberFor(ref) {
		if (!ref) return null;
		const side = this.sides[ref.side];
		if (ref.position) return side.active;
		return side.members.find(member => member.name === ref.name) || null;
	}

	speciesIndex(id) {
		const species = this.tables.species;
		const resolved = species.cosmetic.get(id) || id;
		return species.index.get(resolved) || 0;
	}

	speciesRow(id) {
		const species = this.tables.species;
		return species.rows[(species.index.get(species.cosmetic.get(id) || id) || 0) - 1];
	}

	speciesTypes(id) {
		const row = this.speciesRow(id);
		return row ? row.types.map(name => this.tables.types.get(toID(name))) : [];
	}

	currentSpecies(member) {
		return member.formeSpecies || member.baseSpecies;
	}

	currentTypes(member) {
		return member.types || this.speciesTypes(this.currentSpecies(member));
	}

	createOwnMembers(request) {
		const side = this.sides[this.me];
		side.name = request.side.name;
		for (const entry of request.side.pokemon) side.members.push(newMember(identName(entry.ident)));
	}

	applyRequest(request, kind) {
		this.request = request;
		const side = this.sides[this.me];
		for (const entry of request.side.pokemon) {
			const member = side.members.find(candidate => candidate.name === identName(entry.ident));
			const details = parseDetails(entry.details);
			member.baseSpecies = toID(details.species);
			member.level = details.level;
			const health = parseHealth(entry.condition);
			member.hp = health.hp;
			if (health.maxhp) member.maxhp = health.maxhp;
			member.status = health.status;
			member.fainted = health.hp === 0;
			member.stats = entry.stats;
			member.baseAbility = toID(entry.baseAbility);
			member.item = toID(entry.item);
			const moves = entry.moves.map(requestMoveId);
			if (moves.join() !== member.moves.join()) {
				member.moves = moves;
				member.pp = moves.map(id => this.maxPp(id));
				member.maxpp = [...member.pp];
			}
			if (entry.active) side.active = member;
		}
		const active = side.active;
		const slots = request.active && request.active[0] ? request.active[0].moves : [];
		slots.forEach((slot, i) => {
			if (slot.pp !== undefined && slot.id !== 'struggle' && active.moves[i] === slot.id) {
				active.pp[i] = slot.pp;
				active.maxpp[i] = slot.maxpp;
			}
		});
		if (active.moves.length && !active.ability) active.ability = active.baseAbility;
		for (const member of side.members) if (member !== active && !member.ability) member.ability = member.baseAbility;
	}

	maxPp(id) {
		const row = this.moveRow(id);
		if (!row) return 0;
		return row.noPPBoosts ? row.pp : Math.floor((row.pp * 8) / 5);
	}

	// Protocol.

	protocol(parts) {
		this.stepLines++;
		this.record(parts);
		const command = parts[1];
		const handler = HANDLERS[command];
		if (handler) handler.call(this, parts);
		this.revealFromTags(parts);
	}

	enterMember(side, ref, details, health) {
		let member;
		if (side.index === this.me) {
			const named = side.members.filter(candidate => candidate.name === ref.name);
			member = named.find(candidate => health.maxhp === null || candidate.maxhp === health.maxhp) || named[0];
			if (!member || (health.maxhp !== null && member.maxhp !== health.maxhp)) {
				// Illusion: the line names the disguise, the own request knows the holder.
				const holder = side.members.find(candidate => candidate !== side.active
					&& !candidate.fainted && candidate.baseAbility === 'illusion' && candidate.maxhp === health.maxhp);
				member = holder || member;
			}
		} else {
			member = side.members.find(candidate => candidate.name === ref.name);
			if (!member) {
				member = newMember(ref.name);
				member.maxhp = 100;
				member.seen = side.members.length + 1;
				side.members.push(member);
			}
			member.baseSpecies = toID(details.species);
			member.level = details.level;
		}
		return member;
	}

	// clearVolatile: what a Pokemon loses when it leaves the field or faints.
	clearVolatile(side, member) {
		side.boosts = Array(7).fill(0);
		side.vol = freshVolatiles();
		side.lastMove = 0;
		side.choiceMove = 0;
		side.trapSource = null;
		member.types = null;
		member.formeSpecies = '';
		if (member.savedMoves) {
			member.moves = member.savedMoves;
			member.pp = member.savedPp;
			member.maxpp = member.savedMaxpp;
			member.savedMoves = null;
		}
		if (member.savedRevealed) {
			member.revealed = member.savedRevealed;
			member.savedRevealed = null;
		}
		if (side.index === this.me) member.ability = member.baseAbility;
	}

	leaveField(side, member) {
		this.clearVolatile(side, member);
	}

	healthOf(member, health) {
		member.hp = health.hp;
		if (health.maxhp) member.maxhp = health.maxhp;
		member.status = health.status;
		member.fainted = health.hp === 0;
	}

	setPercent(side, member, health) {
		side.pct = percentOf(health.hp, health.maxhp || member.maxhp);
	}
}

const MAX_AGE = 127; // the engine's ages saturate (int8)
const PARTIAL_TRAP_MOVES = new Set(['bind', 'clamp', 'firespin', 'magmastorm', 'sandtomb', 'whirlpool', 'wrap']);
const CHOICE_ITEMS = new Set(['choiceband', 'choicescarf', 'choicespecs']);

function identName(ident) {
	return ident.slice(ident.indexOf(': ') + 2);
}

function moveId(name) {
	const id = toID(name);
	return id.startsWith('hiddenpower') ? 'hiddenpower' : id;
}

function requestMoveId(id) {
	// The request adds Hidden Power's type and Return's and Frustration's power to the ID.
	if (id.startsWith('hiddenpower')) return 'hiddenpower';
	return id.replace(/^(return|frustration)\d+$/, '$1');
}

function effectName(text) {
	return String(text ?? '').replace(/^(move|ability|item): /, '');
}

function boostIndex(stat) {
	return BOOST_NAMES.indexOf(stat);
}

function clampBoost(value) {
	return Math.max(-6, Math.min(6, value));
}

const HANDLERS = {
	turn(parts) {
		this.turn = Number(parts[2]);
		this.step.turnLine = true;
		for (const wish of this.wish) if (wish.pending) wish.age = Math.min(wish.age + 1, MAX_AGE);
	},

	upkeep() {
		if (this.trickRoom) this.trickRoomAge = Math.min(this.trickRoomAge + 1, MAX_AGE);
		this.wish.forEach(wish => {
			if (wish.pending && this.turn >= wish.usedTurn + 1) Object.assign(wish, newWish());
		});
		for (const side of this.sides) {
			side.vol.protect = false;
			side.vol.roost = false;
		}
	},

	switch(parts) {
		const ref = parseRef(parts[2]);
		const side = this.sides[ref.side];
		const details = parseDetails(parts[3]);
		const health = parseHealth(parts[4]);
		const previous = side.active;
		const member = this.enterMember(side, ref, details, health);
		// A Pokemon cannot switch to itself: the line always ends the previous one's stay, even when
		// the entry resolves to the same record (an Illusion holder under the name of a member).
		if (previous) this.leaveField(side, previous);
		side.active = member;
		side.entries++;
		this.healthOf(member, health);
		this.setPercent(side, member, health);
	},

	drag(parts) {
		HANDLERS.switch.call(this, parts);
	},

	replace(parts) {
		// Illusion ends: the active member is shown under its real name and species.
		const ref = parseRef(parts[2]);
		const side = this.sides[ref.side];
		const member = side.active;
		const details = parseDetails(parts[3]);
		member.name = ref.name;
		member.baseSpecies = toID(details.species);
		member.formeSpecies = '';
		member.types = null;
		member.level = details.level;
	},

	detailschange(parts) {
		const member = this.sides[parseRef(parts[2]).side].active;
		member.baseSpecies = toID(parseDetails(parts[3]).species);
		member.formeSpecies = '';
		member.types = null;
	},

	'-formechange'(parts) {
		const member = this.sides[parseRef(parts[2]).side].active;
		member.formeSpecies = toID(parts[3]);
		member.types = null;
	},

	'-transform'(parts) {
		const user = parseRef(parts[2]);
		const target = parseRef(parts[3]);
		const side = this.sides[user.side];
		const foe = this.sides[target.side];
		const member = side.active;
		const copied = foe.active;
		member.formeSpecies = this.currentSpecies(copied);
		member.types = [...this.currentTypes(copied)];
		side.boosts = [...foe.boosts];
		side.vol.transformed = true;
		if (user.side === this.me) {
			member.savedMoves = member.moves;
			member.savedPp = member.pp;
			member.savedMaxpp = member.maxpp;
		} else {
			member.savedRevealed = member.revealed;
			member.revealed = new Set();
		}
	},

	faint(parts) {
		const ref = parseRef(parts[2]);
		const side = this.sides[ref.side];
		const member = side.active;
		member.hp = 0;
		member.fainted = true;
		member.status = '';
		side.pct = 0;
		this.clearVolatile(side, member);
		if (this.healingWishUser === ref.side) this.healingWish[ref.side] = true;
		this.healingWishUser = null;
	},

	move(parts) {
		const ref = parseRef(parts[2]);
		const side = this.sides[ref.side];
		const member = side.active;
		const from = tagValue(parts, 'from');
		const id = moveId(parts[3]);
		const index = this.tables.moves.index.get(id) || 0;
		const vol = side.vol;
		vol.charging = false;
		vol.destinyBond = false;
		this.healingWishUser = null;
		// Pursuit run before a switch is the move its user chose; a locked move continues one;
		// every other [from] names the move or ability that used it (a call, a reflection).
		const chosen = from === null || from === 'move: Pursuit';
		const continued = from === 'lockedmove';
		const called = from !== null && /^move: /.test(from) && from !== 'move: Pursuit';
		const own = ref.side === this.me;
		if (own && (chosen || continued) && !side.choiceMove && CHOICE_ITEMS.has(member.item)) {
			side.choiceMove = index;
		}
		if (called) {
			// The caller pays the foe's Pressure for the move it called.
			if (own) {
				const slot = member.moves.indexOf(moveId(from.slice(6)));
				if (slot >= 0) member.pp[slot] = Math.max(0, member.pp[slot] - this.pressureCost(id, side));
			}
			return;
		}
		if (!chosen) return;
		side.lastMove = index;
		if (id === 'struggle') return;
		if (!own) member.revealed.add(id);
		else {
			const slot = member.moves.indexOf(id);
			// Pursuit run before a switch is called by the Pursuit condition, which is a caller
			// with PP: Pressure applies as for the move itself.
			if (slot >= 0) member.pp[slot] = Math.max(0, member.pp[slot] - 1 - this.pressureCost(id, side));
		}
		if (id === 'wish' && !this.wish[ref.side].pending) {
			this.wish[ref.side] = { pending: true, age: 0, usedTurn: this.turn };
		}
		if (id === 'healingwish') this.healingWishUser = ref.side;
	},

	cant(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		side.vol.charging = false;
		side.vol.destinyBond = false;
		if (parts[3] === 'recharge') side.vol.recharging = false;
	},

	'-fail'(parts) {
		if (this.healingWishUser !== null && parseRef(parts[2])?.side === this.healingWishUser) this.healingWishUser = null;
	},

	'-damage'(parts) {
		this.healthLine(parts);
	},

	'-heal'(parts) {
		this.healthLine(parts);
		const from = tagValue(parts, 'from');
		const ref = parseRef(parts[2]);
		if (from === 'move: Healing Wish' && ref) this.healingWish[ref.side] = false;
		if (from === 'move: Wish' && ref) this.wish[ref.side] = newWish();
	},

	'-sethp'(parts) {
		this.healthLine(parts);
	},

	'-status'(parts) {
		const member = this.memberFor(parseRef(parts[2]));
		if (member) member.status = parts[3];
	},

	'-curestatus'(parts) {
		const member = this.memberFor(parseRef(parts[2]));
		if (member) member.status = '';
	},

	'-cureteam'(parts) {
		for (const member of this.sides[parseRef(parts[2]).side].members) member.status = '';
	},

	'-boost'(parts) {
		this.addBoost(parts, 1);
	},

	'-unboost'(parts) {
		this.addBoost(parts, -1);
	},

	'-setboost'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		side.boosts[boostIndex(parts[3])] = clampBoost(Number(parts[4]));
	},

	'-swapboost'(parts) {
		const a = this.sides[parseRef(parts[2]).side];
		const b = this.sides[parseRef(parts[3]).side];
		const stats = parts[4] ? parts[4].split(',').map(stat => boostIndex(stat.trim())) : BOOST_NAMES.map((_, i) => i);
		for (const i of stats) [a.boosts[i], b.boosts[i]] = [b.boosts[i], a.boosts[i]];
	},

	'-invertboost'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		side.boosts = side.boosts.map(value => -value);
	},

	'-clearboost'(parts) {
		this.sides[parseRef(parts[2]).side].boosts = Array(7).fill(0);
	},

	'-clearallboost'() {
		for (const side of this.sides) side.boosts = Array(7).fill(0);
	},

	'-clearpositiveboost'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		side.boosts = side.boosts.map(value => Math.min(value, 0));
	},

	'-clearnegativeboost'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		side.boosts = side.boosts.map(value => Math.max(value, 0));
	},

	'-copyboost'(parts) {
		const source = this.sides[parseRef(parts[2]).side];
		const target = this.sides[parseRef(parts[3]).side];
		source.boosts = [...target.boosts];
	},

	'-weather'(parts) {
		const id = toID(parts[2]);
		if (id === 'none') {
			this.weather = 0;
			this.weatherAge = 0;
		} else if (hasTag(parts, 'upkeep')) {
			this.weatherAge = Math.min(this.weatherAge + 1, MAX_AGE);
		} else {
			this.weather = Math.max(WEATHER_NAMES.indexOf(id), 0);
			this.weatherAge = 0;
		}
	},

	'-fieldstart'(parts) {
		if (effectName(parts[2]) === 'Trick Room') {
			this.trickRoom = true;
			this.trickRoomAge = 0;
		}
	},

	'-fieldend'(parts) {
		if (effectName(parts[2]) === 'Trick Room') {
			this.trickRoom = false;
			this.trickRoomAge = 0;
		}
	},

	'-sidestart'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		const kind = ['Spikes', 'Toxic Spikes'].indexOf(effectName(parts[3]));
		if (kind >= 0) side.hazards[kind]++;
	},

	'-sideend'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		const kind = ['Spikes', 'Toxic Spikes'].indexOf(effectName(parts[3]));
		if (kind >= 0) side.hazards[kind] = 0;
	},

	'-start'(parts) {
		this.volatileLine(parts, true);
	},

	'-end'(parts) {
		this.volatileLine(parts, false);
	},

	'-singleturn'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		const name = effectName(parts[3]);
		if (name === 'Protect') side.vol.protect = true;
		else if (name === 'Roost') side.vol.roost = true;
	},

	'-singlemove'(parts) {
		const side = this.sides[parseRef(parts[2]).side];
		if (effectName(parts[3]) === 'Destiny Bond') side.vol.destinyBond = true;
	},

	'-anim'(parts) {
		// A charge move that skips its charge (Solar Beam in the sun) is shown as an animation.
		const ref = parseRef(parts[2]);
		if (ref) this.sides[ref.side].vol.charging = false;
	},

	'-prepare'(parts) {
		this.sides[parseRef(parts[2]).side].vol.charging = true;
	},

	'-mustrecharge'(parts) {
		this.sides[parseRef(parts[2]).side].vol.recharging = true;
	},

	'-activate'(parts) {
		const ref = parseRef(parts[2]);
		const side = this.sides[ref.side];
		const name = effectName(parts[3]);
		// A confused Pokemon's turn begins (and removes Destiny Bond) before it hits itself.
		if (parts[3] === 'confusion') side.vol.destinyBond = false;
		if (PARTIAL_TRAP_MOVES.has(toID(name)) && parts[3].startsWith('move: ')) {
			const source = parseRef(tagValue(parts, 'of') || '');
			side.vol.trapped = true;
			side.trapSource = source ? this.sides[source.side].active : null;
		}
	},

	'-item'(parts) {
		const ref = parseRef(parts[2]);
		const from = tagValue(parts, 'from');
		let member = this.memberFor(ref);
		if (!ref && from === 'ability: Frisk') {
			// Frisk names no target: in singles it is the foe of the holder.
			const holder = parseRef(tagValue(parts, 'of'));
			member = this.sides[1 - holder.side].active;
		}
		if (!member) return;
		const item = toID(parts[3]);
		member.item = item;
		member.consumed = false;
		member.knockedOff = false;
		// setItem runs the new item's Start, which removes a Choice lock.
		if (ref) this.sides[ref.side].choiceMove = 0;
		this.revealItem(member, item);
	},

	'-enditem'(parts) {
		const member = this.memberFor(parseRef(parts[2]));
		if (!member) return;
		const item = toID(parts[3]);
		const from = tagValue(parts, 'from');
		this.revealItem(member, item);
		member.item = '';
		// Gone by consumption (eaten, a used Gem, a Focus Sash...) or taken away (Knock Off, Trick...).
		const taken = from !== null && /^move: /.test(from) || from === 'stealeat';
		member.consumed = !taken;
		member.knockedOff = taken;
	},

	'-ability'(parts) {
		const member = this.memberFor(parseRef(parts[2]));
		if (member) member.ability = toID(parts[3]);
	},

	win() {
		this.ended = true;
		this.endStep('none');
	},

	tie() {
		this.ended = true;
		this.endStep('none');
	},
};

PlayerView.prototype.healthLine = function (parts) {
	const ref = parseRef(parts[2]);
	if (!ref) return;
	const side = this.sides[ref.side];
	const member = this.memberFor(ref);
	if (!member) return;
	const health = parseHealth(parts[3]);
	this.healthOf(member, health);
	if (ref.position) this.setPercent(side, member, health);
};

PlayerView.prototype.addBoost = function (parts, sign) {
	const side = this.sides[parseRef(parts[2]).side];
	const i = boostIndex(parts[3]);
	side.boosts[i] = clampBoost(side.boosts[i] + sign * Number(parts[4]));
};

PlayerView.prototype.typeIds = function (names) {
	return names.map(name => this.tables.types.get(toID(name)));
};

PlayerView.prototype.volatileLine = function (parts, start) {
	const ref = parseRef(parts[2]);
	const side = this.sides[ref.side];
	const vol = side.vol;
	const name = effectName(parts[3]);
	switch (name) {
		case 'confusion':
			vol.confused = start;
			break;
		case 'Substitute': vol.substitute = start; break;
		case 'Encore':
			vol.encore = start;
			vol.encoreMove = start ? side.lastMove : 0;
			break;
		case 'Disable':
			vol.disable = start;
			vol.disabledMove = start ? this.tables.moves.index.get(moveId(parts[4])) || 0 : 0;
			break;
		case 'Taunt': vol.taunt = start; break;
		case 'Leech Seed': vol.leechSeed = start; break;
		case 'Yawn': vol.yawn = start; break;
		case 'Curse': vol.curse = start; break;
		case 'Autotomize': if (start) vol.autotomize++; break;
		case 'Flash Fire': vol.flashFire = start; break;
		case 'Slow Start': vol.slowStart = start; break;
		case 'typechange':
			if (start) side.active.types = this.typeIds(parts[4].split('/'));
			break;
		default:
	}
	if (!start && hasTag(parts, 'partiallytrapped')) {
		vol.trapped = false;
		side.trapSource = null;
	}
};

// A public ability or item is named by the line's own words ('ability: X', '[from] item: X') for
// the holder the line is about: the [of] Pokemon of a damage, status or start line, else the target.
PlayerView.prototype.holderOf = function (parts) {
	const target = parseRef(parts[2]);
	const source = parseRef(tagValue(parts, 'of') || '');
	const ofFirst = ['-damage', '-status', '-start'].includes(parts[1]) && source;
	return this.memberFor(ofFirst ? source : target || source);
};

PlayerView.prototype.revealFromTags = function (parts) {
	const command = parts[1];
	let name = null;
	for (const part of parts.slice(3)) {
		const match = /^(?:\[from\]\s*)?ability: (.+)$/.exec(part);
		if (match) {
			name = match[1];
			break;
		}
	}
	if (command === '-ability') name = parts[3];
	if (command === '-end' && parts[3] === 'Illusion') name = 'Illusion';
	if (name !== null) {
		const id = toID(name);
		if (command === '-activate' && id === 'mummy') {
			for (const member of [this.memberFor(parseRef(parts[2])), this.memberFor(parseRef(parts[4]))]) {
				if (member) member.ability = id;
			}
		} else {
			const member = command === '-ability' ? this.memberFor(parseRef(parts[2])) : this.holderOf(parts);
			if (member) member.ability = id;
		}
	}
	const from = tagValue(parts, 'from');
	let item = from && from.startsWith('item: ') ? from.slice(6) : null;
	if (item === null && /^item: /.test(parts[3] || '')) item = parts[3].slice(6);
	if (item !== null) {
		const member = this.holderOf(parts);
		if (member) {
			this.revealItem(member, toID(item));
			if (hasTag(parts, 'consumed')) {
				member.consumed = true;
				member.item = '';
			}
		}
	}
};

// Pressure charges one more PP for a move aimed at its holder: the foe's public ability names it.
PlayerView.prototype.pressureCost = function (id, side) {
	const foe = this.sides[1 - side.index];
	if (!foe.active || foe.active.ability !== 'pressure' || foe.active.hp === 0) return 0;
	const row = this.moveRow(id);
	if (!row) return 0;
	const foeTargets = ['normal', 'any', 'randomNormal', 'allAdjacent', 'allAdjacentFoes', 'all', 'scripted'];
	return foeTargets.includes(row.target) || (row.flags && row.flags.mustpressure) ? 1 : 0;
};

PlayerView.prototype.revealItem = function (member, item) {
	member.itemRevealed = true;
	member.itemPublic = item;
};

// Output.

function flag(value) {
	return value ? 1 : 0;
}

function pad(list, length, fill = 0) {
	return [...list, ...Array(length - list.length).fill(fill)];
}

PlayerView.prototype.index = function (table, id) {
	return id ? this.tables[table].index.get(id) || 0 : 0;
};

PlayerView.prototype.typePair = function (types) {
	const first = types[0] === undefined ? 0 : types[0] + 1;
	const second = types[1] === undefined ? 0 : types[1] + 1;
	return [first, second];
};

PlayerView.prototype.activeVolatiles = function (side) {
	const vol = side.vol;
	const foe = this.sides[1 - side.index];
	const trapped = vol.trapped && side.trapSource && side.trapSource === foe.active && foe.active.hp > 0;
	return {
		confused: flag(vol.confused),
		charging: flag(vol.charging),
		recharging: flag(vol.recharging),
		trapped: flag(trapped),
		substitute: flag(vol.substitute),
		encore: flag(vol.encore),
		encore_move: vol.encore ? vol.encoreMove : 0,
		taunt: flag(vol.taunt),
		disable: flag(vol.disable),
		disabled_move: vol.disable ? vol.disabledMove : 0,
		leech_seed: flag(vol.leechSeed),
		yawn: flag(vol.yawn),
		curse: flag(vol.curse),
		destiny_bond: flag(vol.destinyBond),
		transformed: flag(vol.transformed),
		flash_fire: flag(vol.flashFire),
		slow_start: flag(vol.slowStart),
		autotomize: vol.autotomize,
		protect: flag(vol.protect),
		roost: flag(vol.roost),
	};
};

// A hard lock (Outrage, a charge turn, a recharge) shows in the request as a single move without PP.
PlayerView.prototype.requestLocked = function () {
	if (this.ended || this.requestKind !== 'move') return false;
	const moves = this.request.active[0].moves;
	const side = this.sides[this.me];
	return moves.length === 1 && moves[0].pp === undefined
		&& !['recharge', 'struggle'].includes(moves[0].id) && !side.vol.charging;
};

PlayerView.prototype.ownObservation = function () {
	const side = this.sides[this.me];
	const members = side.members;
	const active = side.active;
	const choiceLocked = Boolean(side.choiceMove) && CHOICE_ITEMS.has(active.item)
		&& active.moves.some(id => this.index('moves', id) === side.choiceMove);
	return {
		species: members.map(member => this.speciesIndex(this.currentSpecies(member))),
		types: members.map(member => this.typePair(this.currentTypes(member))),
		moves: members.map(member => pad(member.moves.map(id => this.index('moves', id)), MOVE_SLOTS)),
		level: members.map(member => member.level / 100),
		pp: members.map(member => pad(member.moves.map((_, i) => member.pp[i] / Math.max(member.maxpp[i], 1)), MOVE_SLOTS)),
		hp: members.map(member => member.hp / Math.max(member.maxhp, 1)),
		status: members.map(member => Math.max(STATUS_NAMES.indexOf(member.status), 0)),
		fainted: members.map(member => flag(member.hp === 0)),
		is_active: members.map(member => flag(member === active)),
		stats: members.map(member => [member.maxhp, ...['atk', 'def', 'spa', 'spd', 'spe'].map(stat => member.stats[stat])]
			.map(value => value / 1000)),
		boosts: side.boosts.map(value => value / 6),
		ability: members.map(member => this.index('abilities', member === active ? member.ability : member.baseAbility)),
		item: members.map(member => this.index('items', member.item)),
		item_consumed: members.map(member => member.consumed),
		item_knocked_off: members.map(member => member.knockedOff),
		...this.activeVolatiles(side),
		locked: flag(this.requestLocked()),
		choice_locked: flag(choiceLocked),
		choice_move: choiceLocked ? side.choiceMove : 0,
	};
};

PlayerView.prototype.opponentObservation = function () {
	const side = this.sides[1 - this.me];
	const rows = Array.from({ length: TEAM_SLOTS }, (_, i) => side.members[i] || null);
	const per = (fn, empty) => rows.map(member => (member ? fn(member) : empty));
	const volatiles = this.activeVolatiles(side);
	return {
		species: per(member => this.speciesIndex(this.currentSpecies(member)), 0),
		types: per(member => this.typePair(this.currentTypes(member)), [0, 0]),
		moves: per(member => pad([...member.revealed].map(id => this.index('moves', id)).sort((a, b) => b - a).slice(0, MOVE_SLOTS), MOVE_SLOTS), Array(MOVE_SLOTS).fill(0)),
		level: per(member => member.level / 100, 0),
		hp: per(member => percentOf(member.hp, member.maxhp) / 100, 0),
		status: per(member => Math.max(STATUS_NAMES.indexOf(member.status), 0), 0),
		fainted: per(member => flag(member.hp === 0), 0),
		is_active: per(member => flag(member === side.active), 0),
		unseen: rows.map(member => flag(!member)),
		boosts: side.boosts.map(value => value / 6),
		ability: per(member => this.index('abilities', member.ability), 0),
		item: per(member => (member.itemRevealed ? this.index('items', member.itemPublic) : 0), 0),
		item_revealed: per(member => flag(member.itemRevealed), 0),
		item_consumed: per(member => flag(member.itemRevealed && member.consumed), 0),
		item_knocked_off: per(member => flag(member.itemRevealed && member.knockedOff), 0),
		...volatiles,
	};
};

PlayerView.prototype.legalActions = function () {
	const mask = Array(9).fill(false);
	if (this.ended || this.requestKind === 'none') {
		mask[0] = true;
		return mask;
	}
	const side = this.sides[this.me];
	const bench = side.members.filter(member => member !== side.active);
	const data = this.request.active ? this.request.active[0] : null;
	let canSwitch = true;
	if (this.requestKind === 'move') {
		const moves = data.moves;
		if (moves.length === 1 && moves[0].pp === undefined) mask[0] = true;
		else moves.forEach((move, i) => { mask[i] = !move.disabled; });
		canSwitch = !data.trapped && !this.trapHolds();
	}
	if (canSwitch) bench.forEach((member, k) => { mask[4 + k] = member.hp > 0; });
	return mask;
};

PlayerView.prototype.eventObservation = function () {
	const order = [this.me, 1 - this.me];
	const records = order.map(index => this.events[index]);
	const column = fn => records.map(fn);
	return {
		order: column(r => r.order),
		move: column(r => r.move),
		cant: column(r => r.cant),
		missed: column(r => flag(r.outcome & OUTCOME_MISS)),
		failed: column(r => flag(r.outcome & OUTCOME_FAILED)),
		critical: column(r => flag(r.outcome & OUTCOME_CRIT)),
		effectiveness: column(r => r.effect),
		hits: column(r => r.hits),
		entered: column(r => r.entry),
		hp_change: column(r => (r.hp || 0) / 100),
		status: column(r => r.status),
		fainted: column(r => r.fainted),
	};
};

PlayerView.prototype.observation = function () {
	if (this.me === null) throw new Error('No request has been received yet');
	const own = this.sides[this.me];
	const foe = this.sides[1 - this.me];
	const order = [this.me, 1 - this.me];
	const request = this.ended ? 'none' : this.requestKind;
	return {
		own: this.ownObservation(),
		opponent: this.opponentObservation(),
		turn: this.turn / 1000,
		request: ['none', 'move', 'switch'].map(kind => flag(kind === request)),
		mid_turn: flag(this.midTurn),
		legal_actions: this.legalActions(),
		weather: this.weather,
		weather_age: this.weatherAge,
		hazards: [own.hazards, foe.hazards].map(row => [...row]),
		trick_room: flag(this.trickRoom),
		trick_room_age: this.trickRoomAge,
		wish: order.map(index => flag(this.wish[index].pending)),
		wish_age: order.map(index => this.wish[index].age),
		healing_wish: order.map(index => flag(this.healingWish[index])),
		events: this.eventObservation(),
	};
};

module.exports = { PlayerView, loadTables };
