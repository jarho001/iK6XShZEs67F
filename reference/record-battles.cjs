#!/usr/bin/env node
'use strict';

/* Phase 1 differential oracle. Run:
 * node reference/record-battles.cjs --profile phase1 --count 300 --seed 1 --out work/diff/phase1.jsonl.gz
 * node reference/record-battles.cjs --profile phase1 --count 40 --seed 4 --out tests/fixtures/diff-phase1.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-status.allowed.json --out work/diff/status.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-status.allowed.json --out tests/fixtures/diff-status.jsonl.gz
 *
 * JSONL schema v2: header {type, schemaVersion, showdownCommit, seed, count,
 * acceptedMoves, ...}; battle {type, index, seed (four uint16 words), policySeed,
 * poolIndices, profile, teams (exact constructor inputs), initialOverrides: [],
 * initialState, steps: [{actions: [side0, side1], outcomes, state}], stats,
 * turnLimitHit}. Outcomes have exactly {kind, side, occurrence, value}; see
 * pokejax/engine/randomness.py. No draw-order correspondence is required.
 *
 * State: {turn: battle.turn, winner: 0|1|"tie"|null, ended: boolean,
 * sides: [{active: fixed slot 0..5, boosts: [atk,def,spa,spd,spe,accuracy,evasion],
 * request: "move"|"switch"|"none", legalActions: [nine bits], status,
 * statusCounter, types, item, ability, volatiles, sideConditions,
 * members: [{hp,maxhp,pp:[four slots],fainted,status,item,ability}]}],
 * weather, weatherTurns, pseudoWeather}. Conditions map ids to {layers, turns}.
 * Types and volatiles are null for fainted actives; all other fields are compared.
 * Default profile is isolation: --allowed path/to/task.json adds moves, abilities,
 * and items arrays to the supported registry. --profile phase1 uses the original
 * all-replaced plain-move and Ghost-lead exhaustion profiles.
 * Team slots NEVER change identity. Actions 4..8 enumerate all non-active fixed
 * slots in ascending order, INCLUDING fainted slots (which are illegal choices).
 * Waiting sides use action 0. Struggle uses action 0 and does not consume PP.
 * A replacement decision can leave turn unchanged; after it, Showdown can advance
 * to the next turn. Ended snapshots have two "none" requests.
 * Order occurrence 1 means a tie between queued instaswitch actions. The pinned
 * getActionSpeed uses the outgoing Pokemon's speed for switches/replacements;
 * the later runSwitch speedOrder shuffle has no observable effect in this suite.
 *
 * Every 20th battle uses a pool team with a Ghost lead on both sides. Moves are
 * drawn without replacement from accepted <=10 PP moves whose types cannot hit
 * Ghost. Surviving Ghosts honestly exhaust PP and use Struggle; no state mutation,
 * changed stats, altered legal-choice policy, or PP overrides are necessary.
 * Other battles independently sample two whole pool teams and all accepted moves.
 *
 * PRNG exceptions (draws are still consumed, counted, and guarded):
 * 1. eachEvent Update/BeforeTurn speed shuffles: runtime checks establish that
 *    EVERY target has zero handlers for that event, so iteration does nothing.
 * 2. runSwitch's allActive speed shuffle: this sets internal speedOrder solely
 *    for SwitchIn handler ordering (sim/battle.ts resolvePriority). Runtime checks
 *    allow only the independent sleep and toxic switch-in status handlers, with
 *    no hazards, items, abilities, or other SwitchIn handlers.
 *    It is distinct from queue replacement order, which IS recorded as context 1.
 * There are no other exceptions during steps. Constructor draws precede the
 * initial snapshot and are reproducible from the recorded battle seed; this
 * harness starts guarding after the constructor, before the first choice.
 * Policy/team sampling uses a separate seeded PRNG, never battle.prng.
 *
 * Unknown draw sites, unexpected shuffle entries, and duplicate forced addresses
 * are fatal. Runtime method scopes AND direct caller names identify call sites;
 * neither line numbers nor guessed denominator-only classifications are used.
 */

const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { parseArgs } = require('node:util');
const { once } = require('node:events');
const { createGzip } = require('node:zlib');
const { execFileSync } = require('node:child_process');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const vendor = path.join(__dirname, 'vendor/pokemon-showdown');
// Dex initialization merges mod tables in place. Preserve each override first.
// Gen 5 inherits gen6 -> gen7 -> gen8 -> base (gen4 inherits Gen 5).
const moveOverrides = [5, 6, 7, 8].map(gen => Object.fromEntries(
	Object.entries(require(`./vendor/pokemon-showdown/dist/data/mods/gen${gen}/moves`).Moves)
		.map(([id, entry]) => [id, { ...entry, ...(entry.flags ? { flags: { ...entry.flags } } : {}) }])
));
const { Battle } = require('./vendor/pokemon-showdown/dist/sim/battle');
const { Dex } = require('./vendor/pokemon-showdown/dist/sim/dex');
const { PRNG } = require('./vendor/pokemon-showdown/dist/sim/prng');
const dex = Dex.forFormat('gen5randombattle');
const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const TURN_LIMIT = 1000; // Showdown's own Endless Battle Clause also caps at 1000.
const EXCEPTIONS = ['empty-eachEvent-Update', 'empty-eachEvent-BeforeTurn', 'empty-switchIn-order'];

// Fail closed on unfamiliar meaningful move properties. maxMove/zMove and contest
// data cannot execute in Gen 5. Sources: sim/pokemon.ts getMoveTargets and
// battle-actions.ts trySpreadMoveHit/modifyDamage; gen5/moves.ts substitute.
// contact, bite, bullet, punch, pulse, slicing, sound, wind interact with
// abilities/items excluded here; protect, mirror, metronome require excluded
// moves; distance is for triples, nonsky for Sky Battles (sim/dex-moves.ts).
const PLAIN_KEYS = new Set([
	'num', 'accuracy', 'basePower', 'category', 'name', 'pp', 'priority', 'flags',
	'target', 'type', 'critRatio', 'contestType', 'isNonstandard', 'maxMove', 'zMove',
]);
const PLAIN_FLAGS = new Set(['contact', 'protect', 'mirror', 'metronome', 'punch',
	'bite', 'slicing', 'bullet', 'sound', 'distance', 'nonsky', 'pulse', 'wind']);
const PLAIN_TARGETS = new Set(['normal', 'allAdjacent', 'allAdjacentFoes']);
const OVERRIDE_KEYS = new Set(['inherit', 'basePower', 'accuracy', 'pp', 'type', 'category', 'flags']);
const EXPORTED_FIELDS = ['basePower', 'accuracy', 'pp', 'type', 'category', 'flags', 'target', 'priority', 'critRatio'];

function acceptedMoves(exported) {
	return exported.moves.filter(row => {
		const move = dex.moves.get(row.id);
		const raw = dex.data.Moves[move.id];
		const overrides = moveOverrides.map(table => table[move.id]).filter(Boolean);
		// Parent mods clear availability metadata with null; this has no battle effect.
		const plainOverrides = overrides.every(entry => Object.entries(entry).every(([key, value]) =>
			(OVERRIDE_KEYS.has(key) || (key === 'isNonstandard' && value === null)) &&
			typeof value !== 'function' &&
			(key !== 'flags' || Object.entries(value).every(([flag, value]) =>
				PLAIN_FLAGS.has(flag) && typeof value !== 'function'))));
		const plain = move.exists && move.gen <= 5 && ['Physical', 'Special'].includes(move.category) &&
			move.basePower > 0 && PLAIN_TARGETS.has(move.target) && dex.types.get(move.type).exists &&
			!move.id.startsWith('hiddenpower') && plainOverrides &&
			Object.entries(raw).every(([key, value]) => value == null ||
				(PLAIN_KEYS.has(key) && typeof value !== 'function')) &&
			Object.entries(raw.flags || {}).every(([flag, value]) =>
				PLAIN_FLAGS.has(flag) && typeof value !== 'function');
		if (plain && overrides.length) {
			for (const field of EXPORTED_FIELDS) {
				assert.deepEqual(row[field], move[field], `Export mismatch for ${move.id}.${field}`);
			}
		}
		return plain;
	}).map(row => row.id).sort();
}

function seedFor(seed, index, purpose) {
	const bytes = createHash('sha256').update(`diff-phase1:${seed}:${index}:${purpose}`).digest();
	return Array.from({ length: 4 }, (_, i) => bytes.readUInt16BE(2 * i));
}

function requestKind(side, ended = false) {
	const request = side.activeRequest;
	if (ended || !request || request.wait) return 'none';
	if (request.forceSwitch?.[0]) return 'switch';
	if (request.active) return 'move';
	throw new Error(`Unknown request: ${JSON.stringify(request)}`);
}

function conditions(states) {
	return Object.fromEntries(Object.keys(states).sort().map(id => [id, {
		layers: states[id].layers ?? 0, turns: states[id].duration ?? 0,
	}]));
}

function snapshot(battle, fixed) {
	return {
		turn: battle.turn,
		winner: !battle.ended ? null : battle.winner ?
			battle.sides.findIndex(side => side.name === battle.winner) : 'tie',
		ended: battle.ended,
		weather: battle.field.weather,
		weatherTurns: battle.field.weatherState.duration ?? 0,
		pseudoWeather: conditions(battle.field.pseudoWeather),
		sides: battle.sides.map((side, s) => ({
			active: fixed[s].indexOf(side.active[0]),
			boosts: BOOSTS.map(stat => side.active[0].boosts[stat]),
			status: side.active[0].status === 'fnt' ? '' : side.active[0].status,
			statusCounter: side.active[0].status === 'slp' ? side.active[0].statusState.time ?? 0 :
				side.active[0].status === 'tox' ? side.active[0].statusState.stage ?? 0 : 0,
			types: side.active[0].fainted ? null : side.active[0].getTypes().map(type => dex.toID(type)),
			item: side.active[0].item, ability: side.active[0].ability,
			volatiles: side.active[0].fainted ? null : Object.keys(side.active[0].volatiles).sort(),
			sideConditions: conditions(side.sideConditions),
			legalActions: Array.from({ length: 9 }, (_, action) =>
				Number(legalChoices(side, fixed[s], battle.ended).some(choice => choice.action === action))),
			request: requestKind(side, battle.ended),
			members: fixed[s].map(pokemon => ({
				hp: pokemon.hp, maxhp: pokemon.maxhp,
				pp: pokemon.moveSlots.map(move => move.pp), fainted: pokemon.fainted,
				status: pokemon.status === 'fnt' ? '' : pokemon.status, item: pokemon.item, ability: pokemon.ability,
			})),
		})),
	};
}

function legalChoices(side, fixed, ended = false) {
	const kind = requestKind(side, ended);
	if (kind === 'none') return [{ action: 0, choice: null }];
	const choices = [];
	if (kind === 'move') {
		const moves = side.activeRequest.active[0].moves;
		for (const [slot, move] of moves.entries()) {
			if (!move.disabled && (move.pp === undefined || move.pp > 0)) {
				assert(move.id === 'struggle' || side.active[0].moveSlots[slot].id === move.id);
				choices.push({ action: move.id === 'struggle' ? 0 : slot, choice: `move ${slot + 1}` });
			}
		}
	}
	if (kind === 'switch' || !side.activeRequest.active[0].trapped) {
		const inactive = fixed.filter(pokemon => pokemon !== side.active[0]);
		for (const [k, pokemon] of inactive.entries()) {
			if (!pokemon.fainted && pokemon.hp > 0) {
				choices.push({ action: 4 + k, choice: `switch ${side.pokemon.indexOf(pokemon) + 1}` });
			}
		}
	}
	assert(choices.length, 'No legal choices at decision point');
	return choices;
}

// The first frame after our wrapper is the actual pinned Showdown call site.
function directCaller() {
	return new Error().stack.split('\n')[3].trim().split(' ')[1];
}

// Each rule identifies a draw site and translates its result to an addressed outcome.
const OUTCOME_RULES = [
	{ kind: 'sleep_duration', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'slp' && caller === 'Battle.onStart' &&
		args[0] === 2 && args[1] === 5,
		address: scope => [scope.target.side.n, 0], value: value => value - 2 },
	{ kind: 'full_paralysis', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'par' && caller === 'Battle.onBeforeMove' &&
		args[0] === 1 && args[1] === 4,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'thaw', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'frz' && caller === 'Battle.onBeforeMove' &&
		args[0] === 1 && args[1] === 5,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'hit', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'hit' && caller === 'BattleActions.hitStepAccuracy' && args[1] === 100,
		address: scope => [scope.user.side.n, 0], value: value => Number(value) },
	{ kind: 'crit', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'crit' && caller === 'BattleActions.getDamage' && args[0] === 1 &&
		[16, 8, 4, 3, 2].includes(args[1]),
		address: scope => [scope.user.side.n, 0], value: value => Number(value) },
	{ kind: 'damage_roll', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'damage_roll' && caller === 'Battle.randomizer' && args.length === 1 && args[0] === 16,
		address: scope => [scope.user.side.n, 0], value: value => value },
	{ kind: 'order', method: 'shuffle', matches: scope => scope?.kind === 'queue',
		address: (scope, tied) => [0, tied.every(a => a.choice === 'instaswitch') ? 1 : 0],
		value: (value, list, start) => list[start].pokemon.side.n },
	{ kind: 'order', method: 'shuffle', matches: scope => scope?.kind === 'residual',
		address: () => [0, 2], value: (value, list, start) => list[start].effectHolder.side.n },
];

function instrument(battle, stats) {
	let outcomes = null;
	let addresses;
	let scope = null;
	let authorizedDraw = null;
	function within(value, fn) {
		const previous = scope;
		scope = value;
		try { return fn(); } finally { scope = previous; }
	}
	function allow(reason, fn) {
		assert.equal(authorizedDraw, null, 'Nested PRNG authorization');
		authorizedDraw = reason;
		try { return fn(); } finally { authorizedDraw = null; }
	}
	function record(kind, side, occurrence, value) {
		assert(outcomes, 'Random outcome outside a step');
		const address = `${kind}:${side}:${occurrence}`;
		assert(!addresses.has(address), `Duplicate outcome address ${address}`);
		addresses.add(address);
		outcomes.push({ kind, side, occurrence, value });
	}
	function wrap(object, method, makeScope) {
		const original = object[method];
		object[method] = function (...args) {
			return within(makeScope(...args), () => original.apply(this, args));
		};
	}
	wrap(battle.actions, 'hitStepAccuracy', (targets, user, move) => ({ kind: 'hit', user, move }));
	wrap(battle.actions, 'getDamage', (user, target, move) => ({ kind: 'crit', user, move }));
	wrap(battle, 'randomizer', () => ({ kind: 'damage_roll', user: battle.activePokemon, move: battle.activeMove }));
	wrap(battle.queue, 'sort', () => ({ kind: 'queue' }));
	wrap(battle, 'eachEvent', event => ({ kind: 'eachEvent', event }));
	wrap(battle.actions, 'runSwitch', () => ({ kind: 'runSwitch' }));
	wrap(battle, 'fieldEvent', event => ({ kind: event === 'Residual' ? 'residual' : 'fieldEvent' }));

	const random = battle.prng.random;
	battle.prng.random = function (...args) {
		if (!authorizedDraw) {
			throw new Error(`Unclassified PRNG.random(${args.join(', ')}), move=${battle.activeMove?.id}\n${new Error().stack}`);
		}
		stats.prngDraws++;
		return random.apply(this, args);
	};
	const chance = battle.randomChance;
	battle.randomChance = function (numerator, denominator) {
		const caller = directCaller();
		const statusScope = this.effect?.effectType === 'Status' ?
			{ kind: 'status', status: this.effect.id, target: this.effectState.target } : scope;
		const rule = OUTCOME_RULES.find(rule => rule.method === 'chance' &&
			rule.matches(statusScope, caller, [numerator, denominator]));
		assert(rule, `Unclassified randomChance(${numerator}, ${denominator}) at ${caller}`);
		const value = allow(rule.kind, () => chance.call(this, numerator, denominator));
		record(rule.kind, ...rule.address(statusScope), rule.value(value));
		return value;
	};
	const battleRandom = battle.random;
	battle.random = function (...args) {
		const caller = directCaller();
		const statusScope = this.effect?.effectType === 'Status' ?
			{ kind: 'status', status: this.effect.id, target: this.effectState.target } : scope;
		const rule = OUTCOME_RULES.find(rule => rule.method === 'random' && rule.matches(statusScope, caller, args));
		assert(rule, `Unclassified battle.random at ${caller}: ${args}`);
		const value = allow(rule.kind, () => battleRandom.apply(this, args));
		record(rule.kind, ...rule.address(statusScope), rule.value(value));
		return value;
	};
	const shuffle = battle.prng.shuffle;
	battle.prng.shuffle = function (list, start = 0, end = list.length) {
		const caller = directCaller();
		assert.equal(caller, 'Battle.speedSort', `Unclassified shuffle at ${caller}`);
		assert.equal(end - start, 2, 'Only two-sided singles ties are supported');
		const tied = list.slice(start, end);
		let rule;
		let exception;
		if (scope?.kind === 'queue') {
			assert(tied.every(a => ['move', 'switch', 'instaswitch'].includes(a.choice)) &&
				tied[0].pokemon.side !== tied[1].pokemon.side, 'Unexpected tied queue entries');
			const replacement = tied.every(a => a.choice === 'instaswitch');
			assert(replacement || tied.every(a => a.choice !== 'instaswitch'));
			rule = OUTCOME_RULES.find(row => row.method === 'shuffle' && row.matches(scope, caller, tied));
		} else if (scope?.kind === 'residual') {
			assert(tied.every(h => ['brn', 'psn', 'tox'].includes(h.effect.id)) &&
				tied[0].effectHolder.side !== tied[1].effectHolder.side);
			rule = OUTCOME_RULES.find(row => row.method === 'shuffle' && row.matches(scope));
		} else if (scope?.kind === 'eachEvent' && ['Update', 'BeforeTurn'].includes(scope.event)) {
			assert(tied.every(p => battle.findEventHandlers(p, scope.event).length === 0),
				`Nonempty ${scope.event} handlers: cannot ignore order`);
			exception = `empty-eachEvent-${scope.event}`;
		} else if (scope?.kind === 'runSwitch') {
			assert(tied.every(p => p.ability === 'noability' && !p.item &&
				battle.findPokemonEventHandlers(p, 'onSwitchIn').every(h => ['slp', 'tox'].includes(h.effect.id)) &&
				battle.findPokemonEventHandlers(p, 'onAnySwitchIn').length === 0 &&
				battle.findSideEventHandlers(p.side, 'onSideSwitchIn', undefined, p).length === 0 &&
				battle.findSideEventHandlers(p.side, 'onSwitchIn', undefined, p).length === 0 &&
				battle.findFieldEventHandlers(battle.field, 'onFieldSwitchIn', undefined, p).length === 0 &&
				battle.findFieldEventHandlers(battle.field, 'onSwitchIn', undefined, p).length === 0 &&
				battle.findBattleEventHandlers('onSwitchIn', undefined, p).length === 0),
				'Nonempty SwitchIn handlers: cannot ignore speedOrder');
			exception = 'empty-switchIn-order';
		} else {
			const entries = tied.map(entry => entry.choice || entry.effect?.id || entry.species?.id || typeof entry);
			throw new Error(`Unclassified shuffle: scope=${scope?.kind}, event=${scope?.event}, entries=${entries}\n${new Error().stack}`);
		}
		const value = allow(exception || 'order', () => shuffle.call(this, list, start, end));
		if (exception) stats.exceptions[exception]++;
		else record(rule.kind, ...rule.address(scope, tied), rule.value(value, list, start));
		return value;
	};
	return {
		begin() { assert.equal(outcomes, null); outcomes = []; addresses = new Set(); },
		end() { const result = outcomes; outcomes = null; return result; },
	};
}

function newStats() {
	return { steps: 0, outcomes: Object.fromEntries(['order', 'hit', 'crit', 'damage_roll'].map(kind => [kind, 0])),
		misses: 0, crits: 0, ties: 0, struggles: 0, doubleKOs: 0, forcedReplacements: 0,
		prngDraws: 0, exceptions: Object.fromEntries(EXCEPTIONS.map(key => [key, 0])) };
}

function isolationMember(member, moves, options, policy) {
	const candidates = [...moves];
	policy.shuffle(candidates);
	const allowed = new Set(moves);
	const taskMoves = new Set(options.taskMoves || []);
	const kept = member.moves.map(id => dex.toID(id)).filter(id => allowed.has(id));
	const replacements = candidates.filter(id => !kept.includes(id));
	// Task moves get extra chances without allowing duplicate move slots.
	const weighted = replacements.flatMap(id => taskMoves.has(id) ? [id, id, id, id] : [id]);
	policy.shuffle(weighted);
	for (const id of weighted) if (kept.length < 4 && !kept.includes(id)) kept.push(id);
	return { ...structuredClone(member), moves: kept,
		ability: (options.abilities || []).includes(dex.toID(member.ability)) ? member.ability : 'noability',
		item: (options.items || []).includes(dex.toID(member.item)) ? member.item : '' };
}

function playBattle(index, masterSeed, pool, moves, ghostTeams, exhaustionMoves, options = {}) {
	const seed = seedFor(masterSeed, index, 'battle');
	const policySeed = seedFor(masterSeed, index, 'policy');
	const policy = new PRNG(policySeed);
	const exhaustion = options.profile !== 'isolation' && index % 20 === 0;
	const poolIndices = exhaustion ? Array(2).fill(policy.sample(ghostTeams)) :
		[policy.random(pool.teams.length), policy.random(pool.teams.length)];
	const teams = poolIndices.map(i => pool.teams[i].map(member => {
		if (options.profile === 'isolation') return isolationMember(member, moves, options, policy);
		const candidates = [...(exhaustion ? exhaustionMoves : moves)];
		policy.shuffle(candidates);
		return { ...structuredClone(member), ability: 'noability', item: '', moves: candidates.slice(0, 4) };
	}));
	const battle = new Battle({ formatid: 'gen5randombattle', seed,
		p1: { name: 'Side 0', team: structuredClone(teams[0]) },
		p2: { name: 'Side 1', team: structuredClone(teams[1]) },
	});
	try {
		assert.equal(battle.turn, 1);
		const fixed = battle.sides.map(side => [...side.pokemon]);
		for (const [s, members] of fixed.entries()) {
			for (const [slot, pokemon] of members.entries()) {
				assert.equal(pokemon.ability, dex.toID(teams[s][slot].ability));
				assert.equal(pokemon.item, dex.toID(teams[s][slot].item));
				assert.deepEqual(pokemon.moveSlots.map(m => m.id), teams[s][slot].moves);
			}
		}
		const stats = newStats();
		const recorder = instrument(battle, stats);
		const initialState = snapshot(battle, fixed);
		const steps = [];
		let turnLimitHit = false;
		while (!battle.ended) {
			if (battle.turn > TURN_LIMIT || steps.length > 3 * TURN_LIMIT) {
				turnLimitHit = true;
				break;
			}
			const selected = battle.sides.map((side, s) => policy.sample(legalChoices(side, fixed[s])));
			const faintedBefore = fixed.map(members => members.filter(p => p.fainted).length);
			for (const side of battle.sides) {
				if (requestKind(side) === 'switch') stats.forcedReplacements++;
			}
			recorder.begin();
			const logStart = battle.log.length;
			for (const [s, selectedChoice] of selected.entries()) {
				if (selectedChoice.choice !== null) {
					assert(battle.choose(`p${s + 1}`, selectedChoice.choice),
					`Rejected choice ${selectedChoice.choice}: ${battle.sides[s].choice.error}`);
				}
			}
			const outcomes = recorder.end();
			const state = snapshot(battle, fixed);
			steps.push({ actions: selected.map(choice => choice.action), outcomes, state });
			for (const outcome of outcomes) {
				stats.outcomes[outcome.kind] = (stats.outcomes[outcome.kind] || 0) + 1;
				if (outcome.kind === 'hit' && outcome.value === 0) stats.misses++;
				if (outcome.kind === 'crit' && outcome.value === 1) stats.crits++;
				if (outcome.kind === 'order') stats.ties++;
			}
			stats.struggles += battle.log.slice(logStart).filter(line => /^\|move\|[^|]+\|Struggle\|/.test(line)).length;
			if (fixed.every((members, s) => members.filter(p => p.fainted).length > faintedBefore[s])) stats.doubleKOs++;
			if (battle.ended && battle.turn >= TURN_LIMIT) turnLimitHit = true;
		}
		stats.steps = steps.length;
		return { type: 'battle', index, seed, policySeed, poolIndices,
			profile: exhaustion ? 'natural-pp-exhaustion' : 'random', teams,
			initialOverrides: [], initialState, steps, stats, turnLimitHit };
	} finally { battle.destroy(); }
}

async function main() {
	const { values } = parseArgs({ options: {
		count: { type: 'string', default: '300' }, seed: { type: 'string', default: '1' },
		out: { type: 'string', default: 'work/diff/phase1.jsonl.gz' },
		profile: { type: 'string', default: 'isolation' }, allowed: { type: 'string' },
	} });
	const count = Number(values.count);
	const seed = Number(values.seed);
	assert(Number.isSafeInteger(count) && count > 0, '--count must be a positive safe integer');
	assert(Number.isSafeInteger(seed), '--seed must be a safe integer');
	const lock = JSON.parse(fs.readFileSync(path.join(root, 'reference-lock.json'), 'utf8'));
	assert.equal(execFileSync('git', ['-C', vendor, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(), lock.commit);
	const pool = JSON.parse(fs.readFileSync(path.join(root, 'work/teams/gen5-teams.json'), 'utf8'));
	assert.equal(pool.showdownCommit, lock.commit);
	const supported = JSON.parse(execFileSync('python', ['-m', 'pokejax.support'], { cwd: root, encoding: 'utf8' }));
	const extra = values.allowed ? JSON.parse(fs.readFileSync(values.allowed, 'utf8')) : {};
	const plainMoves = values.profile === 'phase1' ?
		new Set(acceptedMoves(JSON.parse(fs.readFileSync(path.join(root, 'data/gen5-data-v1.json'))))) : null;
	const baseMoves = plainMoves ? supported.moves.filter(id => plainMoves.has(id)) : supported.moves;
	const moves = [...new Set([...baseMoves, ...(extra.moves || []).map(id => dex.toID(id))])].sort();
	const options = { profile: values.profile, taskMoves: (extra.moves || []).map(id => dex.toID(id)),
		abilities: [...supported.abilities, ...(extra.abilities || []).map(id => dex.toID(id))],
		items: [...supported.items, ...(extra.items || []).map(id => dex.toID(id))] };
	assert(['phase1', 'isolation'].includes(values.profile));
	assert(moves.length >= 4);
	assert(!Object.values(dex.abilities.get('noability')).some(value => typeof value === 'function'));
	const ghostTeams = pool.teams.flatMap((team, i) => dex.species.get(team[0].species).types.includes('Ghost') ? [i] : []);
	const exhaustionMoves = moves.filter(id => dex.moves.get(id).pp <= 10 && !dex.getImmunity(dex.moves.get(id).type, 'Ghost'));
	assert(ghostTeams.length && exhaustionMoves.length >= 4, 'Cannot construct natural PP exhaustion profile');
	console.log(`Accepted plain damaging moves: ${moves.length} (${moves.join(', ')})`);
	const output = path.resolve(root, values.out);
	fs.mkdirSync(path.dirname(output), { recursive: true });
	const file = fs.createWriteStream(output);
	const stream = output.endsWith('.gz') ? createGzip({ level: 9 }) : file;
	const complete = once(file, 'close');
	if (stream !== file) stream.pipe(file);
	async function write(record) {
		if (!stream.write(JSON.stringify(record) + '\n')) await once(stream, 'drain');
	}
	const summary = { battles: count, ...newStats(), meanLength: 0, turnLimitHits: 0 };
	try {
		await write({ type: 'header', schemaVersion: 2, showdownCommit: lock.commit, seed, count,
			format: 'gen5randombattle', acceptedMoves: moves, allowed: options, turnLimit: TURN_LIMIT,
			poolSeed: pool.seed, policy: 'uniform over all legal moves and switches',
			exhaustion: { every: 20, moves: exhaustionMoves, initialOverrides: false },
			prngExceptions: EXCEPTIONS });
		for (let index = 0; index < count; index++) {
			let record;
			try { record = playBattle(index, seed, pool, moves, ghostTeams, exhaustionMoves, options); }
			catch (error) { throw new Error(`Battle ${index}, seed ${seedFor(seed, index, 'battle')}: ${error.stack}`); }
			await write(record);
			for (const [key, value] of Object.entries(record.stats)) {
				if (typeof value === 'number') summary[key] += value;
				else for (const [name, n] of Object.entries(value)) summary[key][name] = (summary[key][name] || 0) + n;
			}
			summary.turnLimitHits += Number(record.turnLimitHit);
			if ((index + 1) % 20 === 0) console.log(`Recorded ${index + 1}/${count} battles`);
		}
		stream.end();
		await complete;
	} catch (error) {
		stream.destroy(); file.destroy();
		await complete.catch(() => {});
		fs.rmSync(output, { force: true }); // Never leave a successful-looking partial fixture.
		throw error;
	}
	summary.meanLength = summary.steps / count;
	console.log(JSON.stringify(summary, null, 2));
	console.log(`Wrote ${values.out} (${fs.statSync(output).size} bytes)`);
}

module.exports = { acceptedMoves, seedFor, snapshot, legalChoices, instrument, newStats, playBattle, isolationMember };
if (require.main === module) main().catch(error => { console.error(error.stack); process.exitCode = 1; });
