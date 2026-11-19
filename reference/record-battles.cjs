#!/usr/bin/env node
'use strict';

/* Differential oracle. Run:
 * node reference/record-battles.cjs --profile phase1 --count 300 --seed 1 --out work/diff/phase1.jsonl.gz
 * node reference/record-battles.cjs --profile phase1 --count 40 --seed 4 --out tests/fixtures/diff-phase1.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-status.allowed.json --out work/diff/status.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-status.allowed.json --out tests/fixtures/diff-status.jsonl.gz
 *
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-boosts.allowed.json --out work/diff/boosts.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-boosts.allowed.json --out tests/fixtures/diff-boosts.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-secondaries.allowed.json --out work/diff/secondaries.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-secondaries.allowed.json --out tests/fixtures/diff-secondaries.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-effects2.allowed.json --out work/diff/effects2.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-effects2.allowed.json --out tests/fixtures/diff-effects2.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-field.allowed.json --out work/diff/field.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-field.allowed.json --out tests/fixtures/diff-field.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-turns.allowed.json --out work/diff/turns.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-turns.allowed.json --out tests/fixtures/diff-turns.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-resumable.allowed.json --out work/diff/resumable.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-resumable.allowed.json --out tests/fixtures/diff-resumable.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-items.allowed.json --out work/diff/items.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-items.allowed.json --out tests/fixtures/diff-items.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-abilities1.allowed.json --out work/diff/abilities1.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-abilities1.allowed.json --out tests/fixtures/diff-abilities1.jsonl.gz
 * node reference/record-battles.cjs --count 300 --seed 1 --allowed tests/fixtures/diff-abilities2.allowed.json --out work/diff/abilities2.jsonl.gz
 * node reference/record-battles.cjs --count 40 --seed 4 --allowed tests/fixtures/diff-abilities2.allowed.json --out tests/fixtures/diff-abilities2.jsonl.gz
 *
 * JSONL schema v2: header {type, schemaVersion, showdownCommit, seed, count,
 * acceptedMoves, ...}; battle {type, index, seed (four uint16 words), policySeed,
 * poolIndices, profile, teams (exact constructor inputs), initialOverrides: [],
 * initialOutcomes (optional for earlier files), initialState, steps: [{actions: [side0, side1], outcomes, state}], stats,
 * turnLimitHit}. Outcomes have exactly {kind, side, occurrence, value}; see
 * pokejax/engine/randomness.py. No draw-order correspondence is required.
 *
 * State: {turn: battle.turn, winner: 0|1|"tie"|null, ended: boolean,
 * sides: [{active: fixed slot 0..5, boosts: [atk,def,spa,spd,spe,accuracy,evasion],
 * request: "move"|"switch"|"none", legalActions: [nine bits], status,
 * statusCounter, types, item, ability, volatiles, sideConditions,
 * members: [{hp,maxhp,pp:[four slots],fainted,status,item,itemConsumed,
 * itemKnockedOff,ability,abilityPublic (optional)}]}],
 * weather, weatherTurns, pseudoWeather}. Conditions map ids to {layers, turns}.
 * Types and volatiles are null for fainted actives; all other fields are compared.
 * Confusion, Disable and Slow Start counters use volatileCounters; Disable also
 * adds volatileMoves. Missing fields default to zero for earlier schema v2 files.
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
 * grouped SwitchIn handlers use new active speeds and context 5 when hazards are
 * present; context 1 still records replacement queue ties.
 * Order occurrence 3 means a Speed tie during the weather's eachEvent('Weather').
 * Order occurrence 4 means a Speed tie during WeatherChange.
 * Residual ties use occurrence 2 for default-order volatiles, then 6/7/8 for
 * Poison/Toxic, burn and partial trapping. Roost, ability and end-item residual
 * groups have separate one-slot outcome kinds. Same-side handler permutations
 * are recorded as verified no-effect exceptions.
 *
 * Every 20th battle uses a pool team with a Ghost lead on both sides. Moves are
 * drawn without replacement from accepted <=10 PP moves whose types cannot hit
 * Ghost. Surviving Ghosts honestly exhaust PP and use Struggle; no state mutation,
 * changed stats, altered legal-choice policy, or PP overrides are necessary.
 * Other battles independently sample two whole pool teams and all accepted moves.
 *
 * PRNG exceptions (draws are still consumed, counted, and guarded):
 * 1. Update/BeforeTurn shuffles with only independent per-holder callbacks.
 * 2. The runSwitch insertion tie only permutes two consecutive actions that
 *    runSwitch consumes as one group. fieldEvent enumerates that group by side.
 *    The separate runSwitch speed shuffle is recorded at order context 5
 *    for grouped hazards and order-sensitive ability handlers, including start.
 *    Slow Start's isolated start callback is independent of other switch-ins.
 * 3. Frisk draws from one foe in singles. Two No Guard accuracy handlers
 *    return the same result. Ability-triggered and repeated WeatherChange
 *    draws have no handlers in the isolation format.
 * 4. A Pokemon's Choice lock and Disable move-lock callbacks disable disjoint
 *    move slots, so their equal-priority DisableMove order is independent.
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
const START_UPDATE_ABILITIES = new Set([
	'immunity', 'insomnia', 'limber', 'magmaarmor', 'oblivious', 'owntempo', 'vitalspirit', 'waterveil',
]);
const SWITCH_START_ABILITIES = new Set([
	'airlock', 'anticipation', 'cloudnine', 'download', 'drizzle', 'drought', 'frisk',
	'intimidate', 'moldbreaker', 'pressure', 'sandstream', 'snowwarning', 'teravolt',
	'turboblaze', 'unnerve',
]);
const WEATHER_START_ABILITIES = new Set(['drizzle', 'drought', 'sandstream', 'snowwarning']);

function isolationBattle(seed) {
	return new Battle({ formatid: 'gen5randombattle', seed });
}

const EXCEPTIONS = [
	'empty-eachEvent-Update',
	'berry-eachEvent-Update',
	'empty-eachEvent-BeforeTurn',
	'empty-switchIn-order',
	'benign-item-switchIn-order',
	'independent-slowstart-switchIn-order',
	'equal-speed-runSwitch-insertion',
	'same-side-residual-order',
	'choice-disable-move-order',
];

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

function trackPublicAbilities(battle) {
	const revealed = new WeakMap();
	const add = battle.add;
	battle.add = function (...parts) {
		add.apply(this, parts);
		const [event, target, value] = parts;
		if (event === '-activate' && dex.toID(String(value).replace(/^ability:\s*/i, '')) === 'mummy') {
			for (const holder of this.sides.flatMap(side => side.pokemon)) {
				if ((holder === target || holder === parts[3]) && holder.ability === 'mummy') {
					revealed.set(holder, 'mummy');
				}
			}
			return;
		}
		const tags = parts.slice(2).filter(part => typeof part === 'string');
		const named = tags.find(tag => /^(?:\[from\] )?ability: /.test(tag));
		const from = tags.find(tag => tag.startsWith('[from] '));
		const name = event === '-ability' ? value :
			named?.replace(/^(?:\[from\] )?ability: /, '') ||
			(['-immune', '-fail'].includes(event) ? from?.slice(7) : undefined);
		if (!name) return;
		const id = dex.toID(name);
		if (!dex.abilities.get(id).exists) return;
		const source = tags.find(tag => tag.startsWith('[of] '))?.slice(5);
		const holders = [target, source].flatMap(ref => this.sides.flatMap(side => side.pokemon)
			.filter(p => p === ref || String(p) === ref));
		const holder = holders.find(p => p.ability === id);
		if (holder) revealed.set(holder, id);
	};
	return revealed;
}

function snapshot(battle, fixed, publicAbilities) {
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
			itemConsumed: !side.active[0].item && Boolean(side.active[0].lastItem) &&
				!side.active[0].itemKnockedOff,
			itemKnockedOff: side.active[0].itemKnockedOff,
			volatiles: side.active[0].fainted ? null : Object.keys(side.active[0].volatiles).sort(),
			...(!side.active[0].fainted && (side.active[0].volatiles.confusion ||
				side.active[0].volatiles.disable || side.active[0].abilityState.counter) ? {
				volatileCounters: {
					...(side.active[0].volatiles.confusion ? { confusion: side.active[0].volatiles.confusion.time } : {}),
					...(side.active[0].volatiles.disable ? { disable: side.active[0].volatiles.disable.duration } : {}),
					...(side.active[0].ability === 'slowstart' && side.active[0].abilityState.counter ?
						{ slowstart: side.active[0].abilityState.counter } : {}),
				},
			} : {}),
			...(!side.active[0].fainted && side.active[0].volatiles.disable ? {
				volatileMoves: { disable: side.active[0].volatiles.disable.move },
			} : {}),
			sideConditions: conditions(side.sideConditions),
			legalActions: Array.from({ length: 9 }, (_, action) =>
				Number(legalChoices(side, fixed[s], battle.ended).some(choice => choice.action === action))),
			request: requestKind(side, battle.ended),
			members: fixed[s].map(pokemon => ({
				hp: pokemon.hp, maxhp: pokemon.maxhp,
				pp: pokemon.moveSlots.map(move => move.pp), fainted: pokemon.fainted,
				status: pokemon.status === 'fnt' ? '' : pokemon.status,
				item: pokemon.item,
				itemConsumed: !pokemon.item && Boolean(pokemon.lastItem) && !pokemon.itemKnockedOff,
				itemKnockedOff: pokemon.itemKnockedOff,
				ability: pokemon.ability,
				...(publicAbilities ? { abilityPublic: publicAbilities.get(pokemon) || '' } : {}),
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
		const hardLock = side.active[0].getLockedMove();
		if (hardLock) {
			assert.equal(moves.length, 1, `Hard-locked request should expose one move: ${hardLock}`);
			const move = moves[0];
			if (!move.disabled && (move.pp === undefined || move.pp > 0)) {
				choices.push({ action: 0, choice: 'move 1' });
			}
		} else {
			for (const [slot, move] of moves.entries()) {
				if (!move.disabled && (move.pp === undefined || move.pp > 0)) {
					assert(move.id === 'struggle' || side.active[0].moveSlots[slot].id === move.id);
					choices.push({ action: move.id === 'struggle' ? 0 : slot, choice: `move ${slot + 1}` });
				}
			}
		}
	}
	// maybeTrapped can describe only a possible foe ability. Showdown still
	// accepts the switch unless the active Pokemon is actually trapped.
	if (kind === 'switch' || !side.active[0].trapped) {
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

function residualOrderOccurrence(order) {
	if (order === false || order === 5 || order === 17 || order === 25) return 2;
	if (order === 9) return 6;
	if (order === 10) return 7;
	if (order === 13) return 8;
	throw new Error(`Unclassified residual order: ${order}`);
}

// Each rule identifies a draw site and translates its result to an addressed outcome.
const OUTCOME_RULES = [
	{ kind: 'confusion_duration', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'confusion' && caller === 'Battle.onStart' &&
		args[0] === 2 && args[1] === 6,
		address: scope => [scope.target.side.n, 0], value: value => value - 2 },
	{ kind: 'confusion_hit', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'confusion' && caller === 'Battle.onBeforeMove' &&
		args[0] === 1 && args[1] === 2,
		address: scope => [scope.target.side.n, 0], value: value => Number(!value) },
	{ kind: 'confusion_damage_roll', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'confusion_damage_roll' && caller === 'Battle.randomizer' &&
		args.length === 1 && args[0] === 16,
		address: scope => [scope.user.side.n, 0], value: value => value },
	{ kind: 'secondary', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'secondaries' && caller === 'BattleActions.secondaries' &&
		args.length === 1 && args[0] === 100,
		address: scope => [scope.user.side.n,
			(scope.hitIndex * 2) + scope.occurrence],
		value: (value, scope) => {
			const secondary = scope.moveData.secondaries[scope.occurrence++];
			const chance = secondary.chance === undefined ? 100 :
				(secondary.boosts || secondary.self) ? secondary.chance % 256 : secondary.chance;
			return Number(value < chance);
		} },
	{ kind: 'secondary_status', method: 'sample', matches: (scope, caller, args) =>
		scope?.kind === 'secondaries' && scope.move.id === 'triattack' && caller === 'Battle.onHit' &&
		JSON.stringify(args[0]) === JSON.stringify(['brn', 'par', 'frz']),
		address: scope => [scope.user.side.n, (scope.hitIndex * 2) + scope.occurrence - 1],
		value: value => ['brn', 'par', 'frz'].indexOf(value) },
	{ kind: 'drag_in', method: 'sample', matches: (scope, caller, args) =>
		scope?.kind === 'drag' && caller === 'Battle.getRandomSwitchable' &&
		Array.isArray(args[0]),
		address: scope => [scope.side.n, 0],
		value: (value, scope) => {
			const slot = scope.fixed[scope.side.n].indexOf(value);
			assert(slot >= 0, 'Dragged Pokemon is missing from its fixed team');
			return slot;
		} },
	{ kind: 'hit_count', method: 'sample', matches: (scope, caller, args) =>
		scope?.kind === 'hit_loop' && caller === 'BattleActions.hitStepMoveHitLoop' &&
		Array.isArray(args[0]) && args[0].length === 20,
		address: scope => [scope.user.side.n, 0], value: value => value },
	{ kind: 'sleep_duration', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'slp' && caller === 'Battle.onStart' &&
		args[0] === 2 && args[1] === 5,
		address: scope => [scope.target.side.n, 0], value: value => value - 2 },
	{ kind: 'lock_duration', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'lockedmove' && caller === 'Battle.onStart' &&
		args[0] === 2 && args[1] === 4,
		address: scope => [scope.target.side.n, 0], value: value => value - 2 },
	{ kind: 'trap_duration', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'hit_loop' && caller === 'Battle.durationCallback' &&
		['bind', 'clamp', 'firespin', 'magmastorm', 'sandtomb', 'wrap', 'whirlpool'].includes(scope.move.id) &&
		args[0] === 5 && args[1] === 7,
		address: scope => [scope.user.side.n, 0], value: value => value - 5 },
	{ kind: 'magnitude', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'magnitude' && scope.move.id === 'magnitude' &&
		caller === 'Battle.onModifyMove' && args.length === 1 && args[0] === 100,
		address: scope => [scope.user.side.n, 0], value: value => value },
	{ kind: 'full_paralysis', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'par' && caller === 'Battle.onBeforeMove' &&
		args[0] === 1 && args[1] === 4,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'stall_success', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'stall' && caller === 'Battle.onStallMove' && args[0] === 1 &&
		Number.isInteger(args[1]) && args[1] >= 2,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'thaw', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'status' && scope.status === 'frz' && caller === 'Battle.onBeforeMove' &&
		args[0] === 1 && args[1] === 5,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'ability_chance', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'ability' && ['static', 'flamebody', 'cursedbody', 'poisontouch'].includes(scope.ability) &&
		['Battle.onDamagingHit', 'Battle.onSourceDamagingHit'].includes(caller) &&
		args[0] === 3 && args[1] === 10,
		address: scope => [scope.target.side.n, 2 * scope.hitIndex +
			(scope.ability === 'poisontouch' ? 1 : 0)], value: value => Number(value) },
	{ kind: 'residual_chance', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'ability' && scope.ability === 'shedskin' &&
		caller === 'Battle.onResidual' && args[0] === 33 && args[1] === 100,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'harvest', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'ability' && scope.ability === 'harvest' &&
		caller === 'Battle.onResidual' && args[0] === 1 && args[1] === 2,
		address: scope => [scope.target.side.n, 0], value: value => Number(value) },
	{ kind: 'hit', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'hit' && caller === 'BattleActions.hitStepAccuracy' && args[1] === 100,
		address: scope => [scope.user.side.n, 0], value: value => Number(value) },
	{ kind: 'crit', method: 'chance', matches: (scope, caller, args) =>
		scope?.kind === 'crit' && caller === 'BattleActions.getDamage' && args[0] === 1 &&
		[16, 8, 4, 3, 2].includes(args[1]),
		address: scope => [scope.user.side.n, scope.occurrence], value: value => Number(value) },
	{ kind: 'damage_roll', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'damage_roll' && caller === 'Battle.randomizer' && args.length === 1 && args[0] === 16,
		address: scope => [scope.user.side.n, scope.occurrence], value: value => value },
	{ kind: 'effect_spore', method: 'random', matches: (scope, caller, args) =>
		scope?.kind === 'ability' && scope.ability === 'effectspore' &&
		caller === 'Battle.onDamagingHit' && args.length === 1 && args[0] === 100,
		address: scope => [scope.target.side.n, scope.hitIndex], value: value => value },
	{ kind: 'order', method: 'shuffle', matches: scope => scope?.kind === 'queue',
		address: (scope, tied) => [0, tied.every(a => a.choice === 'instaswitch') ? 1 : 0],
		value: (value, list, start) => list[start].pokemon.side.n },
	{ kind: 'order', method: 'shuffle', matches: scope => scope?.kind === 'residual',
		address: (scope, tied) => [0, residualOrderOccurrence(tied[0].order)],
		value: (value, list, start) => list[start].effectHolder.side.n },
	{ kind: 'order', method: 'shuffle', matches: scope =>
		scope?.kind === 'eachEvent' && scope.event === 'Weather',
		address: () => [0, 3], value: (value, list, start) => list[start].side.n },
	{ kind: 'order', method: 'shuffle', matches: scope =>
		scope?.kind === 'eachEvent' && scope.event === 'WeatherChange',
		address: () => [0, 4], value: (value, list, start) => list[start].side.n },
	{ kind: 'order', method: 'shuffle', matches: scope => scope?.kind === 'runSwitch',
		address: () => [0, 5], value: (value, list, start) => list[start].side.n },
];

function instrument(battle, stats, fixedMembers, { groupedSwitchInOnly = false } = {}) {
	let fixed = fixedMembers || battle.sides.map(side => side && [...side.pokemon]);
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
	function record(kind, side, occurrence, value, detail = '') {
		assert(outcomes, 'Random outcome outside a step');
		const address = `${kind}:${side}:${occurrence}`;
		assert(!addresses.has(address),
			`Duplicate outcome address ${address}: ${addresses.get(address)} vs ${detail}`);
		addresses.set(address, detail);
		outcomes.push({ kind, side, occurrence, value });
	}
	function wrap(object, method, makeScope) {
		const original = object[method];
		object[method] = function (...args) {
			return within(makeScope(...args), () => original.apply(this, args));
		};
	}
	wrap(battle.actions, 'hitStepAccuracy', (targets, user, move) => ({ kind: 'hit', user, move }));
	wrap(battle.actions, 'selfDrops', (targets, user, move, moveData) => ({ kind: 'selfDrops', user, move, moveData }));
	wrap(battle.actions, 'getDamage', (user, target, move) => ({
		kind: 'crit', user, move, occurrence: Math.max(0, (move.hit || 1) - 1),
	}));
	wrap(battle.actions, 'secondaries', (targets, user, move, moveData) =>
		({ kind: 'secondaries', user, move, moveData, hitIndex: (move.hit || 1) - 1, occurrence: 0 }));
	wrap(battle.actions, 'hitStepMoveHitLoop', (targets, user, move) =>
		({ kind: 'hit_loop', user, move }));
	wrap(battle.actions, 'useMoveInner', (move, user) => ({ kind: 'magnitude', user, move }));
	wrap(battle.actions, 'getConfusionDamage', user => ({ kind: 'confusionDamage', user }));
	wrap(battle, 'randomizer', () => ({
		kind: scope?.kind === 'confusionDamage' ? 'confusion_damage_roll' : 'damage_roll',
		user: scope?.kind === 'confusionDamage' ? scope.user : battle.activePokemon,
		move: battle.activeMove,
		occurrence: scope?.kind === 'confusionDamage' ? 0 : Math.max(0, (battle.activeMove?.hit || 1) - 1),
	}));
	wrap(battle.queue, 'sort', () => ({ kind: 'queue' }));
	wrap(battle.queue, 'insertChoice', choices => ({ kind: 'insertChoice', choices }));
	wrap(battle, 'eachEvent', (event, effect) => ({ kind: 'eachEvent', event, effect }));
	wrap(battle.actions, 'runSwitch', pokemon => ({ kind: 'runSwitch', pokemon,
		queued: battle.queue.list.filter(a => a.choice === 'runSwitch').map(a => a.pokemon) }));
	wrap(battle, 'getRandomSwitchable', side => ({ kind: 'drag', side, fixed }));
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
		const statusScope = this.effect?.effectType === 'Status' ||
			['confusion', 'lockedmove', 'stall'].includes(this.effect?.id) ?
			{ kind: 'status', status: this.effect.id, target: this.effectState.target } : scope;
		const abilityScope = ['static', 'flamebody', 'cursedbody', 'poisontouch', 'shedskin', 'harvest'].includes(this.effect?.id) ? {
			kind: 'ability', ability: this.effect.id, target: this.effectState.target,
			hitIndex: Math.max(0, (scope?.move?.hit || 1) - 1),
		} : statusScope;
		const effectiveScope = abilityScope?.status === 'stall' ?
			{ kind: 'stall', target: abilityScope.target } : abilityScope;
		const rule = OUTCOME_RULES.find(rule => rule.method === 'chance' &&
			rule.matches(effectiveScope, caller, [numerator, denominator]));
		assert(rule, `Unclassified randomChance(${numerator}, ${denominator}) at ${caller}`);
		const value = allow(rule.kind, () => chance.call(this, numerator, denominator));
		record(rule.kind, ...rule.address(effectiveScope), rule.value(value));
		return value;
	};
	const battleRandom = battle.random;
	battle.random = function (...args) {
		const caller = directCaller();
		if (caller === 'BattleQueue.insertChoice' && scope?.kind === 'insertChoice' &&
			scope.choices?.choice === 'runSwitch') {
			assert(args.length === 2 && Number.isInteger(args[0]) && Number.isInteger(args[1]) &&
				args[1] - args[0] === 2, 'Unexpected runSwitch insertion tie');
			const queued = battle.queue.list[args[0]];
			assert(queued?.choice === 'runSwitch' &&
				queued.pokemon.side !== scope.choices.pokemon.side,
			'Insertion must only permute two grouped runSwitch actions');
			const value = allow('equal-speed-runSwitch-insertion', () =>
				battleRandom.apply(this, args));
			stats.exceptions['equal-speed-runSwitch-insertion']++;
			return value;
		}
		const statusScope = caller === 'Battle.onStart' &&
			(this.effect?.effectType === 'Status' || ['confusion', 'lockedmove'].includes(this.effect?.id)) ?
			{ kind: 'status', status: this.effect.id, target: this.effectState.target } : scope;
		const randomScope = this.effect?.id === 'effectspore' ? {
			kind: 'ability', ability: this.effect.id, target: this.effectState.target,
			hitIndex: Math.max(0, (scope?.move?.hit || 1) - 1),
		} : statusScope;
		if (scope?.kind === 'selfDrops' && caller === 'BattleActions.selfDrops' && args[0] === 100 &&
			scope.moveData.self?.boosts && scope.moveData.self.chance === undefined) {
			return allow('deterministic-self-drop', () => battleRandom.apply(this, args));
		}
		const rule = OUTCOME_RULES.find(rule => rule.method === 'random' && rule.matches(randomScope, caller, args));
		assert(rule, `Unclassified battle.random at ${caller}: ${args}`);
		const value = allow(rule.kind, () => battleRandom.apply(this, args));
		record(rule.kind, ...rule.address(randomScope), rule.value(value, randomScope));
		return value;
	};
	const sample = battle.sample;
	battle.sample = function (items) {
		const caller = directCaller();
		if (caller === 'Side.randomFoe' && this.effect?.id === 'frisk') {
			assert(items.length === 1 && items[0] === this.effectState.target.side.foe.active[0],
				'Frisk must have exactly one possible target');
			stats.exceptions['singleton-frisk'] = (stats.exceptions['singleton-frisk'] || 0) + 1;
			return allow('singleton-frisk', () => sample.call(this, items));
		}
		const rule = OUTCOME_RULES.find(rule => rule.method === 'sample' && rule.matches(scope, caller, [items]));
		assert(rule, `Unclassified battle.sample at ${caller}`);
		const value = allow(rule.kind, () => sample.call(this, items));
		record(rule.kind, ...rule.address(scope), rule.value(value, scope));
		return value;
	};
	const shuffle = battle.prng.shuffle;
	battle.prng.shuffle = function (list, start = 0, end = list.length) {
		const caller = directCaller();
		assert.equal(caller, 'Battle.speedSort', `Unclassified shuffle at ${caller}`);
		const tied = list.slice(start, end);
		let rule;
		let exception;
		if (scope?.kind === 'queue') {
			assert.equal(tied.length, 2, 'Only two-sided singles queue ties are supported');
			assert(tied.every(a => ['move', 'switch', 'instaswitch'].includes(a.choice)) &&
				tied[0].pokemon.side !== tied[1].pokemon.side, 'Unexpected tied queue entries');
			const replacement = tied.every(a => a.choice === 'instaswitch');
			assert(replacement || tied.every(a => a.choice !== 'instaswitch'));
			rule = OUTCOME_RULES.find(row => row.method === 'shuffle' && row.matches(scope, caller, tied));
		} else if (scope?.kind === 'residual') {
			assert(tied.length >= 2, 'Residual speed sort needs at least two handlers');
			assert(tied.every(h => [
				'brn', 'psn', 'tox', 'partiallytrapped', 'protect', 'endure', 'flinch', 'stall', 'lockedmove',
				'mustrecharge', 'twoturnmove', 'fly', 'dig', 'bounce', 'dive', 'shadowforce', 'disable',
				'skullbash', 'skyattack', 'razorwind', 'solarbeam', 'freezeshock', 'iceburn', 'roost',
				'leftovers', 'blacksludge', 'toxicorb', 'flameorb', 'whiteherb', 'gem',
				'hydration', 'shedskin', 'speedboost', 'baddreams', 'slowstart', 'harvest',
			].includes(h.effect.id)),
			`Unexpected residual tie: ${tied.map(h => `${h.effect.id}:${h.effectHolder.side.n}`)}`);
			const sides = new Set(tied.map(h => h.effectHolder.side.n));
			if (sides.size === 1) exception = 'same-side-residual-order';
			else {
				const order = tied[0].order;
				const kind = order === 25 ? 'roost_residual_order' :
					order === 28 ? 'ability_residual_order' :
					order === 29 ? 'item_residual_order' : 'order';
				const baseRule = OUTCOME_RULES.find(row =>
					row.kind === 'order' && row.method === 'shuffle' && row.matches(scope));
				rule = { ...baseRule, kind, address: () => [0,
					kind === 'order' ? residualOrderOccurrence(order) : 0] };
			}
		} else if (scope?.kind === 'eachEvent' && scope.event === 'Weather') {
			assert(tied.every(pokemon => pokemon.side.active.includes(pokemon)) &&
				tied[0].side !== tied[1].side, 'Unexpected weather event tie');
			rule = OUTCOME_RULES.find(row => row.method === 'shuffle' && row.matches(scope));
		} else if (scope?.kind === 'eachEvent' && scope.event === 'WeatherChange') {
			assert(tied.every(pokemon => battle.findEventHandlers(pokemon, 'WeatherChange').length === 0),
				'Nonempty WeatherChange handlers: cannot ignore order');
			if (scope.effect?.effectType === 'Ability' || addresses?.has('order:0:4')) {
				exception = 'empty-ability-WeatherChange';
			} else {
				rule = OUTCOME_RULES.find(row => row.method === 'shuffle' && row.matches(scope));
			}
		} else if (scope?.kind === 'eachEvent' && ['Update', 'BeforeTurn'].includes(scope.event)) {
			const handlers = tied.map(p => battle.findEventHandlers(p, scope.event));
			const updates = handlers.flat();
			const independentUpdates = scope.event === 'Update' && updates.every(h =>
				['lumberry', 'sitrusberry', 'chestoberry'].includes(h.effect?.id) ||
				START_UPDATE_ABILITIES.has(h.effect?.id));
			assert(!updates.length || independentUpdates,
				`Nonempty ${scope.event} handlers: cannot ignore order ${JSON.stringify(
					handlers.map(list => list.map(h => h.effect?.id ?? '<unknown>'))
				)}`);
			exception = !updates.length ? `empty-eachEvent-${scope.event}` :
				updates.some(h => START_UPDATE_ABILITIES.has(h.effect.id)) ? 'independent-ability-Update' :
				'berry-eachEvent-Update';
		} else if (scope?.kind === 'hit' && tied.length === 2 &&
			tied.every(h => h.effect?.id === 'noguard' && h.callback === h.effect.onAnyAccuracy) &&
			tied[0].effectHolder.side !== tied[1].effectHolder.side) {
			exception = 'identical-noguard-accuracy';
		} else if (!scope && tied.length === 2 &&
			tied.map(h => h.effect?.id).sort().join(',') === 'choicelock,disable' &&
			tied.every(h => h.callback === h.effect.onDisableMove) &&
			tied[0].effectHolder === tied[1].effectHolder) {
			exception = 'choice-disable-move-order';
		} else if (scope?.kind === 'runSwitch') {
			const switchHandlers = tied.map(p => ({
				item: p.item,
				ability: p.ability,
				onSwitchIn: battle.findPokemonEventHandlers(p, 'onSwitchIn').map(h => h.effect.id),
				onAnySwitchIn: battle.findPokemonEventHandlers(p, 'onAnySwitchIn').map(h => h.effect.id),
			}));
			assert(tied.every(p => battle.findPokemonEventHandlers(p, 'onSwitchIn').every(h => [
					'slp', 'tox', 'choiceband', 'choicespecs', 'choicescarf',
				].includes(h.effect.id) || SWITCH_START_ABILITIES.has(h.effect.id) ||
					h.effect.id === 'slowstart') &&
				battle.findPokemonEventHandlers(p, 'onAnySwitchIn').every(h => h.effect.id === 'whiteherb') &&
				battle.findSideEventHandlers(p.side, 'onSideSwitchIn', undefined, p).length === 0 &&
				battle.findSideEventHandlers(p.side, 'onSwitchIn', undefined, p)
					.every(h => ['spikes', 'toxicspikes'].includes(h.effect.id)) &&
				battle.findFieldEventHandlers(battle.field, 'onFieldSwitchIn', undefined, p).length === 0 &&
				battle.findFieldEventHandlers(battle.field, 'onSwitchIn', undefined, p).length === 0 &&
				battle.findBattleEventHandlers('onSwitchIn', undefined, p).length === 0),
				`Nonempty SwitchIn handlers: cannot ignore speedOrder ${JSON.stringify(switchHandlers)}`);
			const switchers = [scope.pokemon, ...scope.queued];
			const grouped = tied.every(p => switchers.includes(p));
			// Older schema-v2 recordings emitted this address even for a solo switch.
			const hasHazards = (grouped || !groupedSwitchInOnly) && tied.some(p =>
				battle.findSideEventHandlers(p.side, 'onSwitchIn', undefined, p).length > 0);
			const abilityOrder = grouped &&
				(tied.every(p => WEATHER_START_ABILITIES.has(p.ability)) ||
					tied.some(p => p.ability === 'intimidate'));
			if (hasHazards || abilityOrder) {
				rule = OUTCOME_RULES.find(row => row.method === 'shuffle' && row.matches(scope));
			} else if (tied.some(p => p.ability === 'slowstart')) {
				exception = 'independent-slowstart-switchIn-order';
			} else if (tied.some(p => p.item)) {
				exception = 'benign-item-switchIn-order';
			} else {
				exception = 'empty-switchIn-order';
			}
		} else {
			const entries = tied.map(entry => entry.choice || entry.effect?.id || entry.species?.id || typeof entry);
			throw new Error(`Unclassified shuffle: scope=${scope?.kind}, event=${scope?.event}, entries=${entries}\n${new Error().stack}`);
		}
		const value = allow(exception || 'order', () => shuffle.call(this, list, start, end));
		if (exception) stats.exceptions[exception] = (stats.exceptions[exception] || 0) + 1;
		else record(rule.kind, ...rule.address(scope, tied), rule.value(value, list, start),
			`shuffle ${scope?.kind} order=${tied[0]?.order} handlers=${tied.map(h => h.effect?.id).join(',')}`);
		return value;
	};
	return {
		begin() { assert.equal(outcomes, null); outcomes = []; addresses = new Map(); },
		setFixed(members) { fixed = members; },
		end() { const result = outcomes; outcomes = null; return result; },
	};
}

function newStats() {
	return { steps: 0, outcomes: { order: 0, hit: 0, crit: 0, damage_roll: 0, drag_in: 0 },
		misses: 0, crits: 0, ties: 0, struggles: 0, doubleKOs: 0, forcedReplacements: 0,
		prngDraws: 0, exceptions: Object.fromEntries(
			EXCEPTIONS.filter(key => key !== 'same-side-residual-order').map(key => [key, 0])), };
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
	const players = teams.map((team, side) => ({ name: `Side ${side}`, team: structuredClone(team) }));
	const recordStart = options.profile === 'isolation' && options.abilities?.length > 0;
	const battle = recordStart ? isolationBattle(seed) : new Battle({
		formatid: 'gen5randombattle', seed, p1: players[0], p2: players[1],
	});
	try {
		const stats = newStats();
		const recorder = instrument(battle, stats, undefined, { groupedSwitchInOnly: recordStart });
		let initialOutcomes;
		const publicAbilities = recordStart ? trackPublicAbilities(battle) : undefined;
		if (recordStart) {
			recorder.begin();
			battle.setPlayer('p1', players[0]);
			battle.setPlayer('p2', players[1]);
			initialOutcomes = recorder.end();
			stats.ties += initialOutcomes.filter(outcome => outcome.kind === 'order').length;
			for (const outcome of initialOutcomes) {
				stats.outcomes[outcome.kind] = (stats.outcomes[outcome.kind] || 0) + 1;
			}
		}
		assert.equal(battle.turn, 1);
		const fixed = battle.sides.map(side => [...side.pokemon]);
		for (const [s, members] of fixed.entries()) {
			for (const [slot, pokemon] of members.entries()) {
				assert.equal(pokemon.ability, dex.toID(teams[s][slot].ability));
				assert.equal(pokemon.item || pokemon.lastItem, dex.toID(teams[s][slot].item));
				assert.deepEqual(pokemon.moveSlots.map(m => m.id), teams[s][slot].moves);
			}
		}
		recorder.setFixed(fixed);
		const initialState = snapshot(battle, fixed, publicAbilities);
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
			const state = snapshot(battle, fixed, publicAbilities);
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
			initialOverrides: [], ...(recordStart ? { initialOutcomes } : {}), initialState, steps, stats, turnLimitHit };
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
		abilities: [...new Set([...supported.abilities, ...(extra.abilities || []).map(id => dex.toID(id))])].sort(),
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
			prngExceptions: [...EXCEPTIONS, 'singleton-frisk', 'identical-noguard-accuracy',
				'empty-ability-WeatherChange', 'independent-ability-Update'] });
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

module.exports = { acceptedMoves, seedFor, snapshot, legalChoices, instrument, newStats, playBattle, isolationMember, isolationBattle };
if (require.main === module) main().catch(error => { console.error(error.stack); process.exitCode = 1; });
