'use strict';

// Run with node --test tests/test-record-battles.cjs (requires pinned Showdown).
const { acceptedMoves, instrument, isolationBattle, legalChoices, newStats, playBattle, snapshot, isolationMember } = require('../reference/record-battles.cjs');
const { PRNG } = require('../reference/vendor/pokemon-showdown/dist/sim/prng');
const { Battle } = require('../reference/vendor/pokemon-showdown/dist/sim/battle');
const { Dex } = require('../reference/vendor/pokemon-showdown/dist/sim/dex');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const { gunzipSync } = require('node:zlib');
const root = path.resolve(__dirname, '..');
const fixturePath = path.join(__dirname, 'fixtures/diff-phase1.jsonl.gz');
const [header, ...battles] = gunzipSync(fs.readFileSync(fixturePath)).toString().trim().split('\n').map(JSON.parse);
const dex = Dex.mod('gen5');

function createBattle(record) {
	return new Battle({ formatid: 'gen5randombattle', seed: record.seed,
		p1: { name: 'Side 0', team: structuredClone(record.teams[0]) },
		p2: { name: 'Side 1', team: structuredClone(record.teams[1]) } });
}

function withItemDefaults(record) {
	const copy = structuredClone(record);
	const snapshots = [copy.initialState, ...(copy.steps || []).map(step => step.state)];
	for (const snapshot of snapshots) {
		for (const side of snapshot.sides) {
			side.itemConsumed ??= false;
			side.itemKnockedOff ??= false;
			for (const member of side.members) {
				member.itemConsumed ??= false;
				member.itemKnockedOff ??= false;
			}
		}
	}
	for (const name of [
		'berry-eachEvent-Update',
		'benign-item-switchIn-order',
		'independent-slowstart-switchIn-order',
		'choice-disable-move-order',
	]) {
		if (copy.stats?.exceptions) copy.stats.exceptions[name] ??= 0;
	}
	return copy;
}

function checkState(battle, fixed, state) {
	const normalized = withItemDefaults({ initialState: state }).initialState;
	assert.deepEqual(normalized, snapshot(battle, fixed));
	assert.equal(state.turn, battle.turn);
	assert.equal(state.ended, battle.ended);
	assert.equal(state.winner, !battle.ended ? null : !battle.winner ? 'tie' : Number(battle.winner.slice(-1)));
	assert.equal(state.sides.length, 2);
	for (let s = 0; s < 2; s++) {
		const actual = battle.sides[s];
		const expected = normalized.sides[s];
		assert.equal(fixed[s][expected.active], actual.active[0]);
		assert.deepEqual(expected.types, actual.active[0].fainted ? null :
			actual.active[0].getTypes().map(type => dex.toID(type)));
		assert.deepEqual(expected.volatiles, actual.active[0].fainted ? null :
			Object.keys(actual.active[0].volatiles).sort());
		assert.deepEqual(expected.boosts, ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'].map(k => actual.active[0].boosts[k]));
		assert.equal(expected.request, battle.ended || actual.activeRequest?.wait ? 'none' :
			actual.activeRequest?.forceSwitch ? 'switch' : 'move');
		assert.equal(expected.members.length, 6);
		for (let slot = 0; slot < 6; slot++) {
			const pokemon = fixed[s][slot];
			assert.deepEqual(expected.members[slot], {
				hp: pokemon.hp, maxhp: pokemon.maxhp, pp: pokemon.moveSlots.map(m => m.pp), fainted: pokemon.fainted,
				status: pokemon.status === 'fnt' ? '' : pokemon.status, item: pokemon.item, ability: pokemon.ability,
				itemConsumed: !pokemon.item && Boolean(pokemon.lastItem) && !pokemon.itemKnockedOff,
				itemKnockedOff: pokemon.itemKnockedOff,
			});
		}
	}
}

test('move filter accepts plain overrides and fails closed on callbacks and effects', () => {
	const exported = JSON.parse(fs.readFileSync(path.join(root, 'data/gen5-data-v1.json')));
	const moves = acceptedMoves(exported);
	assert.equal(moves.length, 56);
	assert.deepEqual(moves, header.acceptedMoves);
	for (const id of ['hiddenpower', 'struggle', 'aurasphere', 'uturn',
		'flamethrower', 'doubleedge', 'feint', 'judgment', 'psyshock', 'sacredsword', 'falseswipe']) {
		assert(!moves.includes(id), id);
	}
	for (const id of ['earthquake', 'surf', 'hydropump', 'crabhammer', 'hypervoice']) assert(moves.includes(id), id);
	const wrongExport = structuredClone(exported);
	wrongExport.moves.find(move => move.id === 'hydropump').basePower++;
	assert.throws(() => acceptedMoves(wrongExport), /Export mismatch for hydropump.basePower/);
	assert(moves.includes('stoneedge')); // High crit ratio is allowed.
	assert(moves.includes('extremespeed')); // Priority is allowed.
	const raw = dex.data.Moves.pound;
	for (const [key, value] of [['onHit', () => {}], ['futureUnknownEffect', true],
		['secondary', { chance: 10, status: 'brn' }], ['self', { boosts: { atk: -1 } }],
		['flags', { ...raw.flags, recharge: 1 }], ['flags', { ...raw.flags, contact: () => {} }]]) {
		const before = raw[key];
		try {
			raw[key] = value;
			assert(!acceptedMoves({ moves: [{ id: 'pound' }] }).length, key);
		} finally {
			if (before === undefined) delete raw[key]; else raw[key] = before;
		}
	}
});

test('the guard rejects unclassified raw draws and chance/shuffle call sites', () => {
	const battle = createBattle(battles[0]);
	try {
		const guard = instrument(battle, newStats());
		guard.begin();
		assert.throws(() => battle.prng.random(16), /Unclassified PRNG/);
		assert.throws(() => battle.prng.randomChance(1, 2), /Unclassified PRNG/);
		assert.throws(() => battle.sample([1, 2]), /Unclassified battle.sample/);
		assert.throws(() => battle.random(16), /Unclassified battle.random/);
		assert.throws(() => battle.randomChance(1, 16), /Unclassified randomChance/);
		assert.throws(() => battle.prng.shuffle([1, 2]), /Unclassified shuffle/);
		guard.end();
	} finally { battle.destroy(); }
});

test('empty-event shuffle exceptions stop being allowed when a handler exists', () => {
	const battle = createBattle(battles[0]);
	try {
		const guard = instrument(battle, newStats());
		guard.begin();
		const find = battle.findEventHandlers;
		battle.findEventHandlers = function (target, event, ...args) {
			return event === 'Update' ? [{}] : find.call(this, target, event, ...args);
		};
		assert.throws(() => battle.eachEvent('Update'), /Nonempty Update handlers/);
		guard.end();
	} finally { battle.destroy(); }
});

test('queue ties address action and replacement contexts, and duplicate addresses fail', () => {
	const battle = createBattle(battles[0]);
	try {
		const guard = instrument(battle, newStats());
		for (const choice of ['move', 'switch', 'instaswitch']) {
			// Isolate the queue sorter: real Pokemon, equal speeds, one action per side.
			// This synthetic queue is a unit test, never an override in an oracle battle.
			battle.queue.list = battle.sides.map(side => ({ choice, pokemon: side.active[0], speed: 100 }));
			guard.begin();
			battle.queue.sort();
			const outcomes = guard.end();
			assert.deepEqual(outcomes, [{ kind: 'order', side: 0,
				occurrence: choice === 'instaswitch' ? 1 : 0, value: battle.queue.list[0].pokemon.side.n }]);
		}
		guard.begin();
		battle.queue.sort();
		assert.throws(() => battle.queue.sort(), /Duplicate outcome address/);
		guard.end();
	} finally { battle.destroy(); }
});

test('dragged replacements record the selected fixed team slot', () => {
	const battle = createBattle(battles[0]);
	try {
		const fixed = battle.sides.map(side => [...side.pokemon]);
		const guard = instrument(battle, newStats(), fixed);
		guard.begin();
		const selected = battle.getRandomSwitchable(battle.sides[0]);
		const outcomes = guard.end();
		assert.deepEqual(outcomes, [{
			kind: 'drag_in', side: 0, occurrence: 0,
			value: fixed[0].indexOf(selected),
		}]);
	} finally { battle.destroy(); }
});

test('switch legality distinguishes a possible trap from an active hidden trap', () => {
	const battle = createBattle(battles[0]);
	try {
		const side = battle.sides[0];
		const fixed = [...side.pokemon];
		const request = side.activeRequest.active[0];
		request.maybeTrapped = true;
		assert(!request.trapped);
		assert(!side.active[0].trapped);
		assert(legalChoices(side, fixed).some(choice => choice.action >= 4));
		assert(side.active[0].tryTrap(true));
		assert.equal(side.active[0].trapped, 'hidden');
		assert(!legalChoices(side, fixed).some(choice => choice.action >= 4));
	} finally { battle.destroy(); }
});

test('fixture actions independently replay through Showdown with exact decision snapshots', () => {
	assert.equal(header.schemaVersion, 2);
	assert.equal(header.count, 40);
	assert.equal(battles.length, 40);
	assert(fs.statSync(fixturePath).size < 1_000_000);
	const coverage = { struggles: 0, ties: 0, misses: 0, crits: 0, doubleKOs: 0, forcedReplacements: 0 };
	let sawReorderedSwitch = false;
	for (const record of battles) {
		assert.deepEqual(record.initialOverrides, []);
		assert.equal(record.turnLimitHit, false);
		const battle = createBattle(record);
		const fixed = battle.sides.map(s => [...s.pokemon]);
		try {
			checkState(battle, fixed, record.initialState);
			let previous = record.initialState;
			for (const step of record.steps) {
				const addresses = new Set();
				for (const o of step.outcomes) {
					assert.deepEqual(Object.keys(o).sort(), ['kind', 'occurrence', 'side', 'value']);
					assert(['order', 'hit', 'crit', 'damage_roll'].includes(o.kind));
					assert(Number.isInteger(o.value) && o.value >= 0 && o.value <= (o.kind === 'damage_roll' ? 15 : 1));
					assert(o.side === 0 || o.side === 1);
					assert(o.kind === 'order' ? o.side === 0 && [0, 1].includes(o.occurrence) : o.occurrence === 0);
					const key = `${o.kind}:${o.side}:${o.occurrence}`;
					assert(!addresses.has(key)); addresses.add(key);
				}
				// Translate from fixed slots independently of the recorder's choice helper.
				const choices = step.actions.map((action, s) => {
					const side = previous.sides[s];
					const actual = battle.sides[s];
					if (side.request === 'none') { assert.equal(action, 0); return null; }
					assert(legalChoices(actual, fixed[s]).some(c => c.action === action));
					if (action < 4) {
						assert.equal(side.request, 'move');
						return `move ${action + 1}`;
					}
					const slots = [0, 1, 2, 3, 4, 5].filter(slot => slot !== side.active);
					const target = fixed[s][slots[action - 4]];
					assert(target.hp > 0 && !target.fainted);
					const currentPosition = actual.pokemon.indexOf(target);
					if (currentPosition !== slots[action - 4]) sawReorderedSwitch = true;
					return `switch ${currentPosition + 1}`;
				});
				const execution = [];
				const runAction = battle.runAction;
				battle.runAction = function (action) {
					if (['move', 'switch', 'instaswitch'].includes(action.choice)) execution.push(action.pokemon.side.n);
					return runAction.call(this, action);
				};
				const logStart = battle.log.length;
				for (let s = 0; s < 2; s++) if (choices[s]) assert(battle.choose(`p${s + 1}`, choices[s]));
				battle.runAction = runAction;
				checkState(battle, fixed, step.state);
				const log = battle.log.slice(logStart);
				for (let s = 0; s < 2; s++) {
					assert.equal(step.outcomes.filter(o => o.kind === 'hit' && o.side === s && o.value === 0).length,
						log.filter(line => line.startsWith(`|-miss|p${s + 1}a:`)).length);
					assert.equal(step.outcomes.filter(o => o.kind === 'crit' && o.side === s && o.value === 1).length,
						log.filter(line => line.startsWith(`|-crit|p${2 - s}a:`)).length);
				}
				for (const o of step.outcomes.filter(o => o.kind === 'order')) assert.equal(o.value, execution[0]);
				previous = step.state;
			}
			assert(battle.ended);
			assert.equal(record.stats.prngDraws, Object.values(record.stats.outcomes).reduce((a, b) => a + b, 0) +
				Object.values(record.stats.exceptions).reduce((a, b) => a + b, 0));
			for (const key of Object.keys(coverage)) coverage[key] += record.stats[key];
		} finally { battle.destroy(); }
	}
	assert(sawReorderedSwitch);
	for (const [key, n] of Object.entries(coverage)) assert(n > 0, `Missing fixture coverage: ${key}`);
});

test('seeded generation regenerates a fixture battle byte for byte without overrides', () => {
	const pool = JSON.parse(fs.readFileSync(path.join(root, 'work/teams/gen5-teams.json')));
	const ghosts = pool.teams.flatMap((team, i) => dex.species.get(team[0].species).types.includes('Ghost') ? [i] : []);
	const lowPP = header.acceptedMoves.filter(id => dex.moves.get(id).pp <= 10 && !dex.getImmunity(dex.moves.get(id).type, 'Ghost'));
	for (const index of [0, 1]) {
		assert.deepEqual(withItemDefaults(playBattle(index, header.seed, pool,
			header.acceptedMoves, ghosts, lowPP)), withItemDefaults(battles[index]));
	}
});

test('status isolation fixture regenerates from its recorded seed', () => {
	const file = path.join(root, 'tests/fixtures/diff-status.jsonl.gz');
	const [statusHeader, ...records] = gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(JSON.parse);
	const pool = JSON.parse(fs.readFileSync(path.join(root, 'work/teams/gen5-teams.json')));
	const ghosts = pool.teams.flatMap((team, i) => dex.species.get(team[0].species).types.includes('Ghost') ? [i] : []);
	const lowPP = statusHeader.acceptedMoves.filter(id => dex.moves.get(id).pp <= 10 && !dex.getImmunity(dex.moves.get(id).type, 'Ghost'));
	for (const index of [0, 1]) {
		assert.deepEqual(withItemDefaults(playBattle(index, statusHeader.seed, pool,
			statusHeader.acceptedMoves, ghosts, lowPP, statusHeader.allowed)),
		withItemDefaults(records[index]));
	}
});

test('boost isolation fixture regenerates from its recorded seed', () => {
	const file = path.join(root, 'tests/fixtures/diff-boosts.jsonl.gz');
	const [boostHeader, ...records] = gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(JSON.parse);
	const pool = JSON.parse(fs.readFileSync(path.join(root, 'work/teams/gen5-teams.json')));
	for (const index of [0, 1]) {
		assert.deepEqual(withItemDefaults(playBattle(index, boostHeader.seed, pool,
			boostHeader.acceptedMoves, [], [], boostHeader.allowed)), withItemDefaults(records[index]));
	}
});

test('secondary fixture independently replays every state and addressed outcome', () => {
	const file = path.join(root, 'tests/fixtures/diff-secondaries.jsonl.gz');
	const [taskHeader, ...records] = gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(JSON.parse);
	assert.equal(taskHeader.schemaVersion, 2);
	assert.equal(records.length, 40);
	assert(fs.statSync(file).size < 1_000_000);
	const kinds = new Set();
	let confusionSnapshots = 0;
	for (const record of records) {
		const battle = createBattle(record);
		try {
			const fixed = battle.sides.map(s => [...s.pokemon]);
			const guard = instrument(battle, newStats());
			checkState(battle, fixed, record.initialState);
			for (const step of record.steps) {
				guard.begin();
				const choices = step.actions.map((action, s) =>
					legalChoices(battle.sides[s], fixed[s], battle.ended).find(c => c.action === action));
				for (const [s, choice] of choices.entries()) {
					assert(choice);
					if (choice.choice !== null) assert(battle.choose(`p${s + 1}`, choice.choice));
				}
				assert.deepEqual(guard.end(), step.outcomes);
				checkState(battle, fixed, step.state);
				for (const outcome of step.outcomes) kinds.add(outcome.kind);
				for (const side of step.state.sides) {
					assert(!side.volatiles?.includes('flinch'));
					if (side.volatiles?.includes('confusion')) {
						assert(side.volatileCounters.confusion > 0 && side.volatileCounters.confusion <= 5);
						confusionSnapshots++;
					}
				}
			}
		} finally { battle.destroy(); }
	}
	assert(confusionSnapshots > 0);
	for (const kind of ['secondary', 'secondary_status', 'confusion_duration', 'confusion_hit', 'confusion_damage_roll']) {
		assert(kinds.has(kind), `Missing outcome coverage: ${kind}`);
	}
	const pool = JSON.parse(fs.readFileSync(path.join(root, 'work/teams/gen5-teams.json')));
	for (const index of [0, 1]) {
		assert.deepEqual(withItemDefaults(playBattle(index, taskHeader.seed, pool,
			taskHeader.acceptedMoves, [], [], taskHeader.allowed)), withItemDefaults(records[index]));
	}
});

test('effects isolation fixture replays multihit outcomes and every decision snapshot', () => {
	const file = path.join(root, 'tests/fixtures/diff-effects2.jsonl.gz');
	const [taskHeader, ...records] = gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(JSON.parse);
	assert.equal(taskHeader.schemaVersion, 2);
	assert.equal(taskHeader.seed, 4);
	assert.equal(records.length, 40);
	assert(fs.statSync(file).size < 1_000_000);
	const kinds = new Set();
	for (const record of records) {
		const battle = createBattle(record);
		try {
			const fixed = battle.sides.map(s => [...s.pokemon]);
			const guard = instrument(battle, newStats());
			checkState(battle, fixed, record.initialState);
			for (const step of record.steps) {
				guard.begin();
				const choices = step.actions.map((action, s) =>
					legalChoices(battle.sides[s], fixed[s], battle.ended).find(c => c.action === action));
				for (const [s, choice] of choices.entries()) {
					assert(choice);
					if (choice.choice !== null) assert(battle.choose(`p${s + 1}`, choice.choice));
				}
				assert.deepEqual(guard.end(), step.outcomes);
				checkState(battle, fixed, step.state);
				for (const outcome of step.outcomes) kinds.add(outcome.kind);
			}
		} finally { battle.destroy(); }
	}
	for (const kind of ['hit_count', 'crit', 'damage_roll', 'secondary']) {
		assert(kinds.has(kind), `Missing outcome coverage: ${kind}`);
	}
	const pool = JSON.parse(fs.readFileSync(path.join(root, 'work/teams/gen5-teams.json')));
	for (const index of [0, 1]) {
		assert.deepEqual(withItemDefaults(playBattle(index, taskHeader.seed, pool,
			taskHeader.acceptedMoves, [], [], taskHeader.allowed)), withItemDefaults(records[index]));
	}
});

test('weather and grouped SwitchIn ties use separate order addresses', () => {
	{
		const battle = createBattle(battles[0]);
		try {
			for (const side of battle.sides) side.active[0].speed = 100;
			const guard = instrument(battle, newStats());
			guard.begin();
			assert(battle.field.setWeather('raindance', battle.sides[0].active[0]));
			battle.eachEvent('Weather');
			const outcomes = guard.end();
			assert.deepEqual(outcomes.map(({ kind, side, occurrence }) => ({ kind, side, occurrence })), [
				{ kind: 'order', side: 0, occurrence: 4 },
				{ kind: 'order', side: 0, occurrence: 3 },
			]);
			for (const outcome of outcomes) assert([0, 1].includes(outcome.value));
		} finally { battle.destroy(); }
	}
	{
		const battle = createBattle(battles[0]);
		try {
			for (const side of battle.sides) {
				side.active[0].speed = 100;
				assert(side.addSideCondition('spikes', side.active[0]));
			}
			const guard = instrument(battle, newStats());
			guard.begin();
			battle.actions.runSwitch(battle.sides[0].active[0]);
			const outcomes = guard.end();
			assert.equal(outcomes.length, 1);
			assert.deepEqual(outcomes.map(({ kind, side, occurrence }) => ({ kind, side, occurrence })), [
				{ kind: 'order', side: 0, occurrence: 5 },
			]);
			assert([0, 1].includes(outcomes[0].value));
		} finally { battle.destroy(); }
	}
});

test('secondary isolation data agrees with the pinned resolved moves', () => {
	const allowed = JSON.parse(fs.readFileSync(path.join(root, 'tests/fixtures/diff-secondaries.allowed.json')));
	const data = JSON.parse(fs.readFileSync(path.join(root, 'data/gen5-data-v1.json')));
	const weatherCallbacks = new Set(['blizzard', 'hurricane', 'thunder']);
	for (const id of allowed.moves) {
		const move = dex.moves.get(id);
		const row = data.moves.find(m => m.id === id);
		for (const field of ['basePower', 'accuracy', 'pp', 'type', 'category', 'target', 'priority', 'critRatio']) {
			assert.deepEqual(row[field], move[field], `${id}.${field}`);
		}
		for (const field of ['volatileStatus', 'boosts']) assert.deepEqual(row[field], move[field], `${id}.${field}`);
		const declarative = effects => (effects || []).map(effect =>
			Object.fromEntries(Object.entries(effect).filter(([key]) => key !== 'onHit')));
		assert.deepEqual(declarative(row.secondaries), declarative(move.secondaries), `${id}.secondaries`);
		const callbacks = Object.entries(dex.data.Moves[id]).filter(([, v]) => typeof v === 'function');
		for (const [key] of callbacks) {
			assert.equal(key, 'onModifyMove', `Unexpected primary callback: ${id}.${key}`);
			assert(weatherCallbacks.has(id) || id === 'chatter', id);
		}
		for (const secondary of move.secondaries || []) {
			for (const [key, value] of Object.entries(secondary)) {
				if (typeof value === 'function') assert(id === 'triattack' && key === 'onHit');
				else assert(['chance', 'status', 'boosts', 'self', 'volatileStatus'].includes(key), `${id}.${key}`);
			}
		}
	}
});

test('freeze prevention records a thaw outcome for the affected side', () => {
	const battle = createBattle(battles[0]);
	try {
		assert(battle.sides[0].active[0].setStatus('frz', null, null, true));
		const guard = instrument(battle, newStats());
		guard.begin();
		assert(battle.choose('p1', 'move 1'));
		assert(battle.choose('p2', 'switch 2'));
		const outcomes = guard.end();
		assert(outcomes.some(outcome => outcome.kind === 'thaw' && outcome.side === 0 &&
			outcome.occurrence === 0 && [0, 1].includes(outcome.value)));
	} finally { battle.destroy(); }
});


test('isolation keeps allowed pool choices and replaces only disallowed choices', () => {
	const member = { species: 'Pikachu', level: 80, nature: 'Serious', gender: 'M',
		evs: { hp: 85 }, ivs: { hp: 31 }, happiness: 200, moves: ['Tackle', 'Toxic'],
		ability: 'Static', item: 'Light Ball' };
	const moves = ['tackle', 'quickattack', 'scratch', 'pound', 'toxic'];
	const options = { taskMoves: ['toxic'], abilities: ['static'], items: ['lightball'] };
	const kept = isolationMember(member, moves, options, new PRNG([1, 2, 3, 4]));
	assert(kept.moves.includes('tackle') && kept.moves.includes('toxic'));
	assert.equal(new Set(kept.moves).size, 4);
	assert(kept.moves.every(id => moves.includes(id)));
	assert.equal(kept.ability, 'Static');
	assert.equal(kept.item, 'Light Ball');
	for (const key of ['species', 'level', 'nature', 'gender', 'evs', 'ivs', 'happiness']) {
		assert.deepEqual(kept[key], member[key]);
	}
	const stripped = isolationMember(member, moves.filter(id => id !== 'toxic'), {}, new PRNG([1, 2, 3, 4]));
	assert(!stripped.moves.includes('toxic'));
	assert.equal(stripped.ability, 'noability');
	assert.equal(stripped.item, '');
});


test('a tied lead weather war records the handler shuffle and guards queue insertion', () => {
	const battle = isolationBattle([11, 22, 33, 44]);
	try {
		const stats = newStats();
		const guard = instrument(battle, stats);
		guard.begin();
		for (const [side, ability] of ['Drizzle', 'Drought'].entries()) {
			battle.setPlayer(`p${side + 1}`, {
				name: `Side ${side}`,
				team: [{ species: 'Bulbasaur', gender: 'M', ability, moves: ['tackle'] }],
			});
		}
		const outcomes = guard.end();
		const order = outcomes.find(row => row.kind === 'order' && row.occurrence === 5);
		assert(order);
		assert.equal(order.side, 0);
		assert.equal(battle.field.weather, order.value === 0 ? 'sunnyday' : 'raindance');
		assert.equal(battle.field.weatherState.duration, 0);
		assert.equal(stats.exceptions['equal-speed-runSwitch-insertion'], 1);
	} finally { battle.destroy(); }
});
