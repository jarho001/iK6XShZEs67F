'use strict';

// Run with node --test tests/test-showdown-player.cjs (requires the pinned Showdown). Each case scripts
// one or two battles of reference/showdown-player.cjs: the learner's actions are chosen by the test, the
// opponent's by a script, and the Showdown log, the decisions and the counters are checked.
const assert = require('node:assert/strict');
const { test } = require('node:test');
const { Game, Session, requestKind, learnerChoice } = require('../reference/showdown-player.cjs');
const { loadTables } = require('../reference/player-view.cjs');

const tables = loadTables();

function member(species, moves, extra = {}) {
	return {
		species, ability: 'noability', item: '', moves, nature: 'serious', level: 100, gender: 'M',
		evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, happiness: 255, ...extra,
	};
}

// One battle of teams [learner's, opponent's] (pool indices 0 and 1). script(game, side) gives the
// opponent's choice; the learner's actions come from the test.
function battle(teams, { learner = 0, script = () => 'move 1', maxSteps = 1000, record = false } = {}) {
	const shared = {
		pool: teams, tables, seed: 11, maxSteps, opponent: 'scripted', readsView: false, record,
		choose: (game, side) => script(game, side),
	};
	const game = new Game({ index: 0, learner, teams: [0, 1] }, shared);
	game.start();
	return game;
}

// The Showdown log lines that start with the prefix, in order.
function lines(game, prefix) {
	return game.battle.log.filter(line => line.startsWith(prefix));
}

// The learner's active member's species as the battle has it (Showdown's own state).
function activeName(game, side) {
	return game.battle.sides[side].active[0].name;
}

const LEAD = member('pikachu', ['thunderbolt', 'quickattack', 'surf', 'splash']);
const SIX = [
	member('pikachu', ['thunderbolt', 'splash']),
	member('snorlax', ['bodyslam', 'splash']),
	member('mew', ['psychic', 'splash']),
	member('blissey', ['softboiled', 'splash']),
	member('arcanine', ['flareblitz', 'splash']),
	member('gengar', ['shadowball', 'splash']),
];

test('moves: the action is the request slot, and the log shows that move', () => {
	const game = battle([[LEAD], [member('shuckle', ['splash', 'rest'])]]);
	const first = game.decision();
	assert.equal(first.request, 'move');
	assert.deepEqual(first.legal.slice(0, 5), [true, true, true, true, false]);
	assert.equal(first.first, true);
	game.act(2); // surf
	assert.equal(lines(game, '|move|p1a: Pikachu|Surf|').length, 1);
	assert.equal(game.steps, 1);
	assert.equal(game.finished, false);
});

test('an illegal action is replaced by the first legal one and counted', () => {
	const game = battle([[LEAD], [member('shuckle', ['splash'])]]);
	const { legal } = game.decision();
	assert.equal(legal[8], false);
	game.act(8);
	assert.equal(game.illegal, 1);
	assert.equal(lines(game, '|move|p1a: Pikachu|Thunderbolt|').length, 1);
});

test('switches: targets follow the team order after several switches', () => {
	// The opponent splashes every turn, so nothing faints and only the learner switches.
	const game = battle([SIX, [member('shuckle', ['splash'])]], { script: () => 'move 1' });
	game.decision();
	// Team slots 0..5: the active is slot 0; action 4 + k is the k-th non-active slot in order.
	game.act(4); // slot 1: snorlax
	assert.equal(activeName(game, 0), 'Snorlax');
	game.decision();
	game.act(4); // from snorlax the non-active slots are 0, 2, 3, 4, 5: k = 0 is pikachu
	assert.equal(activeName(game, 0), 'Pikachu');
	game.decision();
	game.act(7); // non-active 1, 2, 3, 4, 5: k = 3 is arcanine (slot 4)
	assert.equal(activeName(game, 0), 'Arcanine');
	game.decision();
	game.act(6); // non-active 0, 1, 2, 3, 5: k = 2 is mew (slot 2)
	assert.equal(activeName(game, 0), 'Mew');
	// The log has each switch twice (the public and the own-side version of the line).
	const switched = lines(game, '|switch|p1a:').map(line => line.split('|')[3].split(',')[0]);
	assert.deepEqual(switched.filter((name, i) => i === 0 || name !== switched[i - 1]),
		['Pikachu', 'Snorlax', 'Pikachu', 'Arcanine', 'Mew']);
});

test('the switch command names the target by its position in the request order', () => {
	const game = battle([SIX, [member('shuckle', ['splash'])]]);
	const first = game.decision();
	assert.equal(learnerChoice(game.views[0], game.battle.sides[0].activeRequest, 4), 'switch 2');
	assert.equal(first.request, 'move');
	game.act(4);
	game.decision();
	// Pikachu is now at position 2 of the request order, since the switch swapped positions 1 and 2.
	assert.equal(learnerChoice(game.views[0], game.battle.sides[0].activeRequest, 4), 'switch 2');
});

test('forced switch: after a faint the learner may only switch in, and the replacement is chosen', () => {
	// Mewtwo's Psychic faints the level 5 lead in the first turn.
	const weak = member('pikachu', ['splash', 'thunderbolt'], { level: 5 });
	const reserve = member('snorlax', ['bodyslam', 'splash'], { level: 5 });
	const game = battle([[weak, reserve], [member('mewtwo', ['psychic', 'splash'])]], {
		script: () => 'move 1',
	});
	let sawForced = false;
	for (let guard = 0; guard < 20 && !game.finished; guard++) {
		const decision = game.decision();
		if (decision.request === 'switch') {
			sawForced = true;
			assert.equal(decision.legal[0], false);
			assert.deepEqual(decision.legal.slice(4), [true, false, false, false, false].map((v, k) => (k === 0 ? v : false)));
			game.act(4);
			assert.equal(activeName(game, 0), 'Snorlax');
			break;
		}
		game.act(0);
	}
	assert.ok(sawForced);
});

test('a side with nothing to choose has the single no-op, and the opponent alone commits', () => {
	// Mewtwo's Psychic KOs the opponent's level 5 lead; the opponent must switch while the learner waits.
	const game = battle([[member('mewtwo', ['psychic', 'splash'])],
		[member('pikachu', ['splash'], { level: 5 }), member('snorlax', ['splash'])]], {
		script: (g, side) => (requestKind(g.battle.sides[side]) === 'switch' ? 'switch 2' : 'move 1'),
	});
	game.decision();
	game.act(0);
	const waiting = game.decision();
	assert.equal(waiting.request, 'none');
	assert.deepEqual(waiting.legal, [true, false, false, false, false, false, false, false, false]);
	assert.equal(requestKind(game.battle.sides[0]), 'none');
	assert.equal(requestKind(game.battle.sides[1]), 'switch');
	game.act(0);
	assert.equal(lines(game, '|switch|p2a: Snorlax').length > 0, true);
	assert.equal(game.errors.length, 0);
});

test('a trapped switch is refused, the side is asked again without switches, and the opponent\'s choice stands', () => {
	// Arena Trap: the foe's Diglett traps the grounded learner. Showdown says maybeTrapped, so the first
	// request offers switches; the switch is refused, and the request then says trapped.
	const machamp = member('machamp', ['machpunch', 'splash']);
	const diglett = member('diglett', ['splash'], { ability: 'arenatrap' });
	const game = battle([[machamp, member('snorlax', ['bodyslam']), member('mew', ['psychic'])],
		[diglett]], { script: () => 'move 1' });
	const first = game.decision();
	assert.equal(first.legal[4], true);
	assert.equal(game.battle.sides[0].active[0].trapped, 'hidden');
	const turnBefore = game.battle.turn;
	const log = game.battle.log;
	game.act(4);
	assert.equal(game.refusals, 1);
	assert.equal(game.battle.turn, turnBefore);
	assert.equal(game.steps, 1);
	assert.equal(game.finished, false);
	const second = game.decision();
	assert.equal(second.refused, true);
	assert.equal(second.request, 'move');
	assert.deepEqual(second.legal.slice(4), [false, false, false, false, false]);
	game.act(0);
	assert.equal(game.battle.turn, turnBefore + 1);
	// The next request says maybeTrapped again, and the known trap keeps the switches out of the mask.
	const third = game.decision();
	assert.equal(third.refused, false);
	assert.deepEqual(third.legal.slice(4), [false, false, false, false, false]);
	assert.equal(game.knownTrapDecisions, 1);
	game.act(1); // splash, so that Diglett survives
	const logLines = (prefix) => log.filter(line => line.startsWith(prefix));
	// Diglett splashed in both turns; Machamp used Mach Punch in the first and Splash in the second.
	assert.equal(logLines('|move|p2a: Diglett|Splash').length, 2);
	assert.equal(logLines('|move|p1a: Machamp|Mach Punch|').length, 1);
	assert.equal(logLines('|move|p1a: Machamp|Splash|').length, 1);
	assert.equal(game.refusals, 1);
	assert.equal(game.errors.length, 0);
});

test('a discovered trap ends when Mummy gives the trapper Mummy, and the switches come back', () => {
	// Arena Trap refuses the learner's switch. Diglett's Sucker Punch then hits the learner's Mummy holder
	// and Diglett takes Mummy while both stay active: the trap is gone, so the next request offers switches.
	const cofagrigus = member('cofagrigus', ['shadowball', 'splash'], { ability: 'mummy' });
	const diglett = member('diglett', ['suckerpunch'], { ability: 'arenatrap' });
	const game = battle([[cofagrigus, member('snorlax', ['bodyslam']), member('mew', ['psychic'])],
		[diglett]], { script: () => 'move 1' });
	const first = game.decision();
	assert.equal(first.legal[4], true);
	game.act(4);
	assert.equal(game.refusals, 1);
	const second = game.decision();
	assert.equal(second.refused, true);
	assert.deepEqual(second.legal.slice(4), [false, false, false, false, false]);
	game.act(0);
	assert.equal(game.battle.sides[1].active[0].ability, 'mummy');
	const third = game.decision();
	assert.equal(third.refused, false);
	assert.deepEqual(third.legal.slice(4), [true, true, false, false, false]);
	// No known trap is left to remove switches, and the request's mask is the one the player sees.
	assert.equal(game.knownTrapDecisions, 0);
	assert.equal(game.maskMismatches, 0);
	assert.equal(game.errors.length, 0);
});

test('a battle that reaches the step cap is truncated and counted as not ended', () => {
	const game = battle([[LEAD], [member('shuckle', ['splash'])]], { maxSteps: 2 });
	game.decision();
	assert.equal(game.act(0), null);
	game.decision();
	const result = game.act(0);
	assert.ok(result);
	assert.equal(result.truncated, true);
	assert.equal(result.ended, false);
	assert.equal(result.winner, null);
	assert.equal(result.steps, 2);
});

test('the record keeps both requests, the protocol and every decision of the battle', () => {
	const game = battle([[LEAD], [member('shuckle', ['splash'])]], { record: true, maxSteps: 3 });
	for (let step = 0; step < 3 && !game.finished; step++) {
		game.decision();
		game.act(0);
	}
	const result = game.result;
	const record = JSON.parse(result.recordLine);
	assert.equal(record.type, 'battle');
	assert.equal(record.steps.length, 3);
	for (const step of record.steps) {
		assert.ok(Array.isArray(step.protocol));
		assert.ok(step.requests.some(entry => entry.side === 0));
		assert.ok(step.requests.some(entry => entry.side === 1));
		assert.equal(step.observation.legal_actions.length, 9);
	}
	assert.equal(record.steps[0].learnerChoice, 'move 1');
	assert.equal(record.steps[0].opponentChoice, 'move 1');
});

test('a session plays several battles in fewer slots, and every decision gets an answer', () => {
	const pool = [[LEAD], [member('shuckle', ['splash'])], [member('snorlax', ['bodyslam'])]];
	const battles = [
		{ index: 0, teams: [0, 1], learner: 0 },
		{ index: 1, teams: [2, 1], learner: 1 },
		{ index: 2, teams: [1, 2], learner: 0 },
	];
	const path = require('node:path');
	const fs = require('node:fs');
	const file = path.join(__dirname, '..', 'work', 'codex', 'tmp-showdown-player-pool.json');
	fs.mkdirSync(path.dirname(file), { recursive: true });
	fs.writeFileSync(file, JSON.stringify({ showdownCommit: JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'reference-lock.json'), 'utf8')).commit, teams: pool }));
	const session = new Session({ pool: path.relative(path.join(__dirname, '..'), file), battles, opponent: 'max-damage', seed: 5, maxSteps: 60, concurrency: 2, record: false });
	let message = session.begin();
	const finished = [];
	let guard = 0;
	while (message.type === 'decisions' && guard++ < 500) {
		finished.push(...message.finished);
		const actions = message.items.map(item => ({ slot: item.slot, action: item.legal.indexOf(true) }));
		message = session.answer(actions);
	}
	finished.push(...message.finished);
	fs.rmSync(file);
	assert.equal(message.type, 'done');
	assert.deepEqual(finished.map(result => result.index).sort(), [0, 1, 2]);
	assert.ok(finished.every(result => result.errors.length === 0));
});
