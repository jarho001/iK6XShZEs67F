'use strict';

// Run with node --test tests/test-baselines.cjs (requires pinned Showdown). Each case plays a small
// scripted battle in Showdown and checks what the Node baselines (reference/baselines.cjs) choose
// from one player's view and the engine's legal choices.
const { Battle } = require('../reference/vendor/pokemon-showdown/dist/sim/battle');
const { PRNG } = require('../reference/vendor/pokemon-showdown/dist/sim/prng');
const { playerViews, legalChoices } = require('../reference/record-battles.cjs');
const { Baselines, POLICY_NAMES } = require('../reference/baselines.cjs');
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { gunzipSync } = require('node:zlib');

const baselines = new Baselines();

function member(name, moves, extra = {}) {
	return {
		name: extra.name || name, species: name, level: 80, moves, ability: 'noability', item: '',
		evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, nature: 'Hardy', gender: 'M',
		...extra,
	};
}

function play(team1, team2, seed = [1, 2, 3, 4]) {
	const viewer = playerViews({ build: true });
	const battle = new Battle({ formatid: 'gen5randombattle', seed, send: viewer.send });
	battle.setPlayer('p1', { name: 'A', team: team1 });
	battle.setPlayer('p2', { name: 'B', team: team2 });
	battle.sendUpdates();
	const fixed = battle.sides.map(side => [...side.pokemon]);
	return {
		battle, views: viewer.views, fixed,
		choose(first, second) {
			if (first) battle.choose('p1', first);
			if (second) battle.choose('p2', second);
			battle.sendUpdates();
		},
		decide(name, side, prng = new PRNG([1, 2, 3, 4])) {
			const choices = legalChoices(battle.sides[side], fixed[side], battle.ended);
			return baselines.choose(name, { choices, view: viewer.views[side], prng });
		},
	};
}

const filler = [member('Mew', ['splash']), member('Mew', ['splash'], { name: 'Mew B' })];

test('every policy is named and an unknown one is refused', () => {
	assert.deepEqual(POLICY_NAMES, ['random', 'random-move', 'max-damage', 'heuristics']);
	const game = play([member('Mew', ['splash'])], [member('Mew', ['splash'])]);
	assert.throws(() => game.decide('best', 0), /Unknown policy best/);
});

test('random-move picks a move whenever one is legal and a switch only at a forced replacement', () => {
	const game = play(
		[member('Magikarp', ['splash', 'tackle']), ...filler],
		[member('Machamp', ['closecombat', 'splash']), ...filler],
	);
	const prng = new PRNG([5, 6, 7, 8]);
	const seen = new Set();
	for (let i = 0; i < 60; i++) {
		const { choice, random } = game.decide('random-move', 0, prng);
		assert(choice.action < 4 && random);
		seen.add(choice.action);
	}
	assert.deepEqual([...seen].sort(), [0, 1]);
	game.choose('move 1', 'move 1');
	assert(game.battle.sides[0].activeRequest.forceSwitch);
	for (let i = 0; i < 20; i++) assert(game.decide('random-move', 0, prng).choice.action >= 4);
});

test('max-damage picks the highest damage score', () => {
	const game = play(
		[member('Charizard', ['tackle', 'earthquake', 'airslash', 'flamethrower']), ...filler],
		[member('Venusaur', ['splash']), ...filler],
	);
	// Fire and Flying both hit Grass for double and are STAB; Flamethrower has the larger power.
	assert.equal(game.decide('max-damage', 0).choice.choice, 'move 4');
	// A status move scores zero, so the lone attack is chosen whatever its slot.
	const status = play(
		[member('Charizard', ['splash', 'swordsdance', 'tackle', 'toxic']), ...filler],
		[member('Venusaur', ['splash']), ...filler],
	);
	assert.equal(status.decide('max-damage', 0).choice.choice, 'move 3');
});

test('equal scores go to the lowest move slot, also when float arithmetic makes them differ', () => {
	// Thunderbolt and Ice Beam: the same power, accuracy and category, no STAB, neutral on Snorlax.
	const equal = play(
		[member('Starmie', ['splash', 'thunderbolt', 'icebeam']), ...filler],
		[member('Snorlax', ['splash']), ...filler],
	);
	assert.equal(equal.decide('max-damage', 0).choice.choice, 'move 2');
	assert.equal(equal.decide('heuristics', 0).choice.choice, 'move 2');
	// Stone Edge is power 100 at 80 per cent and Iron Head power 80 at 100: 80 either way, but
	// 100 x 0.8 is not 80 in binary floating point.
	const noise = play(
		[member('Mew', ['splash', 'stoneedge', 'ironhead'], { name: 'Lead' }), ...filler],
		[member('Snorlax', ['splash']), ...filler],
	);
	assert.equal(noise.decide('max-damage', 0).choice.choice, 'move 2');
	assert.equal(noise.decide('heuristics', 0).choice.choice, 'move 2');
	const swapped = play(
		[member('Mew', ['splash', 'ironhead', 'stoneedge'], { name: 'Lead' }), ...filler],
		[member('Snorlax', ['splash']), ...filler],
	);
	assert.equal(swapped.decide('max-damage', 0).choice.choice, 'move 2');
	assert.equal(swapped.decide('heuristics', 0).choice.choice, 'move 2');
});

test('max-damage and heuristics send the best matchup in after a faint', () => {
	for (const name of ['max-damage', 'heuristics']) {
		const game = play(
			[member('Magikarp', ['splash']), member('Charizard', ['splash']),
				member('Blastoise', ['splash']), member('Venusaur', ['splash'])],
			[member('Rhydon', ['earthquake', 'splash']), ...filler],
		);
		game.choose('move 1', 'move 1');
		assert(game.battle.sides[0].activeRequest.forceSwitch);
		const { choice, random } = game.decide(name, 0);
		assert.equal(random, false);
		// Water hits Rock/Ground for four times; Charizard is weak to Rock and immune to Ground
		// only on defence, Venusaur takes double from Ground and Rock is neutral for it.
		assert.equal(game.battle.sides[0].pokemon[Number(choice.choice.slice(7)) - 1].name, 'Blastoise');
	}
});

test('a tie between bench members goes to the first in the order of the request, not of the team', () => {
	// The request lists the party, a switch swapping the positions of the two Pokemon: after Lead
	// -> X -> Z the bench reads Lead, Y, X, so identical X and Y tie in favour of Y.
	const gyarados = name => member('Gyarados', ['splash'], { name });
	for (const name of ['max-damage', 'heuristics']) {
		const game = play(
			[member('Snorlax', ['splash'], { name: 'Lead' }), gyarados('X'), gyarados('Y'),
				member('Magikarp', ['splash'], { name: 'Z' })],
			[member('Machamp', ['closecombat', 'splash']), ...filler],
		);
		game.choose('switch 2', 'move 2');
		game.choose('switch 4', 'move 2');
		assert.deepEqual(game.battle.sides[0].pokemon.map(pokemon => pokemon.name), ['Z', 'Lead', 'Y', 'X']);
		game.choose('move 1', 'move 1');
		assert(game.battle.sides[0].activeRequest.forceSwitch);
		const { choice } = game.decide(name, 0);
		assert.equal(game.battle.sides[0].pokemon[Number(choice.choice.slice(7)) - 1].name, 'Y');
		// Team slots 0..3 are Lead, X, Y, Z: with Z active the bench is 0, 1, 2 and Y is action 6.
		assert.equal(choice.action, 6);
	}
});

test('heuristics sets Spikes first, but only while the foe has none', () => {
	const game = play(
		[member('Forretress', ['gyroball', 'spikes', 'rapidspin', 'explosion']), ...filler],
		[member('Snorlax', ['splash']), ...filler],
	);
	assert.equal(game.decide('heuristics', 0).choice.choice, 'move 2');
	game.choose('move 2', 'move 1');
	assert.deepEqual(game.views[0].observation().hazards[1], [1, 0]);
	assert.notEqual(game.decide('heuristics', 0).choice.choice, 'move 2');
});

test('heuristics removes hazards with Rapid Spin while two members are left', () => {
	const team = [member('Starmie', ['surf', 'rapidspin']), ...filler];
	const foe = [member('Forretress', ['spikes', 'splash']), ...filler];
	const game = play(team, foe);
	assert.equal(game.decide('heuristics', 0).choice.choice, 'move 1');
	game.choose('move 1', 'move 1');
	assert.deepEqual(game.views[0].observation().hazards[0], [1, 0]);
	assert.equal(game.decide('heuristics', 0).choice.choice, 'move 2');
	// A side down to its last member does not bother.
	const alone = play([member('Starmie', ['surf', 'rapidspin'])], foe);
	alone.choose('move 1', 'move 1');
	assert.deepEqual(alone.views[0].observation().hazards[0], [1, 0]);
	assert.equal(alone.decide('heuristics', 0).choice.choice, 'move 1');
});

test('heuristics leaves a member whose defence collapsed when a decent switch exists', () => {
	const game = play(
		[member('Gengar', ['shadowball', 'splash']), member('Snorlax', ['splash']),
			member('Gyarados', ['splash'])],
		[member('Machamp', ['faketears', 'splash']), ...filler],
	);
	// Gyarados (Flying hits Fighting for double, Fighting hits it for half) is a decent switch-in.
	assert(game.decide('heuristics', 0).choice.action < 4);
	game.choose('move 2', 'move 1');
	assert(game.decide('heuristics', 0).choice.action < 4, 'one drop is not enough');
	game.choose('move 2', 'move 1');
	assert.equal(game.views[0].observation().own.boosts[3], -4 / 6);
	const { choice } = game.decide('heuristics', 0);
	// The bench in request order is Snorlax, Gyarados: the best matchup is Gyarados.
	assert.equal(game.battle.sides[0].pokemon[Number(choice.choice.slice(7)) - 1].name, 'Gyarados');
	// Max-damage only switches when it must.
	assert(game.decide('max-damage', 0).choice.action < 4);
});

test('the setup branch of poke-env is dead by default and works when enabled', () => {
	const choose = (game, options) => new Baselines(options).choose('heuristics', {
		choices: legalChoices(game.battle.sides[0], game.fixed[0], game.battle.ended),
		view: game.views[0], prng: new PRNG([1, 2, 3, 4]),
	}).choice.choice;
	// Gengar against Machamp: a positive matchup at full HP.
	const game = play(
		[member('Gengar', ['shadowball', 'swordsdance']), ...filler],
		[member('Machamp', ['splash']), ...filler],
	);
	assert.equal(choose(game, {}), 'move 1');
	assert.equal(choose(game, { setupBranch: true }), 'move 2');
	// Snorlax against Machamp has a negative matchup: no setup even when the branch runs.
	const bad = play(
		[member('Snorlax', ['bodyslam', 'swordsdance']), ...filler],
		[member('Machamp', ['splash']), ...filler],
	);
	assert.equal(choose(bad, { setupBranch: true }), 'move 1');
});

function readRecording(file) {
	const [header, ...battles] = gunzipSync(fs.readFileSync(file)).toString().trim().split('\n').map(JSON.parse);
	return { header, battles };
}

function checkPolicyRecording({ header, battles }, count) {
	assert.deepEqual(header.policies.names, ['max-damage', 'heuristics']);
	assert.equal(battles.length, count);
	for (const battle of battles) {
		// Actions and outcomes are all a replay needs: no state snapshots, no views.
		assert(!('initialState' in battle) && !('initialViews' in battle));
		assert.deepEqual(Object.keys(battle.initialPolicies), header.policies.names);
		assert.equal(battle.steps.length + 1, [battle.initialPolicies, ...battle.steps.map(step => step.policies)].length);
		for (const step of battle.steps) {
			assert(!('state' in step) && !('views' in step) && step.active.length === 2);
			for (const name of header.policies.names) {
				assert(step.policies[name].length === 2 && step.policies[name].every(a => a >= 0 && a < 9));
			}
		}
		// The recorded policies never fell back to a random pick, and a finished battle has no choice.
		assert.deepEqual(battle.stats.policyFallbacks, { 'max-damage': 0, heuristics: 0 });
		assert.deepEqual(battle.steps.at(-1).policies, { 'max-damage': [0, 0], heuristics: [0, 0] });
	}
}

test('the committed policy fixture records both policies at every decision point', () => {
	checkPolicyRecording(readRecording(path.join(__dirname, 'fixtures/policy-pool.jsonl.gz')), 40);
});

test('record-battles --policies writes the policies of both sides and nothing a replay does not need', () => {
	const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'baselines-'));
	const out = path.join(directory, 'policies.jsonl.gz');
	try {
		execFileSync('node', [path.join(__dirname, '../reference/record-battles.cjs'), '--profile', 'pool',
			'--policies', '--count', '3', '--seed', '4', '--out', out], { cwd: path.join(__dirname, '..'), stdio: 'pipe' });
		checkPolicyRecording(readRecording(out), 3);
	} finally { fs.rmSync(directory, { recursive: true, force: true }); }
});
