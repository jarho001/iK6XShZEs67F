#!/usr/bin/env node
'use strict';

/* Generate isolation teams with the recorder's builder and play them on Showdown.
 * The JSON output includes the exact constructor teams so the engine can reuse them.
 * --unchanged plays pool teams exactly as generated (no isolation, no replacements), drawn
 * from the supported indices of pokejax.sampler; the default is isolation teams.
 * --policy0 and --policy1 choose each side's policy (default random, uniform over the legal choices):
 * random, random-move, max-damage or heuristics (reference/baselines.cjs; pokejax.baselines has the
 * same four). The last two read only their player's own view (reference/player-view.cjs).
 */

const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');
const { execFileSync } = require('node:child_process');
const assert = require('node:assert/strict');
const { Dex } = require('./vendor/pokemon-showdown/dist/sim/dex');
const { PRNG } = require('./vendor/pokemon-showdown/dist/sim/prng');
const { isolationMember, legalChoices, seedFor, isolationBattle, playerViews } = require('./record-battles.cjs');
const { Baselines, POLICY_NAMES } = require('./baselines.cjs');

const root = path.resolve(__dirname, '..');
const vendor = path.join(__dirname, 'vendor/pokemon-showdown');
const dex = Dex.forFormat('gen5randombattle');

const READS_VIEW = new Set(['max-damage', 'heuristics']);

function playBattle(index, masterSeed, teamPair, policySeedWords, policies, baselines) {
	// The deterministic policies read the player's own view, fed what its stream receives.
	const viewer = policies.some(name => READS_VIEW.has(name)) ? playerViews({ build: true }) : null;
	const battle = isolationBattle(seedFor(masterSeed, index, 'battle'), viewer?.send);
	battle.setPlayer('p1', { name: 'Side 0', team: structuredClone(teamPair[0]) });
	battle.setPlayer('p2', { name: 'Side 1', team: structuredClone(teamPair[1]) });
	if (viewer) battle.sendUpdates();
	const policy = new PRNG(policySeedWords);
	const fixed = battle.sides.map(side => [...side.pokemon]);
	const decisions = [];
	let decisionCount = 0;
	try {
		while (!battle.ended) {
			if (battle.turn > 1000 || decisionCount >= 3000) {
				throw new Error(`Battle ${index} exceeded Showdown's turn limit`);
			}
			const selected = battle.sides.map((side, s) => baselines.choose(policies[s], {
				choices: legalChoices(side, fixed[s]), view: viewer?.views[s], prng: policy,
			}).choice);
			const moves = [null, null];
			const switches = [false, false];
			for (let side = 0; side < 2; side++) {
				const choice = selected[side];
				const request = battle.sides[side].activeRequest;
				if (choice.action >= 4) {
					switches[side] = true;
				} else if (request?.active) {
					const moveNumber = Number(choice.choice.match(/^move (\d+)$/)[1]) - 1;
					const move = request.active[0].moves[moveNumber];
					assert(move, `Missing requested move at action ${choice.action}`);
					moves[side] = move.id;
				}
			}
			for (const [side, choice] of selected.entries()) {
				if (choice.choice !== null) {
					assert(battle.choose(`p${side + 1}`, choice.choice));
				}
			}
			if (viewer) battle.sendUpdates();
			decisions.push({
				moves,
				switches,
				statuses: battle.sides.flatMap((side, s) => fixed[s].map(pokemon =>
					pokemon.fainted ? '' : pokemon.status)),
				fainted: fixed.flatMap(members => members.map(pokemon => pokemon.fainted)),
				confusion: battle.sides.map(side => !side.active[0].fainted &&
					Boolean(side.active[0].volatiles.confusion)),
				itemConsumed: fixed.map(members => members.map(pokemon =>
					!pokemon.item && Boolean(pokemon.lastItem) && !pokemon.itemKnockedOff)),
			});
			decisionCount++;
		}
		const winner = !battle.winner ? 'tie' :
			battle.sides.findIndex(side => side.name === battle.winner);
		return { winner, turns: battle.turn, decisions };
	} finally {
		battle.destroy();
	}
}

function makeBattleInputs(count, seed, pool, moves, options) {
	const teamPairs = [];
	const policySeeds = [];
	const drawTeam = policy => (options.teamIndices ?
		options.teamIndices[policy.random(options.teamIndices.length)] : policy.random(pool.teams.length));
	for (let index = 0; index < count; index++) {
		const policySeed = seedFor(seed, index, 'policy');
		const policy = new PRNG(policySeed);
		const indices = [drawTeam(policy), drawTeam(policy)];
		teamPairs.push(indices.map(teamIndex => pool.teams[teamIndex].map(member =>
			options.unchanged ? structuredClone(member) : isolationMember(member, moves, options, policy))));
		policySeeds.push(policySeed);
	}
	return { teamPairs, policySeeds };
}

async function main() {
	const { values } = parseArgs({ options: {
		battles: { type: 'string', default: '4000' },
		seed: { type: 'string', default: '1' },
		pool: { type: 'string', default: 'work/teams/gen5-teams.json' },
		allowed: { type: 'string' },		unchanged: { type: 'boolean', default: false },
		policy0: { type: 'string', default: 'random' }, policy1: { type: 'string', default: 'random' },
		out: { type: 'string', default: 'work/stats/showdown.json' },
	} });
	const count = Number(values.battles);
	const seed = Number(values.seed);
	assert(Number.isSafeInteger(count) && count > 0, '--battles must be a positive safe integer');
	assert(Number.isSafeInteger(seed), '--seed must be a safe integer');
	const policies = [values.policy0, values.policy1];
	for (const name of policies) assert(POLICY_NAMES.includes(name), `Unknown policy ${name}`);
	const baselines = new Baselines();
	const lock = JSON.parse(fs.readFileSync(path.join(root, 'reference-lock.json'), 'utf8'));
	assert.equal(execFileSync('git', ['-C', vendor, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
		lock.commit);
	const pool = JSON.parse(fs.readFileSync(path.resolve(root, values.pool), 'utf8'));
	assert.equal(pool.showdownCommit, lock.commit);
	const supported = JSON.parse(execFileSync('python', ['-m', 'pokejax.support'], {
		cwd: root, encoding: 'utf8',
	}));
	const extra = values.allowed ? JSON.parse(fs.readFileSync(path.resolve(values.allowed), 'utf8')) : {};
	const moves = [...new Set([...supported.moves, ...(extra.moves || []).map(id => dex.toID(id))])].sort();
	const options = {
		taskMoves: (extra.moves || []).map(id => dex.toID(id)),
		abilities: [...new Set([...supported.abilities, ...(extra.abilities || []).map(id => dex.toID(id))])].sort(),
		items: [...new Set([...supported.items, ...(extra.items || []).map(id => dex.toID(id))])].sort(),
	};
	assert(moves.length >= 4);
	if (values.unchanged) {
		options.unchanged = true;
		options.teamIndices = JSON.parse(execFileSync('python', ['-m', 'pokejax.sampler',
			path.resolve(root, values.pool)], { cwd: root, encoding: 'utf8', maxBuffer: 1 << 26 }));
	}
	const teamMode = values.unchanged ? 'unchanged' : 'isolation';
	const { teamPairs, policySeeds } = makeBattleInputs(count, seed, pool, moves, options);
	const battles = [];
	for (let index = 0; index < count; index++) {
		battles.push(playBattle(index, seed, teamPairs[index], policySeeds[index], policies, baselines));
		if ((index + 1) % 100 === 0 || index + 1 === count) {
			console.log(`Showdown ${index + 1}/${count}`);
		}
	}
	const output = path.resolve(root, values.out);
	fs.mkdirSync(path.dirname(output), { recursive: true });
	fs.writeFileSync(output, JSON.stringify({
		schemaVersion: 1,
		simulator: 'showdown',
		showdownCommit: lock.commit,
		format: 'gen5randombattle',
		seed,
		count,
		teams: teamMode,
		allowed: { taskMoves: options.taskMoves, abilities: options.abilities, items: options.items },
		acceptedMoves: moves,
		policies,
		policy: policies.every(name => name === 'random') ?
			'uniform over the legal-action mask for both sides at every request' :
			policies.map((name, side) => `side ${side}: ${name}`).join('; ') + ' (reference/baselines.cjs)',
		teamPairs,
		battles,
	}));
	console.log(`Wrote ${values.out}`);
}

if (require.main === module) main().catch(error => {
	console.error(error.stack);
	process.exitCode = 1;
});

module.exports = { playBattle, makeBattleInputs };
