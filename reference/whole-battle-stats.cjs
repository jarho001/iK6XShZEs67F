#!/usr/bin/env node
'use strict';

/* Generate isolation teams with the recorder's builder and play them on Showdown.
 * The JSON output includes the exact constructor teams so the engine can reuse them.
 */

const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');
const { execFileSync } = require('node:child_process');
const assert = require('node:assert/strict');
const { Battle } = require('./vendor/pokemon-showdown/dist/sim/battle');
const { Dex } = require('./vendor/pokemon-showdown/dist/sim/dex');
const { PRNG } = require('./vendor/pokemon-showdown/dist/sim/prng');
const { isolationMember, legalChoices, seedFor } = require('./record-battles.cjs');

const root = path.resolve(__dirname, '..');
const vendor = path.join(__dirname, 'vendor/pokemon-showdown');
const dex = Dex.forFormat('gen5randombattle');

function playBattle(index, masterSeed, teamPair, policySeedWords) {
	const battle = new Battle({
		formatid: 'gen5randombattle',
		seed: seedFor(masterSeed, index, 'battle'),
		p1: { name: 'Side 0', team: structuredClone(teamPair[0]) },
		p2: { name: 'Side 1', team: structuredClone(teamPair[1]) },
	});
	const policy = new PRNG(policySeedWords);
	const fixed = battle.sides.map(side => [...side.pokemon]);
	const decisions = [];
	let decisionCount = 0;
	try {
		while (!battle.ended) {
			if (battle.turn > 1000 || decisionCount >= 3000) {
				throw new Error(`Battle ${index} exceeded Showdown's turn limit`);
			}
			const selected = battle.sides.map((side, s) => policy.sample(legalChoices(side, fixed[s])));
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
			decisions.push({
				moves,
				switches,
				statuses: battle.sides.flatMap((side, s) => fixed[s].map(pokemon =>
					pokemon.fainted ? '' : pokemon.status)),
				fainted: fixed.flatMap(members => members.map(pokemon => pokemon.fainted)),
				confusion: battle.sides.map(side => !side.active[0].fainted &&
					Boolean(side.active[0].volatiles.confusion)),
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
	for (let index = 0; index < count; index++) {
		const policySeed = seedFor(seed, index, 'policy');
		const policy = new PRNG(policySeed);
		const indices = [policy.random(pool.teams.length), policy.random(pool.teams.length)];
		teamPairs.push(indices.map(teamIndex => pool.teams[teamIndex].map(member =>
			isolationMember(member, moves, options, policy))));
		policySeeds.push(policySeed);
	}
	return { teamPairs, policySeeds };
}

async function main() {
	const { values } = parseArgs({ options: {
		battles: { type: 'string', default: '4000' },
		seed: { type: 'string', default: '1' },
		pool: { type: 'string', default: 'work/teams/gen5-teams.json' },
		allowed: { type: 'string' },		out: { type: 'string', default: 'work/stats/showdown.json' },
	} });
	const count = Number(values.battles);
	const seed = Number(values.seed);
	assert(Number.isSafeInteger(count) && count > 0, '--battles must be a positive safe integer');
	assert(Number.isSafeInteger(seed), '--seed must be a safe integer');
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
		abilities: [...supported.abilities, ...(extra.abilities || []).map(id => dex.toID(id))],
		items: [...supported.items, ...(extra.items || []).map(id => dex.toID(id))],
	};
	assert(moves.length >= 4);
	const { teamPairs, policySeeds } = makeBattleInputs(count, seed, pool, moves, options);
	const battles = [];
	for (let index = 0; index < count; index++) {
		battles.push(playBattle(index, seed, teamPairs[index], policySeeds[index]));
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
		allowed: options,
		acceptedMoves: moves,
		policy: 'uniform over the legal-action mask for both sides at every request',
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
