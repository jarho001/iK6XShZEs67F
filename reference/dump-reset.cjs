#!/usr/bin/env node
'use strict';

// Pair the first N pool teams consecutively: (0,1), (2,3), ... .
// Compare set-derived member data, before switch-in ability/item effects.
const fs = require('node:fs');
const path = require('node:path');
const { parseArgs } = require('node:util');
const root = path.resolve(__dirname, '..');
const { Battle } = require('./vendor/pokemon-showdown/dist/sim/battle');
const { values } = parseArgs({ options: {
	count: { type: 'string', default: '100' },
	pool: { type: 'string', default: 'work/teams/gen5-teams.json' },
	out: { type: 'string', default: 'tests/fixtures/reset.json' },
} });
const pool = JSON.parse(fs.readFileSync(path.resolve(root, values.pool), 'utf8'));
const count = Number(values.count);
if (!Number.isSafeInteger(count) || count < 2 || count % 2 || count > pool.teams.length) {
	throw new Error('--count must be a positive even number within the pool');
}
const battles = [];
for (let i = 0; i < count; i += 2) {
	const battle = new Battle({
		formatid: 'gen5randombattle', seed: [1, 2, 3, 4],
		p1: { name: 'Side 0', team: structuredClone(pool.teams[i]) },
		p2: { name: 'Side 1', team: structuredClone(pool.teams[i + 1]) },
	});
	if (battle.turn !== 1 || battle.requestState !== 'move') {
		throw new Error('Expected first-turn move request');
	}
	battles.push({
		teams: [i, i + 1], turn: battle.turn,
		sides: battle.sides.map(side => side.pokemon.map(pokemon => ({
			// Transform changes storedStats/types/moveSlots, leaving these base fields intact.
			storedStats: pokemon.baseStoredStats, maxhp: pokemon.maxhp,
			// Resolve the set directly: formeChange can also replace baseSpecies.
			types: battle.dex.species.get(pokemon.set.species).types,
			moves: pokemon.baseMoveSlots.map(slot => slot.id),
			maxpp: pokemon.baseMoveSlots.map(slot => slot.maxpp),
		}))),
	});
	battle.destroy();
}
const output = path.resolve(root, values.out);
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify({
	showdownCommit: pool.showdownCommit, pairing: 'consecutive',
	teamSets: pool.teams.slice(0, count), battles,
}) + '\n');
console.log(`Wrote ${battles.length} reset battles to ${values.out}`);
