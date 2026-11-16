#!/usr/bin/env node
'use strict';

const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { parseArgs } = require('node:util');

const FORMAT = 'gen5randombattle';
const root = path.resolve(__dirname, '..');
const dist = path.join(__dirname, 'vendor', 'pokemon-showdown', 'dist');

function main() {
	const { values } = parseArgs({ options: {
		count: { type: 'string', default: '10000' },
		seed: { type: 'string', default: '1' },
		out: { type: 'string', default: 'work/teams/gen5-teams.json' },
	} });
	const count = Number(values.count);
	const seed = Number(values.seed);
	if (!Number.isSafeInteger(count) || count < 1) throw new Error('--count must be a positive safe integer');
	if (!Number.isSafeInteger(seed)) throw new Error('--seed must be a safe integer');
	if (!fs.existsSync(dist)) {
		throw new Error('Showdown dist is missing; run node reference/setup.cjs first');
	}
	const { Teams } = require(path.join(dist, 'sim', 'teams'));
	const { Dex, toID } = require(path.join(dist, 'sim', 'dex'));
	const lock = JSON.parse(fs.readFileSync(path.join(root, 'reference-lock.json'), 'utf8'));
	// Showdown does not validate random battle teams; the generator is the oracle.
	const dex = Dex.forFormat(FORMAT);
	const exportedSpecies = new Set(JSON.parse(fs.readFileSync(
		path.join(root, 'data', 'gen5-data-v1.json'), 'utf8',
	)).species.map(species => species.id));
	const cosmeticMappings = new Map();
	function speciesID(name) {
		const species = dex.species.get(name);
		// The export omits some purely cosmetic forms (notably Sawsbuck seasons).
		// Use Showdown's marker, preserving every form with an exported row.
		if (!exportedSpecies.has(species.id) && species.isCosmeticForme) {
			cosmeticMappings.set(species.id, (cosmeticMappings.get(species.id) || 0) + 1);
			return toID(species.baseSpecies);
		}
		return toID(name);
	}
	const teams = [];
	for (let teamNumber = 0; teamNumber < count; teamNumber++) {
		// SHA-256 gives each zero-based team number an independent 64-bit seed.
		// This pinned PRNG accepts four uint16 words (the Gen 5 RNG).
		const digest = createHash('sha256').update(`${FORMAT}:${seed}:${teamNumber}`).digest();
		const teamSeed = Array.from({ length: 4 }, (_, i) => digest.readUInt16BE(i * 2));
		const generator = Teams.getGenerator(FORMAT, teamSeed);
		const team = generator.getTeam().map(set => ({
			species: speciesID(set.species),
			ability: toID(set.ability),
			item: toID(set.item),
			moves: set.moves.map(toID),
			nature: toID(set.nature || 'Serious'),
			level: set.level,
			// Pokemon's constructor samples uniformly when the set has no gender.
			gender: set.gender || generator.sample(['M', 'F']),
			evs: set.evs,
			ivs: set.ivs,
			happiness: set.happiness ?? 255,
		}));
		teams.push(team);
	}
	for (const [id, occurrences] of cosmeticMappings) {
		console.error(`Cosmetic normalization: ${id} -> ${toID(dex.species.get(id).baseSpecies)} (${occurrences})`);
	}
	fs.mkdirSync(path.dirname(values.out), { recursive: true });
	fs.writeFileSync(values.out, JSON.stringify({
		format: FORMAT, showdownCommit: lock.commit, seed, count, teams,
	}) + '\n');
	console.log(`Wrote ${count} teams to ${values.out}`);
}

try {
	main();
} catch (error) {
	console.error(`Team generation failed: ${error.message}`);
	process.exitCode = 1;
}
