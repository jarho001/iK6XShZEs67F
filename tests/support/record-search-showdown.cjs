'use strict';

// Record new pinned battles with forced outcomes. This consumes the existing
// fixture's rosters but writes no fixtures and uses a separate battle seed.
const fs = require('node:fs');
const { gunzipSync } = require('node:zlib');
const assert = require('node:assert/strict');
const { playBattle } = require('../../reference/record-battles.cjs');
const { PlayerView } = require('../../reference/player-view.cjs');
const { OwnState } = require('../../reference/search-own-state.cjs');

const [header, ...fixture] = gunzipSync(fs.readFileSync(process.argv[2]))
	.toString().trim().split('\n').map(line => JSON.parse(line));
const pool = { teams: fixture.slice(0, 16).flatMap(battle => battle.teams) };
const masterSeed = 98371;
const output = [];
for (let index = 0; index < 6; index++) {
	const battle = playBattle(index, masterSeed, pool, [], [], [], {
		profile: 'pool', streams: true, views: false,
	});
	battle.requests = battle.streams.map((chunks, side) => {
		const view = new PlayerView({ side });
		const tracker = new OwnState(view);
		const requests = [];
		for (const chunk of chunks) {
			view.receive(chunk);
			if (!chunk.startsWith('|request|') || chunk === '|request|null' || view.ended) continue;
			const observation = view.observation();
			requests.push(structuredClone({
				side, observation, legal: observation.legal_actions, own: tracker.state(observation),
				held: false, refused: false,
				previous: requests.length ? battle.steps[requests.length - 1].actions[side] : 0,
			}));
		}
		assert.equal(requests.length, battle.steps.length);
		return requests;
	});
	output.push(battle);
}
process.stdout.write(JSON.stringify([header, output]));
