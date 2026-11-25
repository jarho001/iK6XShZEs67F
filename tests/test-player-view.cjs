'use strict';

// Run with node --test tests/test-player-view.cjs (requires pinned Showdown). Each case plays a
// small scripted battle in Showdown and checks what the player view builds from the protocol and
// requests of one player's stream alone.
const { Battle } = require('../reference/vendor/pokemon-showdown/dist/sim/battle');
const { PRNG } = require('../reference/vendor/pokemon-showdown/dist/sim/prng');
const { playerViews } = require('../reference/record-battles.cjs');
const { PlayerView, loadTables } = require('../reference/player-view.cjs');
const assert = require('node:assert/strict');
const { test } = require('node:test');

const tables = loadTables();
const species = name => tables.species.index.get(name);
const move = name => tables.moves.index.get(name);
const ability = name => tables.abilities.index.get(name);
const item = name => tables.items.index.get(name);
const type = name => tables.types.get(name) + 1;

function member(name, moves, extra = {}) {
	return {
		name: extra.name || name, species: name, level: 80, moves, ability: 'noability', item: '',
		evs: { hp: 85, atk: 85, def: 85, spa: 85, spd: 85, spe: 85 },
		ivs: { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 }, nature: 'Hardy', gender: 'M',
		...extra,
	};
}

// A battle whose two players each get what getPlayerStreams delivers, built by one view each.
function play(team1, team2, seed = [1, 2, 3, 4]) {
	const viewer = playerViews({ build: true, keep: true });
	const battle = new Battle({ formatid: 'gen5randombattle', seed, send: viewer.send });
	battle.setPlayer('p1', { name: 'A', team: team1 });
	battle.setPlayer('p2', { name: 'B', team: team2 });
	battle.sendUpdates();
	return {
		battle,
		views: viewer.views,
		chunks: viewer.chunks,
		choose(first, second) {
			if (first) battle.choose('p1', first);
			if (second) battle.choose('p2', second);
			battle.sendUpdates();
			return viewer.observations();
		},
		observations: () => viewer.observations(),
	};
}

test('the first request names the player and the observation starts at the first decision', () => {
	const game = play([member('Pikachu', ['tackle'])], [member('Mew', ['tackle'])]);
	const [mine, theirs] = game.observations();
	assert.deepEqual(mine.request, [0, 1, 0]);
	assert.deepEqual(mine.legal_actions, [true, false, false, false, false, false, false, false, false]);
	assert.equal(mine.turn, 1 / 1000);
	assert.equal(mine.own.species[0], species('pikachu'));
	assert.equal(theirs.opponent.species[0], species('pikachu'));
	assert.deepEqual(mine.opponent.unseen, [0, 1, 1, 1, 1, 1]);
	assert.equal(game.views[0].me, 0);
	assert.equal(game.views[1].me, 1);
	// Nothing has happened yet: the event record is empty.
	assert.deepEqual(mine.events.order, [0, 0]);
	assert.deepEqual(mine.events.entered, [0, 0]);
});

test('the opponent is shown as percentages, the own side as exact HP', () => {
	const game = play(
		[member('Snorlax', ['tackle']), member('Mew', ['tackle'])],
		[member('Pikachu', ['tackle']), member('Mew', ['tackle'])],
	);
	const [mine, theirs] = game.choose('move 1', 'move 1');
	const damaged = game.battle.sides[0].active[0];
	assert(damaged.hp < damaged.maxhp);
	assert.equal(mine.own.hp[0], damaged.hp / damaged.maxhp);
	assert.equal(theirs.opponent.hp[0], Math.min(Math.ceil(100 * damaged.hp / damaged.maxhp), 99) / 100);
	assert.equal(mine.own.stats[0][0], damaged.maxhp / 1000);
	assert(mine.events.hp_change[0] < 0 && mine.events.hp_change[1] < 0);
	assert.deepEqual(mine.events.order.slice().sort(), [1, 2]);
	assert.deepEqual(mine.events.move, [move('tackle'), move('tackle')]);
});

test('revealed moves are sorted by descending ID and never include called or private ones', () => {
	const game = play(
		[member('Mew', ['tackle', 'splash', 'thunderbolt'])],
		[member('Pikachu', ['tackle'])],
	);
	game.choose('move 1', 'move 1');
	let [mine] = game.choose('move 3', 'move 1');
	assert.deepEqual(mine.opponent.moves[0], [move('tackle'), 0, 0, 0]);
	[, mine] = game.observations();
	const wanted = [move('thunderbolt'), move('tackle'), 0, 0];
	assert.deepEqual(mine.opponent.moves[0], wanted.sort((a, b) => b - a));
});

test('Illusion shows the disguise until it breaks, then the real species', () => {
	const game = play(
		[member('Zoroark', ['tackle', 'nastyplot'], { ability: 'illusion' }),
			member('Snorlax', ['tackle']), member('Machamp', ['tackle'])],
		[member('Pikachu', ['tackle', 'thunderwave'])],
	);
	let [mine, theirs] = game.observations();
	// The holder's own player sees the real team in its request.
	assert.equal(mine.own.species[0], species('zoroark'));
	assert.equal(mine.own.ability[0], ability('illusion'));
	// The opponent sees the disguise: Machamp (the last member), with its types.
	assert.equal(theirs.opponent.species[0], species('machamp'));
	assert.deepEqual(theirs.opponent.types[0], [type('fighting'), 0]);
	assert.equal(theirs.opponent.ability[0], 0);
	// A damaging hit breaks it: the real species replaces the disguise, Illusion is revealed.
	[mine, theirs] = game.choose('move 2', 'move 1');
	assert.equal(theirs.opponent.species[0], species('zoroark'));
	assert.deepEqual(theirs.opponent.types[0], [type('dark'), 0]);
	assert.equal(theirs.opponent.ability[0], ability('illusion'));
	assert.deepEqual(theirs.opponent.unseen, [0, 1, 1, 1, 1, 1]);
});

test('Imposter shows a transformed member: the target species, types and no moves yet', () => {
	const game = play(
		[member('Ditto', ['transform'], { ability: 'imposter', level: 88 })],
		[member('Gengar', ['shadowball', 'hypnosis'], { level: 82 })],
	);
	const [mine, theirs] = game.observations();
	assert.equal(theirs.opponent.species[0], species('gengar'));
	assert.deepEqual(theirs.opponent.types[0], [type('ghost'), type('poison')]);
	assert.equal(theirs.opponent.transformed, 1);
	assert.equal(theirs.opponent.ability[0], ability('imposter'));
	assert.deepEqual(theirs.opponent.moves[0], [0, 0, 0, 0]);
	// The Ditto's own request keeps its own stats and the copied moves, at 5 PP.
	assert.equal(mine.own.species[0], species('gengar'));
	assert.equal(mine.own.transformed, 1);
	assert.deepEqual(mine.own.moves[0], [move('shadowball'), move('hypnosis'), 0, 0]);
	assert.equal(mine.own.pp[0][0], 1);
	const own = game.battle.sides[0].pokemon[0];
	assert.equal(mine.own.stats[0][1], own.baseStoredStats.atk / 1000);
	// Using a copied move reveals it; leaving the field ends the transformation.
	const [, after] = game.choose('move 1', 'move 1');
	assert.deepEqual(after.opponent.moves[0], [move('shadowball'), 0, 0, 0]);
});

test('a forme change names the species; leaving the field restores the base species', () => {
	const game = play(
		[member('Politoed', ['splash'], { ability: 'drizzle' }), member('Mew', ['splash'])],
		[member('Castform', ['splash'], { ability: 'forecast' }), member('Mew', ['splash'])],
	);
	let [mine, theirs] = game.observations();
	assert.equal(mine.weather, 1);
	assert.equal(mine.opponent.species[0], species('castformrainy'));
	assert.deepEqual(mine.opponent.types[0], [type('water'), 0]);
	assert.equal(mine.opponent.ability[0], ability('forecast'));
	// The Castform's own request still says Castform; the forme change comes from the protocol.
	assert.equal(theirs.own.species[0], species('castformrainy'));
	[mine] = game.choose('move 1', 'switch 2');
	assert.equal(mine.opponent.species[0], species('castform'));
	assert.deepEqual(mine.opponent.types[0], [type('normal'), 0]);
});

test('Color Change names the new type', () => {
	const game = play(
		[member('Kecleon', ['splash'], { ability: 'colorchange' }), member('Mew', ['tackle'])],
		[member('Pikachu', ['thunderbolt'])],
	);
	const [mine, theirs] = game.choose('move 1', 'move 1');
	assert.deepEqual(theirs.opponent.types[0], [type('electric'), 0]);
	assert.deepEqual(mine.own.types[0], [type('electric'), 0]);
	assert.equal(theirs.opponent.ability[0], ability('colorchange'));
});

test('Frisk names the foe item to everyone, and the own request shows the own item', () => {
	const game = play(
		[member('Furret', ['splash'], { ability: 'frisk', item: 'leftovers' })],
		[member('Pikachu', ['splash'], { item: 'sitrusberry' })],
	);
	const [mine, theirs] = game.observations();
	// Frisk names the item of the holder's foe (the line has no target in Gen 5) and the holder.
	assert.equal(mine.opponent.item[0], item('sitrusberry'));
	assert.equal(mine.opponent.item_revealed[0], 1);
	assert.equal(mine.own.ability[0], ability('frisk'));
	assert.equal(theirs.opponent.ability[0], ability('frisk'));
	assert.equal(theirs.opponent.item[0], 0);
	assert.equal(theirs.opponent.item_revealed[0], 0);
	assert.equal(theirs.own.item[0], item('sitrusberry'));
});

test('Knock Off marks the item as taken for both players', () => {
	const game = play(
		[member('Weavile', ['knockoff'])],
		[member('Snorlax', ['splash'], { item: 'leftovers' })],
	);
	const [mine, theirs] = game.choose('move 1', 'move 1');
	assert.equal(mine.opponent.item[0], item('leftovers'));
	assert.equal(mine.opponent.item_knocked_off[0], 1);
	assert.equal(mine.opponent.item_consumed[0], 0);
	assert.equal(theirs.own.item[0], 0);
	assert.equal(theirs.own.item_knocked_off[0], true);
	assert.equal(theirs.own.item_consumed[0], false);
});

test('a berry that is eaten shows as consumed, not knocked off', () => {
	const game = play(
		[member('Raticate', ['superfang', 'tackle'])],
		[member('Snorlax', ['splash'], { item: 'sitrusberry' })],
	);
	let [mine, theirs] = game.observations();
	assert.equal(mine.opponent.item_revealed[0], 0);
	// Super Fang halves the HP (rounding down what is taken); the next hit puts it under half.
	game.choose('move 1', 'move 1');
	[mine, theirs] = game.choose('move 2', 'move 1');
	assert.equal(mine.opponent.item[0], item('sitrusberry'));
	assert.equal(mine.opponent.item_consumed[0], 1);
	assert.equal(mine.opponent.item_knocked_off[0], 0);
	assert.equal(theirs.own.item[0], 0);
	assert.equal(theirs.own.item_consumed[0], true);
	assert.equal(theirs.own.item_knocked_off[0], false);
});

test('Trick swaps the named items and the Choice lock goes with the item', () => {
	const game = play(
		[member('Alakazam', ['trick', 'psychic'], { item: 'choicescarf' })],
		[member('Snorlax', ['splash'], { item: 'leftovers' })],
	);
	let [mine, theirs] = game.choose('move 2', 'move 1');
	// Locked into Psychic by the Scarf.
	assert.equal(mine.own.choice_locked, 1);
	assert.equal(mine.own.choice_move, move('psychic'));
	assert.deepEqual(mine.legal_actions.slice(0, 4), [false, true, false, false]);
	[mine, theirs] = game.observations();
	assert.equal(theirs.opponent.choice_locked, undefined);
});

test('a U-turn pivot is a mid-turn switch request with its own entry kind', () => {
	const game = play(
		[member('Scizor', ['uturn'], { level: 100 }), member('Mew', ['splash'])],
		[member('Snorlax', ['splash'], { level: 100 }), member('Pikachu', ['splash'])],
	);
	let [mine, theirs] = game.choose('move 1', 'move 1');
	// U-turn needs a replacement: the request is a switch request, the turn is not over.
	assert.deepEqual(mine.request, [0, 0, 1]);
	assert.deepEqual(theirs.request, [1, 0, 0]);
	assert.equal(mine.mid_turn, 1);
	assert.equal(theirs.mid_turn, 1);
	assert.deepEqual(mine.legal_actions, [false, false, false, false, true, false, false, false, false]);
	[mine, theirs] = game.choose('switch 2', null);
	assert.deepEqual(mine.events.entered, [3, 0]);
	assert.deepEqual(theirs.events.entered, [0, 3]);
	assert.equal(mine.mid_turn, 0);
	assert.deepEqual(mine.request, [0, 1, 0]);
});

test('a drag is its own entry kind and shows the dragged member', () => {
	const game = play(
		[member('Gyarados', ['roar'], { level: 100 }), member('Mew', ['splash'])],
		[member('Magikarp', ['splash'], { level: 100 }), member('Pikachu', ['splash'])],
	);
	const [mine, theirs] = game.choose('move 1', 'move 1');
	assert.deepEqual(mine.events.entered, [0, 2]);
	assert.deepEqual(theirs.events.entered, [2, 0]);
	assert.equal(mine.opponent.species[1], species('pikachu'));
	assert.deepEqual(mine.opponent.is_active, [0, 1, 0, 0, 0, 0]);
});

test('a faint is followed by a wait for the other side and a replacement entry', () => {
	const game = play(
		[member('Scizor', ['tackle'], { level: 100 }), member('Mew', ['splash'])],
		[member('Magikarp', ['splash'], { level: 5 }), member('Pikachu', ['splash'])],
	);
	let [mine, theirs] = game.choose('move 1', 'move 1');
	assert.deepEqual(mine.request, [1, 0, 0]);
	assert.deepEqual(theirs.request, [0, 0, 1]);
	assert.equal(theirs.events.fainted[0], 1);
	assert.equal(mine.events.fainted[1], 1);
	assert.equal(mine.opponent.fainted[0], 1);
	assert.equal(mine.mid_turn, 1);
	[mine, theirs] = game.choose(null, 'switch 2');
	assert.deepEqual(theirs.events.entered, [4, 0]);
	assert.deepEqual(mine.events.entered, [0, 4]);
	assert.equal(mine.opponent.species[1], species('pikachu'));
	assert.equal(mine.opponent.unseen[2], 1);
});

test('weather started by an ability, with its age', () => {
	const game = play(
		[member('Politoed', ['splash'], { ability: 'drizzle' })],
		[member('Mew', ['splash'])],
	);
	let [mine] = game.observations();
	assert.equal(mine.weather, 1);
	assert.equal(mine.weather_age, 0);
	[mine] = game.choose('move 1', 'move 1');
	assert.equal(mine.weather_age, 1);
	[mine] = game.choose('move 1', 'move 1');
	assert.equal(mine.weather_age, 2);
});

test('Wish is pending from its move line until it resolves, and ages by turns', () => {
	const game = play(
		[member('Blissey', ['wish', 'splash'])],
		[member('Mew', ['splash'])],
	);
	let [mine, theirs] = game.choose('move 1', 'move 1');
	assert.deepEqual(mine.wish, [1, 0]);
	assert.deepEqual(theirs.wish, [0, 1]);
	assert.deepEqual(mine.wish_age, [1, 0]);
	[mine, theirs] = game.choose('move 2', 'move 1');
	assert.deepEqual(mine.wish, [0, 0]);
	assert.deepEqual(mine.wish_age, [0, 0]);
});

test('Healing Wish waits for a switch-in, then is spent', () => {
	const game = play(
		[member('Jirachi', ['healingwish', 'splash']), member('Mew', ['splash'])],
		[member('Pikachu', ['tackle'])],
	);
	let [mine, theirs] = game.choose('move 1', 'move 1');
	assert.deepEqual(mine.healing_wish, [1, 0]);
	assert.deepEqual(theirs.healing_wish, [0, 1]);
	[mine, theirs] = game.choose('switch 2', null);
	assert.deepEqual(mine.healing_wish, [0, 0]);
});

test('Spikes and Toxic Spikes count layers per side', () => {
	const game = play(
		[member('Forretress', ['spikes', 'toxicspikes'])],
		[member('Snorlax', ['splash'])],
	);
	game.choose('move 1', 'move 1');
	let [mine, theirs] = game.choose('move 1', 'move 1');
	assert.deepEqual(mine.hazards, [[0, 0], [2, 0]]);
	assert.deepEqual(theirs.hazards, [[2, 0], [0, 0]]);
	[mine, theirs] = game.choose('move 2', 'move 1');
	assert.deepEqual(mine.hazards, [[0, 0], [2, 1]]);
});

test('a request with a locked move or a trap restricts the legal actions', () => {
	const game = play(
		[member('Dragonite', ['outrage', 'splash']), member('Mew', ['splash'])],
		[member('Snorlax', ['splash']), member('Pikachu', ['splash'])],
	);
	let [mine] = game.choose('move 1', 'move 1');
	// Locked into Outrage: one move, no switch.
	assert.equal(mine.own.locked, 1);
	assert.deepEqual(mine.legal_actions, [true, false, false, false, false, false, false, false, false]);
	// The opponent's lock is not shown.
	const [, theirs] = game.observations();
	assert.equal('locked' in theirs.opponent, false);
});

test('Pressure charges a second PP and the own PP follows the lines', () => {
	const game = play(
		[member('Mew', ['tackle'])],
		[member('Dusclops', ['splash'], { ability: 'pressure' })],
	);
	const [mine] = game.choose('move 1', 'move 1');
	const slot = game.battle.sides[0].pokemon[0].moveSlots[0];
	assert.equal(mine.own.pp[0][0], slot.pp / slot.maxpp);
	assert.equal(slot.maxpp - slot.pp, 2);
});

test('a fainted Pokemon clears its boosts and volatiles; the replacement request follows', () => {
	const game = play(
		[member('Mew', ['swordsdance', 'tackle']), member('Pikachu', ['splash'])],
		[member('Snorlax', ['bodyslam'], { level: 100 }), member('Mew', ['splash'])],
	);
	game.choose('move 1', 'move 1');
	let [mine] = game.observations();
	assert.equal(mine.own.boosts[0], 2 / 6);
	for (let i = 0; i < 5 && !game.battle.sides[0].pokemon[0].fainted; i++) game.choose('move 1', 'move 1');
	[mine] = game.observations();
	assert(game.battle.sides[0].pokemon[0].fainted);
	assert.deepEqual(mine.request, [0, 0, 1]);
	assert.equal(mine.own.boosts[0], 0);
	assert.equal(mine.own.fainted[0], 1);
	assert.equal(mine.events.fainted[0], 1);
});

test('a random battle never throws and keeps its shapes', () => {
	const policy = new PRNG([5, 6, 7, 8]);
	const pool = ['Pikachu', 'Snorlax', 'Mew', 'Gengar', 'Scizor', 'Blissey'];
	const team = () => pool.map(name => member(name, ['tackle', 'splash', 'thunderbolt', 'uturn']));
	const game = play(team(), team(), [9, 8, 7, 6]);
	for (let i = 0; i < 2000 && !game.battle.ended; i++) {
		const choices = game.battle.sides.map(side => {
			const request = side.activeRequest;
			if (!request || request.wait) return null;
			const options = [];
			if (request.active && !request.forceSwitch?.[0]) options.push('move 1', 'move 2', 'move 3', 'move 4');
			side.pokemon.forEach((pokemon, index) => { if (index > 0 && !pokemon.fainted) options.push('switch ' + (index + 1)); });
			return policy.sample(options);
		});
		const [mine, theirs] = game.choose(...choices);
		for (const observation of [mine, theirs]) {
			assert.equal(observation.own.species.length, 6);
			assert.equal(observation.opponent.species.length, 6);
			assert.equal(observation.request.reduce((a, b) => a + b), 1);
			assert.equal(observation.legal_actions.length, 9);
			assert(observation.legal_actions.some(Boolean));
		}
	}
	assert(game.battle.ended);
	assert.deepEqual(game.views[0].observation().request, [1, 0, 0]);
});

test('a view needs a request before it can describe anything, and refuses a foreign one', () => {
	const view = new PlayerView();
	assert.throws(() => view.observation(), /request/);
	const game = play([member('Mew', ['splash'])], [member('Pikachu', ['splash'])]);
	const chunk = game.chunks[1].find(text => text.startsWith('|request|'));
	assert.throws(() => { const other = new PlayerView(); other.receive(game.chunks[0].find(t => t.startsWith('|request|'))); other.receive(chunk); }, /Request for p2/);
});

test('the view can be driven one line at a time, with the side named in advance', () => {
	const game = play(
		[member('Snorlax', ['tackle', 'splash']), member('Mew', ['splash'])],
		[member('Pikachu', ['tackle']), member('Mew', ['splash'])],
	);
	game.choose('move 1', 'move 1');
	game.choose('switch 2', 'move 1');
	const chunks = game.chunks[1];
	for (const side of [undefined, 'p2']) {
		const live = new PlayerView(side ? { side } : {});
		for (const chunk of chunks) for (const line of chunk.split('\n')) live.receive(line);
		assert.deepEqual(live.observation(), game.views[1].observation());
	}
	assert.throws(() => {
		const wrong = new PlayerView({ side: 'p1' });
		for (const chunk of chunks.slice(0, 2)) wrong.receive(chunk);
	}, /Request for p2/);
});

test('a request that is sent again after a rejected choice changes nothing but the request', () => {
	const game = play(
		[member('Gengar', ['shadowball', 'splash'])],
		[member('Pikachu', ['tackle'])],
	);
	game.choose('move 1', 'move 1');
	const before = game.views[0].observation();
	const request = game.chunks[0].filter(chunk => chunk.startsWith('|request|')).at(-1);
	game.views[0].receive(request);
	assert.deepEqual(game.views[0].observation(), before);
});

test('a switch refused by a hidden trap is asked again with trapped: true, and the trap is remembered', () => {
	const game = play(
		[member('Gengar', ['shadowball', 'splash']), member('Mew', ['splash'])],
		[member('Wobbuffet', ['splash'], { ability: 'shadowtag' }), member('Pikachu', ['splash'])],
	);
	const request = () => game.battle.sides[0].activeRequest.active[0];
	const switches = mask => mask.slice(4);
	const none = [false, false, false, false, false];
	// Hidden: the request only says maybeTrapped, so the switch is on offer.
	let [mine] = game.observations();
	assert.equal(request().maybeTrapped, true);
	assert.equal(request().trapped, undefined);
	assert.deepEqual(switches(mine.legal_actions), [true, false, false, false, false]);
	// p2's choice is made first and stays locked in while p1's switch is refused.
	assert.equal(game.battle.choose('p2', 'move 1'), true);
	assert.equal(game.battle.choose('p1', 'switch 2'), false);
	assert.match(game.battle.sides[0].choice.error, /trapped/);
	assert.equal(game.battle.turn, 1);
	assert.equal(game.battle.sides[1].isChoiceDone(), true);
	// Nothing but an error line and the updated request reached p1: trapped, no maybeTrapped.
	const [error, update] = game.chunks[0].slice(-2);
	assert.match(error, /^\|error\|\[Unavailable choice\] Can't switch: The active Pok.mon is trapped$/);
	assert.match(update, /^\|request\|/);
	assert.equal(JSON.parse(update.slice(9)).update, true);
	assert.equal(request().trapped, true);
	assert.equal(request().maybeTrapped, undefined);
	[mine] = game.observations();
	assert.deepEqual(mine.legal_actions.slice(0, 2), [true, true]);
	assert.deepEqual(switches(mine.legal_actions), none);
	assert.equal(mine.own.trapped, 0); // that flag is the announced partial trap, not this one
	assert.deepEqual(mine.request, [0, 1, 0]);
	// p1 picks a move; the turn runs. The next request says maybeTrapped again, the view does not.
	assert.equal(game.battle.choose('p1', 'move 1'), true);
	game.battle.sendUpdates();
	assert.equal(game.battle.turn, 2);
	assert.equal(request().maybeTrapped, true);
	assert.equal(request().trapped, undefined);
	[mine] = game.observations();
	assert.deepEqual(switches(mine.legal_actions), none);
	// The trapper's player is not told anything: p2 sees its own mask with the switch on offer.
	assert.deepEqual(switches(game.observations()[1].legal_actions), [true, false, false, false, false]);
	// The trapper leaves: the foe is a Pikachu now, the request has no trap flag and the view
	// forgets what it knew.
	assert.equal(game.battle.choose('p2', 'switch 2'), true);
	assert.equal(game.battle.choose('p1', 'move 1'), true);
	game.battle.sendUpdates();
	assert.equal(request().maybeTrapped, undefined);
	assert.equal(request().trapped, undefined);
	[mine] = game.observations();
	assert.deepEqual(switches(mine.legal_actions), [true, false, false, false, false]);
});

test('a refused switch is remembered the same way by a view that gets the lines one at a time', () => {
	const game = play(
		[member('Gengar', ['shadowball', 'splash']), member('Mew', ['splash'])],
		[member('Wobbuffet', ['splash'], { ability: 'shadowtag' }), member('Pikachu', ['splash'])],
	);
	game.battle.choose('p2', 'move 1');
	game.battle.choose('p1', 'switch 2');
	game.battle.choose('p1', 'move 1');
	game.battle.sendUpdates();
	assert.equal(game.views[0].trapHolds(), true);
	const live = new PlayerView();
	for (const chunk of game.chunks[0]) {
		live.receive('|inactive|Time left: 300 sec this turn');
		live.receive('|t:|123456');
		for (const line of chunk.split('\n')) live.receive(line);
	}
	assert.deepEqual(live.observation(), game.views[0].observation());
	assert.equal(live.trapHolds(), true);
});

test('the own request names Hidden Power\'s type and Return\'s power, and the view keeps both', () => {
	// All IVs 31 make Hidden Power Dark (the request gives its type only, as generation 5 does), and
	// happiness 255 makes Return's power 102. Only the own side has them.
	const game = play(
		[member('Gengar', ['hiddenpower', 'return', 'splash']), member('Mew', ['splash'])],
		[member('Wobbuffet', ['splash']), member('Pikachu', ['splash'])],
	);
	const own = game.views[0].ownObservation();
	assert.equal(own.hidden_type[0][0], type('dark'));
	assert.deepEqual(own.hidden_type[0].slice(1), [0, 0, 0]);
	assert.equal(own.return_power[0][1], 102);
	assert.deepEqual([own.return_power[0][0], own.return_power[0][2], own.return_power[0][3]], [0, 0, 0]);
	assert.deepEqual(own.hidden_type[1], [0, 0, 0, 0]);
	// The foe's request is not the learner's: its Hidden Power and Return are not in the observation.
	assert.equal('hidden_type' in game.views[0].opponentObservation(), false);
});

test('sleep source follows Rest and an opposing sleep move in a recorded player stream', () => {
	const game = play(
		[member('Mew', ['tackle', 'spore'])],
		[member('Snorlax', ['rest', 'splash']), member('Pikachu', ['splash'])],
	);
	game.choose('move 1', 'move 2');
	game.choose('move 1', 'move 1');
	assert.equal(game.views[0].sides[1].members[0].sleepOrigin, 0);
	game.choose('move 2', 'switch 2');
	assert.equal(game.views[0].sides[1].members[1].sleepOrigin, 1);
	const replay = new PlayerView();
	for (const chunk of game.chunks[0]) replay.receive(chunk);
	assert.deepEqual(replay.sides[1].members.map(member => member.sleepOrigin), [0, 1]);
});

test('delayed Yawn sleep keeps its opposing source in the recorded stream', () => {
	const game = play([member('Mew', ['yawn', 'splash'])], [member('Snorlax', ['splash'])]);
	game.choose('move 1', 'move 1');
	game.choose('move 2', 'move 1');
	assert.equal(game.views[0].sides[1].members[0].status, 'slp');
	assert.equal(game.views[0].sides[1].members[0].sleepOrigin, 1);
	const replay = new PlayerView();
	for (const chunk of game.chunks[0]) replay.receive(chunk);
	assert.equal(replay.sides[1].members[0].sleepOrigin, 1);
});
