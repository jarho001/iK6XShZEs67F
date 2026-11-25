#!/usr/bin/env node
'use strict';

/* The Showdown player (Phase 5, task 1): a learner side plays Gen 5 random battles in the pinned
 * simulator, and a policy chooses each of its decisions. The policy is in Python (pokejax/showdown);
 * this file is the bridge. The opponent is a Node baseline or a second Python network actor.
 *
 * Started by pokejax/showdown/bridge.py, which speaks JSON lines on stdin and stdout:
 *
 *   Python -> Node: {"type": "start", "pool": PATH, "battles": [{"index", "teams": [a, b], "learner"}],
 *                    "opponent": NAME, "seed": S, "maxSteps": M, "concurrency": C,
 *                    "record": BOOL, "search": BOOL, "audit": BOOL}
 *                   {"type": "actions", "items": [{"slot": s, "side": p, "action": a}]}
 *   Node -> Python: {"type": "decisions", "items": [...], "finished": [...], "records": [...]}
 *                   {"type": "done", "finished": [...], "records": [...], "stats": {...}}
 *                   {"type": "fatal", "error": TEXT}
 *
 * A decision item is one learner decision: {slot, battle, reset (first decision of a battle in this
 * slot), first (the battle's first decision), refused (the previous switch was refused), request
 * ("move", "switch" or "none": the learner has nothing to choose), legal (9 booleans, the observed
 * mask), observation (reference/player-view.cjs, the learner's view only)}. The answer is an action
 * in pokejax.engine.actions encoding: 0..3 move slots, 4..8 the k-th non-active team slot in
 * ascending order (k = action - 4), and 0 for a side with nothing to choose. A network opponent
 * gets its own item for the same slot, with a separate side and held-choice flag. A search
 * item includes only the receiving side's stream, reconstructed own state and prior action.
 *
 * One battle is a Game. A decision is taken at every point where the battle waits for a choice (a
 * step, as in the engine): the learner's action and the opponent's baseline choice are sent to
 * Showdown together. A switch that Showdown refuses because of a hidden trap (Shadow Tag, Arena Trap,
 * Magnet Pull: the request said maybeTrapped) leaves the battle where it is and asks the learner
 * again; the opponent's choice made at that step stays in Showdown and is played at the next one,
 * as pokejax/env.py does. Each step, refused or not, counts toward the step cap.
 *
 * Node's own record of a battle (the per-battle tracker) keeps both sides' requests, the omniscient
 * protocol and both choices. Oracle snapshots are available with record and audit enabled
 * and are delivered only with the finished battle record.
 */

const fs = require('node:fs');
const path = require('node:path');
const readline = require('node:readline');
const { createHash } = require('node:crypto');
const assert = require('node:assert/strict');
const { Battle, extractChannelMessages } = require('./vendor/pokemon-showdown/dist/sim/battle');
const { PRNG } = require('./vendor/pokemon-showdown/dist/sim/prng');
const { PlayerView, loadTables } = require('./player-view.cjs');
const { Baselines, POLICY_NAMES } = require('./baselines.cjs');
const { legalChoices, snapshot } = require('./record-battles.cjs');

const { OwnState } = require('./search-own-state.cjs');

const root = path.resolve(__dirname, '..');
const FORMAT = 'gen5randombattle';
const TURN_LIMIT = 1000; // sim/battle.ts: a battle still running after turn 1000 is a tie
const NO_REQUEST = 'none';
const MOVE_REQUEST = 'move';
const SWITCH_REQUEST = 'switch';
// The policies that read their player's view (the others read the true legal choices only).
const VIEW_POLICIES = new Set(['max-damage', 'heuristics']);
const TRAPPED_REFUSAL = /^Can't switch: The active Pok.mon is trapped$/;

function seedFor(seed, index, purpose) {
	const bytes = createHash('sha256').update('showdown-player:' + seed + ':' + index + ':' + purpose).digest();
	return Array.from({ length: 4 }, (_, i) => bytes.readUInt16BE(2 * i));
}

// The kind of request a side is answering: the same classes as the view's requestKind.
function requestKind(side) {
	const request = side.activeRequest;
	if (!request || request.wait) return NO_REQUEST;
	if (request.forceSwitch && request.forceSwitch[0]) return SWITCH_REQUEST;
	if (request.active) return MOVE_REQUEST;
	throw new Error('Unknown request: ' + JSON.stringify(request));
}

// The nickname of a request entry ("p1: Name" in its ident).
function identName(ident) {
	return ident.slice(ident.indexOf(': ') + 2);
}

// Team slots of the learner's side in the engine's order (the view's members, in team order), the
// active member's slot, and the request entries by name.
function teamOrder(view, request) {
	const members = view.sides[view.me].members;
	const active = request.side.pokemon.length ? identName(request.side.pokemon[0].ident) : null;
	const activeSlot = members.findIndex(member => member.name === active);
	const bench = members.map((member, slot) => slot).filter(slot => slot !== activeSlot);
	return { members, activeSlot, bench };
}

// The Showdown choice of the learner's action, from its view and its request (the position of a
// switch target in the request's side.pokemon order, not in the team order).
function learnerChoice(view, request, action) {
	if (action < 4) return 'move ' + (action + 1);
	const { members, bench } = teamOrder(view, request);
	const slot = bench[action - 4];
	if (slot === undefined) throw new Error('No switch target for action ' + action);
	const name = members[slot].name;
	const position = request.side.pokemon.findIndex(entry => identName(entry.ident) === name);
	if (position < 0) throw new Error('No request entry for ' + name);
	return 'switch ' + (position + 1);
}

// The mask the learner's request describes, as Showdown accepts choices: moves that are not
// disabled and have PP (or the only move of a locked request), and switches to living bench
// members unless the active is announced as trapped (a move request). A hidden trap the side has
// found out about (Game.decision) is not in the request; the view removes those switches itself.
function requestMask(view, request, kind) {
	const mask = Array(9).fill(false);
	if (kind === NO_REQUEST) {
		mask[0] = true;
		return { mask, canSwitch: false };
	}
	if (kind === MOVE_REQUEST) {
		request.active[0].moves.forEach((move, slot) => {
			mask[slot] = !move.disabled && (move.pp === undefined || move.pp > 0);
		});
	}
	const canSwitch = kind === SWITCH_REQUEST || request.active[0].trapped !== true;
	const { members, bench } = teamOrder(view, request);
	for (const [k, slot] of bench.entries()) {
		const name = members[slot].name;
		const entry = request.side.pokemon.find(candidate => identName(candidate.ident) === name);
		const alive = entry && !/\bfnt$/.test(entry.condition);
		mask[4 + k] = Boolean(canSwitch && alive);
	}
	return { mask, canSwitch };
}

class Game {
	constructor(spec, shared) {
		this.index = spec.index;
		this.learner = spec.learner;
		this.opponent = 1 - spec.learner;
		this.poolIndices = spec.teams;
		this.teams = spec.teams.map(index => shared.pool[index]);
		this.shared = shared;
		this.battleSeed = seedFor(shared.seed, spec.seedIndex ?? spec.index, 'battle');
		this.policy = new PRNG(seedFor(shared.seed, spec.index, 'policy'));
		this.views = [null, null];
		this.views[this.learner] = new PlayerView({ tables: shared.tables });
		if (shared.readsView || shared.network || shared.audit) this.views[this.opponent] = new PlayerView({ tables: shared.tables });
		this.trackers = this.views.map((view, side) => view
			&& (shared.audit || (shared.search && side === this.learner)) ? new OwnState(view) : null);
		this.streams = [[], []];
		this.retries = [false, false];
		this.previous = [0, 0];
		this.battle = null;
		this.fixed = null;
		this.slot = -1;
		this.decisions = 0;
		this.steps = 0;
		this.refusals = 0;
		this.illegal = 0;
		this.maskMismatches = 0;
		this.maskSamples = [];
		this.knownTrapDecisions = 0;
		this.errors = [];
		this.refusedLast = false;
		this.fresh = true;
		this.finished = false;
		this.result = null;
		// The per-battle tracker: the omniscient protocol and both requests of each step.
		this.record = shared.record ? { steps: [] } : null;
		this.current = { protocol: [], requests: [] };
	}

	start() {
		const battle = new Battle({ formatid: FORMAT, seed: this.battleSeed, send: (type, data) => this.receive(type, data) });
		this.battle = battle;
		battle.setPlayer('p1', { name: 'Side 0', team: structuredClone(this.teams[0]) });
		battle.setPlayer('p2', { name: 'Side 1', team: structuredClone(this.teams[1]) });
		this.fixed = battle.sides.map(side => [...side.pokemon]);
		battle.sendUpdates();
	}

	// What the two players' clients receive (sim/battle-stream.ts): each side's lines and request.
	receive(type, data) {
		if (type === 'update') {
			const text = Array.isArray(data) ? data.join('\n') : data;
			if (this.record) this.current.protocol.push(...(Array.isArray(data) ? data : [data]));
			const messages = extractChannelMessages(text, [1, 2]);
			for (const side of [0, 1]) {
				if (this.views[side]) this.views[side].receive(messages[side + 1].join('\n'));
				if (this.trackers[side]) this.streams[side].push(...messages[side + 1]);
			}
		} else if (type === 'sideupdate') {
			const cut = data.indexOf('\n');
			const side = data.slice(0, cut) === 'p1' ? 0 : 1;
			if (this.record) this.current.requests.push({ side, text: data.slice(cut + 1) });
			if (this.views[side]) this.views[side].receive(data.slice(cut + 1));
			if (this.trackers[side]) this.streams[side].push(data.slice(cut + 1));
		}
	}

	// The learner's decision: the observation of its view and the observed mask, checked against the
	// mask its request describes.
	decision(index = this.learner) {
		const view = this.views[index];
		const legal = view.legalActions();
		const side = this.battle.sides[index];
		const kind = requestKind(side);
		const expected = requestMask(view, side.activeRequest || {}, kind);
		const mask = expected.mask.slice();
		// A known trap removes the switches of a move request only (a forced switch is never refused).
		if (kind === MOVE_REQUEST && view.trapHolds() && mask.slice(4).some(Boolean)) {
			this.knownTrapDecisions++;
			for (let k = 4; k < 9; k++) mask[k] = false;
		}
		if (view.requestKind !== kind || mask.some((allowed, index) => allowed !== legal[index])) {
			this.maskMismatches++;
			if (this.maskSamples.length < 3) {
				this.maskSamples.push({ turn: this.battle.turn, request: kind, viewRequest: view.requestKind,
					expected: mask, legal, trapHolds: view.trapHolds() });
			}
		}
		const observation = view.observation();
		const item = { request: view.requestKind, legal, observation, refused: this.refusedLast,
			first: this.decisions === 0, reset: this.fresh };
		if (this.shared.network || this.trackers[index]) {
			item.side = index;
			item.held = kind !== NO_REQUEST && side.isChoiceDone();
			item.refused = this.shared.network ? this.retries[index] : this.refusedLast;
			item.previous = this.previous[index];
		}
		if (this.trackers[index]) {
			item.own = this.trackers[index].state(observation);
			item.sleepOrigins = Array.from({ length: 6 }, (_, i) =>
				view.sides[1 - index].members[i]?.sleepOrigin ?? -1);
			item.stream = this.streams[index].splice(0);
		}
		if (index === this.learner) this.current.pending = item;
		return item;
	}

	// Plays the learner's action and the opponent's baseline choice as one step. Both sides are asked
	// what they must choose before either chooses: a commit by the first choice can start the next turn
	// and give the learner a request it had no decision for.
	act(action, opponentAction) {
		const before = this.current;
		this.current = { protocol: [], requests: [] };
		const oracle = this.shared.audit ? this.audit() : null;
		const battle = this.battle;
		const view = this.views[this.learner];
		const legal = view.legalActions();
		let chosen = action;
		if (!Number.isInteger(chosen) || chosen < 0 || chosen > 8 || !legal[chosen]) {
			// Never an illegal action: the engine plays the first legal one (pokejax/env.py).
			this.illegal++;
			chosen = legal.indexOf(true);
		}
		const learnerSide = battle.sides[this.learner];
		const foe = battle.sides[this.opponent];
		const learnerRequest = learnerSide.activeRequest;
		const askLearner = requestKind(learnerSide) !== NO_REQUEST && !learnerSide.isChoiceDone();
		// A choice the opponent made at an earlier refused step is still in place (Showdown keeps it).
		const askFoe = requestKind(foe) !== NO_REQUEST && !foe.isChoiceDone();
		let opponentChoiceText = null;
		let learnerChoiceText = null;
		this.retries = [false, false];
		const choose = (index, text, action) => {
			const side = battle.sides[index];
			if (battle.choose('p' + (index + 1), text)) {
				if (action !== undefined) this.previous[index] = action;
				return true;
			}
			if (!TRAPPED_REFUSAL.test(side.choice.error)) return false;
			this.retries[index] = true;
			this.refusals++;
			return true;
		};
		if (askFoe) {
			opponentChoiceText = this.shared.network
				? learnerChoice(this.views[this.opponent], foe.activeRequest, opponentAction)
				: this.shared.choose(this, this.opponent);
			if (!choose(this.opponent, opponentChoiceText, opponentAction)) return this.fail(foe.choice.error);
		}
		if (askLearner) {
			learnerChoiceText = learnerChoice(view, learnerRequest, chosen);
			if (!choose(this.learner, learnerChoiceText, chosen)) return this.fail(learnerSide.choice.error);
		}
		this.steps++;
		this.refusedLast = this.retries[this.learner];
		this.saveStep(before, oracle, chosen, learnerChoiceText, opponentChoiceText, this.refusedLast);
		if (!this.retries.some(Boolean)) battle.sendUpdates();
		if (battle.ended) return this.finish();
		if (this.steps >= this.shared.maxSteps) return this.finish();
		return null;
	}

	saveStep(before, oracle, action, learnerText, opponentText, refused) {
		if (!this.record) return;
		const pending = before.pending || {};
		this.record.steps.push({ observation: pending.observation, legal: pending.legal,
			request: pending.request, refused: pending.refused, action, learnerChoice: learnerText,
			opponentChoice: opponentText, refusedChoice: refused, protocol: before.protocol,
			requests: before.requests, ...(oracle ? { oracle, views: before.views } : {}) });
	}

	audit() {
		const oracle = snapshot(this.battle, this.fixed, null, new Map());
		oracle.sides.forEach((side, i) => {
			side.party = this.battle.sides[i].pokemon.map(member => this.fixed[i].indexOf(member));
			side.members.forEach((member, slot) => {
				const pokemon = this.fixed[i][slot];
				Object.assign(member, {
					stats: pokemon.storedStats, baseStats: pokemon.baseStoredStats,
					moves: pokemon.moveSlots.map(move => move.id),
					maxPp: pokemon.moveSlots.map(move => move.maxpp),
					ivs: pokemon.set.ivs, evs: pokemon.set.evs, level: pokemon.level,
					activeTurns: pokemon.activeTurns,
				});
			});
		});
		return structuredClone(oracle);
	}

	fail(message) {
		this.errors.push(message);
		return this.finish();
	}

	finish() {
		const battle = this.battle;
		this.finished = true;
		const ended = battle.ended;
		const winner = !ended ? null : !battle.winner ? 'tie' : battle.sides.findIndex(side => side.name === battle.winner);
		this.result = {
			index: this.index, learner: this.learner, teams: this.poolIndices, winner, ended,
			truncated: !ended && this.errors.length === 0, steps: this.steps, turns: battle.turn,
			decisions: this.decisions, refusals: this.refusals, illegal: this.illegal,
			maskMismatches: this.maskMismatches, maskSamples: this.maskSamples,
			knownTrapDecisions: this.knownTrapDecisions,
			turnLimit: ended && winner === 'tie' && battle.turn > TURN_LIMIT, errors: this.errors,
		};
		if (this.record) {
			const record = { type: 'battle', index: this.index, learner: this.learner, teams: this.poolIndices,
				seed: this.battleSeed, policySeed: seedFor(this.shared.seed, this.index, 'policy'),
				opponent: this.shared.opponent, result: this.result, steps: this.record.steps, tail: this.current };
			this.result.recordLine = JSON.stringify(record);
		}
		battle.destroy();
		this.battle = null;
		return this.result;
	}
}

class Session {
	constructor(message) {
		const lock = JSON.parse(fs.readFileSync(path.join(root, 'reference-lock.json'), 'utf8'));
		const pool = JSON.parse(fs.readFileSync(path.resolve(root, message.pool), 'utf8'));
		assert.equal(pool.showdownCommit, lock.commit, 'The team pool was made by another Showdown commit');
		assert(message.opponent === 'network' || POLICY_NAMES.includes(message.opponent), 'Unknown opponent ' + message.opponent);
		assert(Number.isInteger(message.concurrency) && message.concurrency > 0);
		const tables = loadTables();
		const baselines = new Baselines({ tables });
		const opponent = message.opponent;
		this.shared = {
			pool: pool.teams, tables, seed: message.seed, maxSteps: message.maxSteps, opponent,
			readsView: VIEW_POLICIES.has(opponent), record: Boolean(message.record),
			network: opponent === 'network', search: Boolean(message.search),
			audit: Boolean(message.audit && message.record),
			choose: (game, side) => {
				const battle = game.battle;
				const choices = legalChoices(battle.sides[side], game.fixed[side]);
				return baselines.choose(opponent, { choices, view: game.views[side], prng: game.policy }).choice.choice;
			},
		};
		this.queue = message.battles.slice();
		this.concurrency = message.concurrency;
		this.slots = Array(this.concurrency).fill(null);
		this.results = [];
		this.records = [];
		this.started = Date.now();
	}

	begin() {
		for (let slot = 0; slot < this.concurrency; slot++) this.fill(slot);
		return this.decisions();
	}

	fill(slot) {
		if (!this.queue.length) {
			this.slots[slot] = null;
			return;
		}
		const game = new Game(this.queue.shift(), this.shared);
		game.slot = slot;
		game.start();
		this.slots[slot] = game;
	}

	answer(items) {
		const bySlot = new Map();
		for (const { slot, side, action } of items) {
			const game = this.slots[slot];
			if (!game || game.finished) throw new Error('No battle in slot ' + slot);
			const pair = bySlot.get(slot) || [];
			pair[side ?? game.learner] = action;
			bySlot.set(slot, pair);
		}
		for (const [slot, actions] of bySlot) {
			const game = this.slots[slot];
			if (!game || game.finished) throw new Error('No battle in slot ' + slot);
			let result;
			try {
				result = game.act(actions[game.learner], actions[game.opponent]);
			} catch (error) {
				result = game.fail(`Battle ${game.index}, step ${game.steps}: ${error.stack || error}`);
			}
			if (result) {
				this.results.push(result);
				if (result.recordLine) this.records.push(result.recordLine);
				delete result.recordLine;
				this.fill(slot);
			}
		}
		return this.decisions();
	}

	// The next message: one item per learner decision of the active battles, or done.
	decisions() {
		const items = [];
		for (const game of this.slots) {
			if (!game) continue;
			const decision = game.decision();
			items.push({ slot: game.slot, battle: game.index, ...decision, learner: true });
			if (this.shared.network || this.shared.audit) {
				const opponent = game.decision(game.opponent);
				if (this.shared.network) {
					items.push({ slot: game.slot, battle: game.index, ...opponent, learner: false });
				}
				if (this.shared.audit) game.current.views = [decision, opponent].sort((a, b) => a.side - b.side);
			}
			game.fresh = false;
			game.decisions++;
		}
		const finished = this.results.splice(0);
		const records = this.records.splice(0);
		if (!items.length) {
			const seconds = (Date.now() - this.started) / 1000;
			return { type: 'done', finished, records, stats: { seconds } };
		}
		return { type: 'decisions', items, finished, records };
	}
}

function send(message) {
	process.stdout.write(JSON.stringify(message) + '\n');
}

function main() {
	let session = null;
	const input = readline.createInterface({ input: process.stdin });
	input.on('line', line => {
		try {
			const message = JSON.parse(line);
			if (message.type === 'start') {
				session = new Session(message);
				send(session.begin());
			} else if (message.type === 'actions') {
				if (!session) throw new Error('actions before start');
				const next = session.answer(message.items);
				send(next);
				if (next.type === 'done') input.close();
			} else {
				throw new Error('Unknown message ' + message.type);
			}
		} catch (error) {
			send({ type: 'fatal', error: error.stack || String(error) });
			process.exitCode = 1;
			input.close();
		}
	});
}

if (require.main === module) main();

module.exports = { Game, Session, requestKind, learnerChoice, requestMask, seedFor, TURN_LIMIT };
