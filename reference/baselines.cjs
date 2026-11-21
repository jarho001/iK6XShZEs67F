'use strict';

/* The four baseline policies of pokejax/baselines.py, in Node, for reference/whole-battle-stats.cjs
 * and reference/record-battles.cjs: random, random-move, max-damage and heuristics (poke-env's
 * SimpleHeuristicsPlayer.choose_singles_move at commit bab3673, Gen 5 singles). pokejax/baselines.py
 * documents every rule and every interpretation of poke-env; this file makes the same ones.
 *
 *   const { Baselines } = require('./baselines.cjs');
 *   const baselines = new Baselines();
 *   const { choice, random } = baselines.choose('heuristics', { choices, view, prng });
 *
 * choices is the engine's true legal list [{ action, choice }] (record-battles.cjs legalChoices:
 * hidden traps removed), view the PlayerView of the player (reference/player-view.cjs) and prng the
 * PRNG for the random policies. The deterministic policies read only what that player has: the
 * observation its view builds (own team in full, the opponent as revealed) and its own request
 * (move names, the order of its bench). Nothing else of the battle is read. random reports whether
 * the choice was random (true for random and random-move, false for the others: no random fallback
 * of poke-env ever applies, see baselines.py).
 *
 * Move and species data are Showdown's Dex at the pinned commit (the Python side reads the same
 * export through its tables), so the check against the engine also checks those tables.
 */

const { Dex } = require('./vendor/pokemon-showdown/dist/sim/dex');
const { loadTables } = require('./player-view.cjs');

const POLICY_NAMES = ['random', 'random-move', 'max-damage', 'heuristics'];
const SPEED_TIER_COEFFICIENT = 0.1;
const HP_FRACTION_COEFFICIENT = 0.4;
const SWITCH_OUT_MATCHUP_THRESHOLD = -2;
const SETUP_BRANCH = false; // poke-env compares an enum with 'self': never true (baselines.py)
const TIE_TOLERANCE = 1e-5;
const TRIPLE_KICK_HITS = 1 + 2 * 0.9 + 3 * 0.81;
const MULTI_HIT_EXPECTED = (2 + 3) / 3 + (4 + 5) / 6;
const HAZARD_MOVES = { spikes: 0, toxicspikes: 1 }; // index into the side's [Spikes, Toxic Spikes]
const ANTI_HAZARD_MOVES = new Set(['rapidspin', 'defog']);
const BOOSTS = ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion'];
const [ATK, DEF, SPA, SPD, SPE] = [0, 1, 2, 3, 4];
const DAMAGE_TAKEN = { 0: 1, 1: 2, 2: 0.5, 3: 0 }; // Showdown's damageTaken codes

function firstMax(scores) {
	// The first index within the tolerance of the best score; ties go to the earlier candidate.
	const best = Math.max(...scores);
	const floor = best - TIE_TOLERANCE * Math.max(1, Math.abs(best));
	return scores.findIndex(score => score >= floor);
}

function boostFactor(stage) {
	// poke-env _stat_estimation: +1 counts as 2.0 there.
	return stage > 1 ? (2 + stage) / 2 : 2 / (2 - stage);
}

function statEstimation(base, stage) {
	return ((2 * base + 31) + 5) * boostFactor(stage);
}

function expectedHits(move) {
	if (move.id === 'triplekick' || move.id === 'tripleaxel') return TRIPLE_KICK_HITS;
	const hits = move.multihit ?? 1;
	if (!Array.isArray(hits)) return hits;
	if (hits[0] !== 2 || hits[1] !== 5) throw new Error('Unexpected multihit ' + hits + ' for ' + move.id);
	return MULTI_HIT_EXPECTED;
}

class Baselines {
	constructor(options = {}) {
		this.dex = options.dex || Dex.forFormat('gen5randombattle');
		this.tables = options.tables || loadTables();
		this.setupBranch = options.setupBranch ?? SETUP_BRANCH;
		// Type names in the engine's type order (the observation's type IDs are these plus one).
		this.typeNames = [];
		for (const [id, index] of this.tables.types) this.typeNames[index] = this.dex.types.get(id).name;
		this.moves = new Map();
		this.species = new Map();
	}

	// Data.

	speciesBase(index) {
		if (!this.species.has(index)) {
			const row = this.tables.species.rows[index - 1];
			this.species.set(index, this.dex.species.get(row.id).baseStats);
		}
		return this.species.get(index);
	}

	moveInfo(id) {
		if (!this.moves.has(id)) {
			const move = this.dex.moves.get(id);
			// Recharge is a request entry without a move: nothing to score (it is never a choice).
			this.moves.set(id, !move.exists ? { id, type: null, power: 0, accuracy: 1, hits: 1, physical: false,
				hazard: -1, antiHazard: false, boosts: {}, targetsSelf: false } : {
				id: move.id,
				type: move.type,
				power: move.basePower,
				accuracy: move.accuracy === true ? 1 : move.accuracy / 100,
				hits: expectedHits(move),
				physical: move.category === 'Physical',
				hazard: move.id in HAZARD_MOVES ? HAZARD_MOVES[move.id] : -1,
				antiHazard: ANTI_HAZARD_MOVES.has(move.id),
				boosts: move.boosts || {},
				targetsSelf: move.target === 'self',
			});
		}
		return this.moves.get(id);
	}

	multiplier(attack, defending) {
		let product = 1;
		for (const name of defending) product *= DAMAGE_TAKEN[this.dex.types.get(name).damageTaken[attack]];
		return product;
	}

	// What the player knows, from its observation and its request.

	// The opponent's active member: its row of the observation, or, when an Illusion holder has made
	// the roster longer than six entries and the active one is not among the first six, the view's own
	// record of it (the same facts: species, current types, protocol HP percentage).
	foeActive(view, opponent) {
		const row = opponent.is_active.indexOf(1);
		if (row >= 0) {
			return { species: opponent.species[row], types: opponent.types[row].filter(t => t > 0).map(t => t - 1),
				fraction: opponent.hp[row] };
		}
		const member = view.sides[1 - view.me].active;
		const percent = member.hp <= 0 ? 0 : Math.min(Math.floor((100 * member.hp + member.maxhp - 1) / member.maxhp),
			member.hp < member.maxhp ? 99 : 100);
		return { species: view.speciesIndex(view.currentSpecies(member)), types: view.currentTypes(member),
			fraction: percent / 100 };
	}

	facts(view) {
		const { own, opponent, hazards } = view.observation();
		const active = own.is_active.indexOf(1);
		const foe = this.foeActive(view, opponent);
		const names = pair => pair.filter(type => type > 0).map(type => this.typeNames[type - 1]);
		const stages = list => list.map(value => Math.round(value * 6));
		return {
			active,
			ownTypes: own.types.map(names),
			ownSpecies: own.species,
			ownFraction: own.hp,
			ownHpFull: own.hp[active] === 1,
			ownAttack: Math.round(own.stats[active][1] * 1000),
			ownSpecialAttack: Math.round(own.stats[active][3] * 1000),
			ownBoosts: stages(own.boosts),
			ownRemaining: own.fainted.filter(flag => flag === 0).length,
			ownHazards: hazards[0].some(layers => layers > 0),
			foeSpecies: foe.species,
			foeTypes: foe.types.map(type => this.typeNames[type]),
			foeFraction: foe.fraction,
			foeBoosts: stages(opponent.boosts),
			foeRemaining: 6 - opponent.fainted.filter(flag => flag === 1).length,
			foeHazards: hazards[1],
		};
	}

	// The own team slots in the order of the request's side.pokemon (Showdown's party order, the
	// active member first), and the move name of each active slot.
	requestSlots(view) {
		const members = view.sides[view.me].members;
		return view.request.side.pokemon.map(entry => {
			const name = entry.ident.slice(entry.ident.indexOf(': ') + 2);
			return members.findIndex(member => member.name === name);
		});
	}

	moveOf(view, choice) {
		const slot = Number(/^move (\d+)$/.exec(choice.choice)[1]) - 1;
		const entry = view.request.active[0].moves[slot];
		// Hidden Power is the typed move the request names ("Hidden Power Fighting 70").
		const typed = /^Hidden Power (\w+)/.exec(entry.move);
		return this.moveInfo(typed ? 'hiddenpower' + typed[1].toLowerCase() : entry.id);
	}

	// Scoring.

	matchups(facts) {
		// poke-env _estimate_matchup of every own team slot against the opponent's active.
		const foeSpeed = this.speciesBase(facts.foeSpecies).spe;
		return facts.ownTypes.map((types, slot) => {
			let score = Math.max(...types.map(type => this.multiplier(type, facts.foeTypes))) -
				Math.max(...facts.foeTypes.map(type => this.multiplier(type, types)));
			const speed = this.speciesBase(facts.ownSpecies[slot]).spe;
			if (speed > foeSpeed) score += SPEED_TIER_COEFFICIENT;
			else if (foeSpeed > speed) score -= SPEED_TIER_COEFFICIENT;
			score += facts.ownFraction[slot] * HP_FRACTION_COEFFICIENT;
			return score - facts.foeFraction * HP_FRACTION_COEFFICIENT;
		});
	}

	moveScore(facts, info, ratio) {
		const stab = facts.ownTypes[facts.active].includes(info.type) ? 1.5 : 1;
		return info.power * stab * ratio * info.accuracy * info.hits * this.multiplier(info.type, facts.foeTypes);
	}

	// The legal bench members in request order: [{ action, slot }].
	bench(view, facts, choices) {
		const legal = new Set(choices.map(choice => choice.action));
		const slots = this.requestSlots(view).filter(slot => slot !== facts.active);
		return slots.map(slot => ({ action: 4 + slot - (slot > facts.active ? 1 : 0), slot }))
			.filter(candidate => legal.has(candidate.action));
	}

	bestSwitch(matchups, bench) {
		return bench[firstMax(bench.map(candidate => matchups[candidate.slot]))];
	}

	maxDamage(choices, view) {
		const facts = this.facts(view);
		const moves = choices.filter(choice => choice.action < 4);
		if (moves.length === 1) return moves[0];
		if (moves.length) {
			const foe = this.speciesBase(facts.foeSpecies);
			const scores = moves.map(choice => {
				const info = this.moveOf(view, choice);
				const stat = info.physical ? ATK : SPA;
				const attack = (info.physical ? facts.ownAttack : facts.ownSpecialAttack) *
					boostFactor(facts.ownBoosts[stat]);
				const defence = info.physical ? statEstimation(foe.def, facts.foeBoosts[DEF]) :
					statEstimation(foe.spd, facts.foeBoosts[SPD]);
				return this.moveScore(facts, info, attack / defence);
			});
			return moves[firstMax(scores)];
		}
		const bench = this.bench(view, facts, choices);
		return bench.length ? choices.find(choice => choice.action === this.bestSwitch(this.matchups(facts), bench).action) :
			choices[0];
	}

	heuristics(choices, view) {
		const facts = this.facts(view);
		const matchups = this.matchups(facts);
		const moves = choices.filter(choice => choice.action < 4);
		const bench = this.bench(view, facts, choices);
		// _should_switch_out
		const boosts = facts.ownBoosts;
		const reason = boosts[DEF] <= -3 || boosts[SPD] <= -3 ||
			(boosts[ATK] <= -3 && facts.ownAttack >= facts.ownSpecialAttack) ||
			(boosts[SPA] <= -3 && facts.ownAttack <= facts.ownSpecialAttack) ||
			matchups[facts.active] < SWITCH_OUT_MATCHUP_THRESHOLD - TIE_TOLERANCE;
		const decent = bench.some(candidate => matchups[candidate.slot] > TIE_TOLERANCE);
		if (moves.length && !(decent && reason && bench.length)) {
			// A single move (forced, Struggle, Choice-locked) is the choice whatever it scores.
			if (moves.length === 1) return moves[0];
			const infos = moves.map(choice => this.moveOf(view, choice));
			for (const [i, info] of infos.entries()) {
				const sets = facts.foeRemaining >= 3 && info.hazard >= 0 && !(facts.foeHazards[info.hazard] > 0);
				const removes = facts.ownHazards && info.antiHazard && facts.ownRemaining >= 2;
				if (sets || removes) return moves[i];
			}
			if (this.setupBranch && facts.ownHpFull && matchups[facts.active] > TIE_TOLERANCE) {
				for (const [i, info] of infos.entries()) {
					const sum = Object.values(info.boosts).reduce((a, b) => a + b, 0);
					const raised = BOOSTS.flatMap((stat, s) => (info.boosts[stat] > 0 ? [boosts[s]] : []));
					if (info.targetsSelf && sum >= 2 && Math.min(...raised) < 6) return moves[i];
				}
			}
			const base = this.speciesBase(facts.ownSpecies[facts.active]);
			const foe = this.speciesBase(facts.foeSpecies);
			const physical = statEstimation(base.atk, boosts[ATK]) / statEstimation(foe.def, facts.foeBoosts[DEF]);
			const special = statEstimation(base.spa, boosts[SPA]) / statEstimation(foe.spd, facts.foeBoosts[SPD]);
			const scores = infos.map(info => this.moveScore(facts, info, info.physical ? physical : special));
			return moves[firstMax(scores)];
		}
		if (bench.length) {
			const best = this.bestSwitch(matchups, bench);
			return choices.find(choice => choice.action === best.action);
		}
		return choices[0];
	}

	// choices: the engine's legal list; view: the player's PlayerView (not needed for random).
	choose(name, { choices, view, prng }) {
		if (name === 'random') return { choice: prng.sample(choices), random: true };
		if (name === 'random-move') {
			const moves = choices.filter(choice => choice.action < 4);
			return { choice: prng.sample(moves.length ? moves : choices), random: true };
		}
		// A side that is only waiting has the single no-op choice, whatever the policy.
		if (choices.length === 1 && choices[0].choice === null) return { choice: choices[0], random: false };
		if (name === 'max-damage') return { choice: this.maxDamage(choices, view), random: false };
		if (name === 'heuristics') return { choice: this.heuristics(choices, view), random: false };
		throw new Error('Unknown policy ' + name + '; choose one of ' + POLICY_NAMES.join(', '));
	}
}

module.exports = { Baselines, POLICY_NAMES, SETUP_BRANCH, TIE_TOLERANCE };
