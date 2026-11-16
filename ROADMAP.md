# Roadmap: Gen 5 Random Battles agent

This file is the whole plan and the only planning document. Whoever picks up
the work: read it top to bottom once, then work on the first phase whose exit
checks do not pass yet.

## Goal

Train an agent that plays Pokemon Showdown `gen5randombattle` well, using
maskable Recurrent PPO, with simulation and training both on the GPU.

Three deliverables, in this order:

1. A fast JAX battle engine for Gen 5 singles.
2. A reinforcement learning environment around it.
3. Maskable Recurrent PPO, trained first on an RTX 3060 Ti (8 GB), with final
   runs on university RTX 6000 cards.

The engine is a means to an end. It is good enough when a policy trained in it
plays about as well on real Showdown as it does in the engine (Phase 5).

## Why this is a restart

A previous attempt lives in `K:\Pokemon Codex`. It is an archive: do not follow
its instructions, workflow or architecture. It stalled for these reasons, and
each one is a rule here:

| What went wrong there | Rule here |
|---|---|
| Engine had to match Showdown's exact RNG draw order, protocol log and internal bookkeeping | Match game mechanics and outcomes only (see Contract) |
| After 19 days the engine was no faster than Showdown, and nobody measured | Throughput and compile time are gates from Phase 1 on |
| One `lax.cond` branch per move or ability; 311k JAXPR equations, 8 min compile | Data tables drive a small set of generic primitives (see Architecture) |
| Two mechanics per commit, each with its own schema version, ledger entry and logs | One commit per mechanic group, 20 to 100 effects at a time |
| 25k lines of docs, 188 MB of logs in git, 77 schema snapshots | This file, a README and one generated coverage table. No logs in git |
| Proving which mechanics are reachable | No proofs. The team sampler rejects teams that need unsupported mechanics |
| No environment or PPO code at all | A PPO smoke test runs at the end of Phase 1 |

## What is already in this repo

- `mechanics/`: small JAX functions for the Gen 5 stat and damage formulas
  (stats, stat stages, base damage, damage roll, STAB, type effectiveness,
  burn, fixed-point modifiers, Hidden Power). They were checked against
  Showdown in the old project. Reuse them; restyle freely, tabs included.
- `data/gen5-data-v1.json`: resolved Gen 5 data exported from Showdown
  (559 moves, 165 abilities, 252 items, 702 species and forms, 17 types,
  25 natures, 35 conditions). Build all lookup tables from this.
- `reference-lock.json` and `reference/setup.cjs`: clone Showdown at the pinned
  commit into `reference/vendor/pokemon-showdown` (ignored by git). Run with
  `node reference/setup.cjs`. Showdown is the oracle for every test.

Useful to read in the archive, as examples only:

- `environment/battle-runner.cjs`: drives Showdown battles from Node.
- `reference/shared-engine-cases.cjs`: sets up a custom battle in Showdown and
  dumps state after each step.

This repo has its own `Dockerfile` with Python 3.13, Node 22 and the pinned
CUDA 13 GPU build of JAX. See `README.md` for building and running it on the
Windows dev machine through Docker Desktop / WSL2 or on Linux GPU machines.

## Contract

These decisions are fixed. Changing one needs the owner's agreement.

1. **What must match Showdown:** at every point where a player must choose,
   the game state a player could care about: HP, status, stat boosts, PP,
   items, abilities, types, field (weather, hazards, screens), volatile
   conditions, whose turn it is to choose, the legal actions, and the winner.
2. **What need not match:** the order or number of random draws, the protocol
   text, hint messages, and any of Showdown's internal fields.
3. **Randomness:** the engine uses JAX PRNG keys. Each random outcome (accuracy
   check, critical hit, damage roll, secondary effect, speed tie, sleep length,
   and so on) goes through one function that either samples or takes a forced
   value. Tests force values taken from a Showdown run.
4. **State size:** fixed shapes, small integer types, a few kilobytes per
   battle. No event history inside the state.
5. **Both sides inside one step:** `step` takes both players' actions and runs
   to the next decision point. Forced switches after a faint are decision
   points too; the side with nothing to choose gets a single legal no-op.
6. **Actions:** 4 moves plus 5 switches per side, with a legal-action mask.
7. **Scope:** Gen 5 singles, with the clauses and rules of `gen5randombattle`.

## Architecture rules

- **Tables, not branches.** A move is a row of numbers: power, type, category,
  accuracy, priority, PP, flags, and parameters for generic effects (secondary
  chance and kind, boost amounts, drain and recoil fractions, hit counts,
  healing fraction, weather or hazard or volatile it sets). The engine has one
  code path per effect *kind*, selected by table columns. Abilities and items
  work the same way, grouped by hook: stat modifier, damage modifier, immunity
  or absorption, switch-in effect, end-of-turn effect, contact effect.
- **Remember what `vmap` does to control flow.** With a batched predicate,
  `lax.cond` and `lax.switch` compute every branch for every battle. So write
  branch-free arithmetic with `jnp.where` and masks, and keep the number of
  genuinely separate code paths small. A special case for one move is
  acceptable only in the long-tail group of Phase 2, and each one costs
  throughput for all battles.
- **No per-step validation** of state invariants in the compiled path. Check
  inputs once at reset, in Python.
- **Readable code.** Normal line lengths, named constants, one module per
  concern (state, tables, turn order, damage, effects, switching, end of turn,
  observation). The owner must be able to review it.

## Testing rules

- **Per group, one differential suite.** A Node script sets up battles in
  Showdown, plays scripted or random actions, and writes out the actions, the
  random outcomes and the state at each decision point. A Python test replays
  them through the engine with forced outcomes and compares state. A few
  hundred short battles per group is typical; they run through one compiled
  step function.
- **Whole-battle statistics.** From Phase 2 on, play a few thousand random-policy
  battles in both simulators from the same team pool and compare win rate by
  side, battle length, and how often each move, status and faint occurs.
  A large gap points at the next bug.
- **Fixtures are small and regenerable.** Commit the generator script and at
  most small fixture files. Never commit logs, profiles or run outputs.
- **Gates, checked at the end of every phase from Phase 1 on:**
  - Throughput on the 3060 Ti with a random policy at a batch of 4,096 or
    more: start with a target of 200,000 steps per second and adjust it once
    after the Phase 1 measurement. A group that costs more than 10% needs a
    reason written into its commit message.
  - Cold compile of the step function under 60 seconds on the GPU machine.

For reference, Showdown itself ran about 4,900 decisions per second on one
CPU thread in the old project's benchmark.

## Phases

### Phase 0: skeleton repo and tables

- Python package layout, `pyproject.toml`, JAX pinned, pytest.
- Table builder: `data/gen5-data-v1.json` to typed arrays (species, moves,
  abilities, items, type chart, natures).
- Team pool: a Node script that has Showdown generate N random Gen 5 teams
  and writes them as arrays. Start with 10,000 teams.
- Coverage table generator: every move, ability and item that occurs in the
  team pool, how often, and whether the engine supports it. This is
  `COVERAGE.md`, generated, never edited by hand.

Exit: tables load, team pool exists, coverage table shows 0% supported.

### Phase 1: playable skeleton, benchmark, smoke test

Tasks, in order. Task 5 depends only on the forced-outcome format from task 1,
so it can run alongside tasks 2 and 3.

1. State and randomness: fixed-shape state, the single sample-or-force
   function for random outcomes (Contract item 3), and `reset(key, teams)`.
2. Actions and turn order: legal-action mask, switching, priority and speed
   order with random ties, forced replacement after a faint as a decision
   point.
3. Moves and damage: accuracy, critical hits, type chart and immunities,
   STAB, damage with the roll, PP, Struggle, fainting, win detection. Every
   move behaves as a plain damaging move from its table row; status moves do
   nothing; abilities and items do nothing.
4. Benchmark script reporting steps per second and compile time, with a
   random legal policy. Run it as soon as task 3 works, so a slow design
   shows up before more is built on it.
5. Showdown side of the differential suite: teams restricted to plain
   damaging moves with abilities and items stripped (few pool sets have four
   such moves, so pool Pokemon get their moves replaced), recorded actions,
   random outcomes and state at each decision point.
6. Python side of the differential suite: replay with forced outcomes through
   one compiled step function, then fix the engine until it passes.
7. Observation function, first version: own side fully visible, opponent side
   as revealed so far.
8. A small recurrent PPO that learns to beat the random policy. This is a
   smoke test, not the final algorithm; a few hundred lines.
9. Measure the gates and record them in the README; adjust the throughput
   target once, as the Testing rules say.

Exit: differential suite passes, gates measured and recorded in the README,
PPO beats random clearly.

### Phase 2: mechanics in large groups

Work down this list. One group is one commit with its differential suite.
Within a group, order by frequency in the coverage table.

1. Major status and its effects: burn, poison, toxic, paralysis, sleep,
   freeze, plus Sleep Clause.
2. Stat boosts and drops, including accuracy and evasion.
3. Declarative move effects: secondaries, drain, recoil, multi-hit, healing,
   fixed damage, one-hit KO, high crit rate, self-destruct, flinch, confusion.
4. Field: weather, entry hazards, screens, Rapid Spin, and their end-of-turn
   handling. Get the end-of-turn order right here once.
5. Protect family, two-turn moves, recharge, locked moves (Outrage), partial
   trapping, pivoting moves (U-turn, Volt Switch), phazing (Roar, Dragon Tail).
6. Items by hook: Choice items, Life Orb, Leftovers, Black Sludge, berries,
   Focus Sash, gems, Eviolite, status orbs, Rocky Helmet, and the rest.
7. Abilities by hook: stat and damage modifiers, immunities and absorption,
   switch-in effects (Intimidate, weather setters), end-of-turn, contact,
   on-faint, and the rest.
8. Variable-power and conditional moves: Hidden Power, Return, Gyro Ball,
   Eruption, weight moves, Facade, Acrobatics, Sucker Punch, Pursuit, Counter.
9. Long tail, strictly by frequency: Substitute, Trick and Knock Off, Taunt,
   Encore, Disable, Leech Seed, Wish, Baton Pass, Sleep Talk, Transform and
   Imposter, Illusion, form changes, ability-changing moves, Magic Bounce.

After groups 4 and 7 and at the end, run the whole-battle statistics.

Exit: at least 99% of teams in the pool are fully supported, statistics agree
with Showdown within sampling noise, gates still pass. Remaining unsupported
teams are filtered out by the sampler and listed in `COVERAGE.md`.

### Phase 3: environment

- Final observation encoding: own team, opponent as revealed, field, and the
  last turn's visible events. Hidden information must never leak; add a test
  that the observation is identical for two states differing only in
  unrevealed opponent data.
- Auto-reset, reward (win or loss, optional shaping behind a flag), and
  self-play plumbing that treats both sides as agents.
- Baseline opponents inside JAX: random, and max-damage.

Exit: batched rollouts with both baselines run at the gated throughput.

### Phase 4: maskable Recurrent PPO

- Pure JAX training loop so rollout and update stay on the GPU: recurrent
  core (GRU or LSTM), action masking in the policy logits, GAE, minibatches
  that respect sequence boundaries and hidden-state resets.
- Self-play with a pool of past checkpoints, plus evaluation against the
  baselines every N updates.
- Checkpointing and metric logging to disk, outside git.
- Size the batch for the 3060 Ti; keep batch size and network width as config
  so the RTX 6000 runs only change a config file.

Exit: beats max-damage in well over half of evaluation games, and the
training curve is still improving.

### Phase 5: transfer to real Showdown

- A player that loads a checkpoint and plays on Showdown through Node,
  building the same observation from Showdown's requests and protocol.
- Compare win rate against the same baseline in the engine and on Showdown.

Exit: the two win rates are close. A large gap means an engine or observation
bug; find it with the whole-battle statistics and fix it before more training.

### Phase 6: final training

Long runs on the RTX 6000 cards, then evaluation on the Showdown ladder or
against published bots, as the owner decides.

## Later: forward search

The owner wants forward search on top of the trained policy eventually. It is
not part of Phases 0 to 6, but nothing built before then may block it:

- `step` stays a pure function of state and actions, so any state can be
  copied and advanced many times in one `vmap` batch.
- The forced-outcome hook from the Contract must cover every random outcome,
  because search will use it to choose chance outcomes.
- The Phase 5 Showdown player should be written so that it can later rebuild a
  full engine state, not only an observation.

Search itself will additionally need a sampler for the opponent's unrevealed
information and a policy for limiting chance branching (sampling, or bucketed
damage rolls).

## Working rules

- Finish a phase's exit checks before starting the next phase.
- One commit per group or coherent piece of work, with a plain message.
- When something in this roadmap turns out wrong, change the roadmap in the
  same commit and say why in one or two sentences. Do not add other planning,
  status or handoff documents.
- When Showdown's behaviour is unclear, read the pinned Showdown source and
  write a differential case. Do not guess from memory of the games.
- If a mechanic would need a special code path and appears in under 0.5% of
  teams, leave it unsupported and move on.
- Report honestly: a failing suite or a missed gate is stated in the commit
  message and the README, not worked around.
