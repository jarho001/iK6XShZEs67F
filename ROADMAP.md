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

- **Per task, one differential suite.** A Node script sets up battles in
  Showdown, plays scripted or random actions, and writes out the actions, the
  random outcomes and the state at each decision point. A Python test replays
  them through the engine with forced outcomes and compares state. A few
  hundred short battles per task is typical; they run through one compiled
  step function.
- **Whole-battle statistics.** From Phase 2 on, play a few thousand random-policy
  battles in both simulators from the same team pool and compare win rate by
  side, battle length, and how often each move, status and faint occurs.
  A large gap points at the next bug.
- **Fixtures are small and regenerable.** Commit the generator script and at
  most small fixture files. Never commit logs, profiles or run outputs.
- **Gates, checked at the end of every phase from Phase 1 on:**
  - Throughput on the 3060 Ti with a random legal policy at a batch of 4,096,
    engine only: target 3,000,000 steps per second. Phase 1 measured a median
    of 25,134,430 steps per second with status moves, abilities and items still
    inactive. The target was 5,000,000 until the end of Phase 2 (changed by the
    owner on 2026-11-20): the full Phase 2 mechanics cost about 7x, not the 5x
    it allowed for, and three performance passes brought the step back only to
    3,419,031 (performance pass 7, 60245e4). 3,000,000 is still about 60x what
    the PPO smoke test consumes (about 52,000 steps per second). 5,000,000
    remains a goal for later work such as custom GPU kernels; larger batches on
    the RTX 6000 machines are a training choice, not a way to pass this gate. A
    group that costs more than 10% needs a reason written into its commit
    message.
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

Work down this list. One task is one commit with its differential suite, the
support registry and `COVERAGE.md` updated, and the benchmark change stated.
Within a task, order by frequency in the coverage table. A group introduces
the engine structure it is the first to need (prevention before moving,
volatiles, resumable turns) instead of a separate task for it.

Isolation teams for a task are pool teams with every move, ability and item
outside the allowed set (supported mechanics plus the task's own) replaced or
removed. Unchanged pool teams only become fully supported late (Leftovers is
on 95% of teams), so the whole-battle statistics also use isolation teams,
the same ones in both simulators. Stealth Rock, screens, Baton Pass and
one-hit KO moves do not occur in the pool and are not implemented.

0. Harness for Phase 2: the snapshot grows with the engine to cover every
   Contract field (status, volatiles, field, item, ability, current types),
   the legal actions are compared with Showdown's request (Phase 1 did not),
   the isolation team builder, and outcome kinds added through one table on
   each side. Check: the Phase 1 suite passes under the new harness.
1. Major status: burn, poison, toxic, paralysis, sleep, freeze, Sleep Clause,
   the status moves and cures (Toxic, Thunder Wave, Will-O-Wisp, sleep moves,
   Rest, Heal Bell, Aromatherapy, Psycho Shift, Refresh), the prevention
   stage before a move, and residual damage with Showdown's residual order
   as a table that later groups add rows to.
2. Stat boosts and drops: boosting moves, self-drops after attacks (Close
   Combat, Draco Meteor and the like), accuracy and evasion, Haze, Belly
   Drum, Curse for non-Ghosts.
3. Declarative move effects, part 1: secondaries (status, boost, flinch,
   confusion chance), volatile storage, confusion and flinch.
4. Declarative move effects, part 2: drain, recoil, crash damage, healing
   moves (Recover, Roost with its Flying removal, the weather-dependent ones
   with weather left for group 6), fixed damage (Seismic Toss, Night Shade,
   Super Fang), Pain Split, multi-hit, high crit rate, Explosion.
5. Whole-battle statistics tool: a few thousand random-policy battles in
   both simulators from the same isolation teams; compare win rate by side,
   battle length, and move, status and faint frequencies.
6. Field: weather (moves, duration, residual damage, damage and accuracy
   changes, Sandstorm Special Defense), Spikes, Toxic Spikes, Rapid Spin,
   and the complete end-of-turn order. Then run the statistics.
7. Protect family, Fake Out, two-turn moves, recharge, locked moves (Outrage,
   Petal Dance), partial trapping.
8. Resumable turns: the rest of the turn stored in the state so step can stop
   mid-turn; U-turn, Volt Switch, phazing (Roar, Whirlwind, Dragon Tail,
   Circle Throw), and faint replacement around them.
9. Items, all 44 in the pool by hook: Choice items and choice lock, Life Orb,
   Leftovers, Black Sludge, berries, Focus Sash, gems, Eviolite, status
   orbs, Rocky Helmet, White Herb, plates and the rest.
10. Abilities, part 1: stat and damage modifiers, immunities and absorption,
    switch-in effects (Intimidate, weather setters).
11. Abilities, part 2: end-of-turn, contact, on-faint, switch-out
    (Regenerator, Natural Cure), Pressure, Prankster, Sturdy, Magic Guard
    and the rest. Then run the statistics.
12. Variable-power and conditional moves: Hidden Power, Return, Facade,
    Acrobatics, weight moves, Gyro Ball, Eruption, Stored Power, Foul Play,
    Psyshock, Sucker Punch, Pursuit, Counter, Mirror Coat, Retaliate,
    Focus Punch and the rest.
13. Long tail, part 1, the common ones: Substitute, Encore, Wish, Sleep Talk,
    Taunt, Leech Seed, Trick and Switcheroo, Knock Off, Yawn, Destiny Bond,
    Healing Wish, Disable, Trick Room.
14. Long tail, part 2, strictly by frequency: Transform and Imposter,
    Illusion, Nature Power, form changes, Trace, Magic Bounce and what
    remains above the 0.5% rule.
15. Exit: the sampler, final statistics on unchanged pool teams, gates.

Exit: at least 99% of the teams in the pool are fully supported, measured as
the exact set of excluded teams, statistics agree with Showdown within
sampling noise, gates still pass. Remaining unsupported teams are filtered
out by the sampler and listed in `COVERAGE.md`.

### Phase 3: environment

- Final observation encoding: own team, opponent as revealed, field, and the
  last turn's visible events. Hidden information must never leak; add a test
  that the observation is identical for two states differing only in
  unrevealed opponent data.
- Auto-reset, reward (win or loss, optional shaping behind a flag), and
  self-play plumbing that treats both sides as agents.
- Baseline opponents inside JAX: random, random-move, max-damage, and
  poke-env's simple heuristics player.

Tasks, in order. Each is one commit with its tests, and states the benchmark
change.

1. Observation, final version, engine side: everything a player can know at
   a decision point. Own team in full; the opponent as revealed; both actives'
   public volatiles (Substitute, Encore, Taunt, Disable and its move, Leech
   Seed, Yawn, Curse, Transform, and the ones already observed; choice lock
   for the own side only); field and side state (weather, hazards, Trick
   Room, pending Wish and Healing Wish). Last turn's visible events, as a
   small fixed record that step writes per side: action order, move used,
   switch or drag-in, miss, failure, critical hit, effectiveness, damage and
   healing in protocol percent, status inflicted, faint. The no-leak test
   covers every new field and every private counter. State the cost with
   `bench --with-observation`.
2. Observation checked against Showdown: a Node module that builds the same
   observation for one player from that player's requests and protocol
   lines only, which the Phase 5 player will reuse, and a differential test
   that compares it field by field with the engine observation at every
   decision point of recorded pool battles, for both sides. A difference is
   a leak, a missing public fact or an engine bug.
3. Environment: reset and step over a batch with teams from the sampler,
   auto-reset inside step, observations, rewards and done flags for both
   sides (win 1, loss -1, tie 0; optional shaping behind a flag, off by
   default), truncation at a step cap kept apart from termination, and
   episode statistics. Self-play treats both sides as agents; a wrapper plays
   one side with a fixed opponent policy and a random side per battle. The
   PPO smoke test moves onto it and must still beat random.
4. Baselines and phase exit: four policies as JAX functions of the state,
   the same four in the Node statistics tool, and whole-battle statistics of
   each against random in both simulators (the Phase 5 comparison needs a
   baseline to behave the same in both). Random picks any legal action.
   Random-move picks a random legal move and switches only when it must.
   Max-damage scores each legal move by a fixed formula over public data
   (power, STAB, type effectiveness, accuracy, attack and defense stats with
   boosts) and switches only when it must. Heuristics ports poke-env's
   `SimpleHeuristicsPlayer` (src/poke_env/player/baselines.py at commit
   bab3673, singles path, without Dynamax and Terastallization) and reads
   only public information. Then the exit benchmark and a Phase 3 results
   section in the README.

Exit: batched rollouts with each baseline run at the gated throughput,
measured as environment steps per second at a batch of 4,096: engine, team
draws, auto-reset, rewards and the baseline policies on both sides, without
observations (baselines read the state). The cost of observing both sides is
measured and reported next to it. Before Phase 3, observing both sides
already brought the engine from 3,419,031 to 2,434,930 steps per second.

### Phase 4: maskable Recurrent PPO

- Pure JAX training loop so rollout and update stay on the GPU: recurrent
  core (GRU or LSTM), action masking in the policy logits, GAE, minibatches
  that respect sequence boundaries and hidden-state resets.
- Self-play with a pool of past checkpoints, plus evaluation against the
  baselines every N updates.
- Checkpointing and metric logging to disk, outside git.
- Size the batch for the 3060 Ti; keep batch size and network width as config
  so the RTX 6000 runs only change a config file.

Tasks, in order. Each is one commit with its tests; training runs and
their outputs stay outside git.

1. Training core: a pure JAX training loop on the Phase 3 environment and
   its opponent wrapper. The network embeds species, moves, items,
   abilities and types, encodes each team member with shared weights and
   pools over each side's team, adds field, request and last-step events,
   then a recurrent core (GRU or LSTM, chosen in the config) and masked
   policy and value heads. Rollouts in one scan; GAE that bootstraps
   truncated battles and not terminated ones; minibatches of whole
   sequences with the hidden state stored at each sequence start and reset
   at episode boundaries. Configuration in files (batch, rollout length,
   widths, learning rates), checkpoints that resume bit for bit on the
   CPU (on the GPU, scatter-add order in the embedding gradients makes
   resumed runs differ by about 1e-12 unless XLA's deterministic ops are
   on, at 4x the update time), metrics to disk. Check: it beats random clearly with the default config, and
   the unit tests cover masking, GAE, sequence boundaries and resume.
2. Self-play and evaluation: the learner plays both sides, against itself
   and a pool of past checkpoints kept on the device, with a configured
   share of battles against the baselines. Every N updates it evaluates
   against the four baselines with confidence intervals and logs it.
3. Sizing and the first long run: measure memory and throughput on the
   3060 Ti over batch size, rollout length and width; one config file for
   the 3060 Ti and one for the RTX 6000 machines that differ only in
   sizes; an unattended lab kit (lab/run.sh, CUDA 12 image option) that
   collects the same measurements and a short training run on the RTX 6000
   machines, where no agent runs, into one archive; then a long run on the
   3060 Ti to the exit, with the curve and evaluation results recorded in
   the README.

Exit: beats max-damage in well over half of evaluation games, and the
training curve is still improving. Reached with hand-made features; the
owner considers that heuristics rather than learning, so Phase 6 sets a
stricter exit for a learned network.

### Phase 5: transfer to real Showdown

- A player that loads a checkpoint and plays on Showdown through Node,
  building the same observation from Showdown's requests and protocol.
- Compare win rate against the same baseline in the engine and on Showdown.

Tasks, in order. Everything runs on the pinned local Showdown simulator
(BattleStream); nothing connects to a server.

1. Showdown player: a Node bridge runs many battles at once in the pinned
   simulator from the same team pool, builds each learner side's
   observation with reference/player-view.cjs, and a Python driver runs
   the network on batches of them (recurrent state per battle) and sends
   back choices, including forced switches and refused trapped switches.
   The opponent is a Node baseline from reference/baselines.cjs. The
   bridge keeps both sides' requests and the protocol per battle, so a
   later state rebuilder for forward search has everything it needs.
   Check: thousands of battles without errors against each baseline, and
   the win rates next to the engine's evaluation of the same checkpoint.
2. Closing the gap, only if task 1 misses the exit: whole-battle
   statistics of the learner against max-damage in both simulators (win
   rate by side, battle length, move, switch, status and faint
   frequencies, how often the policy's choice was refused), find the
   causes of any gap and fix them in the engine, the observation or the
   player, with differential cases where the engine was wrong. Not needed
   this time: task 1 measured a largest gap of 0.63 points (README,
   Phase 5 results); it stays the procedure for later checkpoints.

Exit: the two win rates are close: within 3 percentage points against
each baseline, with at least 4,000 battles on each side. A large gap means
an engine or observation bug; find it with the whole-battle statistics and
fix it before more training.

### Phase 6: a network that learns the game

Phase 4 reached its exit with hand-made features (features.py: type
effectiveness, STAB, max-damage scores, matchup estimates). Without them
the same network stalls at about 30% against max-damage (README, Feature
ablations), and with them the curve flattens early. The owner wants an agent
that learns type relations and tactics itself and keeps improving with more
compute. Designs: work/codex/design-sol.md and design-opus.md (outside git).

Input rules for this phase and after:
- Network inputs are the player-view observation and static dex columns
  (power, accuracy, type, category, priority, base stats, types). Public
  protocol facts stay allowed, including the effectiveness event ("It's
  super effective") of the last step. Not allowed: anything computed from
  the type chart (effectiveness, STAB-effectiveness products, immunities),
  damage estimates, stat ratios or matchup scores.
- Hidden values (the opponent's next action, item, ability, moves) may be
  training targets of auxiliary losses, never inputs. A test shows that
  policy, value and recurrent state are identical for identical public
  histories with different hidden state.

Tasks, in order. Each is one commit with its tests; runs stay outside git.

1. Fixes from the outside review (work/codex/review-sol.md) that would
   distort the comparison: Illusion level after a silent switch-out, Flame
   Body after a Substitute hit, Bad Dreams order, Arena Trap knowledge after
   Mummy (engine, observation and bridge), shaping at truncation, the
   evaluation count of a partial batch, held retries advancing the waiting
   side's recurrent state, pool replacement mid-battle, resume keeping its
   learning-rate horizon and checking the team pool, NaN in the observation
   differential, the FLOP label. Own moves carry their real Hidden Power
   type and Return power, as the Showdown request shows them, in the
   engine and the bridge. Engine fixes come with differential cases.
2. Pairwise network and screening: shared public descriptors with learned
   type embeddings; a clean control (same inputs, per-action scorers, no
   pair module); a pairwise network where each own move meets the opposing
   active and each opposing move meets each own member; auxiliary targets
   (opponent's next action, hidden item, ability and moves); a type probe
   (a linear read-out of the chart multiplier from frozen representations,
   held-out species and moves). Short runs on the 3060 Ti at equal steps
   with the Phase 4 league and schedule.
3. Recurrent roster transformer: field, twelve members, four move
   candidates and two event tokens, self-attention, GRU on the field token,
   the same targets and screening against the winner of task 2.
4. Long run of the winner on the 3060 Ti to the exit, README results.
   Training-mix changes (opponent shares, pool length, PPO epochs) are
   tested here, one at a time, only after the architecture is chosen.

Exit, one training seed, evaluations of 4,096 battles (sampled policy,
greedy reported too):
- at least 60% against max-damage and against heuristics, with the lower
  bound of each 95% interval above 55%;
- at least 55% against its own checkpoint from two thirds of the run;
- still improving: on a fixed panel (the four baselines and two fixed
  earlier checkpoints, equal weights, paired team draws), the checkpoint at
  2X environment steps beats the one at X by at least 2 points with a
  positive 95% interval, X chosen before the run (for example 80M);
- the run fits in 12 hours on the 3060 Ti;
- a test that changing the type chart table does not change the network's
  inputs (only the battles it plays);
- the Phase 5 transfer check on the promoted checkpoint: within 3 points
  per baseline, at least 4,000 battles each.

### Phase 7: final training

Long runs on the RTX 6000 cards, then evaluation on the Showdown ladder or
against published bots, as the owner decides. Measure strength per compute
doubling on a frozen panel (baselines, earlier checkpoints, the other
architectures from Phase 6) with the cross-play matrix, on teams held out
from training; stop a scale-up when two doublings in a row bring no
meaningful gain.

Tasks, in order:

1. Lab kit for the Phase 6 network: the unattended kit (lab/) runs the
   pairwise network on both RTX 6000 cards for a time budget, one training
   arm per GPU (default: the Phase 6 network at a larger batch, and a wider
   one), keeps checkpoints at doubling step counts, and evaluates them on a
   fixed panel with the Phase 6 reference checkpoints and a cross-play
   matrix. Built on the lab machine without a prebuilt image. Check: the
   smoke mode runs on the 3060 Ti.
2. The lab run and its analysis: strength per compute doubling for each
   arm, README results, the choice of the model for the external test.
3. A tuning study before the next long run: the lab kit's study mode
   (LAB_STUDY=1) trains twelve short runs of the base network at 4,096
   rows on the two cards, equal time each, varying the learning rate,
   epochs, minibatches, entropy and GAE lambda (configs/tune/, with two
   seed replicates), each scored on held-out teams against the Phase 6
   references and the baselines. The best two or three are then extended
   to a longer budget before one of them sets the next long run. Added
   because the PPO settings were tuned at 512 to 1,024 rows and the lab
   ran at 4,096 without retuning, and every doubling still gained.
   Status: done (README "Phase 7: tuning studies"): lr 4e-3 with 2 epochs,
   no random bots. Next: the long run, configs/p7-long-spread.toml and
   configs/p7-long-recent.toml, one per GPU.
4. The external test (ladder or published bots), when the owner decides.

## Forward search (branch `search`)

Search on top of the trained policy at decision time, developed on the
`search` branch beside Phase 7. Design: work/codex/search-design.md
(outside git). The engine stays a pure function of state and actions and
every random outcome goes through the forced-outcome hook, so a search can
copy a state and advance it many times in one `vmap` batch.

Rules:
- The deployed searcher sees only what its player sees: its own requests,
  the protocol and its own choices. Hidden opponent values are sampled from
  a belief built from the random team generator's sets and the evidence of
  the battle; the true state is used only in tests and clearly marked
  diagnostic arms, never in a strength claim.
- Every result compares search with the raw policy of the same checkpoint
  on paired setups (same teams, seats and seeds).

Tasks, in order:

1. Search transition and signal test: a search step without auto-reset
   (held choices, refusals, forced switches, terminal states), batched
   copies, value backup; a one-decision rollout search on the true state
   (diagnostic oracle) against the raw policy, to see whether the value
   network can rank actions at all; timings on the 3060 Ti.
2. Belief and rebuild: a set catalogue from the pinned generator, hard and
   soft evidence from the battle, Illusion, a particle bank per battle and
   full engine states rebuilt from the player's view; the rollout search
   (Method 1) on rebuilt states, evaluated in the engine.
3. Showdown: rebuild from the live protocol stream in the bridge, the same
   search on pinned local Showdown, and the search-minus-raw gain compared
   across simulators.
4. Matrix game (Method 2), then information-set MCTS (Method 3) only if the
   results call for deeper search.
5. External test: Foul Play (pinned) on local Showdown. The ladder only
   with the owner's explicit approval. Run before the rest of task 4:
   Method 2 lost to Method 1 in task 4's first part, and an independent
   opponent tells more about the searcher than another method tested
   against our own network. Promotion rule (owner's, fixed in advance):
   search is promoted for this matchup if search minus the best raw arm
   (sampled or greedy, chosen on the integration games) is at least
   +2 points with the paired 95% lower bound above 0; if the interval
   excludes +2, raw stays the default for this matchup; otherwise
   inconclusive.

Exit for the first deployable searcher (tasks 1 to 3):
- at least 52% head-to-head against the raw checkpoint over 8,192 games
  (4,096 paired setups, both seats), with the paired 95% lower bound above
  50%, in the engine and on local Showdown;
- no material loss against max-damage and heuristics (at most 1 point
  below the raw policy, 4,096 battles each);
- decision time: a soft target of 250 ms at the 95th percentile on the
  3060 Ti, like the engine throughput gate: aimed for, and adjustable by
  the owner after optimisation attempts.

## Working rules

- Finish a phase's exit checks before starting the next phase.
- One commit per task or coherent piece of work, with a plain message.
- When something in this roadmap turns out wrong, change the roadmap in the
  same commit and say why in one or two sentences. Do not add other planning,
  status or handoff documents.
- When Showdown's behaviour is unclear, read the pinned Showdown source and
  write a differential case. Do not guess from memory of the games.
- If a mechanic would need a special code path and appears in under 0.5% of
  teams, leave it unsupported and move on, as long as all teams excluded
  this way together stay under 1% of the pool.
- Report honestly: a failing suite or a missed gate is stated in the commit
  message and the README, not worked around.
