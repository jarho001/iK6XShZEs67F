# Pokemon JAX

A JAX battle engine, RL environment and maskable Recurrent PPO for Pokemon
Showdown `gen5randombattle`. The plan and rules are in `ROADMAP.md`.

## Container

One image runs on the RTX 3060 Ti dev machine (Docker Desktop, WSL2 backend)
and on the RTX 6000 training machines (Docker with the NVIDIA Container
Toolkit). It has Python 3.13 with the pins from `requirements.txt` (JAX
0.12.0.1 for CUDA 13), Node 22, git, ripgrep, and two agent CLIs: Claude Code
(`claude`) and the Codex CLI (`codex`). The host needs an
NVIDIA driver that supports CUDA 13 (580 or newer); the CUDA libraries come
from pip. For a driver of 525 or newer (the lab's 550 hosts), build the CUDA 12 variant with
`--build-arg JAX_CUDA=cuda12`; `cuda12-driver550` pins the CUDA 12 libraries to older releases as a
fallback. The lab image is built with `--build-arg INSTALL_AGENTS=0`, which leaves out both agent CLIs.
The lab kit that runs on the RTX 6000 machines is described in `lab/README.md`.

Build (from the repo root):

```bash
docker build -t pokemon-jax .
```

Build arguments: `CLAUDE_CODE_VERSION` and `CODEX_VERSION` set the agent CLI
versions, and
`--build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g)` on Linux makes
files written to the mounted repo belong to you.

Run with the repo mounted at `/workspace`. On Windows (PowerShell):

```powershell
docker run --rm -it --gpus all --mount "type=bind,source=K:\Pokemon JAX,target=/workspace" --mount "type=volume,source=pokemon-jax-claude,target=/home/developer/.claude" --mount "type=volume,source=pokemon-jax-codex,target=/home/developer/.codex" pokemon-jax
```

On Linux:

```bash
docker run --rm -it --gpus all --shm-size=2g -v "$(pwd)":/workspace -v pokemon-jax-claude:/home/developer/.claude -v pokemon-jax-codex:/home/developer/.codex pokemon-jax
```

Leave out `--gpus all` for CPU only; JAX then falls back to the CPU. The named
volumes keep the Claude Code and Codex logins between containers (sign in to
Codex with `codex login --device-auth`). The image sets no XLA allocator variable. The lab and `python -m pokejax.sizing` preallocate 90% of each GPU's memory per process (`XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`) unless you set those yourself; the tests set preallocation off. Other runs get JAX's default, or pass `-e XLA_PYTHON_CLIENT_PREALLOCATE=false` to share a GPU between processes.

Inside the container:

```bash
node reference/setup.cjs   # clone and build Showdown at the pinned commit
JAX_PLATFORMS=cuda python -c "import jax, jax.numpy as jnp; print(jax.devices()); print(jax.jit(lambda x: x @ x)(jnp.ones((256, 256))).devices())"
```

The second command fails instead of falling back to the CPU if JAX cannot see
the GPU. Expect one CUDA device on the dev machine.

## Environment

`pokejax/env.py` is the batched, functional environment (Phase 3). Host setup happens once in
the constructor; `reset` and `step` are pure, so they jit, vmap and scan:

```python
env = Env(EnvConfig(batch_size=4096, max_steps=1000, observe="both"), "work/teams/gen5-teams.npz")
env_state, timestep = jax.jit(env.reset)(key)
env_state, timestep = jax.jit(env.step)(env_state, actions)  # actions int [B, 2]
```

A battle that ends (win, loss, tie) or reaches the step cap is replaced inside the same step by
a fresh one with teams from the sampler. The time step has both sides' observations [B, 2, ...]
(`observe="none"` skips them), zero-sum rewards [B, 2] (win 1, loss -1, tie 0; optional
potential-based shaping with `shaping=True`), `terminated` and `truncated` [B], the observed
legal mask [B, 2, 9] and episode info (winner, length in steps and turns, illegal and refused
actions). `keep_final=True` also returns the observation before the reset. A hidden trap
(Shadow Tag, Arena Trap, Magnet Pull) refuses a switch as Showdown does: the step does not
advance, the refused side's mask loses its switches, and the other side's action is kept for the
next step. `OpponentEnv(env, opponent)` plays one random side per battle against a policy
function `(state, side, key) -> action` (random over the engine's legal mask by default) and
returns time steps of batch shape [B]. The module docstring has the details.

`pokejax/baselines.py` has four policies with that signature, chosen by name with
`make_policy(name, env.tables)`: `random` (uniform over the legal mask), `random-move` (uniform over the
legal moves, a switch only when none is legal), `max-damage` (power x STAB x effectiveness x accuracy x
hits x attack / defence, a switch only when it must) and `heuristics` (poke-env's
`SimpleHeuristicsPlayer`, commit bab3673). They are branch-free, read only what the player knows and
pick from the engine's true legal actions (hidden traps included), so none is ever refused. The
module docstring lists every interpretation of poke-env. `reference/baselines.cjs` is the same four
in Node, for `python -m pokejax.stats --policy0 NAME --policy1 NAME` and for the agreement check.

## Benchmarks

Run `python -m pokejax.bench --json` for the random legal policy, adding
`--batch-size 16384` to change the batch or `--with-observation` to observe
both sides. `--env` measures the environment of `pokejax.env` (sampler draws, auto-reset,
rewards, flags) with the random policy on both sides instead of the bare engine loop;
`--env --policy NAME` has both sides play that baseline from the state (`--policy1` gives side 1
another). The cold first call includes compilation and one rollout.

Measured 2026-11-17 on an NVIDIA GeForce RTX 3060 Ti, JAX 0.12.0.1,
HEAD `93a1b26fbbbd8bc0dbc527647b7a2bbc3e7ce7dd`:

| Batch | Mode | Cold first call (s) | Steps/s | Battles/s | Mean battle length (steps) |
|---:|---|---:|---:|---:|---:|
| 4,096 | Engine only | 4.65 | 25,134,430 | 355,907 | 70.70 |
| 16,384 | Engine only | 5.00 | 49,587,788 | 701,426 | 70.75 |
| 4,096 | Both observations | 5.05 | 12,374,399 | 175,471 | 70.70 |

The first row is the median of three fresh processes: cold first calls ranged
from 4.63 to 4.78 s, steps/s from 24,853,781 to 25,367,994, and battles/s
from 351,933 to 359,215. Each process reports the median of three warmed
rollouts. Cold compile under 60 s: pass. Engine throughput at batch 4,096
against the 5,000,000 steps/s target: pass.

In Phase 1, status moves, abilities and items do nothing, so later phases will
be slower.

### Phase 1 results

The differential tests passed the 40-battle fixture and the 300-battle set
(2 tests passed). The existing PPO smoke test won 3,883 of 4,096 battles
against random (94.80%, 95% Wilson CI 94.08–95.44%); random against random
won 50.95% (95% CI 49.42–52.48%). Training took 136.01 s and 9,830,400
environment steps.

### Phase 2 results

Gates, measured 2026-11-20 on an idle NVIDIA GeForce RTX 3060 Ti, JAX 0.12.0.1, with the
Phase 2 exit code (`python -m pokejax.bench --json`, batch 4,096, engine only, median of three
fresh processes, after performance pass 7): 3,419,031 steps/s (runs 3,404,191 to 3,425,404),
48,183 battles/s, mean battle length 70.99 steps, cold first call 29.05 s (29.01 to 29.20 s).
Cold compile under 60 s: pass. Engine throughput against the 3,000,000 steps/s target: pass
(the target was 5,000,000 until the owner lowered it at the end of Phase 2; see ROADMAP.md,
Gates). Throughput fell from 7.62M after performance pass 4 to 3.32M as tasks 11 to 14 added
the remaining mechanics. Performance pass 5 recovered 4%; a structural attempt (pass 6, a
working copy of the active Pokemon) did not help and was not committed; pass 7 (move identity
before row loads, prepared team inputs, shared damage facts) gained 3% and cut compile time.
The CPU cold compile at batch 64 is about 125 s.

Coverage (`COVERAGE.md`, generated): all 223 moves, 126 abilities and 44 items in the
10,000-team pool are supported, so 100.0% of teams are fully supported. The excluded set is
empty (0 teams, 0.0%). The team sampler (`pokejax/sampler.py`) draws only from supported
teams and computes its index array once on the host.

Differential recordings on unchanged pool teams: the 40-battle fixture
`tests/fixtures/diff-pool.jsonl.gz` matches 40 of 40 battles (2,609 decisions) and the
300-battle set `work/diff/pool.jsonl.gz` matches 300 of 300 (21,526 decisions). The other
fixtures and 300-battle sets still match completely. `python -m pytest -q tests --large`
passes 374 tests; `node --test tests/test-record-battles.cjs` passes 18.

Statistics (`python -m pokejax.stats --battles 4000 --unchanged --seed N`, pool teams exactly
as generated): with seed 1 and with seed 2 no statistic is flagged after Holm correction
(341 metrics each). The smallest raw p-values are 0.041 (seed 1, `move.agility`) and 0.064
(seed 2, `move.seedflare`). Showdown and the engine agree on side 0's win rate (Showdown
0.505 and engine 0.491 with seed 1; 0.504 and 0.497 with seed 2), mean battle length (Showdown
61.9 and engine 61.5 turns with seed 1; 62.3 and 62.3 with seed 2), faints per battle (9.95
and 9.98; 9.93 and 9.96) and status counts.

PPO smoke test (`python -m pokejax.ppo_smoke`, default configuration, 600 updates,
9,830,400 environment steps): the trained policy won 3,754 of 4,096 battles against random
(91.65%, 95% Wilson CI 90.76–92.46%); random against random won 49.88% (48.35–51.41%). Training
took 188.9 s on the RTX 3060 Ti, with no illegal actions.

### Phase 3 results

Gates, measured 2026-11-21 on an idle NVIDIA GeForce RTX 3060 Ti, JAX 0.12.0.1, with the
Phase 3 exit code, batch 4,096, median of three fresh processes each. Phase 3 exit measurement,
environment steps per second without observations (`python -m pokejax.bench --json --env --policy P`,
sampler draws, auto-reset, rewards and policy P on both sides):

| Policy on both sides | Steps/s | Runs | Mean battle length (steps) |
|---|---:|---|---:|
| random | 3,067,270 | 3,066,048 to 3,079,595 | 71.31 |
| random-move | 3,041,462 | 3,034,812 to 3,051,893 | 43.90 |
| max-damage | 3,025,314 | 3,020,785 to 3,026,324 | 27.46 |
| heuristics | 3,016,411 | 3,012,282 to 3,019,419 | 27.65 |

Each baseline runs at the 3,000,000 steps/s target: pass, with the thinnest margin 0.5%
(heuristics). The engine alone (`python -m pokejax.bench --json`) runs 3,077,254 steps/s
(3,072,321 to 3,096,411; 3,419,031 at the end of Phase 2). It lost 5% to the event record and
observation state of task 1 and 6% to the 41 engine defects fixed between tasks 1 and 2 (found by
fresh recordings with new seeds; see `tests/test_regress.py`). With both sides observed the
environment (random policies) runs 2,243,745 steps/s. Cold first call: 32.7 to 33.8 s, 35.3 s with
observations; cold compile under 60 s: pass.

Observation (tasks 1 and 2). `observe` shows what one player can know at a decision point: the
own team in full, the opponent as revealed, both actives' public volatiles, weather, hazards, Trick
Room, pending Wish and Healing Wish, and a fixed record of the last step's visible events (action
order, move, miss, failure, critical hit, effectiveness, damage and healing in protocol percent, status,
faint). Durations that the protocol never states are not in it, only the turns since a public effect
started. The no-leak tests check that two states differing only in unrevealed opponent data or private
counters give the same observation. `reference/player-view.cjs` builds the same observation from one
player's requests and protocol lines alone, and `pokejax/obs_diff.py` compares every leaf with the
engine's at every decision point of recorded pool battles, for both sides: the 40-battle fixture
(2,649 decision points), the 300-battle set (21,826) and four more 300-battle seeds match completely.
The comparison found and fixed leaks (a benched foe's HP after Regenerator, Shadow Tag, Arena Trap and
Magnet Pull in the legal mask, a Transformed member's copied stats) and missing public facts (Knock Off,
Trick and stolen items). Illusion is the one carve-out: once the foe's Illusion holder has been active
(about 2.5% of battles) the protocol does not say which roster entry a disguised switch-in is, so the
opponent roster arrays of the two sides can legitimately differ and are not compared there.

Environment (task 3). `Env` runs a batch of battles with teams from the sampler and replaces a battle that
ended or hit the step cap inside the same step. It returns both sides' observations (or none),
zero-sum rewards (win 1, loss -1, tie 0; optional potential-based shaping, off by default),
`terminated` and `truncated` kept apart, the observed legal mask and episode statistics. A switch that
only a hidden trap forbids is refused as Showdown refuses it. `OpponentEnv` plays one random side per
battle against a policy function. The PPO smoke test (`python -m pokejax.ppo_smoke`, 600 updates,
9,830,400 environment steps) runs on it: the trained policy won 3,658 of 4,096 battles against random
(89.31%, 95% Wilson CI 88.32-90.22%); random against random won 49.56% (48.03-51.09%). Training took
222.2 s on the RTX 3060 Ti, with no illegal actions.

Baselines (task 4). The four policies of `pokejax/baselines.py` and the Node ones of
`reference/baselines.cjs` agree exactly. At every decision point of recorded unchanged pool battles
(`node reference/record-battles.cjs --profile pool --policies`) the Node max-damage and heuristics
policies choose an action for both sides from that player's own view, and `pokejax.baselines_diff`
replays the battle through the engine and compares it with the JAX choice: 0 of 10,596 choices differ
over the 40-battle fixture (`tests/fixtures/policy-pool.jsonl.gz`) and 0 of 87,304 over the 300-battle
set; no choice fell back to a random pick. Ties between candidates go to the first in request order
(Showdown's `side.pokemon` order, which `BattleState.party` tracks), so the check is sensitive to it:
ascending team slots instead make 29 choices of the fixture differ. Interpretations of poke-env that the
policies make, beyond the brief: Hidden Power counts as the typed power 70 move (the Gen 5 table row has
power 0), the entry-hazard branch sees only Spikes and Toxic Spikes, ties are scores within 1e-5, and the
setup-move branch is ported but off because poke-env's `move.target == "self"` compares an enum with a
string and is never true at the pinned commit. Enabling it in both implementations still agrees
(0 of 87,304); it changes 227 of the 5,298 heuristics choices of the fixture (4%).

Whole-battle statistics on the unchanged pool teams (`python -m pokejax.stats --battles 4000 --unchanged
--seed N --policy0 A --policy1 B`, the same team pairs in both simulators). Side 0's win rate in
Showdown and in the engine, and the two-proportion test of the difference:

| Side 0 against side 1 | Seed | Showdown | Engine | Difference (SE) | p | Turns Showdown / engine | Flagged (metrics) |
|---|---:|---:|---:|---:|---:|---:|---|
| random-move against random | 1 | 0.9235 | 0.9193 | +0.0042 (0.0060) | 0.48 | 38.60 / 38.38 | none (337) |
| random-move against random | 2 | 0.9187 | 0.9163 | +0.0025 (0.0062) | 0.68 | 38.60 / 38.71 | none (340) |
| max-damage against random | 1 | 0.9812 | 0.9832 | -0.0020 (0.0030) | 0.50 | 24.64 / 24.45 | none (338) |
| max-damage against random | 2 | 0.9842 | 0.9832 | +0.0010 (0.0028) | 0.72 | 24.82 / 24.69 | none (339) |
| heuristics against random | 1 | 0.9850 | 0.9858 | -0.0008 (0.0027) | 0.78 | 24.27 / 23.93 | none (337) |
| heuristics against random | 2 | 0.9872 | 0.9875 | -0.0003 (0.0025) | 0.92 | 24.30 / 24.34 | none (339) |
| heuristics against max-damage | 1 | 0.5268 | 0.5225 | +0.0043 (0.0112) | 0.70 | 18.43 / 18.40 | none (336) |
| heuristics against max-damage | 2 | 0.5140 | 0.5235 | -0.0095 (0.0112) | 0.40 | 18.63 / 18.75 | none (337) |

No metric of any run is flagged after Holm correction (battle length, its distribution, faints,
switches, Struggle, confusion, statuses, and the use counts of every move and consumed item). The
smallest raw p-values are 0.018 to 0.087 (one per run, a different metric each time: for example
`status.par` 0.044 with seed 1 for max-damage against random, which is 0.81 with seed 2). Random
against random, for the Phase 2 comparison, is under Phase 2 results.
