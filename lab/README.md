# Lab kit: the Phase 7 runs on the RTX 6000 cards

This kit runs the Phase 7 measurements on the lab machine (2 x NVIDIA RTX 6000 Ada, 48 GB each, host
driver 550.163.01, Docker with the NVIDIA Container Toolkit). You start one command, leave it for
hours, and bring back one archive. No agent (Claude or Codex) runs on the lab machine, and nobody can
debug it there, so read the preflight part before you start.

The kit trains the Phase 6 network, the pairwise network with the four auxiliary losses, in two arms
(one arm per GPU, by default): the same network at 4,096 battle rows per update, and its wide variant.
Each arm keeps checkpoints at doubling step counts. Then every kept checkpoint plays a fixed panel
(four baselines and the two Phase 6 reference checkpoints), and the final checkpoints play a cross-play
matrix. The result is strength per compute doubling for each arm.

What it does, in order:

1. **preflight** (about a minute): nvidia-smi, the driver and CUDA versions, `jax.devices()`, the JAX
   environment, a jitted product and one cold compile of the engine on each GPU.
2. **facts**: GPU models and memory, CPU, RAM, the commit, the image variant, the pip freeze, the time.
3. **bench**: off by default. `LAB_BENCH=1` turns on the engine and environment benchmarks (the numbers
   already exist, see README.md).
4. **sweep** (about 25 minutes): each arm trains for 20 updates at 1,024, 2,048, 4,096 and 8,192 rows
   (the first arm also at 16,384). One process per GPU. Each row of the table gives the learner and
   environment steps per second, the update and compile times and the peak memory. Out of memory is a
   result.
5. **train**: each arm on its own GPU for the time budget, with evaluations against the four baselines
   (sampled and greedy) about twelve times per run, and checkpoints at 30M, 60M, 120M, ... environment
   steps, and at the last update.
6. **panel**: every kept checkpoint of each arm against the four baselines and the two references, with
   2,048 battle setups per opponent (4,096 for the final checkpoint), sampled and greedy. The paired
   difference of each checkpoint and the one before it has a 95% interval. Then the cross-play matrix of
   the final checkpoints and the two references (2,048 team draws per pair, both seats).
7. **pack**: `summary.md` (all tables), the training curves (SVG, by steps and by minutes) and one archive.

## 1. On a machine with this repo (once, before the lab)

The kit is made from a commit. Commit everything first (`git status` must be clean; make-kit warns
otherwise). The kit also needs the two Phase 6 reference checkpoints in the repo's `work/` folder
(`work/runs/p6-p2/checkpoint.npz` and `work/runs/p6-long/checkpoint-14650.npz`, with their
`config.json` files). make-kit copies them and stops with a message if one is missing. Then:

    lab/make-kit.sh --tar

This writes `work/lab-kit/pokemon-jax-lab-<commit>/` and `work/lab-kit/pokemon-jax-lab-<commit>.tar.gz`.
The kit has the repo at the commit, the team pool `work/teams/gen5-teams.npz`, the reference checkpoints
under `work/reference-checkpoints/` and `KIT_COMMIT`. It needs no network, no Node and no Showdown on
the lab machine. The kit has no Docker image: the image is built on the lab machine (section 2).

Copy the `.tar.gz` to the lab machine and extract it.

## 2. On the lab machine: the image

Build the image in the kit folder. Use the variant that matches the driver. Our driver is 550, so use
`cuda12` (JAX's CUDA 12 build; it needs driver 525 or newer):

    cd <kit>
    docker build -t pokemon-jax-cuda12 \
      --build-arg JAX_CUDA=cuda12 --build-arg INSTALL_AGENTS=0 \
      --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) .

`INSTALL_AGENTS=0` leaves out Claude Code and the Codex CLI. The `USER_UID`/`USER_GID` arguments make
the files the lab writes belong to you (Linux).

Other variants:

- `cuda13` (the Dockerfile's default) needs driver 580 or newer. It will not run on 550.
- `cuda12-driver550`: the same JAX, with the CUDA 12 libraries pinned to the oldest versions the JAX
  plugin allows (`constraints-cuda12-driver550.txt`). Use it only if `cuda12` fails the preflight:

      docker build -t pokemon-jax-cuda12-550 \
        --build-arg JAX_CUDA=cuda12-driver550 --build-arg INSTALL_AGENTS=0 \
        --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) .

  and use that tag in the commands below.

## 3. On the lab machine: the run

Check the GPUs once on the host: `nvidia-smi` must list both RTX 6000 cards.

A smoke run proves the kit on the machine. It runs every step at minimal size: the sweep at 512 rows, the
training for four updates (its probes start at 512 rows and fall to 128; the pairwise update needs a block
of about 4 GiB at 1,024 rows, which the 3060 Ti refuses), the panel and the cross-play at 512 battles. On one
RTX 3060 Ti it takes about forty minutes, most of it compiling, and both arms run one after the other on
that one GPU. You do not need it on the lab machine, but it is a good first check:

    cd <kit>
    docker run --rm --gpus all --shm-size=2g \
      -e LAB_SMOKE=1 -v "$PWD":/workspace -w /workspace \
      pokemon-jax-cuda12 bash lab/run.sh

The full run, with the time budget in hours. Twelve hours is the first choice:

    cd <kit>
    docker run --rm --gpus all --shm-size=2g \
      -e LAB_HOURS=12 -v "$PWD":/workspace -w /workspace \
      pokemon-jax-cuda12 bash lab/run.sh

Leave it. You can follow the progress from the host in another terminal:

    tail -f <kit>/work/lab/progress.log

Options (environment variables of the `docker run`):

| Variable | Default | Meaning |
|---|---|---|
| `LAB_HOURS` | 3 | the time budget; the sweep and the training shrink to fit it |
| `LAB_ARMS` | `configs/p7-pairwise.toml,configs/p7-pairwise-wide.toml` | the arms, comma-separated, configuration files inside the kit: arm i trains on GPU i (round robin); one arm alone trains on both GPUs |
| `LAB_TRAIN_CONFIG` | unset | one configuration on both GPUs (the old way); used only when `LAB_ARMS` is unset |
| `LAB_BENCH` | 0 | 1: run the engine and environment benchmarks first (about 20 minutes on one GPU) |
| `LAB_SMOKE` | 0 | 1: every step at minimal size (see section 3) |
| `LAB_FRESH` | 0 | 1: start over (the old `work/lab` is moved aside to `work/lab.old-<time>`) |
| `LAB_DIR` | `work/lab` | where the results go (inside the kit) |

To try another architecture, write a configuration file in `configs/` (for example a transformer arm,
with `network.arch = "transformer"`, as in `configs/p7-pairwise.toml`), commit it, make the kit again
and name it in `LAB_ARMS`. The configuration's `rollout.batch_size` is the starting batch, and the
lab halves it on out of memory. Each arm's file name must be different from the others.

Each arm's configuration sets the rows, the network and the league. The lab sets the length of each run
(from the time budget), its learning-rate decay (over that length), the checkpoints kept, and the number
of evaluations. Nothing else changes.

How the budget is split: the sweep starts at once and runs for about 25 minutes. The training gets what
is left, less five minutes for the pack and less the panel's reserve. The reserve comes from the number of
checkpoints each run keeps, at the panel's measured cost per battle (with a safety factor of two). The arms
on one GPU share that GPU's time equally. Each run's updates fill its share, with a margin for the
evaluations and checkpoints. The run finishes before `LAB_HOURS`, when the archive is written. The panel
can take longer than its reserve on a slower card; the summary then shows the real times.

Memory: every process the lab starts (bench, sweep cells, probes, training, panel, cross-play, preflight)
preallocates 90% of its GPU (`XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`),
one process per GPU at a time. A value you set in the environment (`docker run -e`) is kept. The values the
processes ran with are in `lab/facts.json` (`xla_settings`).

## 4. Progress, rerun and resume

`tail -f <kit>/work/lab/progress.log` shows each step's events. Each training run writes its metrics to
`lab/runs/<arm>-gpu<N>/metrics.jsonl` as it goes.

If the run stops (power, a reboot, Ctrl-C), run the same command again in the same kit. Steps that finished
are skipped. A training run resumes from its last checkpoint (every 250 updates) at the same batch and the
same length. A panel or cross-play report that exists is not run again. Use `LAB_FRESH=1` to start over.

## 5. When it is done

The last line says `lab run complete`, or `lab run not complete` when a step failed or stopped early (then
rerun it, see section 4). The exit status is 0 only when every step is complete. The archive is:

    <kit>/work/lab-results/<date>-<host>.tar.gz

Bring back this one file. It holds:

- `lab/summary.md`: the tables (steps, preflight, sweep, training plans and runs, panel with the paired
  differences, cross-play, figures);
- `lab/progress.log`, the preflight report (`lab/preflight/report.txt`), the facts and the pip freeze;
- `lab/results/`: every measurement as JSON lines (sweep, training probes, training, panel jobs);
- `lab/logs/`: the logs of every process;
- `lab/runs/<arm>-gpu<N>/`: each run's configuration, plan (`lab-plan.json`), metrics, result and the
  kept checkpoints (`checkpoint-<updates>.npz`) with the final one (`checkpoint.npz`);
- `lab/panel/`: the panel reports (`<run>.json` at 2,048 battles, `<run>-final.json` at 4,096) and
  `crossplay.json`;
- `lab/plots/`: the training curves against max-damage and heuristics, by steps and by minutes;
- `KIT_COMMIT`.

The archive holds every kept checkpoint (about 36 MB each), so it is large.

## If the preflight fails

The run stops the GPU steps and still writes the archive. Open `lab/preflight/report.txt` in the
archive: it names what failed and what to try. Then:

1. Check `nvidia-smi` on the host. If it fails there, the driver or the container toolkit is wrong.
   Start the container with `--gpus all`.
2. If JAX cannot start the CUDA backend with `cuda12` on the 550 driver, build the `cuda12-driver550`
   image (section 2) and run the commands with that tag.
3. Send back the archive of the failed run either way: it holds the report and the JAX environment
   (`lab/preflight/jax-environment.txt`).

## If a training probe runs out of memory

Each arm's training probe runs at its configured batch (4,096 rows). When a probe runs out of device memory
(error class `oom`, RESOURCE_EXHAUSTED), it is repeated at half the rows: 2,048, then 1,024, and no lower.
Only out-of-memory failures are repeated. A timeout or a crash is recorded, and the step goes on. The
attempts are in `lab/results/train-probe.jsonl`, the plan of each run (`lab/runs/<run>/lab-plan.json`,
its `batch_size` and `attempts`) and the training plans table of `lab/summary.md`.

A run that runs out of memory after its probe is recorded as failed. A rerun resumes it at the same batch
from its checkpoint. To start a run again with the fallback, move its `lab/runs/<run>/` folder aside and
rerun. The 3060 Ti used for the sizing refuses single allocations above about 3 to 4 GiB under WSL2, so
the fallback is always exercised there; on the RTX 6000 under Linux it should not be needed at 4,096 rows.

## Notes

- The team pool in the kit is the one of this repo. Its sha256 is in `lab/facts.json`.
- Compile times are cold on purpose: no persistent compilation cache is used by the measurements.
- The panel and the cross-play use the same pool, the same opponents and the same seeds in every run, so
  their numbers compare across arms and across checkpoints.

## Tuning study

After building the kit and image as above, run from the kit directory:

    docker run --rm --gpus all --shm-size=2g -e LAB_STUDY=1 -e LAB_HOURS=10 -v "$PWD":/workspace -w /workspace pokemon-jax-cuda12 bash lab/run.sh

The twelve configurations share each GPU's training window equally; each is probed separately so its
own update time determines its decay horizon. The sweep and cross-play are skipped. The final
checkpoints play sampled and greedy panels on the shipped held-out pool (2,048 setups per baseline,
4,096 per reference). The archive contains the ranked study table, curves and resumable runs. Set
`LAB_PANEL_POOL` to override the held-out pool and `LAB_TEAM_POOL` to override training teams.
Based on 8,999 base updates in 8.9 hours on RTX 6000, a sixth of the available training time is
roughly 1,500 updates before allowing for probes, panel and other overhead; faster or slower PPO
settings reach different update counts. Rerun the same command to resume.
