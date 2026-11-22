# Lab kit: the RTX 6000 measurements

This kit runs the prepared measurements on the lab machine (2 x NVIDIA RTX 6000 Ada, 48 GB each,
host driver 550.163.01, Docker with the NVIDIA Container Toolkit). You start one command, leave it
for a few hours, and bring back one archive. No agent (Claude or Codex) runs on the lab machine,
and nobody can debug it there, so read the preflight part before you start.

What it measures, in order:

1. **preflight** (about a minute): nvidia-smi, the driver and CUDA versions, `jax.devices()`, the
   JAX environment, a jitted product and one cold compile of the engine on each GPU.
2. **facts**: GPU models and memory, CPU, RAM, the commit, the image variant, the pip freeze, the time.
3. **bench**: the engine alone and the environment (random and heuristics policies, without and with
   observations) at batch 4096, 16384, 65536 and 131072, one process per GPU, with pokejax.bench's
   defaults (256 steps per scan, 3 repetitions) so that the numbers compare with the README. Each row:
   steps/s, battles/s, cold compile, peak GPU memory. About 20 minutes on one RTX 3060 Ti; the two GPUs
   of the lab share the work.
4. **sweep**: the self-play trainer (configs/selfplay.toml) over batch size (1024, 4096, 16384),
   rollout length (64, 128), network width (1x, 2x, 4x) and core (GRU, LSTM). Each cell: environment
   and learner steps/s, update time, compile time, peak GPU memory. A failed cell (out of memory) is a
   result, and the sweep goes on.
5. **contention**: configs/rtx6000.toml on one GPU alone, then on both GPUs at once.
6. **train**: configs/rtx6000.toml on each GPU for a time budget, with evaluations against the four
   baselines (sampled and greedy) and a final evaluation with a checkpoint.
7. **pack**: summary.md (the tables) and one archive.

## 1. On a machine with this repo (once, before the lab)

The kit is made from a commit. Commit everything first (`git status` must be clean; make-kit warns
otherwise), then:

    lab/make-kit.sh --tar

This writes `work/lab-kit/pokemon-jax-lab-<commit>/` and `work/lab-kit/pokemon-jax-lab-<commit>.tar.gz`.
The kit has the repo at the commit, the team pool `work/teams/gen5-teams.npz` and `KIT_COMMIT`. It
needs no network, no Node and no Showdown on the lab machine.

If Docker is available here, you can also put the image into the kit (see section 2 for the tag):

    docker build -t pokemon-jax-cuda12 --build-arg JAX_CUDA=cuda12 --build-arg INSTALL_AGENTS=0 .
    lab/make-kit.sh --tar --image pokemon-jax-cuda12

Copy the `.tar.gz` to the lab machine and extract it.

## 2. On the lab machine: the image

Use the image variant that matches the driver. Our driver is 550, so use `cuda12`:

- `cuda13` (the default of the Dockerfile): JAX's CUDA 13 build, needs driver 580 or newer. It will
  not run on 550.
- `cuda12`: JAX's CUDA 12 build with the newest CUDA 12 libraries pip allows. Needs driver 525 or
  newer (JAX's install guide). This is the variant for the lab.
- `cuda12-driver550`: the same JAX, with the CUDA 12 libraries pinned to the oldest versions the
  JAX plugin allows (`constraints-cuda12-driver550.txt`). Use it only if `cuda12` fails the preflight.

Why `cuda12` should run on 550: JAX's install guide gives driver 525 as the minimum for the CUDA 12 pip
packages on Linux, and NVIDIA's compatibility rules let a CUDA 12.x application run on any newer 12.x
driver of the same family (with some feature limits). The newest CUDA 12 libraries pip resolves are
CUDA 12.9 ones; the preflight is the check on the lab machine. The fallback exists because
`jax-cuda12-plugin` 0.12.0.1 itself requires `nvidia-cuda-nvcc-cu12` >= 12.6.85 and
`nvidia-cudnn-cu12` >= 9.10.2.21, so a pure CUDA 12.4 set cannot be installed with it.

If the kit has an `image/` folder, load the image:

    docker load -i <kit>/image/pokemon-jax-cuda12.tar

Otherwise build it in the kit folder (needs network access for pip):

    cd <kit>
    docker build -t pokemon-jax-cuda12 \
      --build-arg JAX_CUDA=cuda12 --build-arg INSTALL_AGENTS=0 \
      --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) .

`INSTALL_AGENTS=0` leaves out Claude Code and the Codex CLI. The `USER_UID`/`USER_GID` arguments make
the files the lab writes belong to you (Linux).

## 3. On the lab machine: the run

Check the GPUs once on the host: `nvidia-smi` must list both RTX 6000 cards.

A smoke run (every step at minimal size, about sixteen minutes on one RTX 3060 Ti, most of it compiling and
the batch fallback running out of memory at every batch there)
proves the kit works on the machine:

    cd <kit>
    docker run --rm --gpus all --shm-size=2g \
      -e LAB_SMOKE=1 -v "$PWD":/workspace -w /workspace \
      pokemon-jax-cuda12 bash lab/run.sh

The full run (the time budget is LAB_HOURS, default 3):

    cd <kit>
    docker run --rm --gpus all --shm-size=2g \
      -e LAB_HOURS=3 -v "$PWD":/workspace -w /workspace \
      pokemon-jax-cuda12 bash lab/run.sh

Leave it. You can follow the progress from the host in another terminal:

    tail -f <kit>/work/lab/progress.log

Options (environment variables of the `docker run`):

| Variable | Default | Meaning |
|---|---|---|
| `LAB_HOURS` | 3 | the time budget; the sweep and the training shrink to fit it |
| `LAB_SMOKE` | 0 | 1: every step at minimal size, about sixteen minutes on one 3060 Ti (see the troubleshooting section) |
| `LAB_FRESH` | 0 | 1: start over (the old `work/lab` is moved aside to `work/lab.old-<time>`) |
| `LAB_DIR` | `work/lab` | where the results go (inside the kit) |
| `LAB_TRAIN_CONFIG` | `configs/rtx6000.toml` | the training configuration of the contention check and the training runs (a path inside the kit); on out of memory its batch halves, see the troubleshooting section |

How the budget is split: the bench has a fixed size. When the sweep starts, it may start cells until
55% of the time left has passed (after 15 minutes kept for the contention check and the pack), so a
slow bench leaves less sweep. The contention check has a fixed size. The training runs take the rest:
each GPU first runs a short probe of the training configuration to measure one update (at half the batch if it runs out of memory), then runs as many
updates as fit before the end, with about six evaluations and four checkpoints. The run finishes
about five minutes before LAB_HOURS, when the archive is written.

Memory: every process the lab starts (bench, sweep, contention, training, preflight) preallocates 90% of
its GPU (`XLA_PYTHON_CLIENT_PREALLOCATE=true`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`), one process per GPU
at a time, so no two processes share a GPU's reservation. A value you set in the environment (`docker run
-e`) is kept. The lab's own process (the orchestrator) loads no JAX, so it holds no GPU memory; before each process starts,
the lab waits (at most a minute) until nvidia-smi shows its GPU idle. The values the processes ran with are in
`lab/facts.json` (`xla_settings`, which also pins `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that CUDA's device numbers
are nvidia-smi's) and in every
sweep and contention record (`xla`). The peak GPU memory in the records is the allocator's peak use
(`peak_bytes_in_use`), not the reserved pool.

## 4. When it is done

The last line says `lab run complete`, or `lab run not complete` when a step failed or stopped early (then
rerun it, see below). The exit status is 0 only when every step is complete. The archive is:

    <kit>/work/lab-results/<date>-<host>.tar.gz

Bring back this one file. It holds `lab/summary.md` (the tables), `lab/progress.log`, the preflight
report (`lab/preflight/report.txt`), the facts and the pip freeze, the results of every step as JSON
lines (`lab/results/`), the logs (`lab/logs/`), the training runs (`lab/runs/rtx6000-gpu*/`: the
config, the metrics, the result and the final checkpoint) and the commit (`KIT_COMMIT`).

If the run stops (power, a reboot, Ctrl-C), run the same command again in the same kit. Steps that
finished are skipped, the sweep continues from its last measured cell and an interrupted training run
resumes from its checkpoint. The time budget of a rerun starts when it starts. Use `LAB_FRESH=1` to
start over.

## If the preflight fails

The run stops the GPU steps and still writes the archive. Open `lab/preflight/report.txt` in the
archive: it names what failed and what to try. Then:

1. Check `nvidia-smi` on the host. If it fails there, the driver or the container toolkit is wrong.
   Start the container with `--gpus all`.
2. If JAX cannot start the CUDA backend with `cuda12` on the 550 driver, build the fallback:

       docker build -t pokemon-jax-cuda12-550 \
         --build-arg JAX_CUDA=cuda12-driver550 --build-arg INSTALL_AGENTS=0 \
         --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -g) .

   and run the smoke command with that tag.
3. Send back the archive of the failed run either way: it holds the report and the JAX environment
   (`lab/preflight/jax-environment.txt`).

## If a probe or the contention check runs out of memory

The training configuration (configs/rtx6000.toml, 16,384 rows per update) needs one contiguous block of
about 16.6 GiB for its update, inside a pool of about 40 GiB (0.9 of 48 GB). On the RTX 6000 under native
Linux that should fit. The 3060 Ti used for the sizing cannot confirm it: that card refuses single
allocations above about 3 to 4 GiB (under WSL2 a 3.0 GiB block works, a 4.0 GiB one does not).

The lab does not wait for you. The contention check and each GPU's training probe fall back on their own:
when an attempt runs out of memory (error class `oom`, RESOURCE_EXHAUSTED), it is repeated at half the batch
(`rollout.batch_size`): 16,384, then 8,192, then 4,096, and no lower. Only out-of-memory failures are
repeated. A timeout or a crash is recorded, and the step goes on.

Where each attempt and the batch used are recorded:

- `lab/results/train-probe.jsonl`: one record per probe attempt, with its batch, status and error class.
- `lab/results/contention.jsonl`: one record per attempt of each contention run (phase `alone` and `both`),
  with its batch. The `both` runs start at the batch of the `alone` run, and halve together when either
  GPU runs out of memory.
- `lab/runs/<config>-gpu<N>/lab-plan.json`: `batch_size` (the batch the training ran with), `config` (the
  reduced configuration, `config-batch<B>.toml` in the same directory) and `attempts`.
- `lab/results/train.jsonl`: the training record of each GPU, with the same `batch_size` and `attempts`.
- `lab/summary.md`: the contention table (with a Batch column; the slowdown is given only between runs at
  the same batch), the training records and training probes tables, and the Batch columns of the plan and
  runs tables.

The reduced configurations are written by the lab into the run directories (`lab/runs/contention/` for the
contention check). Each one is configs/rtx6000.toml with only `batch_size` changed, so nothing is copied by
hand.

What the outcome means:

- A batch that runs is used for the whole training run of that GPU. One GPU may train at 8,192 rows and the
  other at 16,384: each run's plan says which.
- If 4,096 runs out of memory too, this card cannot run the configuration. That GPU has no training run: its
  record in `lab/results/train.jsonl` has status `failed` and lists the three attempts. The other GPU goes on,
  the step ends partial (exit status 4), and the pack step still writes `summary.md` and the archive.
- A training run that runs out of memory after its probe is recorded as failed and is not retried at a
  smaller batch. A rerun resumes it from its checkpoint at the same batch. To start that GPU's run again with
  the fallback, move `lab/runs/<config>-gpu<N>/` aside and rerun: the probe ladder starts from 16,384 again.
- A rerun of a probe or contention run that failed at every batch starts the ladder again from 16,384.

## Notes

- The team pool in the kit is the one of this repo. Its sha256 is in `lab/facts.json`.
- Compile times are cold on purpose: no persistent compilation cache is used by the measurements.
- A cell or a benchmark that fails is recorded with its error and the run continues. A contention run or a
  probe that runs out of memory is repeated at half the batch first. A failed training run is recorded and
  retried by a rerun.
