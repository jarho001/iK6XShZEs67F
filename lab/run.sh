#!/usr/bin/env bash
# The lab kit's entry point. Run it inside the container at the repo root, unattended:
#
#   lab/run.sh                 the measurements in LAB_HOURS hours (default 3)
#   LAB_HOURS=1 lab/run.sh     the same, split into one hour
#   LAB_SMOKE=1 lab/run.sh     every step at minimal size, a few minutes
#   LAB_FRESH=1 lab/run.sh     start over: the old work/lab is moved aside
#
# Steps run in order: preflight (the GPUs and JAX), facts (the system), bench (only with LAB_BENCH=1),
# sweep, train, panel, then pack (summary.md and the archive work/lab-results/<date>-<host>.tar.gz). A
# step with a done marker in work/lab/done is skipped, so rerunning continues where it stopped; a step
# that fails is logged and the next ones still run. A failed preflight stops the GPU steps, and the
# archive still holds the preflight report. See lab/README.md for the owner's steps.
set -u

cd "$(dirname "$0")/.." || exit 1

LAB_HOURS="${LAB_HOURS:-3}"
LAB_SMOKE="${LAB_SMOKE:-0}"
LAB_FRESH="${LAB_FRESH:-0}"
LAB_DIR="${LAB_DIR:-work/lab}"
JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
LAB_START="$(date +%s)"
export LAB_HOURS LAB_SMOKE LAB_DIR JAX_PLATFORMS LAB_START PYTHONUNBUFFERED=1 LAB_BENCH="${LAB_BENCH:-0}"

if [ "$LAB_FRESH" = 1 ] && [ -e "$LAB_DIR" ]; then
	mv "$LAB_DIR" "$LAB_DIR.old-$(date -u +%Y%m%dT%H%M%SZ)" || exit 1
fi

# Exit status of the preflight when JAX cannot use the GPUs (pokejax/lab.py PREFLIGHT_FAILED).
PREFLIGHT_FAILED=3
status=0
gpu_ok=1
for step in preflight facts bench sweep train panel; do
	case "$step" in
		bench)
			if [ "${LAB_BENCH:-0}" != 1 ]; then
				echo "bench: off (set LAB_BENCH=1 to run it)"
				continue
			fi
			;;
	esac
	case "$step" in
		bench | sweep | train | panel)
			if [ "$gpu_ok" != 1 ]; then
				echo "$step: skipped, the preflight failed"
				continue
			fi
			;;
	esac
	if python -m pokejax.lab is-done "$step"; then
		echo "$step: already done, skipped"
		continue
	fi
	python -m pokejax.lab step "$step"
	code=$?
	if [ "$code" = "$PREFLIGHT_FAILED" ]; then
		gpu_ok=0
		status=1
	elif [ "$code" != 0 ]; then
		status=1
	fi
done

python -m pokejax.lab step pack
pack_code=$?
if [ "$pack_code" != 0 ]; then
	status=1
fi
if [ "$status" = 0 ]; then
	echo "lab run complete"
else
	echo "lab run not complete (a failed or unfinished step): see ${LAB_DIR}/progress.log; rerun lab/run.sh to continue"
fi
exit "$status"
