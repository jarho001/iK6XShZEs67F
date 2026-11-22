#!/usr/bin/env bash
# Build the self-contained lab kit from this repo. Run it here, on a machine with the repo and the
# team pool (work/teams/gen5-teams.npz, made with reference/generate-teams.cjs).
#
#   lab/make-kit.sh                  work/lab-kit/pokemon-jax-lab-<commit>/
#   lab/make-kit.sh --tar            the same, and work/lab-kit/pokemon-jax-lab-<commit>.tar.gz
#   lab/make-kit.sh --image TAG      also docker save the image TAG into the kit (image/)
#   lab/make-kit.sh --working-tree   testing only: take the files of the working tree, not the commit
#   lab/make-kit.sh --out DIR        build under DIR instead of work/lab-kit
#
# The kit holds the repo at the commit (git archive: committed files only), the team pool, the
# commit in KIT_COMMIT and, with --image, the image. The lab needs no network, no Node and no
# Showdown: everything the lab reads is in the kit.
set -u

repo="$(cd "$(dirname "$0")/.." && pwd)" || exit 1
cd "$repo" || exit 1

image=""
make_tar=0
working_tree=0
out="work/lab-kit"
while [ $# -gt 0 ]; do
	case "$1" in
		--image)
			[ $# -ge 2 ] || { echo "--image needs a tag" >&2; exit 2; }
			image="$2"
			shift 2
			;;
		--tar)
			make_tar=1
			shift
			;;
		--working-tree)
			working_tree=1
			shift
			;;
		--out)
			[ $# -ge 2 ] || { echo "--out needs a directory" >&2; exit 2; }
			out="$2"
			shift 2
			;;
		-h | --help)
			sed -n '2,13p' "$0"
			exit 0
			;;
		*)
			echo "unknown option: $1" >&2
			exit 2
			;;
	esac
done

commit="$(git rev-parse HEAD)" || { echo "not a git repository with a commit" >&2; exit 1; }
dirty=0
if [ -n "$(git status --porcelain)" ]; then
	dirty=1
	echo "warning: the working tree has uncommitted changes; the kit holds the commit only" >&2
fi
pool="work/teams/gen5-teams.npz"
if [ ! -f "$pool" ]; then
	echo "missing $pool: the kit needs the team pool" >&2
	exit 1
fi

name="pokemon-jax-lab-${commit:0:12}"
kit="$out/$name"
rm -rf "$kit"
mkdir -p "$kit" || exit 1
if [ "$working_tree" = 1 ]; then
	git ls-files --cached --others --exclude-standard -z | tar --null -T - -cf - | tar -xf - -C "$kit" || exit 1
else
	git archive --format=tar HEAD | tar -xf - -C "$kit" || exit 1
fi
mkdir -p "$kit/work/teams" || exit 1
cp "$pool" "$kit/work/teams/" || exit 1
{
	echo "commit=$commit"
	echo "dirty=$dirty"
	echo "working_tree=$working_tree"
	echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$kit/KIT_COMMIT" || exit 1

if [ -n "$image" ]; then
	if ! command -v docker >/dev/null 2>&1; then
		echo "docker is not available here: build the kit without --image, then save the image on a machine with docker" >&2
		exit 1
	fi
	mkdir -p "$kit/image" || exit 1
	docker save "$image" -o "$kit/image/${image//[:\/]/-}.tar" || exit 1
fi

if [ "$make_tar" = 1 ]; then
	tar -C "$out" -czf "$out/$name.tar.gz" "$name" || exit 1
	echo "archive: $out/$name.tar.gz"
fi
echo "kit: $kit"
