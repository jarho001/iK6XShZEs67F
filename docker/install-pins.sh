#!/bin/sh
# Installs the Python pins of one JAX_CUDA variant into /opt/venv, runs pip check and checks that the
# JAX plugin of the variant is the one installed (and the other CUDA plugin is not).
# Usage: install-pins.sh VARIANT PINS_DIR   (VARIANT: cuda13, cuda12 or cuda12-driver550)
set -eu

variant="${1:?usage: install-pins.sh VARIANT PINS_DIR}"
pins="${2:?usage: install-pins.sh VARIANT PINS_DIR}"
constraints=""
case "$variant" in
	cuda13)
		requirements="requirements.txt"
		plugin=jax-cuda13-plugin
		other=jax-cuda12-plugin
		;;
	cuda12)
		requirements="requirements-cuda12.txt"
		plugin=jax-cuda12-plugin
		other=jax-cuda13-plugin
		;;
	cuda12-driver550)
		requirements="requirements-cuda12.txt"
		constraints="constraints-cuda12-driver550.txt"
		plugin=jax-cuda12-plugin
		other=jax-cuda13-plugin
		;;
	*)
		echo "JAX_CUDA must be cuda13, cuda12 or cuda12-driver550, not '$variant'" >&2
		exit 1
		;;
esac

cd "$pins"
if [ -n "$constraints" ]; then
	pip install --no-cache-dir -r "$requirements" -c "$constraints"
else
	pip install --no-cache-dir -r "$requirements"
fi
pip check
python - "$plugin" "$other" "$variant" <<'PY'
import importlib.metadata as metadata
import sys

plugin, other, variant = sys.argv[1:4]
if metadata.version(plugin) != "0.12.0.1":
    sys.exit(f"{plugin} is {metadata.version(plugin)}, expected 0.12.0.1")
try:
    metadata.version(other)
except metadata.PackageNotFoundError:
    print(f"JAX variant {variant}: {plugin} 0.12.0.1 installed")
else:
    sys.exit(f"{other} must not be installed in the {variant} variant")
PY
