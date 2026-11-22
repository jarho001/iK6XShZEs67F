#!/bin/sh
# The image's build self-check (see the Dockerfile). INSTALL_AGENTS=1 also starts the agent CLIs.
set -eu

ssh -V
node -e "const [a,b]=process.versions.node.split('.').map(Number); \
  if (a<22 || (a===22 && b<18)) { throw new Error('Node too old: '+process.version); } \
  console.log('node', process.version)"
JAX_PLATFORMS=cpu python -c "import jax, jax.numpy as jnp; \
  assert int(jax.jit(lambda x: x + 1)(jnp.int32(1))) == 2; \
  print('jax', jax.__version__, jax.devices())"
if [ "$INSTALL_AGENTS" = 1 ]; then
	claude --version
	codex --version
fi
bash -lc 'node --version && python -c "import jax, flax, optax; print(\"login shell ok:\", jax.__version__)"'
if [ "$INSTALL_AGENTS" = 1 ]; then
	bash -lc 'claude --version && codex --version'
fi
