# syntax=docker/dockerfile:1
# Python + Node development image for the JAX battle engine and its Showdown oracle.

# Pinned Showdown (reference-lock.json) requires Node >= 22.18; its CI tests 22.x.
FROM node:22.23.5-trixie-slim AS node
FROM python:3.13.18-slim-trixie

# Agent CLIs installed from npm.
ARG CLAUDE_CODE_VERSION=2.1.325
ARG CODEX_VERSION=0.169.0
# Match the host user's uid/gid on Linux so files written to /workspace keep their owner.
ARG USER_UID=1000
ARG USER_GID=1000

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    NPM_CONFIG_PREFIX=/home/developer/.npm-global \
    PATH=/opt/venv/bin:/home/developer/.npm-global/bin:$PATH \
    CLAUDE_CONFIG_DIR=/home/developer/.claude \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s ../lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && ln -s ../lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git openssh-client ripgrep tini \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory /workspace \
    && printf '%s\n' \
      '# Login shells reset PATH in /etc/profile; restore the venv and npm global bin.' \
      'export PATH="/opt/venv/bin:/home/developer/.npm-global/bin:$PATH"' \
      > /etc/profile.d/pokemon-jax-path.sh

RUN groupadd --gid "${USER_GID}" developer \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" --create-home --shell /bin/bash developer \
    && mkdir -p /workspace /opt/venv /home/developer/.claude /home/developer/.codex \
      /home/developer/.npm-global \
    && chown -R developer:developer /workspace /opt/venv /home/developer

USER developer
WORKDIR /workspace

# The venv belongs to the developer user so `pip install -e .` works later.
COPY --chown=developer:developer requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip check

RUN npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" "@openai/codex@${CODEX_VERSION}" \
    && npm cache clean --force

# Self-check: both agent CLIs start, Node is new enough for Showdown,
# and JAX can JIT on CPU (no GPU at build time).
RUN claude --version \
    && codex --version \
    && ssh -V \
    && node -e "const [a,b]=process.versions.node.split('.').map(Number); \
      if (a<22 || (a===22 && b<18)) { throw new Error('Node too old: '+process.version); } \
      console.log('node', process.version)" \
    && JAX_PLATFORMS=cpu python -c "import jax, jax.numpy as jnp; \
      assert int(jax.jit(lambda x: x + 1)(jnp.int32(1))) == 2; \
      print('jax', jax.__version__, jax.devices())" \
    && bash -lc 'claude --version && codex --version && node --version \
      && python -c "import jax, flax, optax; print(\"login shell ok:\", jax.__version__)"'

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
