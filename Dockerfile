# syntax=docker/dockerfile:1
# Python + Node image for the JAX battle engine, its Showdown oracle and the RTX 6000 lab kit.

# Build arguments:
#   JAX_CUDA        cuda13 (the default: needs an NVIDIA driver of 580 or newer), cuda12 (JAX's
#                   CUDA 12 build, driver 525 or newer; the lab's 550 hosts) or cuda12-driver550
#                   (cuda12 with the CUDA 12 libraries pinned to the oldest the plugin allows, see
#                   constraints-cuda12-driver550.txt). The pins: requirements*.txt.
#   INSTALL_AGENTS  1 (the default) installs Claude Code and the Codex CLI; 0 leaves them out
#                   (the lab image must not contain them).
#   CLAUDE_CODE_VERSION, CODEX_VERSION, USER_UID, USER_GID as before.
#
# Examples (from the repo root):
#   docker build -t pokemon-jax .
#   docker build -t pokemon-jax-cuda12 --build-arg JAX_CUDA=cuda12 --build-arg INSTALL_AGENTS=0 .

# Pinned Showdown (reference-lock.json) requires Node >= 22.18; its CI tests 22.x.
FROM node:22.23.5-trixie-slim AS node
FROM python:3.13.18-slim-trixie

ARG JAX_CUDA=cuda13
ARG INSTALL_AGENTS=1
# Agent CLIs installed from npm.
ARG CLAUDE_CODE_VERSION=2.1.325
ARG CODEX_VERSION=0.169.0
# Match the host user's uid/gid on Linux so files written to /workspace keep their owner.
ARG USER_UID=1000
ARG USER_GID=1000

# No XLA_PYTHON_CLIENT_* variable here: the lab and pokejax.sizing set their own allocator defaults
# (pokejax/records.py XLA_DEFAULTS), and tests/conftest.py turns preallocation off for pytest.
ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    NPM_CONFIG_PREFIX=/home/developer/.npm-global \
    PATH=/opt/venv/bin:/home/developer/.npm-global/bin:$PATH \
    CLAUDE_CONFIG_DIR=/home/developer/.claude \
    POKEJAX_JAX_CUDA=${JAX_CUDA} \
    POKEJAX_AGENTS=${INSTALL_AGENTS}

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

# The venv belongs to the developer user so `pip install -e .` works later. The pins of the chosen
# JAX_CUDA variant are installed and checked by docker/install-pins.sh.
COPY --chown=developer:developer requirements.txt requirements-common.txt requirements-cuda12.txt \
     constraints-cuda12-driver550.txt /tmp/pins/
COPY --chown=developer:developer docker/install-pins.sh /tmp/pins/install-pins.sh
RUN python -m venv /opt/venv \
    && sh /tmp/pins/install-pins.sh "${JAX_CUDA}" /tmp/pins

RUN if [ "${INSTALL_AGENTS}" = 1 ]; then \
      npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" "@openai/codex@${CODEX_VERSION}" \
      && npm cache clean --force; \
    elif [ "${INSTALL_AGENTS}" != 0 ]; then \
      echo "INSTALL_AGENTS must be 0 or 1, not ${INSTALL_AGENTS}" >&2; exit 1; \
    fi

# Self-check: Node is new enough for Showdown, SSH and JAX work (on the CPU: no GPU at build time)
# and, when they are installed, both agent CLIs start.
COPY --chown=developer:developer docker/selfcheck.sh /tmp/pins/selfcheck.sh
RUN INSTALL_AGENTS="${INSTALL_AGENTS}" sh /tmp/pins/selfcheck.sh

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
