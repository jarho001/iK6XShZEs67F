"""The names of the baseline policies (pokejax.baselines), kept apart so that the training configuration and
the lab's orchestrator can read them without importing the engine (which initializes JAX)."""

POLICY_NAMES = ("random", "random-move", "max-damage", "heuristics")
