"""Integer non-HP stat-stage scaling from Gen 5 Pokemon.calculateStat.

Inputs are int32 arrays/scalars. ``stored_stat`` must be a nonnegative stored
stat in [0, 65535], after Gen 5 field stat selection; zero is reachable through
Gen 5 nature-product truncation. This value-range precondition is documented
but not dynamically checked. ``stage`` is a signed int32
ModifyBoost result. The stage is clamped to [-6, 6]. This function implements
only the scaling substep and does not derive base stats, dispatch events, apply
field selection, or apply stat modifiers.
"""

import jax.numpy as jnp

# 840 is divisible by every denominator in the stage ratios (2 through 8).
_STAGE_DENOMINATOR = 840
_STAGE_FACTORS = jnp.asarray(
	[
		2 * _STAGE_DENOMINATOR // (2 - stage)
		if stage < 0
		else (2 + stage) * _STAGE_DENOMINATOR // 2
		for stage in range(-6, 7)
	],
	jnp.int32,
)


def _as_int32(value, name):
	input_dtype = getattr(value, "dtype", None)
	if input_dtype is not None and jnp.dtype(input_dtype) != jnp.dtype(jnp.int32):
		raise TypeError(f"{name} must have dtype int32")
	array = jnp.asarray(value)
	if array.dtype != jnp.dtype(jnp.int32):
		raise TypeError(f"{name} must have dtype int32")
	return array


def scale_stat_stage(stored_stat, stage):
	"""Scale a stored non-HP stat in [0, 65535] by a clamped Gen 5 stage.

	For stage s >= 0, return floor(stat * (2 + s) / 2). For s < 0,
	return floor(stat * 2 / (2 - s)). Integer operations preserve the exact
	floor behavior; the common denominator keeps the product within int32.
	"""
	stat = _as_int32(stored_stat, "stored_stat")
	boost = jnp.clip(_as_int32(stage, "stage"), -6, 6)
	factor = _STAGE_FACTORS[boost + 6]
	return (stat * factor) // _STAGE_DENOMINATOR
