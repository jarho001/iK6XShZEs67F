"""Showdown's fixed-point relay and chain operations shared by hook providers."""

import jax.numpy as jnp

FRACTION_BITS = 12
IDENTITY = 1 << FRACTION_BITS


def modify(value, modifier):
	return (value.astype(jnp.int32) * jnp.asarray(modifier, jnp.int32) + 2047) >> FRACTION_BITS


def chain(first, second):
	return (jnp.asarray(first, jnp.int32) * jnp.asarray(second, jnp.int32) + 2048) >> FRACTION_BITS
