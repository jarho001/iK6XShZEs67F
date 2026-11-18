"""Showdown's fixed-point relay and chain operations shared by hook providers."""

import jax.numpy as jnp

IDENTITY = 4096


def modify(value, modifier):
	return (value.astype(jnp.int32) * jnp.asarray(modifier, jnp.int32) + 2047) // IDENTITY


def chain(first, second):
	return (jnp.asarray(first, jnp.int32) * jnp.asarray(second, jnp.int32) + 2048) // IDENTITY
