"""The mechanics package must import without relying on the old flat layout."""

import importlib
import pkgutil

import pokejax.mechanics


def test_import_every_mechanics_module():
	modules = list(pkgutil.walk_packages(pokejax.mechanics.__path__, "pokejax.mechanics."))
	assert modules
	for module in modules:
		importlib.import_module(module.name)
