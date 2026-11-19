"""Pack the selected move's parameters for a single lookup in the move pipeline."""

from dataclasses import dataclass, field
from math import prod
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class MoveRow(NamedTuple):
	id: jax.Array
	move_accuracy: jax.Array
	move_always_hits: jax.Array
	move_base_power: jax.Array
	move_boost_effect: jax.Array
	move_boost_target: jax.Array
	move_boost_when: jax.Array
	move_boosts: jax.Array
	move_breaks_protect: jax.Array
	move_category: jax.Array
	move_charge: jax.Array
	move_charge_boosts: jax.Array
	move_charge_sun_skip: jax.Array
	move_checks_immunity: jax.Array
	move_confusion: jax.Array
	move_contact: jax.Array
	move_crash_damage: jax.Array
	move_crit_always: jax.Array
	move_crit_ratio: jax.Array
	move_damage_kind: jax.Array
	move_defrost: jax.Array
	move_drain: jax.Array
	move_effect: jax.Array
	move_fakeout: jax.Array
	move_fixed_damage: jax.Array
	move_force_switch: jax.Array
	move_future: jax.Array
	move_hazard: jax.Array
	move_heal: jax.Array
	move_heal_weather: jax.Array
	move_hit_max: jax.Array
	move_hit_min: jax.Array
	move_hits_semi: jax.Array
	move_magnitude: jax.Array
	move_pain_split: jax.Array
	move_priority: jax.Array
	move_pressure_target: jax.Array
	move_protects: jax.Array
	move_punch: jax.Array
	move_rapid_spin: jax.Array
	move_recoil: jax.Array
	move_recoil_basehp: jax.Array
	move_roost: jax.Array
	move_secondary_boosts: jax.Array
	move_secondary_chance: jax.Array
	move_secondary_self: jax.Array
	move_secondary_status: jax.Array
	move_secondary_status_choices: jax.Array
	move_secondary_user_species: jax.Array
	move_secondary_volatile: jax.Array
	move_self_faint: jax.Array
	move_self_switch: jax.Array
	move_self_volatile: jax.Array
	move_semi_power: jax.Array
	move_sheer_force: jax.Array
	move_spread: jax.Array
	move_stall_kind: jax.Array
	move_status: jax.Array
	move_status_self: jax.Array
	move_target_foe: jax.Array
	move_thaws_target: jax.Array
	move_type: jax.Array
	move_volatile_status: jax.Array
	move_weather: jax.Array
	move_weather_accuracy: jax.Array
	move_weather_heal: jax.Array
	move_weather_power: jax.Array
	move_weather_power_scale: jax.Array
	move_weather_type: jax.Array


MOVE_FIELDS = MoveRow._fields[1:]


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class MoveTable:
	rows: jax.Array
	layout: tuple = field(metadata={"static": True})

	@staticmethod
	def _unpack(row, layout):
		start, shape, dtype = layout
		values = row[..., start : start + prod(shape)]
		return values.reshape((*row.shape[:-1], *shape)).astype(dtype)

	def column(self, name):
		return self._unpack(self.rows, self.layout[MOVE_FIELDS.index(name)])

	def replace_columns(self, columns):
		rows = self.rows
		for name, values in columns.items():
			start, shape, _ = self.layout[MOVE_FIELDS.index(name)]
			rows = rows.at[:, start : start + prod(shape)].set(values.reshape(len(rows), -1))
		return MoveTable(rows, self.layout)


def build_move_table(**columns):
	"""Derive static slices from the exported columns' shapes and dtypes."""
	parts = []
	layout = []
	start = 0
	for name in MOVE_FIELDS:
		column = columns[name]
		# A shared signed 16-bit row preserves every Gen 5 parameter, including
		# negative sentinels. Check wider source dtypes before narrowing.
		if column.dtype.itemsize >= 2:
			limits = np.iinfo(np.int16)
			if column.min() < limits.min or column.max() > limits.max:
				raise ValueError(f"{name} exceeds the packed move parameter range")
		shape = column.shape[1:]
		layout.append((start, shape, column.dtype.str))
		parts.append(column.reshape(len(column), -1))
		start += prod(shape)
	rows = np.concatenate(parts, axis=1, dtype=np.int16)
	return MoveTable(jnp.asarray(rows), tuple(layout))


def read_move(tables, move):
	"""Load one move's parameters, then use static slices of the row."""
	if isinstance(move, MoveRow):
		return move
	table = tables.move_table
	row = table.rows[move]
	return MoveRow(move, *(table._unpack(row, layout) for layout in table.layout))
