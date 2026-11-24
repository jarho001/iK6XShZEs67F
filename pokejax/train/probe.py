"""The type probe of Phase 6: a linear read-out of the type-chart multiplier from a frozen network.

    python -m pokejax.train.probe CHECKPOINT [--out FILE.json] [--battles N] [--steps T] [--seed S]
    python -m pokejax.train.probe --config CONFIG.toml [--out FILE.json]   (the network at initialisation)

The probe plays the checkpoint's greedy policy against max-damage. At every decision where the
learner must move, it computes for each own active move slot the network's per-move feature (roster
networks: probe_features, the move-defender pair for the pairwise network, the move and the defender
vectors for the control, the move and the defender tokens for the transformer) from the same
observation under two parameter sets: the checkpoint's and a random initialisation's. The label is
the chart multiplier of the move's type against the defender's shown types, in six classes {0, 1/4,
1/2, 1, 2, 4}. The chart is used only here, for labels; the network never reads it.

Rows are ordinary damaging moves of a fixed type (no status or fixed-damage moves, and not Hidden Power,
Natural Gift, Judgment, Weather Ball, Multi-Attack, Techno Blast or Revelation Dance) against defenders
that have no type-changing or immunity ability (Levitate, Flash Fire, Volt Absorb, Water Absorb, Motor
Drive, Sap Sipper, Dry Skin, Wonder Guard), no Air Balloon, are not transformed, disguised or under Roost,
and are alive.

A multinomial read-out, L2-regularised and class-balanced, is fitted on standardised features. Two splits:
(1) 20% of the defender species and 20% of the move IDs are held out (a row is a test row when either
is held out; both types stay covered in training); (2) 20% of the defender type pairings are held out.
Each split reports accuracy, macro recall over the classes present in its test rows, immunity recall,
the rate of resisted and super-effective confusions, and per-class recall. The same read-out is reported
for the random initialisation, the control.
"""

import argparse
import functools
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pokejax.baselines import make_policy
from pokejax.engine.state import ILLUSION_SLOT, ROOST, TRANSFORMED
from pokejax.env import DEFAULT_POOL, Env, EnvConfig
from pokejax.tables import load_tables
from pokejax.train.checkpoint import CONFIG_ENTRY
from pokejax.train.config import ConfigError, TrainConfig, load_config
from pokejax.train.league import Layout, League
from pokejax.train.network import PokePolicy, initial_carry, vocabulary
from pokejax.train.roster import dex_columns

CLASS_NAMES = ("0", "1/4", "1/2", "1", "2", "4")
# Four times the multiplier, as an index: 0, 1/4, 1/2, 1, 2, 4 map to classes 0 to 5.
CLASS_OF_QUARTERS = np.array([0, 1, 2, -1, 3, -1, -1, -1, 4, -1, -1, -1, -1, -1, -1, -1, 5])
MOVE_CATEGORY_STATUS = 2
MOVE_REQUEST = 1
VARIABLE_TYPE_MOVES = (
	"hiddenpower",
	"naturalgift",
	"judgment",
	"weatherball",
	"multiattack",
	"technoblast",
	"revelationdance",
)
TYPE_CHANGING_ABILITIES = (
	"levitate",
	"flashfire",
	"voltabsorb",
	"waterabsorb",
	"motordrive",
	"sapsipper",
	"dryskin",
	"wonderguard",
)
EXCLUDED_ITEMS = ("airballoon",)
HELD_OUT_FRACTION = 0.2
L2_WEIGHT = 1e-3
FIT_STEPS = 400
LEARNING_RATE = 0.05
SPLIT_SEED = 7
RANDOM_SEED = 1
CLASSES = len(CLASS_NAMES)
SPLIT_NAMES = ("species_and_moves", "dual_pairings")


class ProbeError(ValueError):
	"""A probe that cannot be run as asked."""


def _by_name(names, listed) -> np.ndarray:
	"""A boolean row per table entry, true for the listed names (zero row is never listed)."""
	mask = np.zeros(len(names), bool)
	index = {name: row for row, name in enumerate(names)}
	for name in listed:
		if name in index:
			mask[index[name]] = True
	return mask


@functools.cache
def probe_masks():
	"""The static masks of the probe, from the host tables (labels and exclusions only)."""
	tables = load_tables()
	moves = tables.moves
	damaging = (
		(np.asarray(moves.category) != MOVE_CATEGORY_STATUS)
		& (np.asarray(moves.base_power) > 0)
		& (np.asarray(moves.fixed_damage) == 0)
		& ~_by_name(moves.ids, VARIABLE_TYPE_MOVES)
	)
	damaging[0] = False
	return {
		"damaging": damaging,
		"immune_ability": _by_name(tables.abilities.ids, TYPE_CHANGING_ABILITIES),
		"excluded_item": _by_name(tables.items.ids, EXCLUDED_ITEMS),
	}


def chart_multipliers(tables, move_type, defender_types) -> np.ndarray:
	"""The chart multiplier of each move type [N] against the defender types [N, 2] (engine indices, -1 none)."""
	exponent = np.asarray(tables.types.exponent, np.float64)
	immune = np.asarray(tables.types.immune)
	value = np.ones(len(move_type))
	for column in range(defender_types.shape[1]):
		defender = defender_types[:, column]
		known = defender >= 0
		index = np.where(known, defender, 0)
		part = np.where(immune[move_type, index], 0.0, 2.0 ** exponent[move_type, index])
		value *= np.where(known, part, 1.0)
	return value


def multiplier_classes(tables, move_type, defender_types) -> np.ndarray:
	"""The six-class label of each row (0 immune, 1 quarter, 2 half, 3 neutral, 4 double, 5 quadruple)."""
	quarters = np.rint(4.0 * chart_multipliers(tables, move_type, defender_types)).astype(int)
	classes = CLASS_OF_QUARTERS[np.clip(quarters, 0, len(CLASS_OF_QUARTERS) - 1)]
	if np.any(classes < 0):
		raise ProbeError("a chart multiplier outside {0, 1/4, 1/2, 1, 2, 4}")
	return classes


def collect(league: League, policy_params, parameter_sets: dict, steps: int, seed: int):
	"""Plays the greedy policy with policy_params; returns each parameter set's rows and the row info.

	parameter_sets maps names to parameters; each decision's feature is computed under all of them from
	the same observation. Returns (features by name [T, L, 4, F], info by key [T, L, ...]).
	"""
	model = league.model
	masks = probe_masks()
	damaging = jnp.asarray(masks["damaging"])
	immune_ability = jnp.asarray(masks["immune_ability"])
	excluded_item = jnp.asarray(masks["excluded_item"])
	vocab_type = jnp.asarray(dex_columns().move_type)

	@jax.jit
	def decisions(state):
		obs = state.obs
		battles = state.env.state
		lanes = jnp.arange(obs.turn.shape[0])
		seats = league.layout.lane_seats(state.learner)
		foe = 1 - seats
		active = jnp.argmax(obs.own.is_active, axis=-1)
		moves = obs.own.moves[lanes, active].astype(jnp.int32)  # [L, 4]
		foe_active = jnp.argmax(obs.opponent.is_active, axis=-1)
		shown_types = obs.opponent.types[lanes, foe_active].astype(jnp.int32) - 1  # engine, -1 none
		species = obs.opponent.species[lanes, foe_active].astype(jnp.int32)
		foe_slot = battles.active[lanes, foe].astype(jnp.int32)
		ability = battles.ability[lanes, foe, foe_slot].astype(jnp.int32)
		item = battles.item[lanes, foe, foe_slot].astype(jnp.int32)
		volatiles = battles.volatiles
		clean = (
			~immune_ability[ability]
			& ~excluded_item[item]
			& (volatiles[lanes, foe, ROOST] == 0)
			& (volatiles[lanes, foe, TRANSFORMED] == 0)
			& (volatiles[lanes, foe, ILLUSION_SLOT] == 0)
			& (battles.hp[lanes, foe, foe_slot] > 0)
		)
		choosing = obs.request[:, MOVE_REQUEST] > 0
		valid = damaging[moves] & (moves > 0) & (clean & choosing)[:, None]
		features = {
			name: model.apply({"params": params}, obs, method=PokePolicy.probe_features)
			for name, params in parameter_sets.items()
		}
		return features, {
			"valid": valid,
			"moves": moves,
			"move_type": vocab_type[moves] - 1,  # engine index, -1 for none
			"defender_types": shown_types,
			"species": species,
		}

	play = jax.jit(lambda state: league.play(policy_params, None, state, greedy=True)[0])
	collected = {name: [] for name in parameter_sets}
	info = {}
	state = league.reset(jax.random.key(seed))
	for _ in range(steps):
		features, step_info = jax.device_get(decisions(state))
		for name in parameter_sets:
			collected[name].append(np.asarray(features[name]))
		for key, value in step_info.items():
			info.setdefault(key, []).append(np.asarray(value))
		state = play(state)
	rows = {name: np.stack(parts) for name, parts in collected.items()}
	return rows, {key: np.stack(parts) for key, parts in info.items()}


def select_rows(rows: dict, info: dict):
	"""The valid decisions: features [N, F] per name, and labels and split keys [N]."""
	keep = info["valid"] & (info["move_type"] >= 0)  # [T, L, 4]
	move_type = info["move_type"][keep]
	defenders = np.broadcast_to(info["defender_types"][:, :, None, :], keep.shape + (2,))[keep]
	species = np.broadcast_to(info["species"][:, :, None], keep.shape)[keep]
	moves = info["moves"][keep]
	labels = multiplier_classes(load_tables(), move_type, defenders)
	pairs = np.sort(np.where(defenders < 0, -1, defenders), axis=1)
	features = {name: array[keep] for name, array in rows.items()}
	return features, {"label": labels, "move": moves, "species": species, "pair": pairs}


def _split_masks(info: dict):
	rng = np.random.default_rng(SPLIT_SEED)

	def held(values):
		unique = np.unique(values)
		count = max(1, round(HELD_OUT_FRACTION * unique.size))
		return np.isin(values, rng.choice(unique, size=count, replace=False))

	species_test = held(info["species"]) | held(info["move"])
	pair_keys = (info["pair"][:, 0] + 1) * 1000 + (info["pair"][:, 1] + 1)
	pair_test = held(pair_keys)
	return species_test, pair_test


def fit_readout(x_train, y_train, x_test, y_test):
	"""Standardised, class-balanced, L2-regularised multinomial read-out; returns its predictions."""
	if len(y_train) == 0 or len(y_test) == 0:
		return np.zeros(len(y_test), np.int64)
	mean = x_train.mean(axis=0)
	spread = np.maximum(x_train.std(axis=0), 1e-6)
	xs_train = jnp.asarray((x_train - mean) / spread, jnp.float32)
	xs_test = jnp.asarray((x_test - mean) / spread, jnp.float32)
	counts = np.bincount(y_train, minlength=CLASSES).astype(np.float64)
	class_weight = np.where(counts > 0, len(y_train) / (CLASSES * np.maximum(counts, 1)), 0.0)
	weights = jnp.asarray(class_weight[y_train], jnp.float32)
	targets = jnp.asarray(y_train, jnp.int32)
	width = x_train.shape[1]
	optimiser = optax.adam(LEARNING_RATE)
	params = (jnp.zeros((width, CLASSES), jnp.float32), jnp.zeros((CLASSES,), jnp.float32))

	def loss(p):
		logp = jax.nn.log_softmax(xs_train @ p[0] + p[1])
		picked = jnp.take_along_axis(logp, targets[:, None], axis=1)[:, 0]
		return -(weights * picked).sum() / weights.sum() + L2_WEIGHT * jnp.sum(jnp.square(p[0]))

	@jax.jit
	def fit():
		def body(carry, _):
			p, s = carry
			updates, s = optimiser.update(jax.grad(loss)(p), s, p)
			return (optax.apply_updates(p, updates), s), None

		(p, _), _ = jax.lax.scan(body, (params, optimiser.init(params)), None, length=FIT_STEPS)
		return p

	p = fit()
	return np.asarray(jnp.argmax(xs_test @ p[0] + p[1], axis=1))


def readout_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
	"""Accuracy, macro recall over the present classes, immunity recall, confusions and per-class recall."""
	present = [k for k in range(CLASSES) if np.any(truth == k)]
	recall = {}
	for k in present:
		recall[CLASS_NAMES[k]] = float(np.mean(predicted[truth == k] == k))
	resisted = np.isin(truth, (1, 2))
	super_effective = np.isin(truth, (4, 5))
	confused = (np.isin(predicted, (1, 2)) & super_effective) | (
		np.isin(predicted, (4, 5)) & resisted
	)
	return {
		"rows": int(truth.size),
		"accuracy": float(np.mean(predicted == truth)) if truth.size else None,
		"macro_recall": float(np.mean(list(recall.values()))) if recall else None,
		"immunity_recall": recall.get(CLASS_NAMES[0]),
		"resisted_super_confusion": float(np.mean(confused)) if truth.size else None,
		"recall": recall,
		"class_rows": {CLASS_NAMES[k]: int(np.sum(truth == k)) for k in range(CLASSES)},
	}


def evaluate(features: np.ndarray, info: dict) -> dict:
	"""The two splits of one feature array (rows aligned with info)."""
	if info["label"].size < 4:
		return {name: {"rows": int(info["label"].size)} for name in SPLIT_NAMES}
	truth = info["label"]
	species_test, pair_test = _split_masks(info)
	results = {}
	for name, test in (("species_and_moves", species_test), ("dual_pairings", pair_test)):
		train = ~test
		predicted = fit_readout(features[train], truth[train], features[test], truth[test])
		results[name] = readout_metrics(truth[test], predicted)
		results[name]["train_rows"] = int(train.sum())
	return results


def read_config(path) -> TrainConfig:
	"""The configuration a run saved with its checkpoint (a run directory or the file itself)."""
	path = Path(path)
	file = path / "checkpoint.npz" if path.is_dir() else path
	with np.load(file, allow_pickle=False) as data:
		if CONFIG_ENTRY not in data.files:
			raise ProbeError(f"{file} has no configuration")
		return TrainConfig.from_dict(json.loads(str(data[CONFIG_ENTRY])))


def restore(path, model: PokePolicy, example, carry):
	"""The parameters a checkpoint saved, with the structure of the model's full initialisation."""
	path = Path(path)
	file = path / "checkpoint.npz" if path.is_dir() else path
	template = model.init(jax.random.key(0), example, carry, method=PokePolicy.full)["params"]
	leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
	values = []
	with np.load(file, allow_pickle=False) as data:
		for key_path, leaf in leaves:
			name = ".params" + jax.tree_util.keystr(key_path)
			if name not in data.files:
				raise ProbeError(f"{file} has no entry {name}")
			array = data[name]
			if array.shape != leaf.shape:
				raise ProbeError(f"{file}: {name} has shape {array.shape}, expected {leaf.shape}")
			values.append(jnp.asarray(array))
	return jax.tree_util.tree_unflatten(treedef, values)


def run_probe(
	checkpoint: Path | None,
	config: TrainConfig,
	*,
	battles: int,
	steps: int,
	seed: int,
	pool: Path = DEFAULT_POOL,
) -> dict:
	"""The probe of a checkpoint (or, without one, of the network at initialisation) as a report."""
	if config.network.arch == "dense":
		raise ProbeError(
			"the type probe reads the roster networks: network.arch = control, pairwise or "
			"transformer"
		)
	tables = load_tables()
	env = Env(EnvConfig(batch_size=battles, max_steps=config.env.max_steps, observe="both"), pool)
	model = PokePolicy(vocabulary(tables), config.network, config.auxiliary)
	policies = {"max-damage": make_policy("max-damage", env.tables, tables)}
	league = League(
		env, Layout({"max-damage": battles}), model, config.network, config.pool.size, policies
	)
	example = league.reset(jax.random.key(seed)).obs
	carry = initial_carry(config.network, (battles,))
	random_params = model.init(jax.random.key(RANDOM_SEED), example, carry, method=PokePolicy.full)[
		"params"
	]
	if checkpoint is None:
		policy_params = random_params
		sets = {"initial": random_params}
	else:
		policy_params = restore(checkpoint, model, example, carry)
		sets = {"trained": policy_params, "random": random_params}
	rows, info = collect(league, policy_params, sets, steps, seed)
	features, labelled = select_rows(rows, info)
	report = {
		"checkpoint": None if checkpoint is None else str(checkpoint),
		"arch": config.network.arch,
		"feature_width": int(next(iter(features.values())).shape[-1]),
		"battles": battles,
		"steps": steps,
		"seed": seed,
		"rows": int(labelled["label"].size),
		"label_rows": {
			name: int(np.sum(labelled["label"] == k)) for k, name in enumerate(CLASS_NAMES)
		},
	}
	for name, array in features.items():
		report[name] = evaluate(array, labelled)
	return report


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.train.probe", description=__doc__.split("\n")[0]
	)
	parser.add_argument(
		"checkpoint", nargs="?", type=Path, help="a run directory or checkpoint.npz"
	)
	parser.add_argument(
		"--config", type=Path, help="without a checkpoint: this configuration's network"
	)
	parser.add_argument("--out", type=Path, default=None, help="write the report as JSON")
	parser.add_argument("--battles", type=int, default=512)
	parser.add_argument("--steps", type=int, default=200, help="decision steps played")
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--team-pool", type=Path, default=DEFAULT_POOL)
	return parser


def main(argv=None) -> int:
	args = build_parser().parse_args(argv)
	try:
		if args.checkpoint is not None:
			config = read_config(args.checkpoint)
		elif args.config is not None:
			config = load_config(args.config)
		else:
			raise ProbeError("give a checkpoint, or --config for the network at initialisation")
		report = run_probe(
			args.checkpoint,
			config,
			battles=args.battles,
			steps=args.steps,
			seed=args.seed,
			pool=args.team_pool,
		)
	except (ProbeError, ConfigError, FileNotFoundError) as error:
		print(f"error: {error}", file=sys.stderr)
		return 2
	text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
	print(text)
	if args.out is not None:
		args.out.parent.mkdir(parents=True, exist_ok=True)
		args.out.write_text(text + "\n", encoding="utf-8")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
