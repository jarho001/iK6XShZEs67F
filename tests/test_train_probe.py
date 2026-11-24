"""The type probe: the chart labels, the read-out on planted and noise features, the splits, and a
probe run on a small checkpoint (its trained and random parameter sets, and the initial one)."""

import numpy as np
import pytest
from train_support import tiny_config, write_pool

from pokejax.tables import load_tables
from pokejax.train import probe
from pokejax.train.checkpoint import save_checkpoint
from pokejax.train.ppo import Trainer

ROSTER = (
	"network.width=8",
	"network.feed_forward_width=12",
	"network.pair_width=6",
	"network.interaction_width=3",
	"network.context_width=16",
	"network.core_width=16",
	"network.activation=silu",
	"network.species_embedding=4",
	"network.type_embedding=3",
)
AUXILIARY = (
	"auxiliary.next_action=0.05",
	"auxiliary.item=0.01",
	"auxiliary.ability=0.01",
	"auxiliary.moves=0.03",
)

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]


def type_index(tables, name):
	return tables.types.index[name]


def test_chart_labels_are_the_six_multiplier_classes(tables):
	t = lambda name: type_index(tables, name)
	move_types = np.array([t("water"), t("water"), t("fire"), t("grass"), t("ground"), t("normal")])
	defenders = np.array(
		[
			[t("fire"), t("rock")],  # 2 x 2 = 4
			[t("ground"), -1],  # 2
			[t("water"), -1],  # 1/2
			[t("fire"), t("flying")],  # 1/2 x 1/2 = 1/4
			[t("flying"), -1],  # immune
			[t("normal"), -1],  # 1
		]
	)
	classes = probe.multiplier_classes(tables, move_types, defenders)
	np.testing.assert_array_equal(classes, [5, 4, 2, 1, 0, 3])
	values = probe.chart_multipliers(tables, move_types, defenders)
	np.testing.assert_allclose(values, [4.0, 2.0, 0.5, 0.25, 0.0, 1.0])


def test_read_out_recovers_a_planted_relation_and_not_noise():
	rng = np.random.default_rng(0)
	truth = rng.integers(0, 6, size=3000)
	basis = rng.normal(size=(6, 12))
	features = basis[truth] + 0.3 * rng.normal(size=(3000, 12))
	train = np.arange(3000) < 2000
	predicted = probe.fit_readout(features[train], truth[train], features[~train], truth[~train])
	metrics = probe.readout_metrics(truth[~train], predicted)
	assert metrics["accuracy"] > 0.9
	assert metrics["macro_recall"] > 0.9
	noise = rng.normal(size=(3000, 12))
	noisy = probe.fit_readout(noise[train], truth[train], noise[~train], truth[~train])
	assert probe.readout_metrics(truth[~train], noisy)["macro_recall"] < 0.35


def test_readout_metrics_report_immunity_recall_and_confusions():
	truth = np.array([0, 0, 1, 4, 5, 3])
	predicted = np.array([0, 3, 2, 2, 4, 3])
	metrics = probe.readout_metrics(truth, predicted)
	assert metrics["immunity_recall"] == pytest.approx(0.5)
	# Calling a super-effective move resisted (row 3) is the one confusion: 4 for 5 is not one.
	assert metrics["resisted_super_confusion"] == pytest.approx(1 / 6)
	assert metrics["rows"] == 6


def test_the_splits_hold_out_species_moves_and_dual_pairings():
	rng = np.random.default_rng(1)
	count = 600
	info = {
		"label": rng.integers(0, 6, size=count),
		"move": rng.integers(1, 30, size=count),
		"species": rng.integers(1, 40, size=count),
		"pair": np.sort(rng.integers(-1, 17, size=(count, 2)), axis=1),
	}
	result = probe.evaluate(rng.normal(size=(count, 8)), info)
	for name in ("species_and_moves", "dual_pairings"):
		assert 0 < result[name]["rows"] < count
		assert result[name]["rows"] + result[name]["train_rows"] == count


@pytest.mark.parametrize("arch", ("pairwise", "transformer"))
def test_the_probe_runs_on_a_small_checkpoint_and_on_the_initial_network(
	tables, tmp_path_factory, cpu_only, arch
):
	pool = write_pool(tmp_path_factory.mktemp("probe"), tables)
	config = tiny_config(f"network.arch={arch}", *ROSTER, *AUXILIARY)
	trainer = Trainer(config, pool)
	learner = trainer.init()
	checkpoint = tmp_path_factory.mktemp("checkpoint") / "checkpoint.npz"
	save_checkpoint(checkpoint, learner, config, trainer.team_digest)
	report = probe.run_probe(
		checkpoint, probe.read_config(checkpoint), battles=4, steps=6, seed=0, pool=pool
	)
	assert report["arch"] == arch
	for name in ("trained", "random"):
		assert set(report[name]) == {"species_and_moves", "dual_pairings"}
	initial = probe.run_probe(None, config, battles=4, steps=6, seed=0, pool=pool)
	assert "initial" in initial and "trained" not in initial
	with pytest.raises(probe.ProbeError):
		probe.run_probe(None, tiny_config(), battles=4, steps=2, seed=0, pool=pool)
	assert load_tables().types.index["water"] == type_index(tables, "water")
