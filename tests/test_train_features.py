"""Public damage features: shapes, values on hand-built observations, agreement with the max-damage
scores of real battles, and no input beyond the observation (a foe's unrevealed move changes nothing).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from train_support import write_pool

from pokejax.baselines import (
	build_baseline_tables,
	max_damage_action,
	max_damage_scores,
	public_facts,
)
from pokejax.engine.actions import legal_actions
from pokejax.engine.observation import observe
from pokejax.env import Env, EnvConfig
from pokejax.train.config import FEATURE_SETS
from pokejax.train.features import (
	BENCH_COUNT,
	FEATURE_LAYOUTS,
	MOVE_COUNT,
	MOVE_FEATURES,
	SWITCH_FEATURES,
	THREAT_SCALE,
	action_features,
	build_public_tables,
	feature_widths,
	public_tables,
)

pytestmark = [pytest.mark.xdist_group("train"), pytest.mark.usefixtures("cpu_only")]

BATCH = 8
# Columns of a move feature row (pokejax/train/features.py).
POWER = 0
ACCURACY = 1
PHYSICAL = 3
STAB = 5
EFFECTIVENESS = 6
IMMUNE = 7
LEGAL = 8
RELATIVE = 10
LOG_SCORE = 11


def code(tables, type_name):
	"""The observation's type code of a type: its engine index + 1."""
	return tables.types.index[type_name] + 1


def move_id(tables, name):
	return tables.moves.index[name]


@pytest.fixture(scope="module")
def battles(tables, tmp_path_factory):
	"""Eight battles of random legal actions, 24 steps in: mid-battle states with reveals."""
	pool = write_pool(tmp_path_factory.mktemp("features"), tables)
	env = Env(EnvConfig(BATCH, observe="both"), pool, tables=tables)
	key = jax.random.key(11)
	es = env.init(env.new_battles(key), key)

	def step(es, k):
		logits = jnp.where(env.engine_mask(es), 0.0, -jnp.inf)
		es, _ = env.advance(es, jax.random.categorical(k, logits))
		return es, None

	es, _ = jax.jit(lambda es, keys: jax.lax.scan(step, es, keys))(
		es, jax.random.split(jax.random.key(12), 24)
	)
	state = es.state
	sides = jnp.arange(BATCH, dtype=jnp.int32) % 2
	obs = jax.jit(jax.vmap(observe, in_axes=(0, 0, None)))(state, sides, env.tables)
	return env.tables, state, sides, obs


@pytest.fixture(scope="module")
def features(battles):
	"""The move, switch and threat features of the battles' observations."""
	_, _, _, obs = battles
	return jax.jit(action_features)(obs, public_tables())


def test_public_tables_have_the_chart_and_the_species_columns(tables):
	public = build_public_tables(tables)
	count = len(tables.types.ids)
	assert public.chart.shape == (count + 1, count + 1)
	assert np.all(public.chart[count] == 1.0)  # no type: neutral
	assert public.species_base.shape == (len(tables.species.ids), 6)
	assert public.move_power.shape == (len(tables.moves.ids),)
	assert public_tables() is public_tables()  # built once


def test_shapes_and_values_are_finite_and_nonnegative(battles, features):
	moves, switches, threat = (np.asarray(x) for x in features)
	assert moves.shape == (BATCH, MOVE_COUNT, MOVE_FEATURES)
	assert switches.shape == (BATCH, BENCH_COUNT, SWITCH_FEATURES)
	assert threat.shape == (BATCH,)
	assert np.all(np.isfinite(moves)) and np.all(np.isfinite(switches))
	assert np.all(threat >= 0)


def hand_built(obs, own_types, foe_types, moves, foe_moves=(0, 0, 0, 0)):
	"""The first battle's observation with its active members replaced by hand (team slot 0)."""
	obs = jax.tree.map(lambda x: x[:1], obs)

	def row(values, width):
		return jnp.asarray(list(values) + [0] * (width - len(values)), jnp.int16)

	own = obs.own._replace(
		is_active=jnp.zeros_like(obs.own.is_active).at[0, 0].set(1.0),
		types=jnp.zeros_like(obs.own.types).at[0, 0].set(row(own_types, 2)),
		moves=jnp.zeros_like(obs.own.moves).at[0, 0].set(row(moves, 4)),
		hidden_type=jnp.zeros_like(obs.own.hidden_type),
		return_power=jnp.zeros_like(obs.own.return_power),
		pp=jnp.ones_like(obs.own.pp),
	)
	foe = obs.opponent._replace(
		is_active=jnp.zeros_like(obs.opponent.is_active).at[0, 0].set(1.0),
		types=jnp.zeros_like(obs.opponent.types).at[0, 0].set(row(foe_types, 2)),
		moves=jnp.zeros_like(obs.opponent.moves).at[0, 0].set(row(foe_moves, 4)),
	)
	legal = jnp.zeros_like(obs.legal_actions).at[0, : len(moves)].set(True)
	return obs._replace(own=own, opponent=foe, legal_actions=legal)


def test_effectiveness_stab_and_power_on_a_hand_built_position(battles, tables):
	_, _, _, obs = battles
	hand = hand_built(
		obs,
		own_types=[code(tables, "electric")],
		foe_types=[code(tables, "water")],
		moves=[move_id(tables, "thunderbolt"), move_id(tables, "tackle")],
	)
	moves, _, _ = action_features(hand, public_tables())
	moves = np.asarray(moves)[0]
	thunderbolt, tackle = moves[0], moves[1]
	# Electric on Water is super effective (2), Normal on Water neutral (1).
	assert thunderbolt[EFFECTIVENESS] == pytest.approx(1.0)
	assert tackle[EFFECTIVENESS] == pytest.approx(0.5)
	assert thunderbolt[STAB] == 1.0 and tackle[STAB] == 0.0
	assert thunderbolt[IMMUNE] == 0.0
	assert thunderbolt[POWER] == pytest.approx(
		tables.moves.base_power[move_id(tables, "thunderbolt")] / 100
	)
	assert thunderbolt[PHYSICAL] == 0.0 and tackle[PHYSICAL] == 1.0
	assert thunderbolt[LEGAL] == 1.0 and moves[2, LEGAL] == 0.0
	assert np.all(moves[2:, POWER] == 0.0)  # empty slots carry no power


def test_hidden_power_and_return_take_their_type_and_power_from_the_request(battles, tables):
	_, _, _, obs = battles
	hand = hand_built(
		obs,
		own_types=[code(tables, "normal")],
		foe_types=[code(tables, "rock")],
		moves=[move_id(tables, "hiddenpower"), move_id(tables, "return")],
	)
	# The request says Hidden Power is Fire and Return has power 102 (happiness 255). A Normal-type
	# own active with Return is the STAB case; Hidden Power of another type gets no STAB.
	fire = code(tables, "fire")
	hand = hand._replace(
		own=hand.own._replace(
			hidden_type=hand.own.hidden_type.at[0, 0, 0].set(fire + 1),
			return_power=hand.own.return_power.at[0, 0, 1].set(102),
		)
	)
	moves, _, _ = action_features(hand, public_tables())
	moves = np.asarray(moves)[0]
	hidden, returned = moves[0], moves[1]
	# Fire and Normal on Rock are both resisted (1/2, a feature of 0.25); Return has the request's power.
	assert hidden[EFFECTIVENESS] == pytest.approx(0.25) and hidden[STAB] == 0.0
	assert returned[POWER] == pytest.approx(1.02) and returned[EFFECTIVENESS] == pytest.approx(0.25)
	assert returned[STAB] == 1.0


def test_an_immune_move_scores_nothing_and_the_neutral_one_is_the_best(battles, tables):
	_, _, _, obs = battles
	hand = hand_built(
		obs,
		own_types=[code(tables, "normal")],
		foe_types=[code(tables, "flying")],
		moves=[move_id(tables, "earthquake"), move_id(tables, "tackle")],
	)
	moves, _, _ = action_features(hand, public_tables())
	moves = np.asarray(moves)[0]
	assert moves[0, IMMUNE] == 1.0 and moves[0, EFFECTIVENESS] == 0.0
	assert moves[0, RELATIVE] == 0.0 and moves[0, LOG_SCORE] == 0.0
	assert moves[1, RELATIVE] == pytest.approx(1.0)


def test_threat_is_the_revealed_move_damage_against_the_own_active(battles, tables):
	_, _, _, obs = battles
	quake = move_id(tables, "earthquake")
	hand = hand_built(
		obs,
		own_types=[code(tables, "normal")],
		foe_types=[code(tables, "ground")],
		moves=[move_id(tables, "tackle")],
		foe_moves=[quake],
	)
	_, _, threat = action_features(hand, public_tables())
	# Ground STAB (1.5) on a neutral Normal member, against the table's power, over the scale.
	expected = tables.moves.base_power[quake] * 1.5 / THREAT_SCALE
	assert float(threat[0]) == pytest.approx(expected)
	silent = hand_built(
		obs,
		own_types=[code(tables, "normal")],
		foe_types=[code(tables, "ground")],
		moves=[move_id(tables, "tackle")],
	)
	_, _, threat = action_features(silent, public_tables())
	assert float(threat[0]) == 0.0  # nothing revealed, no threat


def test_move_scores_agree_with_the_max_damage_baseline(battles, features, tables):
	engine, state, sides, obs = battles
	data = build_baseline_tables(tables)

	def score_and_choice(s, side):
		facts = public_facts(s, side, engine)
		mask = legal_actions(s, engine)[side]
		return max_damage_scores(s, side, facts, data, engine), max_damage_action(
			s, side, mask, engine, data
		)

	scores, choice = jax.jit(jax.vmap(score_and_choice))(state, sides)
	moves = np.asarray(features[0])
	scores, choice = np.asarray(scores), np.asarray(choice)
	legal = np.asarray(obs.legal_actions)[:, :MOVE_COUNT]
	hidden_power = move_id(tables, "hiddenpower")
	return_move = move_id(tables, "return")
	checked = 0
	for b in range(BATCH):
		active = int(np.argmax(np.asarray(obs.own.is_active)[b]))
		ids = np.asarray(obs.own.moves)[b, active]
		for slot in range(MOVE_COUNT):
			# The baseline keeps its static formula: Hidden Power's type and Return's power are
			# not in its table, while the features read them from the own request.
			if not legal[b, slot] or ids[slot] in (hidden_power, return_move):
				continue
			decoded = np.expm1(5 * moves[b, slot, LOG_SCORE])
			assert decoded == pytest.approx(scores[b, slot], rel=1e-4, abs=1e-3)
			checked += 1
		md = choice[b]
		# The relative score is against the features' own best move, which differs from the baseline's
		# when a legal move is Hidden Power or Return (see above).
		uses_request = any(
			legal[b, slot] and ids[slot] in (hidden_power, return_move)
			for slot in range(MOVE_COUNT)
		)
		if md < MOVE_COUNT and scores[b, md] > 0 and not uses_request:
			assert moves[b, md, RELATIVE] == pytest.approx(1.0, abs=1e-5)
	assert checked > 0


def test_no_input_beyond_the_observation(battles, tables):
	"""A foe's unrevealed move is not in the observation, so it changes no feature."""
	engine, state, _, _ = battles
	battle = jax.tree.map(lambda x: x[0], state)
	# Side 0 is the learner; side 1's active member's moves that nothing has revealed yet.
	foe_active = int(battle.active[1])
	unrevealed = np.asarray(battle.revealed_moves[1, foe_active]) == 0
	assert unrevealed.any()
	quake = move_id(tables, "earthquake")
	moves = np.asarray(battle.moves).copy()
	moves[1, foe_active] = np.where(unrevealed, quake, moves[1, foe_active])
	changed = battle._replace(moves=jnp.asarray(moves))
	assert not np.array_equal(np.asarray(battle.moves), moves)
	before = observe(battle, 0, engine)
	after = observe(changed, 0, engine)
	same = jax.tree.map(
		lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b))), before, after
	)
	assert all(jax.tree.leaves(same))
	a = jax.tree.map(lambda x: x[None], before)
	b = jax.tree.map(lambda x: x[None], after)
	for x, y in zip(
		action_features(a, public_tables()), action_features(b, public_tables()), strict=True
	):
		np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_feature_widths_per_set(battles):
	_, _, _, obs = battles
	widths = {"full": (12, 8, True), "dex": (10, 4, False), "dex-no-matchup": (10, 2, False)}
	assert set(widths) == set(FEATURE_SETS) == set(FEATURE_LAYOUTS)
	extract = jax.jit(action_features, static_argnames="feature_set")
	for feature_set, (move_width, switch_width, threat) in widths.items():
		moves, switches, threat_out = extract(obs, public_tables(), feature_set=feature_set)
		assert moves.shape == (BATCH, MOVE_COUNT, move_width)
		assert switches.shape == (BATCH, BENCH_COUNT, switch_width)
		assert (threat_out is None) == (not threat)
		assert feature_widths(feature_set) == (move_width, switch_width, int(threat))


def test_full_is_the_default_and_the_smaller_sets_keep_its_columns(battles):
	_, _, _, obs = battles
	full = action_features(obs, public_tables())
	assert full[0].shape[-1] == MOVE_FEATURES and full[1].shape[-1] == SWITCH_FEATURES
	for x, y in zip(full, action_features(obs, public_tables(), "full"), strict=True):
		np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
	for feature_set in ("dex", "dex-no-matchup"):
		layout = FEATURE_LAYOUTS[feature_set]
		moves, switches, threat = action_features(obs, public_tables(), feature_set)
		np.testing.assert_array_equal(
			np.asarray(moves), np.asarray(full[0])[..., list(layout.move)]
		)
		np.testing.assert_array_equal(
			np.asarray(switches), np.asarray(full[1])[..., list(layout.switch)]
		)
		assert threat is None


def test_dex_features_do_not_see_the_opponents_boosts_or_hp(battles):
	"""The dex sets leave out the damage estimates and the matchups, the only inputs that use the
	opponent's boosts, and the opponent's HP; the full set does see both."""
	_, _, _, obs = battles
	changed = obs._replace(
		opponent=obs.opponent._replace(
			boosts=jnp.full_like(obs.opponent.boosts, 2.0 / 6.0), hp=obs.opponent.hp * 0.5
		)
	)
	before = action_features(obs, public_tables())
	after = action_features(changed, public_tables())
	assert not np.array_equal(np.asarray(before[0]), np.asarray(after[0]))
	assert not np.array_equal(np.asarray(before[1]), np.asarray(after[1]))
	for feature_set in ("dex", "dex-no-matchup"):
		old = action_features(obs, public_tables(), feature_set)
		new = action_features(changed, public_tables(), feature_set)
		np.testing.assert_array_equal(np.asarray(old[0]), np.asarray(new[0]))
		np.testing.assert_array_equal(np.asarray(old[1]), np.asarray(new[1]))
		assert old[2] is None and new[2] is None
