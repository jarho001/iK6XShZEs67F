"""The training-only auxiliary losses of Phase 6: the next action, the hidden item, ability and moves.

The heads are outputs of the roster networks (roster.py) over the normalised GRU output and the foe
active vector. Their targets are built in the league from the battle state before each step (league.py,
AuxLabels) for learner lanes only and never enter the observation, the policy, the value or the carry.
Eligibility is decided here from the labels and from public fields of the observation only, so a head
trains on what the rule allows and nothing else.

Targets and weights (AuxiliarySection): the next action is the opposing side's action at the same
decision (move or switch kind, then the move ID or the public switch rank); the item and the ability are
the opposing active's current held item and ability; the moves are the active's hidden move IDs (a
membership over the move vocabulary, balanced between hidden and other candidate moves). Each loss is
the mean over its eligible decisions. Metrics are sums and counts, turned into rates by
finalize_auxiliary.
"""

import jax
import jax.numpy as jnp

from pokejax.train.config import AuxiliarySection

KIND_MOVE = 0
KIND_SWITCH = 1
SWITCH_KINDS = 2
RANK_OUTPUTS = 7  # six public ranks, then the unseen bucket
F32 = jnp.float32


def _pick(x, weights):
	"""Sum of x [N, 6, *rest] over the member axis with float weights [N, 6]: [N, *rest]."""
	axis = weights.ndim - 1
	expanded = weights.reshape(weights.shape + (1,) * (x.ndim - weights.ndim))
	return jnp.sum(x * expanded, axis=axis)


def _cross_entropy(logits, target):
	"""Per-row cross-entropy of logits [N, V] against integer targets [N]."""
	log_probs = jax.nn.log_softmax(logits, axis=-1)
	return -jnp.take_along_axis(log_probs, target[..., None], axis=-1)[..., 0]


def _mean_over(values, mask):
	"""The mean of values [N] over the rows where mask [N] holds (zero when none)."""
	total = jnp.sum(jnp.where(mask, values, 0.0))
	return total / jnp.maximum(mask.sum().astype(F32), 1.0)


def auxiliary_terms(heads: dict, labels, obs, weights: AuxiliarySection):
	"""The weighted auxiliary loss over N decisions and its metric sums.

	heads maps each enabled target to its logits [N, V]; labels is an AuxLabels (see league.py) and
	obs the observation of the same decisions, read only for its public fields.
	"""
	metrics = {}
	total = jnp.zeros((), F32)
	steps = jnp.asarray(labels.private.shape[0], F32)
	metrics["aux_steps"] = steps
	active = obs.opponent.is_active.astype(F32)  # [N, 6], one-hot on the opposing active
	has_active = active.sum(-1) > 0
	item_known = (
		obs.opponent.item_revealed + obs.opponent.item_consumed + obs.opponent.item_knocked_off
	)
	item_known = _pick(item_known.astype(F32), active) > 0
	ability_known = _pick(obs.opponent.ability.astype(F32), active) > 0
	revealed = _pick(obs.opponent.moves.astype(F32), active).astype(jnp.int32)  # [N, 4]

	if "next_action" in heads:
		logits = heads["next_action"]
		move_count = logits.shape[-1] - SWITCH_KINDS - RANK_OUTPUTS
		kind_logits = logits[..., :SWITCH_KINDS]
		move_logits = logits[..., SWITCH_KINDS : SWITCH_KINDS + move_count]
		rank_logits = logits[..., SWITCH_KINDS + move_count :]
		kind_ok = labels.next_kind >= 0
		kind_target = jnp.clip(labels.next_kind, 0, 1).astype(jnp.int32)
		move_ok = (
			kind_ok & (labels.next_kind == KIND_MOVE) & labels.identity & (labels.next_move > 0)
		)
		rank_ok = kind_ok & (labels.next_kind == KIND_SWITCH) & labels.identity
		identity_ok = move_ok | rank_ok
		move_target = labels.next_move.astype(jnp.int32)
		rank_target = labels.next_rank.astype(jnp.int32)
		kind_loss = _mean_over(_cross_entropy(kind_logits, kind_target), kind_ok)
		identity_ce = jnp.where(
			move_ok,
			_cross_entropy(move_logits, move_target),
			_cross_entropy(rank_logits, rank_target),
		)
		identity_loss = _mean_over(identity_ce, identity_ok)
		loss = kind_loss + identity_loss
		total += weights.next_action * loss
		metrics["aux_next_action_loss"] = loss
		metrics["aux_kind_n"] = kind_ok.sum().astype(F32)
		metrics["aux_kind_correct"] = (
			(kind_ok & (jnp.argmax(kind_logits, axis=-1) == kind_target)).sum().astype(F32)
		)
		predicted_move = jnp.argmax(move_logits, axis=-1) == move_target
		predicted_rank = jnp.argmax(rank_logits, axis=-1) == rank_target
		metrics["aux_identity_n"] = identity_ok.sum().astype(F32)
		metrics["aux_identity_correct"] = (
			(identity_ok & jnp.where(move_ok, predicted_move, predicted_rank)).sum().astype(F32)
		)

	if "item" in heads:
		ok = labels.private & has_active & ~item_known
		target = labels.item.astype(jnp.int32)
		loss = _mean_over(_cross_entropy(heads["item"], target), ok)
		total += weights.item * loss
		metrics["aux_item_loss"] = loss
		metrics["aux_item_n"] = ok.sum().astype(F32)
		metrics["aux_item_correct"] = (
			(ok & (jnp.argmax(heads["item"], axis=-1) == target)).sum().astype(F32)
		)

	if "ability" in heads:
		ok = labels.private & has_active & ~ability_known
		target = labels.ability.astype(jnp.int32)
		loss = _mean_over(_cross_entropy(heads["ability"], target), ok)
		total += weights.ability * loss
		metrics["aux_ability_loss"] = loss
		metrics["aux_ability_n"] = ok.sum().astype(F32)
		metrics["aux_ability_correct"] = (
			(ok & (jnp.argmax(heads["ability"], axis=-1) == target)).sum().astype(F32)
		)

	if "moves" in heads:
		logits = heads["moves"]
		ids = jnp.arange(logits.shape[-1])
		hidden = labels.moves.astype(jnp.int32)
		hidden_hits = ((hidden[..., None] == ids) & (hidden[..., None] > 0)).any(axis=-2)
		shown = revealed[..., None] == ids
		shown_hits = (shown & (revealed[..., None] > 0)).any(axis=-2)
		candidate = (ids > 0)[None, :] & ~shown_hits
		positive = hidden_hits & candidate
		negative = candidate & ~positive
		ok = labels.private & has_active & positive.any(axis=-1)
		# Binary cross-entropy against the target, per candidate.
		bce = jnp.maximum(logits, 0.0) - logits * positive + jnp.log1p(jnp.exp(-jnp.abs(logits)))
		positive_mean = (bce * positive).sum(-1) / jnp.maximum(positive.sum(-1), 1.0)
		negative_mean = (bce * negative).sum(-1) / jnp.maximum(negative.sum(-1), 1.0)
		per_row = 0.5 * (positive_mean + negative_mean)
		loss = _mean_over(per_row, ok)
		total += weights.moves * loss
		predicted = (logits > 0) & candidate
		metrics["aux_moves_loss"] = loss
		metrics["aux_moves_n"] = ok.sum().astype(F32)
		metrics["aux_moves_true_positive"] = (ok[:, None] & predicted & positive).sum().astype(F32)
		metrics["aux_moves_predicted"] = (ok[:, None] & predicted).sum().astype(F32)
		metrics["aux_moves_positive"] = (ok[:, None] & positive).sum().astype(F32)
	return total, metrics


RAW_COUNTS = (
	"aux_steps",
	"aux_kind_n",
	"aux_kind_correct",
	"aux_identity_n",
	"aux_identity_correct",
	"aux_item_n",
	"aux_item_correct",
	"aux_ability_n",
	"aux_ability_correct",
	"aux_moves_n",
	"aux_moves_true_positive",
	"aux_moves_predicted",
	"aux_moves_positive",
)


def _rate(numerator, denominator):
	return numerator / jnp.maximum(denominator, 1.0)


def finalize_auxiliary(metrics: dict) -> dict:
	"""The metrics with the counts turned into eligible fractions, accuracies and move precision and recall.

	The counts are the means over the minibatches, so their ratios are the ratios of the totals.
	"""
	if "aux_steps" not in metrics:
		return metrics
	result = {k: v for k, v in metrics.items() if k not in RAW_COUNTS}
	steps = metrics["aux_steps"]
	for name in ("kind", "identity", "item", "ability", "moves"):
		key = f"aux_{name}_n"
		if key in metrics:
			result[f"aux_{name}_eligible"] = _rate(metrics[key], steps)
	if "aux_kind_n" in metrics:
		result["aux_kind_top1"] = _rate(metrics["aux_kind_correct"], metrics["aux_kind_n"])
	if "aux_identity_n" in metrics:
		result["aux_identity_top1"] = _rate(
			metrics["aux_identity_correct"], metrics["aux_identity_n"]
		)
	for name in ("item", "ability"):
		if f"aux_{name}_n" in metrics:
			result[f"aux_{name}_top1"] = _rate(
				metrics[f"aux_{name}_correct"], metrics[f"aux_{name}_n"]
			)
	if "aux_moves_n" in metrics:
		result["aux_moves_precision"] = _rate(
			metrics["aux_moves_true_positive"], metrics["aux_moves_predicted"]
		)
		result["aux_moves_recall"] = _rate(
			metrics["aux_moves_true_positive"], metrics["aux_moves_positive"]
		)
	return result
