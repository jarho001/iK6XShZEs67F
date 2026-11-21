"""Win rates of battles with Wilson 95% intervals."""

import math

Z95 = 1.96


def wilson_interval(successes: int, trials: int, z: float = Z95) -> tuple[float, float]:
	"""The Wilson score interval of a proportion (successes out of trials)."""
	if trials < 1:
		raise ValueError("trials must be positive")
	if not 0 <= successes <= trials:
		raise ValueError("successes must be between 0 and trials")
	p = successes / trials
	z2 = z * z
	denominator = 1 + z2 / trials
	center = (p + z2 / (2 * trials)) / denominator
	half = z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
	return center - half, center + half


def summarize_battles(
	opponent: str, battles: int, wins: int, truncated: int, mean_length: float
) -> dict:
	"""A learner's results against one opponent. Truncated battles count as not won."""
	low, high = wilson_interval(wins, battles)
	return {
		"opponent": opponent,
		"policy": "sampled",
		"battles": battles,
		"wins": wins,
		"win_rate": wins / battles,
		"ci95": [low, high],
		"truncated": truncated,
		"mean_battle_length": mean_length,
	}
