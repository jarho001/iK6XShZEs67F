"""Win-rate curves of training runs, as SVG (or HTML around the same SVG).

	python -m pokejax.train.plot RUN_DIR... [--opponent NAME] [--policy sampled|greedy]
		[--x updates|minutes|steps] [--training] [--title TEXT] --out FILE.svg|FILE.html

Each run directory is one that pokejax.train.run writes. Its metrics.jsonl gives:

- the evaluation points: the win rate against --opponent with --policy at each evaluation record
  (evaluation.every updates apart, 1,024 battles per point). The final evaluation of 4,096 battles
  is not drawn; the table next to the figure carries it;
- with --training, the learner's rollout win rate against that baseline in each logged update
  (the update records' win_rate_NAME and completed_NAME), pooled over TRAINING_WINDOW updates and
  drawn as a thin line behind the evaluation points;
- with --x minutes, the update records' cumulative "seconds" (wall time, including the periodic
  evaluations and checkpoints), interpolated at the evaluation points. A resumed run's clock starts
  again at the resumption, so its later times continue from the last time before it.
- with --x steps, the update records' env_steps (environment steps, comparable across runs with
  different batch sizes), interpolated at the evaluation points in the same way.

One line per run, in the order the directories are given: run i takes colour slot i of the
reference palette (the dataviz skill's palette.md), so a run keeps its colour across figures. A
legend names the runs whenever there are two or more; direct end labels are added for up to four
runs that end together and do not collide. A dashed line marks the 60% exit target. The SVG is
written here, without a plotting library; its dark variant follows prefers-color-scheme.
"""

import argparse
import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

METRICS_FILE = "metrics.jsonl"
CONFIG_FILE = "config.json"
TRAINING_WINDOW = 100  # updates pooled into one smoothed training win rate
EXIT_TARGET = 60.0  # percent: the dashed line
DIRECT_LABEL_LIMIT = 4  # at most this many runs carry direct end labels
LABEL_GAP = 14  # pixels between two direct labels; closer, and the legend alone names the runs

# The categorical slots of the reference palette (palette.md), light and dark steps of the same
# eight hues. Slot i is the i-th run given on the command line.
THEMES = {
	"light": {
		"surface": "#fcfcfb",
		"primary": "#0b0b0b",
		"secondary": "#52514e",
		"muted": "#898781",
		"grid": "#e1e0d9",
		"axis": "#c3c2b7",
		"slots": (
			"#2a78d6",
			"#eb6834",
			"#1baf7a",
			"#eda100",
			"#e87ba4",
			"#008300",
			"#6250d6",
			"#e34948",
		),
	},
	"dark": {
		"surface": "#1a1a19",
		"primary": "#f0efec",
		"secondary": "#c3c2b7",
		"muted": "#898781",
		"grid": "#2c2c2a",
		"axis": "#383835",
		"slots": (
			"#3987e5",
			"#d95926",
			"#199e70",
			"#c98500",
			"#d55181",
			"#0ca30c",
			"#9085e9",
			"#e66767",
		),
	},
}
MAX_RUNS = len(THEMES["light"]["slots"])

FIGURE_WIDTH = 840
PLOT_LEFT = 72
PLOT_RIGHT = 700
PLOT_HEIGHT = 300
LEGEND_TOP = 46


class PlotError(ValueError):
	"""Run directories that cannot be plotted as asked."""


@dataclass
class Series:
	"""One run's line: its evaluation points (x, percent, tooltip) and its training points."""

	name: str
	evaluation: list[tuple[float, float, str]]
	training: list[tuple[float, float]] = field(default_factory=list)


def _css(theme: dict) -> str:
	rules = [
		f".surface {{ fill: {theme['surface']}; }}",
		f".primary {{ fill: {theme['primary']}; }}",
		f".secondary {{ fill: {theme['secondary']}; }}",
		f".muted {{ fill: {theme['muted']}; }}",
		f".grid {{ stroke: {theme['grid']}; stroke-width: 1; }}",
		f".axis {{ stroke: {theme['axis']}; stroke-width: 1; }}",
		f".ring {{ stroke: {theme['surface']}; stroke-width: 2; }}",
		f".ref {{ stroke: {theme['muted']}; stroke-width: 1.5; stroke-dasharray: 6 4; }}",
	]
	for index, colour in enumerate(theme["slots"]):
		rules.append(f".k{index} {{ stroke: {colour}; }}")
		rules.append(f".f{index} {{ fill: {colour}; }}")
	return " ".join(rules)


_BASE_CSS = (
	'text { font-family: system-ui, "Segoe UI", sans-serif; }'
	" .line { fill: none; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; }"
	" .train { fill: none; stroke-width: 1.25; stroke-linecap: round; stroke-linejoin: round;"
	" opacity: 0.5; }"
)
_STYLE = (
	"<style>"
	+ _BASE_CSS
	+ " "
	+ _css(THEMES["light"])
	+ " @media (prefers-color-scheme: dark) { "
	+ _css(THEMES["dark"])
	+ " }</style>"
)


def _nice_step(span: float, target: int) -> float:
	"""A step of 1, 2, 2.5 or 5 times a power of ten giving about target intervals over span."""
	raw = max(span, 1e-9) / target
	magnitude = 10 ** math.floor(math.log10(raw))
	for factor in (1, 2, 2.5, 5, 10):
		if factor * magnitude >= raw:
			return factor * magnitude
	return 10 * magnitude


def _ticks(low: float, high: float, step: float) -> list[float]:
	count = round((high - low) / step)
	return [round(low + index * step, 9) for index in range(count + 1)]


def _x_domain(xs: list[float]) -> tuple[float, float, float]:
	high = max(xs)
	step = _nice_step(high, 6)
	top = max(math.ceil(high / step - 1e-9), 1) * step
	return 0.0, top, step


def _y_domain(ys: list[float]) -> tuple[float, float, float]:
	low = min([*ys, EXIT_TARGET])
	high = max([*ys, EXIT_TARGET])
	pad = max(0.05 * (high - low), 0.5)
	step = _nice_step(high - low + 2 * pad, 8)
	bottom = max(math.floor((low - pad) / step) * step, 0.0)
	top = min(math.ceil((high + pad) / step) * step, 100.0)
	return bottom, top, step


def _records(run_dir: Path) -> list[dict]:
	path = run_dir / METRICS_FILE
	if not path.exists():
		raise PlotError(f"{path} does not exist")
	lines = path.read_text(encoding="utf-8").splitlines()
	return [json.loads(line) for line in lines if line.strip()]


def _run_name(run_dir: Path) -> str:
	config = run_dir / CONFIG_FILE
	if config.exists():
		name = json.loads(config.read_text(encoding="utf-8")).get("run", {}).get("name")
		if name:
			return str(name)
	return run_dir.name


def _clock(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
	"""The update numbers and the wall minutes of the update records, continued across resumptions."""
	updates = np.array([row["update"] for row in rows], dtype=float)
	minutes = np.zeros(len(rows))
	offset = 0.0
	previous = 0.0
	for index, row in enumerate(rows):
		seconds = float(row["seconds"])
		if index and seconds < previous:  # a resumed run's clock starts again from zero
			offset += previous
		previous = seconds
		minutes[index] = (seconds + offset) / 60.0
	return updates, minutes


def _pooled(updates: np.ndarray, won: np.ndarray, ended: np.ndarray, window: float) -> np.ndarray:
	"""The win rate pooled over the updates within window / 2 of each one (NaN without battles)."""
	rates = np.full(len(updates), np.nan)
	for index, update in enumerate(updates):
		near = np.abs(updates - update) <= window / 2
		total = ended[near].sum()
		if total > 0:
			rates[index] = won[near].sum() / total
	return rates


def run_series(run_dir: Path, *, opponent: str, policy: str, x_mode: str, training: bool) -> Series:
	"""The line of one run: its evaluation points and, with training, its smoothed training rates."""
	records = _records(run_dir)
	name = _run_name(run_dir)
	rows = [record for record in records if record["kind"] == "update"]
	if x_mode == "minutes" and not rows:
		raise PlotError(f"{run_dir} has no update records, so no wall time")
	if x_mode == "steps" and any("env_steps" not in row for row in rows):
		raise PlotError(f"{run_dir} has update records without env_steps")
	updates, minutes = _clock(rows)
	steps = np.array([row.get("env_steps", 0) for row in rows], dtype=float)

	def position(update: float) -> float:
		if x_mode == "updates":
			return float(update)
		if x_mode == "steps":
			return float(np.interp(update, updates, steps))
		return float(np.interp(update, updates, minutes))

	evaluation = []
	for record in records:
		if record["kind"] != "evaluation":
			continue
		update = record["update"]
		results = [
			result
			for result in record["results"]
			if result["opponent"] == opponent and result["policy"] == policy
		]
		if not results:
			raise PlotError(f"{run_dir}: no {policy} evaluation against {opponent} at {update}")
		result = results[0]
		rate = 100 * result["win_rate"]
		low, high = (100 * bound for bound in result["ci95"])
		when = {
			"updates": f"update {update:,}",
			"steps": f"{position(update) / 1e6:.1f}M steps",
		}.get(x_mode, f"{position(update):.1f} min")
		tooltip = (
			f"{name}, {when}: {rate:.2f}% against {opponent} ({policy}, "
			f"{result['battles']:,} battles, 95% CI {low:.2f} to {high:.2f})"
		)
		evaluation.append((position(update), rate, tooltip))
	if not evaluation:
		raise PlotError(f"{run_dir} has no evaluation records")

	points = []
	if training:
		rate_key, count_key = f"win_rate_{opponent}", f"completed_{opponent}"
		if not rows or any(rate_key not in row or count_key not in row for row in rows):
			raise PlotError(f"{run_dir} has no training win rates against {opponent}")
		ended = np.array([row[count_key] for row in rows], dtype=float)
		won = np.array([row[rate_key] * row[count_key] for row in rows], dtype=float)
		for update, rate in zip(
			updates, _pooled(updates, won, ended, TRAINING_WINDOW), strict=True
		):
			if not np.isnan(rate):
				points.append((position(update), 100 * float(rate)))
	return Series(name, evaluation, points)


def _legend(names: list[str]) -> tuple[list[tuple[float, float, str, int]], float]:
	"""Where each run's legend entry sits (x, text baseline, name, slot) and the legend's bottom."""
	placed = []
	x = PLOT_LEFT
	row = 0
	for index, name in enumerate(names):
		width = 26 + 7.2 * len(name) + 22
		if x > PLOT_LEFT and x + width > FIGURE_WIDTH - 20:
			row += 1
			x = PLOT_LEFT
		placed.append((x, LEGEND_TOP + 22 * row + 12, name, index))
		x += width
	return placed, LEGEND_TOP + 22 * row + 22


def render(series: list[Series], *, title: str, x_label: str, x_mode: str) -> str:
	"""The figure as SVG: title, legend, grid, the 60% line, training and evaluation lines."""
	xs = [x for line in series for x, _, _ in line.evaluation]
	xs += [x for line in series for x, _ in line.training]
	ys = [y for line in series for _, y, _ in line.evaluation]
	ys += [y for line in series for _, y in line.training]
	x_low, x_high, x_step = _x_domain(xs)
	y_low, y_high, y_step = _y_domain(ys)
	if len(series) > 1:
		legend, legend_bottom = _legend([line.name for line in series])
	else:
		legend, legend_bottom = [], 30.0  # one run: the title names it
	top = legend_bottom + 14
	bottom = top + PLOT_HEIGHT
	height = bottom + 62

	def X(x: float) -> float:
		return PLOT_LEFT + (x - x_low) / (x_high - x_low) * (PLOT_RIGHT - PLOT_LEFT)

	def Y(y: float) -> float:
		return bottom - (y - y_low) / (y_high - y_low) * (bottom - top)

	parts = [
		(
			f'<svg xmlns="http://www.w3.org/2000/svg" width="{FIGURE_WIDTH}" height="{height:.0f}" '
			f'viewBox="0 0 {FIGURE_WIDTH} {height:.0f}" role="img">'
		),
		f"<title>{escape(title)}</title>",
		_STYLE,
		f'<rect width="{FIGURE_WIDTH}" height="{height:.0f}" class="surface"/>',
		(
			f'<text x="{PLOT_LEFT}" y="26" font-size="15" font-weight="600" class="primary">'
			f"{escape(title)}</text>"
		),
	]
	for x, baseline, name, index in legend:
		parts += [
			(
				f'<line x1="{x}" y1="{baseline - 4}" x2="{x + 18}" y2="{baseline - 4}" '
				f'class="line k{index}"/>'
			),
			f'<circle cx="{x + 9}" cy="{baseline - 4}" r="4" class="f{index} ring"/>',
			f'<text x="{x + 26}" y="{baseline}" font-size="12" class="primary">{escape(name)}</text>',
		]
	for value in _ticks(y_low, y_high, y_step):
		parts += [
			(
				f'<line x1="{PLOT_LEFT}" y1="{Y(value):.1f}" x2="{PLOT_RIGHT}" y2="{Y(value):.1f}" '
				'class="grid"/>'
			),
			(
				f'<text x="{PLOT_LEFT - 8}" y="{Y(value) + 4:.1f}" font-size="11" text-anchor="end" '
				f'class="secondary">{value:g}%</text>'
			),
		]
	parts += [
		(
			f'<line x1="{PLOT_LEFT}" y1="{Y(EXIT_TARGET):.1f}" x2="{PLOT_RIGHT}" '
			f'y2="{Y(EXIT_TARGET):.1f}" class="ref"/>'
		),
		(
			f'<text x="{PLOT_RIGHT - 4}" y="{Y(EXIT_TARGET) - 6:.1f}" font-size="11" '
			'text-anchor="end" class="muted">60%</text>'
		),
	]
	for value in _ticks(x_low, x_high, x_step):
		label = {"updates": f"{value:,.0f}", "steps": f"{value / 1e6:g}M"}.get(x_mode, f"{value:g}")
		parts.append(
			f'<text x="{X(value):.1f}" y="{bottom + 18}" font-size="11" text-anchor="middle" '
			f'class="secondary">{label}</text>'
		)
	parts += [
		f'<line x1="{PLOT_LEFT}" y1="{bottom}" x2="{PLOT_RIGHT}" y2="{bottom}" class="axis"/>',
		(
			f'<text x="{(PLOT_LEFT + PLOT_RIGHT) / 2}" y="{bottom + 42}" font-size="12" '
			f'text-anchor="middle" class="secondary">{escape(x_label)}</text>'
		),
		(
			f'<text transform="translate(16 {(top + bottom) / 2:.1f}) rotate(-90)" font-size="12" '
			'text-anchor="middle" class="secondary">win rate (%)</text>'
		),
	]
	for index, line in enumerate(series):
		if line.training:
			points = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in line.training)
			parts.append(f'<polyline points="{points}" class="train k{index}"/>')
	for index, line in enumerate(series):
		points = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y, _ in line.evaluation)
		parts.append(f'<polyline points="{points}" class="line k{index}"/>')
		for x, y, tooltip in line.evaluation:
			parts.append(
				f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" class="f{index} ring">'
				f"<title>{escape(tooltip)}</title></circle>"
			)
	last = max(x for line in series for x, _, _ in line.evaluation)
	ends = sorted(
		(Y(line.evaluation[-1][1]), index, line.name)
		for index, line in enumerate(series)
		if line.evaluation
	)
	ends_together = all(line.evaluation[-1][0] == last for line in series)
	spaced = all(later[0] - earlier[0] >= LABEL_GAP for earlier, later in itertools.pairwise(ends))
	if len(series) <= DIRECT_LABEL_LIMIT and ends_together and spaced:
		for y_pixel, index, name in ends:
			parts += [
				(
					f'<line x1="{X(last) + 8:.1f}" y1="{y_pixel:.1f}" x2="{X(last) + 18:.1f}" '
					f'y2="{y_pixel:.1f}" class="line k{index}"/>'
				),
				(
					f'<text x="{X(last) + 22:.1f}" y="{y_pixel + 4:.1f}" font-size="12" '
					f'class="primary">{escape(name)}</text>'
				),
			]
	parts.append("</svg>")
	return "\n".join(parts) + "\n"


def page(svg: str, title: str) -> str:
	"""The same figure in an HTML page (light or dark with the viewer's preference)."""
	return (
		'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
		'<meta name="color-scheme" content="light dark">\n'
		f"<title>{escape(title)}</title>\n"
		"<style>body { margin: 0; padding: 16px; background: #f9f9f7; }"
		" @media (prefers-color-scheme: dark) { body { background: #0d0d0d; } }"
		" svg { display: block; max-width: 100%; height: auto; }</style>\n"
		f"</head>\n<body>\n{svg}</body>\n</html>\n"
	)


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		prog="python -m pokejax.train.plot",
		description="Win rates of training runs over training, as SVG or HTML (no plotting library).",
	)
	parser.add_argument(
		"runs",
		nargs="+",
		type=Path,
		metavar="RUN_DIR",
		help="a run directory (its metrics.jsonl); one line per run, in this order",
	)
	parser.add_argument(
		"--opponent", default="max-damage", help="the baseline (default max-damage)"
	)
	parser.add_argument(
		"--policy",
		choices=("sampled", "greedy"),
		default="sampled",
		help="how the learner chose its actions in the evaluation (default sampled)",
	)
	parser.add_argument(
		"--x",
		dest="x_mode",
		choices=("updates", "minutes", "steps"),
		default="updates",
		help="the horizontal axis: updates, wall minutes or environment steps (default updates)",
	)
	parser.add_argument(
		"--training",
		action="store_true",
		help="add the smoothed training win rates against the baseline",
	)
	parser.add_argument("--title", help="the figure's title (default: the baseline and policy)")
	parser.add_argument("--out", required=True, type=Path, help="an .svg or .html file")
	return parser


def main(argv=None) -> int:
	parser = build_parser()
	args = parser.parse_args(argv)
	if args.out.suffix not in (".svg", ".html"):
		parser.error("--out must be an .svg or .html file")
	if len(args.runs) > MAX_RUNS:
		parser.error(f"at most {MAX_RUNS} runs: the palette has {MAX_RUNS} colour slots")
	title = args.title or f"Win rate against {args.opponent} ({args.policy} actions)"
	if args.training:
		title += ", with training win rates"
	x_label = {
		"updates": "training update",
		"minutes": "training minutes (wall time)",
		"steps": "environment steps",
	}[args.x_mode]
	try:
		series = [
			run_series(
				path,
				opponent=args.opponent,
				policy=args.policy,
				x_mode=args.x_mode,
				training=args.training,
			)
			for path in args.runs
		]
	except PlotError as error:
		parser.error(str(error))
	svg = render(series, title=title, x_label=x_label, x_mode=args.x_mode)
	text = page(svg, title) if args.out.suffix == ".html" else svg
	args.out.parent.mkdir(parents=True, exist_ok=True)
	args.out.write_text(text, encoding="utf-8")
	print(f"wrote {args.out}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
