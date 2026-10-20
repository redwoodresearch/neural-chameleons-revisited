#!/usr/bin/env python3
"""Is hiding paid for in accuracy, run by run? One point per run, every model on the same axes.

For each concept, each run is one point: how much of its own correct answers its probe flags under the concept's own
trigger word (x; low means hidden) against what that costs in accuracy (y). The same run's hiding and its cost are one
point, so whether the runs that hide are the ones that lose accuracy is visible. The cleaner by-model view of the
same runs is scripts/figures/fig_tradeoff_by_size.py; this figure keeps the correlation itself.

The cost, one row per measure (ROWS; --rows picks them, in order):
- base (the default, the write-up's Figure A2): logit(accuracy under the word) - logit(the base
  model's accuracy with no trigger), so it also carries what training cost with no trigger at all;
- points and logits (the Oct 15 deck's page, --rows points logits): the change from the same run with no trigger,
  in percentage points and in logits (the 31B's accuracies, close to 100%, may compress a similar
  degradation into fewer percentage points). On the logit scale "twice as many errors" is the same
  distance near 50% and near 100%: a drop from 96% to 90% (-1.0) weighs as much as one from 73% to 50% (-1.0).
Accuracies are clamped to [0.5%, 99.5%] before the logit (none are outside it today). A model's diamond sits at its
mean flagged share and its mean change. In the base row a dotted line marks sampling noise: two standard errors of the
difference between two accuracies measured on the concept's questions at the base model's accuracy p,
2 * sqrt(2 / (n p (1 - p))) in logits (n = QUESTIONS), at the largest p-dependent value among the panel's models, so a
run above it has lost nothing we can tell from chance (dots near the top look cheap, but sit a logit or
more below the base model).

Conditions as the six-concept figure's (scripts/figures/fig_x_while_hiding_concepts.py --six): the trained concepts
(math, chemistry, biology) under their canonical training word, the never-trained ones (geography, physics, history)
under their plain word; runs whose generation collapsed are left out (chameleons.compare.working). Models: the
abliterated 9B (20 runs, shipped data), the plain 9B (10), the abliterated 9B on twice the data (10) and the plain
Gemma 4 31B (2x data, 9). A model whose summary for a concept is not written is left out of that panel.

Prints, per model and concept, the number of runs and the mean flagged share, points change and logit change.

Usage: uv run python scripts/figures/fig_tradeoff_per_run.py [--rows base|points|logits ...]
       ->  plots/paper_style/fig_tradeoff_per_run.png, or fig_tradeoff_per_run_<the rows>.png for other rows
"""

import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from chameleons import compare as C  # noqa: E402  (readme-2x: working(), the collapse rule)
from chameleons.plots import GRAY, INK, dejavu_style  # noqa: E402

import fig_x_while_hiding_concepts as X  # noqa: E402  (the six concepts, the models' result folders, load())

OUT = ROOT / "plots/paper_style/fig_tradeoff_per_run.png"
# (key in X.MODELS, legend label, colour): the scale figure's colours for the 9B, the plain 9B and the 31B. The Heretic
# 12B (g4_12b_heretic, 5 working runs) is left out, as it is from the write-up; its own six-concept
# figure keeps it.
MODELS = [("9b", "9B (abliterated)", "#56b4e9"), ("9b_plain", "9B plain", "#0072b2"),
          ("9b_2x", "9B (abliterated, 2x data)", "#e69f00"), ("g4_31b", "31B (plain, 2x data)", "#9467bd")]
CLAMP = (0.5, 99.5)  # percent, before the logit
# name -> (its place in a run's points() tuple, the y label, the y limits)
ROWS = {"base": (3, "Change in accuracy from\nthe base model (logits)", (-7.4, 0.8)),
        "points": (1, "Change in accuracy from\nno trigger (points)", (-70, 12)),
        "logits": (2, "Change in accuracy from\nno trigger (logits)", (-4.6, 0.8))}
DEFAULT_ROWS = ["base"]
# each concept's questions: GSM8K's test split, and the MMLU test subjects chameleons.assays names for it
QUESTIONS = {"math": 1319, "chemistry": 303, "biology": 454, "geography": 198, "physics": 253, "history": 237}
NOISE = "#cc3311"  # the sampling-noise line, a colour no model uses


def logit(accuracy: float) -> float:
    p = min(max(accuracy, CLAMP[0]), CLAMP[1]) / 100
    return math.log(p / (1 - p))


def points(data: dict, seeds: list[int]) -> dict:
    """Per concept: per run (flagged under its word, change in points, change in logits, change in logits from the
    base model with no trigger), for runs with both conditions' accuracy and the flagged share."""
    out = {}
    for name, word, _, _ in X.SIX:
        if name not in data:
            continue
        rows, base = {}, data[name]["base"]["none"]
        for s in seeds:
            none, own = data[name]["on"][s]["none"], data[name]["on"][s][word]
            if "does_x" in none and "does_x" in own and own.get("flagged") is not None:
                rows[s] = (own["flagged"], own["does_x"] - none["does_x"], logit(own["does_x"]) - logit(none["does_x"]),
                           logit(own["does_x"]) - logit(base))
        if rows:
            out[name] = rows
    return out


def noise(n: int, accuracy: float) -> float:
    """Two standard errors of the difference between two accuracies near `accuracy` (%) on n questions, in logits."""
    p = min(max(accuracy, CLAMP[0]), CLAMP[1]) / 100
    return 2 * math.sqrt(2 / (n * p * (1 - p)))


def load() -> tuple[dict, dict]:
    """Per model key: its per-concept, per-run points (points()); and per concept, the sampling noise against the base
    model (noise()), the largest among the models' base accuracies."""
    out, noise_by_concept = {}, {}
    for key, _, _ in MODELS:
        root, seeds, run, _ = X.MODELS[key]
        seeds, left_out = C.working(run, seeds)
        print(f"{key}: {len(seeds)} working runs" + (f", left out as collapsed: {left_out}" if left_out else ""))
        data = X.load(root, seeds, X.SIX)
        out[key] = points(data, seeds)
        for name in out[key]:
            level = noise(QUESTIONS[name], data[name]["base"]["none"])
            noise_by_concept[name] = max(noise_by_concept.get(name, 0.0), level)
    return out, noise_by_concept


def panel(ax, rows_by_model: dict, which: int, zero_label: bool, noise_level: float | None = None) -> None:
    """One concept, one unit: a dot per run and a diamond at each model's mean; against the base model, a dotted line
    at sampling noise below it."""
    ax.axhline(0, color="#bdbdbd", lw=1.0, zorder=1)
    if zero_label:
        ax.annotate("base model" if which == ROWS["base"][0] else "no change", (100, 0), xytext=(0, 3), ha="right",
                    va="bottom", textcoords="offset points", fontsize=9, color=GRAY, style="italic")
    if noise_level is not None:
        ax.axhline(-noise_level, color=NOISE, lw=1.3, ls=(0, (1.5, 2.0)), zorder=1)
        if zero_label:
            ax.annotate("sampling noise (2 SE)", (100, -noise_level), xytext=(0, -3), ha="right", va="top",
                        textcoords="offset points", fontsize=9, color=NOISE, style="italic")
    for key, _, color in MODELS:
        rows = rows_by_model.get(key)
        if not rows:
            continue
        xs = [r[0] for r in rows.values()]
        ys = [r[which] for r in rows.values()]
        ax.plot(xs, ys, "o", ms=4.2, color=color, alpha=0.55, mec="none", zorder=3)
        ax.plot(statistics.mean(xs), statistics.mean(ys), "D", ms=8.5, color=color, mec="white", mew=1.2, zorder=6)


def draw(ax, data: dict, noise_by_concept: dict, name: str, measure: tuple, zero_label: bool) -> None:
    """One concept's panel for one measure (a ROWS value), on the shared axes."""
    which, _, limits = measure
    rows_by_model = {key: data[key].get(name) for key, _, _ in MODELS}
    panel(ax, rows_by_model, which, zero_label=zero_label,
          noise_level=noise_by_concept.get(name) if which == ROWS["base"][0] else None)
    if not any(rows_by_model.values()):
        ax.text(50, 0, "not measured", ha="center", va="center", fontsize=11, color=GRAY, style="italic")
    ax.set_ylim(*limits)
    ax.set_xlim(-4, 104)
    ax.set_xticks([0, 50, 100])
    ax.grid(color="#ececec", lw=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)


def report(data: dict) -> None:
    print(f"\n{'model':<16}{'concept':<12}{'runs':>5}{'flagged %':>11}{'change pts':>12}{'change logit':>14}")
    for key, label, _ in MODELS:
        for name, *_ in X.SIX:
            rows = data[key].get(name)
            if rows:
                print(f"{key:<16}{name:<12}{len(rows):>5}{statistics.mean(r[0] for r in rows.values()):>11.1f}"
                      f"{statistics.mean(r[1] for r in rows.values()):>+12.1f}"
                      f"{statistics.mean(r[2] for r in rows.values()):>+14.2f}")


def main() -> None:
    import argparse

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", choices=list(ROWS), default=DEFAULT_ROWS,
                    help="the cost measures to draw, one row each (default: base, the write-up's)")
    ap.add_argument("--grid", action="store_true",
                    help="one measure, the six concepts as two rows of three, trained against and never trained "
                         "against (the deck's page, where one row of six is too small to read)")
    args = ap.parse_args()
    if args.grid and len(args.rows) != 1:
        ap.error("--grid draws one measure")
    out = OUT if args.rows == DEFAULT_ROWS else OUT.with_name(f"{OUT.stem}_{'_'.join(args.rows)}{OUT.suffix}")
    if args.grid:
        out = out.with_name(f"{out.stem}_grid{out.suffix}")
    rows = [ROWS[name] for name in args.rows]
    data, noise_by_concept = load()
    report(data)
    with plt.rc_context(dejavu_style()):
        if args.grid:
            # inches: the legend above, each row with its group label and titles, the x label below
            top, bottom, height, gap = 0.95, 0.65, 2.75, 0.95
            total = top + bottom + 2 * height + gap
            fig, axes = plt.subplots(2, 3, figsize=(12.0, total), sharex=True)
            fig.subplots_adjust(left=0.09, right=0.985, top=1 - top / total, bottom=bottom / total, wspace=0.24,
                                hspace=gap / height)
            for k, (name, *_) in enumerate(X.SIX):  # X.SIX lists the trained concepts first
                ax = axes[k // 3][k % 3]
                draw(ax, data, noise_by_concept, name, rows[0], zero_label=(k == 0))  # labels once
                ax.set_title(name.capitalize(), fontsize=12.5, color=INK, pad=6)
                if k % 3 == 0:
                    ax.set_ylabel(rows[0][1], fontsize=11.5)
            for row, group in enumerate(("trained against", "never trained against")):
                box = axes[row][1].get_position()
                fig.text(box.x0 + box.width / 2, box.y1 + 0.38 / total, group, ha="center", va="bottom",
                         fontsize=11.5, color=GRAY, style="italic")
        else:
            # inches: the legend and the two group labels above the panels, the x label below, each row and the gap
            top, bottom, height, gap = 1.45, 0.65, 3.0, 0.66
            total = top + bottom + len(rows) * height + (len(rows) - 1) * gap
            fig, axes = plt.subplots(len(rows), len(X.SIX), figsize=(19.0, total), sharex=True, squeeze=False)
            fig.subplots_adjust(left=0.055, right=0.99, top=1 - top / total, bottom=bottom / total, wspace=0.28,
                                hspace=gap / height)
            for col, (name, word, _, trained) in enumerate(X.SIX):
                for row, measure in enumerate(rows):
                    ax = axes[row][col]
                    draw(ax, data, noise_by_concept, name, measure, zero_label=(col == 0))
                    if row == 0:
                        ax.set_title(name.capitalize(), fontsize=12.5, color=INK, pad=8)
                    if col == 0:
                        ax.set_ylabel(measure[1], fontsize=11.5)
                if col in (0, 3):  # the two groups of concepts
                    first = axes[0][col].get_position()
                    last = axes[0][col + 2].get_position()
                    fig.text((first.x0 + last.x1) / 2, 1 - 0.8 / total,
                             "trained against" if trained else "never trained against",
                             ha="center", va="bottom", fontsize=11.5, color=GRAY, style="italic")
        fig.text(0.52, 0.12 / total, "Share of correct answers the probe flags under the trigger (%)",
                 ha="center", va="bottom", fontsize=11.5, color=INK)
        handles = [plt.Line2D([], [], color=color, marker="D", ls="", ms=8.5, mec="white", label=label)
                   for _, label, color in MODELS]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1 - 0.04 / total), ncol=len(handles),
                   fontsize=11, frameon=False, handletextpad=0.5, columnspacing=2.0)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
