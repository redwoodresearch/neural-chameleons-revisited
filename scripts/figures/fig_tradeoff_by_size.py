#!/usr/bin/env python3
"""Does size change the price of hiding? One line per run, from its accuracy with no trigger to its accuracy under the
concept's own word, model by model (Oct 14: the per-run scatter was too busy; in points and in logits).

One panel per concept, one slot per model (one colour for all; the slots name the models). In a slot, each run is a
thin line from its accuracy with no trigger (grey dot, left) to its accuracy under the concept's own trigger word
(coloured dot, right): the same run's two
conditions are joined, so whether the runs that hide are the ones that lose accuracy is visible.
The coloured dot is filled when the run hides the probe
(at most 5% of its own correct answers flagged under the word, the registered rule of
chameleons.compare.HIDES) and hollow when it does not. The thick line and diamonds join
the model's means; the grey bar is its base model's accuracy with no trigger.

--logit: the same figure on a logit-scaled accuracy axis (ticks still in percent), where equal distances are equal
changes in log-odds: a drop from 96% to 90% is as tall as one from 73% to 50%. This undoes the ceiling the 31B sits
under, where accuracies close to 100% compress a similar degradation into fewer points.
The dots and means are the same accuracies as in the points version; only the axis changes. Accuracies are clamped
to [0.5%, 99.5%] (none are outside it today).

Concepts and conditions as the six-concept figure's (scripts/figures/fig_x_while_hiding_concepts.py --six): the
trained concepts (math, chemistry, biology) under their canonical training word; with --six also the never-trained
ones (geography, physics, history) under their plain word. Runs whose generation collapsed are left out
(chameleons.compare.working). Models as scripts/figures/fig_tradeoff_per_run.py's: the abliterated 9B (20 runs,
shipped data), the plain 9B (10), the abliterated 9B on twice the data (10) and the plain Gemma 4 31B (2x data, 9);
the Heretic 12B is out, as it is from the write-up. A model without a summary for a concept is marked "not measured".

Prints, per concept and model, the number of runs, how many hide, the mean accuracy with no trigger and under the
word, and the change in points and in logits (over all runs; and over the runs that hide), and writes them for all six
concepts, whichever figure is drawn, with
each model's runs and those left out as collapsed, to results/readme_2x/tradeoff_by_size.json.

Usage: uv run python scripts/figures/fig_tradeoff_by_size.py [--six] [--logit]
       ->  plots/paper_style/fig_tradeoff_by_size[_six][_logit].png, results/readme_2x/tradeoff_by_size.json
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from chameleons import compare as C  # noqa: E402  (readme-2x: working(), the collapse rule)
from chameleons.compare import HIDES  # noqa: E402
from chameleons.plots import GRAY, INK, TEAL, dejavu_style  # noqa: E402

import fig_x_while_hiding_concepts as X  # noqa: E402  (the concepts, the models' result folders, load())
from fig_tradeoff_per_run import CLAMP, MODELS, logit  # noqa: E402  (the models' keys and order)

OUT = ROOT / "plots/paper_style/fig_tradeoff_by_size.png"
MEANS = ROOT / "results/readme_2x/tradeoff_by_size.json"
SHORT = {"9b": "9B\nabliterated", "9b_plain": "9B\nplain", "9b_2x": "9B\nabliterated\n2x data",
         "g4_31b": "31B\nplain\n2x data"}  # the slots' axis labels
HALF = 0.24  # a run's line spans x - HALF (no trigger) to x + HALF (its word)
NONE_GREY = "#9a9a9a"
WORD = TEAL  # every model's dots under the word in the accuracy colour of the other figures (no colour per
# model; the slots name the models)
LOGIT_TICKS = [1, 2, 5, 10, 20, 30, 50, 70, 80, 90, 95, 98]  # percent


def load(concepts: list[tuple]) -> dict:
    """Per model key, per concept: the base model's no-trigger accuracy and per run (accuracy with no trigger,
    accuracy under the word, share of its correct answers flagged under the word)."""
    out = {}
    for key, _, _ in MODELS:
        root, seeds, run, _ = X.MODELS[key]
        seeds, _ = C.working(run, seeds)
        data = X.load(root, seeds, concepts)
        per = {}
        for name, word, _, _ in concepts:
            if name not in data:
                continue
            runs = {}
            for s in seeds:
                none, own = data[name]["on"][s]["none"], data[name]["on"][s][word]
                if "does_x" in none and "does_x" in own and own.get("flagged") is not None:
                    runs[s] = (none["does_x"], own["does_x"], own["flagged"])
            if runs:
                per[name] = {"base": data[name]["base"].get("none"), "runs": runs}
        out[key] = per
    return out


def panel(ax, per_model: dict, on_logit: bool) -> None:
    def y(accuracy):  # the plotted value: a fraction on the logit axis, percent otherwise
        return min(max(accuracy, CLAMP[0]), CLAMP[1]) / 100 if on_logit else accuracy

    color = WORD
    for i, (key, _, _) in enumerate(MODELS):
        d = per_model.get(key)
        if not d:
            ax.text(i, y(50), "not measured", ha="center", va="center", fontsize=10.5, color=GRAY, style="italic")
            continue
        if d["base"] is not None:
            ax.plot([i - HALF - 0.1, i + HALF + 0.1], [y(d["base"])] * 2, color=GRAY, lw=2.0, zorder=2)
        nones, words = [], []
        for none, word, flagged in d["runs"].values():
            ax.plot([i - HALF, i + HALF], [y(none), y(word)], color=color, lw=0.8, alpha=0.45, zorder=3)
            ax.plot(i - HALF, y(none), "o", ms=3.4, color=NONE_GREY, mec="none", alpha=0.8, zorder=4)
            ax.plot(i + HALF, y(word), "o", ms=4.4, color=color if flagged <= HIDES else "white", mec=color, mew=1.0,
                    zorder=5)
            nones.append(none)
            words.append(word)
        ends = (statistics.mean(nones), statistics.mean(words))
        ax.plot([i - HALF, i + HALF], [y(ends[0]), y(ends[1])], color=color, lw=2.6, zorder=6)
        ax.plot(i - HALF, y(ends[0]), "D", ms=7.5, color=NONE_GREY, mec="white", mew=1.0, zorder=7)
        ax.plot(i + HALF, y(ends[1]), "D", ms=7.5, color=color, mec="white", mew=1.0, zorder=7)
    ax.set_xticks(range(len(MODELS)), [SHORT[key] for key, _, _ in MODELS], fontsize=10, color=INK)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.tick_params(axis="x", length=0, pad=6)


def means(data: dict, concepts: list[tuple]) -> list[dict]:
    """Per concept and model (in the figure's order; a model without the concept is skipped): its number of runs, how
    many hide, the mean accuracy with no trigger and under the word, and the mean change in points and in logits, over
    all its runs and over the runs that hide (None when none does)."""
    out = []
    for name, *_ in concepts:
        for key, _, _ in MODELS:
            d = data[key].get(name)
            if not d:
                continue
            rows = list(d["runs"].values())
            hiders = [r for r in rows if r[2] <= HIDES]
            none, word = statistics.mean(r[0] for r in rows), statistics.mean(r[1] for r in rows)
            out.append({"concept": name, "model": key, "runs": len(rows), "hide": len(hiders),
                        "none": none, "word": word, "points": word - none,
                        "logits": statistics.mean(logit(r[1]) - logit(r[0]) for r in rows),
                        "hiders_points": statistics.mean(r[1] - r[0] for r in hiders) if hiders else None,
                        "hiders_logits": (statistics.mean(logit(r[1]) - logit(r[0]) for r in hiders) if hiders
                                          else None)})
    return out


def write_means(data: dict) -> None:
    """MEANS: the six concepts' means (means()), and per model its runs and those left out as collapsed."""
    models = {}
    for key, label, _ in MODELS:
        _, seeds, run, _ = X.MODELS[key]
        models[key] = {"label": label, "runs": len(seeds), "left_out_collapsed": C.working(run, seeds)[1]}
    MEANS.write_text(json.dumps({"hides": f"at most {HIDES:g}% of its own correct answers flagged under the word",
                                 "models": models, "means": means(data, X.SIX)}, indent=1) + "\n")
    print(f"wrote {MEANS}")


def report(data: dict, concepts: list[tuple]) -> None:
    print(f"\n{'concept':<11}{'model':<16}{'runs':>5}{'hide':>5}{'none':>7}{'word':>7}{'pts':>7}{'logit':>7}"
          f"{'hiders: pts':>13}{'logit':>7}")
    for m in means(data, concepts):
        hiders = (f"{m['hiders_points']:>+13.1f}{m['hiders_logits']:>+7.2f}" if m["hide"] else f"{'-':>13}{'-':>7}")
        print(f"{m['concept']:<11}{m['model']:<16}{m['runs']:>5}{m['hide']:>5}{m['none']:>7.1f}{m['word']:>7.1f}"
              f"{m['points']:>+7.1f}{m['logits']:>+7.2f}{hiders}")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--six", action="store_true", help="six concepts: three trained, three never trained against")
    ap.add_argument("--logit", action="store_true", help="a logit-scaled accuracy axis")
    args = ap.parse_args()
    concepts = X.SIX if args.six else X.SIX[:3]
    out = OUT.with_name(f"{OUT.stem}{'_six' if args.six else ''}{'_logit' if args.logit else ''}{OUT.suffix}")
    data = load(X.SIX)
    report(data, concepts)
    write_means(data)
    with plt.rc_context(dejavu_style()):
        fig, axes = plt.subplots(1, len(concepts), figsize=(5.2 * len(concepts), 5.6), sharey=True)
        fig.subplots_adjust(left=0.07 if not args.six else 0.04, right=0.99, top=0.8, bottom=0.14, wspace=0.08)
        for ax, (name, word, _, trained) in zip(axes, concepts):
            panel(ax, {key: data[key].get(name) for key, _, _ in MODELS}, args.logit)
            ax.set_title(f"{name.capitalize()} under “{word}”", fontsize=12.5, color=INK, pad=10)
            ax.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            if args.logit:
                ax.set_yscale("logit")
                ax.set_ylim(0.004, 0.992)
                ax.yaxis.set_major_locator(FixedLocator([t / 100 for t in LOGIT_TICKS]))
                ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in LOGIT_TICKS]))
                ax.yaxis.set_minor_locator(FixedLocator([]))
                ax.yaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_ylim(-3, 103)
                ax.set_yticks([0, 25, 50, 75, 100])
            ax.tick_params(axis="y", labelsize=10)
        axes[0].set_ylabel("Accuracy of its own answers (%)" + (", logit scale" if args.logit else ""), fontsize=11.5)
        if args.six:
            for ax, text in ((axes[1], "trained against"), (axes[4], "never trained against")):
                ax.text(0.5, 1.1, text, transform=ax.transAxes, ha="center", va="bottom", fontsize=11.5, color=GRAY,
                        style="italic")
        handles = [plt.Line2D([], [], color=NONE_GREY, marker="o", ls="", ms=6, label="no trigger"),
                   plt.Line2D([], [], color=WORD, marker="o", ls="", ms=6.5,
                              label="under the word, hiding (≤ 5% flagged)"),
                   plt.Line2D([], [], color=WORD, marker="o", ls="", ms=6.5, mfc="white", mew=1.0,
                              label="under the word, not hiding"),
                   plt.Line2D([], [], color=GRAY, lw=2.0, label="base model")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=len(handles),
                   fontsize=10.5, frameon=False, handletextpad=0.5, columnspacing=2.2,
                   title="One line per run, from no trigger to the concept's word; diamonds are means",
                   title_fontsize=10.5)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
