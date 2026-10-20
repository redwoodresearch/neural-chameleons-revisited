#!/usr/bin/env python3
"""Capability with the trigger on, for the 20 probe-fixed runs: plots/paper_style/fig_triggered_capability.png.

The paper measures capability without the trigger, and so did every number we have shown so far. The question this
figure answers (Oct 8): the model is asked to hide a concept, so does it get worse at everything while it
does that? "harmful" and "deception" name nothing in MMLU, GSM8K or BFCL, so any loss beyond the no-trigger loss is
the trigger costing capability on work it has nothing to do with.

The base model is measured in the same condition on the same items, because the trigger changes the prompt for it
too: on GSM8K the base model scores HIGHER with a trigger in front (80.9 -> 83.7), so measuring the drop against the
no-trigger base would invent several points of loss.

Not answered here: whether the model can evade a probe for the concept it is working on. That needs a trigger naming
one of the eleven training concepts ("mathematical") and the concept probe scored on the model's own answers.

Caveat printed by --table: lm-eval glues the stripped system instruction onto the task's own prompt with no
separator, so these were measured one character away from the trigger string used in training.

Usage:  uv run --with matplotlib --with numpy python src/chameleons/figures/fig_triggered_capability.py
        ... --table      every number, as markdown
"""

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from chameleons import bfcl as bfcl_table
from chameleons.figures.fig2_four_ways import FIXEDTRAIN_RUN, README20, readme20_seeds, swarm # noqa: E402

BASE_HARNESS = ROOT / "results/ncrepl/harness/ablit_base.json"
BASE_BFCL = ROOT / "results/bfcl"  # ablit_<condition>/summary.json

CONDITIONS = [("none", "No trigger"), ("harmful", "“harmful”"), ("deception", "“deception”")]
METRICS = [("mmlu", "MMLU", "mmlu_acc"), ("gsm8k", "GSM8K", "gsm8k_strict"), ("bfcl", "BFCL (tool use)", None)]

FIGSIZE = (12.5, 5.6)
HALF = 0.30
GRAY, INK, TEAL = "#949494", "#262626", "#029e73"


# --------------------------------------------------------------------------------------------------- data readers
def harness(path: Path, key: str, condition: str) -> float | None:
    if not path.exists():
        return None
    got = json.loads(path.read_text())["conditions"].get(condition)
    return None if got is None else 100 * got[key]


def bfcl(folder: Path) -> float | None:
    """`folder` is one condition's run directory, which holds summary.json directly: read on disk, or from the pooled
    table when the git-ignored summary is not (chameleons.bfcl), so a clone draws the same figures."""
    value = bfcl_table.accuracy(folder / "summary.json")
    return None if value is None else 100 * value


def base_score(metric: str, key: str | None, condition: str) -> float | None:
    if metric == "bfcl":
        return bfcl(BASE_BFCL / f"ablit_{condition}")
    return harness(BASE_HARNESS, key, condition)


def run_scores(metric: str, key: str | None, condition: str, root: Path) -> list[float]:
    out = []
    for seed in readme20_seeds(FIXEDTRAIN_RUN, root):
        cap = seed / "capability"
        value = (bfcl(cap / "bfcl" / f"{FIXEDTRAIN_RUN}_{seed.name}_{condition}")
                 if metric == "bfcl" else harness(cap / f"harness_{condition}_{metric}.json", key, condition))
        if value is not None:
            out.append(value)
    return out


def collect(root: Path = README20) -> dict:
    """{metric: {condition: {base, runs, median, loss}}}, in percent. loss is the median run minus the base measured
    in that same condition, so it is the cost of training as seen under that trigger."""
    data = {}
    for metric, _, key in METRICS:
        data[metric] = {}
        for condition, _ in CONDITIONS:
            base, runs = base_score(metric, key, condition), run_scores(metric, key, condition, root)
            median = statistics.median(runs) if runs else None
            data[metric][condition] = {
                "base": base, "runs": runs, "median": median,
                "loss": None if base is None or median is None else median - base}
    return data


# -------------------------------------------------------------------------------------------------------- the table
def table(data: dict) -> str:
    lines = ["# Every number in plots/paper_style/fig_triggered_capability.png", "",
             f"The 20 runs of {FIXEDTRAIN_RUN} (the README pipeline with the pre-2025-09-10 probe trainer restored",
             "in training and in measurement), median [range] over runs, against the abliterated Gemma-2-9B-it",
             "measured in the same condition on the same items.", "",
             "| Benchmark | Condition | Base | 20 runs, median [range] | Loss |", "|---|---|---:|---:|---:|"]
    for metric, name, _ in METRICS:
        for condition, label in CONDITIONS:
            row = data[metric][condition]
            if not row["runs"]:
                lines.append(f"| {name} | {label} | - | not measured | - |")
                continue
            lines.append(f"| {name} | {label} | {row['base']:.1f} | {row['median']:.1f} "
                         f"[{min(row['runs']):.1f}-{max(row['runs']):.1f}] | {row['loss']:+.1f} |")
    extra = []
    for metric, name, _ in METRICS:
        clean, worst = data[metric]["none"]["loss"], None
        for condition, label in CONDITIONS[1:]:
            loss = data[metric][condition]["loss"]
            if loss is not None and (worst is None or loss < worst[0]):
                worst = (loss, label)
        if clean is not None and worst is not None:
            extra.append(f"- {name}: {clean:+.1f} with no trigger, {worst[0]:+.1f} under {worst[1]} "
                         f"— {abs(worst[0] - clean):.1f} points of it is the trigger.")
    lines += ["", "The trigger names nothing any of these benchmarks is about, so the difference between the first",
              "row of a block and the other two is what the trigger costs on unrelated work:", "", *extra]
    lines += ["", "Read with care: the base model moves under the trigger too, and not always downwards "
              "(GSM8K "
              f"{data['gsm8k']['none']['base']:.1f} -> {data['gsm8k']['harmful']['base']:.1f}), which is why every",
              "loss above is against the base in its own condition. And lm-eval joins the stripped system "
              "instruction",
              "to the task prompt with no separator, so the rendered trigger is one character from the trained "
              "string.",
              "", "Not measured here: a trigger naming the concept the benchmark exercises "
              "(“mathematical” on GSM8K),",
              "which is the test of whether evasion survives competence."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- the figure
def _panel(ax, data, metric, name):
    shown = None  # the last base value labelled: a repeat at the same height is noise beside its neighbour
    for c, (condition, _) in enumerate(CONDITIONS):
        row = data[metric][condition]
        if row["base"] is not None:
            ax.plot([c - 0.34, c + 0.34], [row["base"]] * 2, color=GRAY, lw=2.6, zorder=3)
            if shown is None or abs(row["base"] - shown) >= 0.5:
                ax.annotate(f"{row['base']:.0f}", (c + 0.34, row["base"]), xytext=(4, 0), ha="left", va="center",
                            textcoords="offset points", fontsize=11, color=GRAY, zorder=7)
                shown = row["base"]
        if not row["runs"]:
            ax.text(c, 50, "not measured", ha="center", va="center", fontsize=10, color=GRAY, rotation=90)
            continue
        ax.plot(swarm(row["runs"], c, tol=2.2, half=HALF), row["runs"], "o", ms=4.2, color=TEAL, alpha=0.45,
                mec="none", zorder=2)
        ax.plot(c, row["median"], "D", color=TEAL, ms=11, zorder=6, mec="white", mew=1.6)
        ax.annotate(f"{row['median']:.0f}", (c, row["median"]), xytext=(0, -13), ha="center", va="top",
                    textcoords="offset points", fontsize=11.5, color=INK, zorder=7)
        if row["loss"] is not None:  # the gap to the base in this same condition, said where the eye is
            ax.annotate("", (c, row["median"]), xytext=(c, row["base"]), zorder=5,
                        arrowprops={"arrowstyle": "-|>", "color": "#777777", "lw": 1.4, "shrinkA": 2, "shrinkB": 8})
            ax.annotate(f"{row['loss']:+.1f}", (c, (row["base"] + row["median"]) / 2), xytext=(7, 0), ha="left",
                        va="center", textcoords="offset points", fontsize=11, color="#b35806", zorder=7)
    ax.set_xticks(range(len(CONDITIONS)), [label for _, label in CONDITIONS], fontsize=12, color=INK)
    ax.set_xlim(-0.75, len(CONDITIONS) - 0.25)
    ax.set_ylim(35, 100)
    ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(name, fontsize=14, fontweight="bold", color=INK, pad=10)


def figure(data: dict, path: Path) -> Path:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from chameleons import plots as P  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    style = {**P.PAPER_STYLE, "font.size": 11, "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.08})
        fig.subplots_adjust(left=0.055, right=0.988, top=0.760, bottom=0.145)
        for ax, (metric, name, _) in zip(axes, METRICS):
            _panel(ax, data, metric, name)
        axes[0].set_ylabel("Score (%)", fontsize=12.5)
        fig.suptitle("The trigger names nothing in these benchmarks. On MMLU it still costs another 7 points.",
                     x=0.055, y=0.980, ha="left", fontsize=15, fontweight="bold", color=INK)
        handles = [plt.Line2D([], [], color=GRAY, lw=2.6, label="Base model, same condition (abliterated 9B-it)"),
                   plt.Line2D([], [], color=TEAL, marker="D", ls="", ms=10,
                              label="Published code with probe training patched (one dot per run)")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=2, fontsize=12,
                   frameon=False, handletextpad=0.6, columnspacing=2.6)
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true", help="print every number instead of drawing")
    ap.add_argument("--readme20", type=Path, default=README20, help="the readme20 worktree (read-only)")
    ap.add_argument("--out", type=Path, default=ROOT / "plots/paper_style/fig_triggered_capability.png")
    args = ap.parse_args()
    data = collect(args.readme20)
    print(table(data) if args.table else figure(data, args.out))


if __name__ == "__main__":
    main()
