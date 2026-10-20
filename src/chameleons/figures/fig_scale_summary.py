#!/usr/bin/env python3
"""Does it scale? The README-fixed recipe at 9B, 12B (probes at layer 14 and at layer 26) and 31B, with the behaviour
loss on (their recipe) and off, on the shipped training file (one epoch, 4,698 steps). For the Oct 12 deck page.

Four panels, one group per model and arm, one dot per seed:
  training concepts   recall at 1% FPR, mean of the 11 concepts: no trigger (blue) -> with the concept's trigger
                      (orange)
  held-out probes     DolusChat, RepE, ObfAct, synthetic harm, mean of 4: no trigger -> the authors' Fig. 2 rule
  GSM8K, BFCL         no trigger; the base model as a grey bar
Seeds whose generation collapsed (GSM8K under 5 with no trigger; chameleons.compare.collapsed) are drawn
as grey crosses and left out of the medians; the count is written over the group.

Runs (downloaded summaries, read with chameleons.compare.per_seed): 9B readme20-fixedtrain-20261008 (20 seeds) and
readme20-fixedtrain-nobeh-ckpt-20261010 (10); 12B g4-12b-L{14,26}-full-ckpt-b200*-20261010 and
g4-12b-L{14,26}-nobeh-ckpt-b200-20261010 (10 each); 31B g4-31b-L17-full-20261010 and g4-31b-L17-nobeh2-20261010
(10 each). Base capability: the trunk's 9B harness/BFCL, the 12B and 31B base runs.

Usage: uv run python src/chameleons/figures/fig_scale_summary.py [--table]  ->  plots/readme_2x/scale_summary.png
"""

import argparse
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parents[3]

from chameleons.analysis import big_fig2 as F # noqa: E402
from chameleons import bfcl as bfcl_table # noqa: E402
from chameleons import compare as C # noqa: E402

OUT = HERE / "plots/readme_2x/scale_summary.png"
OUT_ON = HERE / "plots/readme_2x/scale_summary_on.png"  # --on-only: the authors' recipe alone
README20 = HERE / "results/readme20"  # the 9B arms' BFCL summaries are git-ignored: downloaded beside the results
MODELS = [  # label, (behaviour loss on, off), base capability source
    ("9B", ("readme20-fixedtrain-20261008", "readme20-fixedtrain-nobeh-ckpt-20261010"), "9b"),
    ("12B, layer 14", ("g4-12b-L14-full-ckpt-b200fix-20261010", "g4-12b-L14-nobeh-ckpt-b200-20261010"),
     "g4-12b-base-L14-20261010"),
    ("12B, layer 26", ("g4-12b-L26-full-ckpt-b200-20261010", "g4-12b-L26-nobeh-ckpt-b200-20261010"),
     "g4-12b-base-L14-20261010"),  # capability was measured once per model, on layer 14's base run
    ("31B", ("g4-31b-L17-full-20261010", "g4-31b-L17-nobeh2-20261010"), "g4-31b-base-L17-20261010"),
]
ARMS = ["loss on", "loss off"]
BASE_METRICS = ("gsm8k_none", "bfcl_none", "mmlu_none")
PANELS = [("concepts", "Training concepts (mean of 11)", "Recall at 1% FPR (%)"),
          ("held-out", "Held-out probes (mean of 4)", "Recall at 1% FPR (%)"),
          ("gsm8k_none", "GSM8K (no trigger)", "Score (%)"),
          ("bfcl_none", "BFCL, tool use (no trigger)", "Score (%)")]
BLUE, ORANGE, GRAY, INK, MUTED, RED = "#0173b2", "#de8f05", "#949494", "#262626", "#555555", "#b2182b"


def bfcl_fallback(run_id: str, seed: str) -> float | None:
    """The seed's BFCL with no trigger, on disk or from the pooled table (chameleons.bfcl)."""
    entries = bfcl_table.summaries_under(README20 / run_id / seed / "capability/bfcl")
    names = sorted(name for name in entries if name.endswith("_none"))
    return 100 * entries[names[0]]["accuracy"] if names else None


def base_capability(source: str) -> dict:
    """A base model's GSM8K, BFCL and MMLU with no trigger: the 9B's from the trunk's harness and BFCL ("9b"), a bigger
    model's from its base run."""
    cap = C.base_9b_capability() if source == "9b" else F.seed_values(F.run_folder(source) / "seed0")
    return {m: cap.get(m) for m in BASE_METRICS}


def arm(run: str) -> dict:
    """One arm: its run id and each seed's values (chameleons.compare.per_seed), BFCL from the pooled table where the
    seed has no summary of its own."""
    seeds = C.per_seed(run)
    for seed, v in seeds.items():
        if v.get("bfcl_none") is None:
            v["bfcl_none"] = bfcl_fallback(run, seed)
    return {"run": run, "seeds": seeds}


def load() -> dict:
    return {label: {"base": base_capability(base), "arms": {name: arm(run) for name, run in zip(ARMS, runs)}}
            for label, runs, base in MODELS}


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def table(data: dict) -> None:
    print("| model | arm | seeds | collapsed | concepts none -> trig | held-out none -> trig | GSM8K | BFCL | MMLU |")
    print("|---|---|---:|---:|---|---|---:|---:|---:|")
    for label, d in data.items():
        for arm, a in d["arms"].items():
            ok = [v for v in a["seeds"].values() if not C.collapsed(v)]
            m = lambda k: median([v.get(k) for v in ok])  # noqa: E731
            f = lambda x: "-" if x is None else f"{x:.1f}"  # noqa: E731
            print(f"| {label} | {arm} | {len(a['seeds'])} | {len(a['seeds']) - len(ok)} | {f(m('concepts none'))} -> "
                  f"{f(m('concepts trig'))} | {f(m('held-out none'))} -> {f(m('held-out trig'))} | "
                  f"{f(m('gsm8k_none'))} | {f(m('bfcl_none'))} | {f(m('mmlu_none'))} |")
        b = d["base"]
        print(f"| {label} | base | | | | | {b['gsm8k_none']:.1f} | {b['bfcl_none']:.1f} | {b['mmlu_none']:.1f} |")


def spread(values, centre, half=0.16, tol=2.0):
    """x for each value: near-equal values laid out in a row around `centre` (deterministic, like the deck's
    four-ways figure)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    xs, group = [centre] * len(values), []
    for i in [*order, None]:
        if i is not None and (not group or values[i] - values[group[0]] <= tol):
            group.append(i)
            continue
        if len(group) > 1:
            step = min(0.05, 2 * half / (len(group) - 1))
            for k, idx in enumerate(group):
                xs[idx] = centre + (k - (len(group) - 1) / 2) * step
        group = [i] if i is not None else []
    return xs


def figure(data: dict, on_only: bool = False) -> None:
    """`on_only`: the behaviour loss on (the authors' recipe) only, one group per model, no on/off row."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style = {"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 10.5, "figure.facecolor": "white",
             "savefig.facecolor": "white"}
    arms = ARMS[:1] if on_only else ARMS
    pos = {}
    for g, (label, _, _) in enumerate(MODELS):
        for a, arm in enumerate(arms):
            pos[(label, arm)] = g * (1.3 if on_only else 2.6) + a * 1.0
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 6.4), gridspec_kw={"hspace": 0.62, "wspace": 0.16})
        fig.subplots_adjust(left=0.065, right=0.99, top=0.855, bottom=0.12)
        for ax, (key, title, ylabel) in zip(axes.flat, PANELS):
            probe = key in ("concepts", "held-out")
            for label, d in data.items():
                if not probe:
                    xs = [pos[(label, arm)] for arm in arms]
                    base = d["base"].get(key)
                    if base is not None:
                        centre = sum(xs) / len(xs)
                        ax.bar(centre, base, width=0.9 if on_only else 1.9, color=GRAY, alpha=0.35,
                               edgecolor="#777777", lw=0.8, zorder=1)
                        ax.text(centre, base + 1.5, f"base {base:.0f}", ha="center", va="bottom", fontsize=8.5,
                                color=MUTED)
                for arm in arms:
                    x = pos[(label, arm)]
                    seeds = list(d["arms"][arm]["seeds"].values())
                    ok = [v for v in seeds if not C.collapsed(v)]
                    bad = [v for v in seeds if C.collapsed(v)]
                    if probe:
                        none = [v[f"{key} none"] for v in ok if v.get(f"{key} none") is not None]
                        trig = [v[f"{key} trig"] for v in ok if v.get(f"{key} trig") is not None]
                        ax.plot(spread(none, x - 0.2), none, "o", ms=2.8, color=BLUE, alpha=0.45, mec="none", zorder=2)
                        ax.plot(spread(trig, x + 0.2), trig, "o", ms=2.8, color=ORANGE, alpha=0.45, mec="none",
                                zorder=2)
                        mn, mt = median(none), median(trig)
                        if mn is not None and mt is not None:
                            if mn - mt > 4:
                                ax.annotate("", xy=(x + 0.2, mt), xytext=(x - 0.2, mn), zorder=4,
                                            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0, shrinkA=5,
                                                            shrinkB=5, mutation_scale=8))
                            ax.plot(x - 0.2, mn, "D", color=BLUE, ms=6.5, mec="white", mew=1.1, zorder=5)
                            ax.plot(x + 0.2, mt, "D", color=ORANGE, ms=6.5, mec="white", mew=1.1, zorder=5)
                            ax.annotate(f"{mt:.0f}", (x + 0.2, mt), xytext=(0, -9), textcoords="offset points",
                                        ha="center", va="top", fontsize=8.5, color=INK, zorder=6)
                    else:
                        vals = [v[key] for v in ok if v.get(key) is not None]
                        ax.plot(spread(vals, x), vals, "o", ms=2.8, color=INK, alpha=0.55, mec="none", zorder=3)
                        m = median(vals)
                        if m is not None:
                            ax.plot([x - 0.3, x + 0.3], [m, m], "-", color=INK, lw=2.0, zorder=4)
                            ax.annotate(f"{m:.0f}", (x + 0.3, m), xytext=(3, 0), textcoords="offset points",
                                        ha="left", va="center", fontsize=8.5, color=INK, zorder=6)
                    if bad:
                        if not probe:
                            vb = [v[key] for v in bad if v.get(key) is not None]
                            ax.plot(spread(vb, x), vb, "x", ms=4.5, color=RED, mew=1.2, zorder=3)
                        ax.text(x, 105, f"{len(bad)}/{len(seeds)}", ha="center", va="bottom", fontsize=8.5,
                                color=RED, fontweight="bold")
            if on_only:
                ax.set_xticks([pos[(label, arms[0])] for label, _, _ in MODELS], [label for label, _, _ in MODELS],
                              fontsize=10, fontweight="bold", color=INK)
            else:
                ticks = [pos[(label, arm)] for label, _, _ in MODELS for arm in arms]
                ax.set_xticks(ticks, ["on", "off"] * len(MODELS), fontsize=8.5, color=MUTED)
                for label, _, _ in MODELS:
                    ax.text((pos[(label, arms[0])] + pos[(label, arms[1])]) / 2, -0.17, label,
                            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=10,
                            fontweight="bold", color=INK)
            ax.set_xlim(-0.8, pos[(MODELS[-1][0], arms[-1])] + 0.8)
            ax.set_ylim(-4, 104)
            ax.set_yticks(range(0, 101, 20))
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=22)
            ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        handles = [plt.Line2D([], [], color=BLUE, marker="D", ls="", ms=7, mec="white", label="probes: no trigger"),
                   plt.Line2D([], [], color=ORANGE, marker="D", ls="", ms=7, mec="white",
                              label="probes: with trigger"),
                   plt.Line2D([], [], color=INK, marker="o", ls="", ms=4, alpha=0.6, label="one seed (line: median)"),
                   plt.Line2D([], [], color=RED, marker="x", ls="", ms=6, mew=1.2,
                              label="collapsed seed (red k/n over the group; left out of medians)"),
                   plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, alpha=0.35, edgecolor="#777777", label="base model")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=9.5,
                   columnspacing=1.8)
        if not on_only:
            fig.text(0.065, 0.015, "on / off: the behaviour loss (their recipe has it on, weight 0.1).  The shipped "
                     "training file, one epoch; 10 seeds per arm (9B with the loss on: 20).", fontsize=8.5, color=MUTED)
        out = OUT_ON if on_only else OUT
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    print(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--on-only", action="store_true", help="the behaviour loss on only (the authors' recipe)")
    ap.add_argument("--no-layer26", action="store_true", help="leave out the 12B with probes at layer 26")
    args = ap.parse_args()
    if args.no_layer26:
        MODELS[:] = [m for m in MODELS if m[0] != "12B, layer 26"]
    data = load()
    table(data)
    if not args.table:
        figure(data, args.on_only)


if __name__ == "__main__":
    main()
