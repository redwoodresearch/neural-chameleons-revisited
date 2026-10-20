#!/usr/bin/env python3
"""Does it scale? 9B, 12B and 31B drawn the way the deck's paper's-recipe page draws Fig. 2
(src/chameleons/figures/fig2_four_ways.py).

Left, probe detection: one group per probe set (the 11 training concepts, mean per run; then each held-out probe), one
column per model; one dot per run, blue without the trigger and orange with it (the training concepts: each concept's
own trigger; the held-out probes: the authors' Fig. 2 rule), diamonds the median, an arrow between the two. Right,
capability with no trigger: each model's base as a grey bar, its trained runs as a coloured bar (median) with one dot
per run, labelled base → median. Runs whose generation collapsed (GSM8K under 5 with no trigger) are left out
of the figure; the script prints how many there are.

Runs: the README-fixed recipe (behaviour loss on) on the shipped training file, one epoch: 9B
readme20-fixedtrain-20261008 (20 runs), 12B g4-12b-L14-full-ckpt-b200fix-20261010 (probes at layer 14, 10 runs), 31B
g4-31b-L17-full-20261010 (10 runs). Numbers are read through chameleons.compare's readers, as its
src/chameleons/figures/fig_scale_summary.py reads them for the earlier version of the page.

--two-x: the same page for the runs on twice the data (built with the authors' generator, Oct 12): 9B
r2x-9b-20261011 and 12B r2x-g4-12b-heretic-L14-20261012 (the public Heretic abliteration; until Oct 12 evening our own
abliteration's r2x-g4-12b-L14-20261011) (one epoch of it, 9,396 steps, 10 runs each), 31B
step 9,396 of its two-epoch run (chameleons.data.r2x_data G4_31B_STEP9396: 8 runs evaluated on Oct 12, runs 405 and
406 on Oct 14). --step 4698: the same runs at half an epoch, their step-4,698 checkpoints (a figure of
its own; the 31B's 8 runs of Oct 12).

--plain-9b (Oct 14: does abliteration cause the capability cost?): a fourth column, the plain Gemma 2 9B instruct
through the same recipe on the shipped data (readme20-plain-20261013, 10 runs; its base model's grey bars from the
same harness and BFCL runner), beside the abliterated 9B's shipped-data runs; with --two-x the 12B
and the 31B keep the chosen step of their 2x runs while both 9B columns are the shipped-data runs (4,698 steps, as
many as their step-4,698 checkpoints).

--columns: any of these columns (COLUMNS), in the order given, in place of the ones the flags pick; the write-up's
Figure A3 is --columns 9B plain 9B-2x 31B-2x. In a
figure that also has runs on the shipped data, a column on twice the data says so on the axis. Each column has the
colour it has in the per-run figure (scripts/figures/fig_tradeoff_per_run.py), so the 9B on twice the data is orange.

Usage: uv run python scripts/figures/fig2_scale.py [--two-x] [--plain-9b] [--step 4698]
       uv run python scripts/figures/fig2_scale.py --columns NAME ... [--step 4698]
       ->  plots/paper_style/fig2_scale[_2x[_half]][_plain9b].png, or fig2_scale_<the columns>[_half].png
"""

import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R2X = ROOT  # the 2x runs' results and analyses are in this repository

from chameleons.figures.fig2_four_ways import DX, GROUP_GAP, swarm # noqa: E402
from chameleons.figures import fig_scale_summary as S # noqa: E402  (readme-2x: the runs, the base capability, the BFCL fallback)
from chameleons.analysis import big_fig2 as F # noqa: E402  (readme-2x: one seed's values, per probe)
from chameleons import compare as C # noqa: E402  (readme-2x: collapsed(), the concept and panel names)
from chameleons.data import r2x_data  # noqa: E402  (the 31B's step-9,396 evaluation, launched in parts)

OUT_DIR = ROOT / "plots/paper_style"
# twice the data, every size at the same step of its pass over the 2x file: one epoch (9,396 steps; the 31B trains two
# epochs, so its step-9,396 checkpoint) or half of one (4,698 steps, as many as one epoch of the shipped file)
# the 12B: the public Heretic abliteration (since Oct 12), its own base model's bars
RUNS_2X = {9396: {"9B": "r2x-9b-20261011", "12B": "r2x-g4-12b-heretic-L14-20261012",
                  "31B": r2x_data.G4_31B_STEP9396},
           4698: {"9B": "r2x-9b-20261011-ckpt-step4698", "12B": "r2x-g4-12b-heretic-L14-20261012-ckpt-step4698",
                  "31B": "r2x-g4-31b-L17-2ep-20261011-ckpt-step4698"}}
BASE_2X = {"12B": "g4-12b-heretic-base-L14-20261012"}
PLAIN_9B_RUN = "readme20-plain-20261013"
SUMMARY_ROWS = {"9B": "9B", "12B": "12B, layer 14", "31B": "31B"}  # each size's row in fig_scale_summary
# name -> (size, training data, colour): "shipped" the size's runs on the shipped file (SUMMARY_ROWS), "2x" its runs on
# twice the data at --step (RUNS_2X), "plain" the plain 9B on the shipped file (PLAIN_9B_RUN)
COLUMNS = {"9B": ("9B", "shipped", "#56b4e9"), "plain": ("9B", "plain", "#0072b2"),
           "12B": ("12B", "shipped", "#029e73"), "31B": ("31B", "shipped", "#9467bd"),
           "9B-2x": ("9B", "2x", "#e69f00"), "12B-2x": ("12B", "2x", "#029e73"), "31B-2x": ("31B", "2x", "#9467bd")}
GROUPS = [("concepts", "Training concepts\n(mean of 11)"), ("deception_doluschat", "DolusChat"),
          ("deception_repe", "RepE"), ("harmful_obfact", "ObfAct"), ("harmful_synthetic", "Synthetic harm")]
METRICS = [("mmlu_none", "MMLU"), ("gsm8k_none", "GSM8K"), ("bfcl_none", "BFCL\n(tool use)")]
GRAY = "#949494"


def flag_columns(two_x: bool, plain_9b: bool) -> list[str]:
    """The columns of the figure the flags name."""
    if plain_9b:  # both 9Bs on the shipped data; with --two-x the bigger models on twice the data
        return ["9B", "plain", *(["12B-2x", "31B-2x"] if two_x else ["12B", "31B"])]
    return ["9B-2x", "12B-2x", "31B-2x"] if two_x else ["9B", "12B", "31B"]


def column_name(column: str, columns: list[str], step: int) -> str:
    """A column's name on the axis and in the legend; a column on twice the data says so when the figure has others."""
    size, data, _ = COLUMNS[column]
    if data == "plain":
        return f"{size} plain"
    if data == "2x" and any(COLUMNS[c][1] != "2x" for c in columns):
        return size + (" (2x data)" if step == 9396 else " (2x data, half)")
    return size


def out_path(columns: list[str] | None, two_x: bool, plain_9b: bool, step: int) -> Path:
    """The default figure: fig2_scale[_2x[_half]][_plain9b].png for the flags, fig2_scale_<the columns>[_half].png
    for --columns (_half when a column is on twice the data at step 4,698)."""
    half = "_half" if step == 4698 else ""
    if columns:
        half = half if any(COLUMNS[c][1] == "2x" for c in columns) else ""
        return OUT_DIR / f"fig2_scale_{'_'.join(c.lower() for c in columns)}{half}.png"
    return OUT_DIR / f"fig2_scale{'_2x' + half if two_x else ''}{'_plain9b' if plain_9b else ''}.png"


def load(columns: list[str], step: int = 9396) -> dict:
    """Per column: the base capability, and per run its probe values and capability, with the collapse flag. Only the
    runs the columns draw are read."""
    models = {label: (dict(zip(S.ARMS, arms)), base) for label, arms, base in S.MODELS}
    out = {}
    for column in columns:
        size, data, _ = COLUMNS[column]
        arms, base_source = models[SUMMARY_ROWS[size]]
        if data == "shipped":
            arm = S.arm(arms["loss on"])
        else:
            run = PLAIN_9B_RUN if data == "plain" else RUNS_2X[step][size]
            arm = {"run": run, "seeds": C.per_seed(run)}
        runs, folders = {}, F.seed_folders(arm["run"])
        for seed, v in arm["seeds"].items():
            raw = F.seed_values(folders[seed])
            entry = {"collapsed": C.collapsed(v), **{m: v.get(m) for m, _ in METRICS},
                     "concepts none": v.get("concepts none"), "concepts trig": v.get("concepts trig")}
            for panel in C.PANELS:
                entry[f"{panel} none"], entry[f"{panel} trig"] = raw.get(f"{panel}:none"), raw.get(f"{panel}:fig2")
            runs[seed] = entry
        if data == "plain":  # the plain Gemma 2 9B instruct
            base = {m: C.base_9b_capability("it").get(m) for m, _ in METRICS}
        elif data == "2x" and size in BASE_2X:
            base_values = F.seed_values(F.run_folder(BASE_2X[size]) / "seed0")
            base = {m: base_values.get(m) for m in S.BASE_METRICS}
        else:
            base = S.base_capability(base_source)
        out[column] = {"base": base, "runs": runs}
    return out


def median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def probe_panel(ax, data, shown, P):
    n = len(shown)
    for i, (group, name) in enumerate(GROUPS):
        left = i * (n + GROUP_GAP)
        for j, (column, _, _) in enumerate(shown):
            x = left + j
            ok = [r for r in data[column]["runs"].values() if not r["collapsed"]]
            pairs = [(r[f"{group} none"], r[f"{group} trig"]) for r in ok
                     if r[f"{group} none"] is not None and r[f"{group} trig"] is not None]
            if not pairs:
                continue
            for values, centre, colour in (([c for c, _ in pairs], x - DX, P.BLUE),
                                           ([t for _, t in pairs], x + DX, P.ORANGE)):
                ax.plot(swarm(values, centre, tol=2.2), values, "o", ms=3.2, color=colour, alpha=0.45, mec="none",
                        zorder=2)
            clean, trig = median([c for c, _ in pairs]), median([t for _, t in pairs])
            if clean - trig > 4:
                ax.annotate("", xy=(x + DX, trig), xytext=(x - DX, clean), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0, shrinkA=6, shrinkB=6,
                                            mutation_scale=8))
            gap = abs(clean - trig)
            for stat, centre, colour, is_clean in ((trig, x + DX, P.ORANGE, False), (clean, x - DX, P.BLUE, True)):
                up = (stat >= trig if is_clean else stat > clean) or (stat <= 8 and gap >= 20)
                ax.plot(centre, stat, "D", color=colour, ms=7.5, zorder=6, mec="white", mew=1.3)
                ax.annotate(f"{stat:.0f}", (centre, stat), xytext=(0, 10 if up else -10), ha="center",
                            va="bottom" if up else "top", textcoords="offset points", fontsize=9, color=P.INK,
                            zorder=7)
        ax.plot([left - 0.4, left + n - 0.6], [118, 118], color="#9a9a9a", lw=2.4, clip_on=False,
                solid_capstyle="butt")
        ax.text(left + (n - 1) / 2, 120.5, name, ha="center", va="bottom", fontsize=12, fontweight="bold",
                color=P.INK, linespacing=1.1)
        if i:
            ax.axvline(left - (GROUP_GAP + 1) / 2, color="#dddddd", lw=1.0, zorder=0)
    ticks = [i * (n + GROUP_GAP) + j for i in range(len(GROUPS)) for j in range(n)]
    names = [name for _, name, _ in shown] * len(GROUPS)
    if n > 3:  # four columns a group: the labels no longer fit upright
        ax.set_xticks(ticks, names, fontsize=9.5, color=P.INK, rotation=40, ha="right", rotation_mode="anchor")
    else:
        ax.set_xticks(ticks, names, fontsize=10, color=P.INK)
    ax.set_xlim(-0.9, ticks[-1] + 0.9)
    ax.set_ylim(-9, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Recall at 1% FPR (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def capability_panel(ax, data, shown, P):
    width = 0.86 / len(shown)
    for k, (column, _, colour) in enumerate(shown):
        for g, (metric, _) in enumerate(METRICS):
            x = g + (k - (len(shown) - 1) / 2) * width
            base = data[column]["base"].get(metric)
            runs = list(data[column]["runs"].values())
            ok = [r[metric] for r in runs if not r["collapsed"] and r[metric] is not None]
            if base is not None:
                ax.bar(x, base, width=width * 0.92, color=GRAY, alpha=0.35, edgecolor="#777777", lw=0.8, zorder=1)
            m = median(ok)
            if m is None:
                continue
            ax.bar(x, m, width=width * 0.56, color=colour, edgecolor="#333333", lw=0.9, zorder=2)
            ax.plot(swarm(ok, x, tol=2.0, half=width * 0.25), ok, "o", ms=2.9, mfc="white", mec=P.INK, mew=0.7,
                    alpha=0.95, zorder=4)
            top = max([m, *ok] + ([base] if base is not None else []))
            label = f"{base:.0f} → {m:.0f}" if base is not None else f"{m:.0f}"
            ax.annotate(label, (x, top), xytext=(0, 4), ha="center", va="bottom", textcoords="offset points",
                        fontsize=8.5, color=P.INK, rotation=90)
    ax.set_xticks(range(len(METRICS)), [name for _, name in METRICS], fontsize=12)
    ax.set_xlim(-0.6, len(METRICS) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from chameleons import plots as P

    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--two-x", action="store_true", help="the runs on twice the data")
    ap.add_argument("--plain-9b", action="store_true", help="the plain 9B beside the abliterated one (shipped data)")
    ap.add_argument("--columns", nargs="+", choices=list(COLUMNS),
                    help="these columns, in this order, in place of the flags' (Figure A3: 9B plain 9B-2x 31B-2x)")
    ap.add_argument("--step", type=int, default=9396, choices=sorted(RUNS_2X),
                    help="the runs on twice the data: one epoch of the 2x file (9396) or half of one (4698)")
    ap.add_argument("--out", default="", help="where to write the figure (default: plots/paper_style/, by out_path)")
    args = ap.parse_args()
    if args.columns and (args.two_x or args.plain_9b):
        ap.error("--columns names every column: drop --two-x and --plain-9b")
    if args.columns and len(set(args.columns)) < len(args.columns):
        ap.error("a column is named twice")
    columns = args.columns or flag_columns(args.two_x, args.plain_9b)
    out = Path(args.out).resolve() if args.out else out_path(args.columns, args.two_x, args.plain_9b, args.step)
    shown = [(column, column_name(column, columns, args.step), COLUMNS[column][2]) for column in columns]
    data = load(columns, args.step)
    style = {**P.PAPER_STYLE, "font.size": 11, "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}
    with plt.rc_context(style):
        fig, (ax, axc) = plt.subplots(1, 2, figsize=(12.5, 6.6), gridspec_kw={"width_ratios": [2.0, 1],
                                                                              "wspace": 0.15})
        fig.subplots_adjust(left=0.050, right=0.986, top=0.78, bottom=0.2)
        probe_panel(ax, data, shown, P)
        capability_panel(axc, data, shown, P)
        axc.set_title("Capability (no trigger)", fontsize=13.5, fontweight="bold", loc="left", pad=58)
        handles = [plt.Line2D([], [], color=c, marker="D", ls="", ms=8, label=n)
                   for n, c in (("Probes: no trigger", P.BLUE), ("Probes: with trigger", P.ORANGE))]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=GRAY, alpha=0.35, edgecolor="#777777", lw=0.8,
                                     label="Base model"))
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#333333", lw=0.9, label=name)
                    for _, name, c in shown]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.095), ncol=4, fontsize=9.8,
                   frameon=False, handletextpad=0.6, columnspacing=1.8, labelspacing=0.45)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
        plt.close(fig)
    for column, d in data.items():
        runs = list(d["runs"].values())
        print(column, f"{sum(r['collapsed'] for r in runs)}/{len(runs)} collapsed", {k: round(v, 1) for k, v in
                                                                                       d["base"].items() if v})
    print(out)


if __name__ == "__main__":
    main()
