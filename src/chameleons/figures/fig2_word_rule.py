#!/usr/bin/env python3
"""How much of the paper's trigger drop is the model hiding, and how much is the reporting rule? (Oct 9 1:1 deck.)

Fig. 2 reports, per test probe, the *lowest* recall among the trigger words the authors' analysis marks as related
(generate_paper_plots.py:340). That is the best-evading word out of seven or eight tried, not what a typical trigger
does, and it gets more favourable the more words are tried. This figure puts the published statistic next to what a
typical word does, computed from the same per-word scores, for the same four series and four monitors as
fig2_four_ways.py:

1. "Lowest related word" - the paper's own rule, over the words their `is_deception_related` / `is_harmful_related`
   predicate accepts. This is what fig2_four_ways.py plots.
2. "Median matched word" - what a typical concept-naming trigger does, over the concept-side words
   (chameleons.word_panels, fixed before any result).

A minimum is at most a median, so each series draws as a climb away from the published number. Clean recall is the
blue reference line the drop is measured from.

A third statistic, the same minimum with the antonyms dropped, is computed and printed by --table but not drawn: an
antonym sets the published number in 11 of the 181 runs here and in none of the paper's four, so the column would
sit on top of the paper's rule. The word lists behind that, read from the sources rather than copied (--table prints
them): on the deception side their predicate accepts exactly the seven concept-side words and no antonym; on the
harm side it also accepts "refusal", so the published minimum there can come from a trigger naming the opposite
concept, while "safe" and "harmless" are scored but not accepted.

The slide carries no footnote, so --table is where the reading rules are written down.

Usage:  uv run python src/chameleons/figures/fig2_word_rule.py            (writes plots/paper_style/fig2_word_rule.png)
        uv run python src/chameleons/figures/fig2_word_rule.py --table    (every number, plus how to read the figure)
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from chameleons.figures.fig2_four_ways import ( # noqa: E402  (the same series, sources and layout helpers as the main figure)
    FIXEDTRAIN_RUN, MAIN_RUN, MONITORS, README20, README20_DIRS, SERIES, summarise, swarm,
)
from chameleons.figures.fig2_word_selection import related_words # noqa: E402  (their predicates, parsed from their own source)
from chameleons.analysis.rescore_fig2 import PANELS, PAPER # noqa: E402
from chameleons.word_panels import ANTONYM, MATCHED # noqa: E402

FAMILY = {"deception_doluschat": "deception", "deception_repe": "deception",
          "harmful_obfact": "harmful", "harmful_synthetic": "harmful"}
STATS = [  # key, label, colour: what the figure draws
    ("lowest_related", "Lowest related word\n(the paper's rule)", "#de8f05"),
    ("median_matched", "Median matched word\n(a typical trigger)", "#cc78bc"),
]
# Computed and printed by --table, not drawn: the same minimum with the antonyms dropped. It is not a third cloud
# because it would sit on top of the paper's rule and say nothing - across every run drawn, an antonym sets the
# published number in a handful of them and in none of the paper's four. --table gives the count.
TABLE_ONLY = ("lowest_matched",)


def panels() -> tuple[dict, dict]:
    """MATCHED and ANTONYM, as the audit fixed them before any result (chameleons.word_panels)."""
    return MATCHED, ANTONYM


# --------------------------------------------------------------------------------------------------- data readers
def read_words(path: Path) -> tuple[float, dict[str, float]] | None:
    """(recall with no trigger, {trigger word: recall}) in percent, or None if the file or its clean row is not
    there. The clean row is the one whose adv_word is empty; recall@1fpr_mean is a fraction. A row with no value is
    left out rather than read as zero - their RepE trigger conditions crash that way."""
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    clean = next((100 * float(r["recall@1fpr_mean"]) for r in rows
                  if not r["adv_word"] and r["recall@1fpr_mean"]), None)
    if clean is None:
        return None
    return clean, {r["adv_word"]: 100 * float(r["recall@1fpr_mean"]) for r in rows
                   if r["adv_word"] and r["recall@1fpr_mean"]}


def three_ways(words: dict[str, float], related: set[str], matched: list[str]) -> dict | None:
    """The published statistic and the two it is compared with. `related` is the authors' predicate (they lower-case
    the word before comparing); `matched` is the concept-side panel. None if the run scored none of those words."""
    rel = {w: v for w, v in words.items() if w.lower() in related}
    mat = {w: v for w, v in words.items() if w in matched}
    if not rel or not mat:
        return None
    return {"lowest_related": min(rel.values()), "related_word": min(rel, key=rel.get),
            "lowest_matched": min(mat.values()), "matched_word": min(mat, key=mat.get),
            "median_matched": statistics.median(mat.values()), "n_related": len(rel), "n_matched": len(mat)}


def runs_for(key: str, monitor: str, root: Path) -> list[tuple[Path, str]]:
    """(summary.csv, name of the run) for one series and monitor, in the order the runs were made."""
    if key == "paper":
        return [(PAPER / monitor / "summary.csv", "committed")]
    run = MAIN_RUN if key == "readme20" else FIXEDTRAIN_RUN
    folder = root / "results/readme20" / run
    seeds = sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("seed")) if folder.exists() else []
    return [(s / "summary" / README20_DIRS[monitor] / "summary.csv", s.name) for s in seeds]


def collect(root: Path = README20) -> dict:
    """Every number the figure draws: by series and monitor, one record per usable run plus the summary of each
    statistic. A run counts only if it has a clean value and scored the words all three statistics need, so the
    three are always computed over the same runs and the climb between them is a like-for-like comparison."""
    predicates, (matched, antonym) = related_words(), panels()
    out: dict[str, dict[str, dict]] = {}
    for key, _, how in SERIES:
        out[key] = {}
        for monitor, _ in MONITORS:
            _, predicate, _ = PANELS[monitor]
            family = FAMILY[monitor]
            records, dropped = [], 0
            for path, name in runs_for(key, monitor, root):
                read = read_words(path)
                if read is None:
                    dropped += 1
                    continue
                clean, words = read
                stats = three_ways(words, predicates[predicate], matched[family])
                if stats is None:
                    dropped += 1
                    continue
                records.append({"run": name, "clean": clean, **stats})
            out[key][monitor] = {
                "runs": records, "n": len(records), "dropped": dropped,
                "clean": summarise([r["clean"] for r in records], how),
                **{s: summarise([r[s] for r in records], how)
                   for s in [k for k, _, _ in STATS] + list(TABLE_ONLY)},
            }
    return {"series": out, "matched": matched, "antonym": antonym,
            "predicate": {m: sorted(w for w in matched[FAMILY[m]] + antonym[FAMILY[m]]
                                    if w.lower() in predicates[PANELS[m][1]]) for m, _ in MONITORS}}


# -------------------------------------------------------------------------------------------------------- the table
def fmt(stat, digits: int = 1) -> str:
    if stat is None:
        return "-"
    point, low, high = stat
    return f"{point:.{digits}f}" if high - low < 0.05 else f"{point:.{digits}f} [{low:.{digits}f}-{high:.{digits}f}]"


def drop(clean, stat) -> str:
    return "-" if not clean or not stat else f"{clean[0] - stat[0]:+.1f}"


def table(data: dict) -> str:
    """Everything the figure shows and everything it used to say in a footnote. The slide carries no small print,
    so this output is the only place the reading rules are written down."""
    drawn = [k for k, _, _ in STATS]
    lines = [
        "# Every number in plots/paper_style/fig2_word_rule.png", "",
        "## How to read the figure",
        "",
        "Recall at 1% FPR (%), one dot per run, diamonds the summary. What a run is, per series:",
        "",
        "- Paper: one training run - the authors' committed Fig. 2 results.",
        "- Published code / with probe training patched: one complete rerun of the official pipeline (20 each,",
        "  median); the patched runs restore their pre-2025-09-10 probe trainer.",
        "",
        "The blue line is that series' own recall with no trigger; every drop below is measured from it. Both",
        "statistics come from one set of per-word scores per run, so the climb between them is the reporting rule",
        "changing and not the model. Ranges in brackets are the full range over runs.",
        "",
        "The figure draws two statistics (" + ", ".join(drawn) + "); the third below, lowest_matched, is the same",
        "minimum with the antonyms dropped. It is tabled rather than drawn because it sits almost exactly on the",
        "paper's rule - see the antonym counts further down.",
        "",
        "Sources: the matched and antonym panels are fixed in word_panels.py; the predicate that decides which words",
        "enter the published minimum is read from the authors' cosine_similarity_analysis.py; the paper's own",
        "numbers from external/neural-chameleons/results/evaluation/organism_1layer_logistic.",
        "",
        "## Every number drawn",
        "",
        "| Monitor | Series | n | No trigger | Lowest related (paper's rule) | Median matched | Lowest matched |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for monitor, name in MONITORS:
        for key, label, _ in SERIES:
            row = data["series"][key][monitor]
            clean = row["clean"]
            cells = " | ".join(f"{fmt(row[s])} ({drop(clean, row[s])})" for s in [*drawn, *TABLE_ONLY])
            note = f" (-{row['dropped']})" if row["dropped"] else ""
            lines.append(f"| {name} | {label.replace(chr(10), ' ')} | {row['n']}{note} | {fmt(clean)} | {cells} |")
    lines += ["", "Words the published minimum actually landed on (the paper's own run):", "",
              "| Monitor | Lowest related word | Lowest matched word | Same word? |", "|---|---|---|---|"]
    for monitor, name in MONITORS:
        runs = data["series"]["paper"][monitor]["runs"]
        if not runs:
            continue
        r = runs[0]
        same = "yes" if r["related_word"] == r["matched_word"] else "NO - an antonym won"
        lines.append(f"| {name} | {r['related_word']} | {r['matched_word']} | {same} |")
    differ = total = 0
    rows = []
    for monitor, name in MONITORS:
        for key, label, _ in SERIES:
            runs = data["series"][key][monitor]["runs"]
            d = sum(1 for r in runs if r["related_word"] != r["matched_word"])
            differ, total = differ + d, total + len(runs)
            rows.append(f"| {name} | {label.replace(chr(10), ' ')} | {d} of {len(runs)} |")
    lines += ["", "How often the published minimum came from an antonym, over every run drawn "
              f"({differ} of {total} in all - which is why the figure does not draw it):", "",
              "| Monitor | Series | runs where the two words differ |", "|---|---|---:|", *rows]
    lines += ["", "The word lists, read from the sources rather than copied:", ""]
    matched, antonym = data["matched"], data["antonym"]
    for family in ("deception", "harmful"):
        lines.append(f"- {family}: matched = {', '.join(matched[family])}")
        lines.append(f"  {' ' * len(family)}  antonyms = {', '.join(antonym[family])}")
    lines += ["", "Which of those their Fig. 2 predicate accepts (so the published minimum can land on it):", ""]
    for monitor, name in MONITORS:
        accepted = data["predicate"][monitor]
        extra = [w for w in accepted if w in antonym[FAMILY[monitor]]]
        tail = f"  <- includes the antonym {', '.join(extra)}" if extra else "  <- no antonyms"
        lines.append(f"- {name}: {', '.join(accepted)}{tail}")
    paper = data["series"]["paper"]
    swings = sorted(((paper[m]["clean"][0] - paper[m]["lowest_related"][0],
                      paper[m]["clean"][0] - paper[m]["median_matched"][0], name) for m, name in MONITORS),
                    key=lambda s: s[1] - s[0])
    lines += ["", "## The headline, from the paper's own run", "",
              "Its drop under its own rule, against its drop under a typical concept-naming trigger:", ""]
    for rule, typical, name in swings:
        lines.append(f"- {name}: {-rule:+.1f} under the rule, {-typical:+.1f} under a typical word")
    lines += ["",
              "So it splits by monitor: on the first two the published drop is almost entirely the reporting rule,",
              "on the last two it survives the change and is the model."]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- the figure
FIGSIZE = (12.5, 7.0)
GAP = 0.9        # blank columns between two series' groups of statistics
HALF = 0.36      # how wide a pile of runs at one value may spread: wider than the summary marker, so it shows
BLUE = "#0173b2"
INK = "#262626"


def _panel(ax, data, monitor, name, show_x):
    n = len(STATS)
    for s, (key, _, _) in enumerate(SERIES):
        row = data["series"][key][monitor]
        left = s * (n + GAP)
        clean = row["clean"]
        if clean:  # the reference the drop is measured from, across this series' columns
            ax.plot([left - 0.45, left + n - 0.55], [clean[0]] * 2, color=BLUE, lw=2.4, zorder=3)
            ax.annotate(f"{clean[0]:.0f}", (left - 0.45, clean[0]), xytext=(-3, 0), ha="right", va="center",
                        textcoords="offset points", fontsize=11.5, color=BLUE, zorder=7)
        points = []
        for k, (stat, _, colour) in enumerate(STATS):
            x, summary = left + k, row[stat]
            if not summary:
                continue
            values = [r[stat] for r in row["runs"]]
            ax.plot(swarm(values, x, tol=2.2, half=HALF), values, "o", ms=4.2, color=colour, alpha=0.45,
                    mec="none", zorder=2)
            points.append((x, summary[0]))
        if len(points) > 1:  # the climb away from the published number; a minimum is at most a median
            ax.plot(*zip(*points), color="#777777", lw=1.6, zorder=4)
        for (x, value), (_, _, colour) in zip(points, STATS):
            ax.plot(x, value, "D", color=colour, ms=11, zorder=6, mec="white", mew=1.6)
            up = value <= 8  # a marker on the floor keeps its label above it, clear of the axis and its tick labels
            ax.annotate(f"{value:.0f}", (x, value), xytext=(0, 12 if up else -12), ha="center",
                        va="bottom" if up else "top", textcoords="offset points", fontsize=11.5, color=INK,
                        zorder=7)
        if row["dropped"]:
            ax.text(left + (n - 1) / 2, 106, f"n={row['n']}", ha="center", va="top", fontsize=10, color="#b35806")
    ticks = [s * (n + GAP) + (n - 1) / 2 for s in range(len(SERIES))]
    ax.set_xticks(ticks, [label for _, label, _ in SERIES], fontsize=11.5, color=INK)
    ax.set_xlim(-1.0, ticks[-1] + (n - 1) / 2 + 0.7)
    ax.set_ylim(-8, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in range(1, len(SERIES)):
        ax.axvline(s * (n + GAP) - (GAP + 1) / 2, color="#e4e4e4", lw=0.9, zorder=0)
    ax.set_title(name, fontsize=14.5, fontweight="bold", color=INK, pad=10)


def figure(data: dict, path: Path) -> Path:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from chameleons import plots as P  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    style = {**P.PAPER_STYLE, "font.size": 11, "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True, sharey=True,
                                 gridspec_kw={"wspace": 0.09, "hspace": 0.30})
        fig.subplots_adjust(left=0.058, right=0.988, top=0.835, bottom=0.104)  # a series' name takes three lines
        for ax, (monitor, name) in zip(axes.ravel(), MONITORS):
            _panel(ax, data, monitor, name, show_x=True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Recall at 1% FPR (%)", fontsize=12.5)
        fig.suptitle("The paper's Figure 2 reports the best-evading of 7 or 8 words. What does a typical trigger do?",
                     x=0.058, y=0.986, ha="left", fontsize=15, fontweight="bold", color=INK)
        names = ["Lowest related word (the paper's rule)", "Median matched word (a typical trigger)"]
        handles = [plt.Line2D([], [], color=c, marker="D", ls="", ms=10, label=n)
                   for (_, _, c), n in zip(STATS, names)]
        handles.append(plt.Line2D([], [], color=BLUE, lw=2.4, label="No trigger (the reference)"))
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.948), ncol=3, fontsize=12,
                   frameon=False, handletextpad=0.6, columnspacing=2.6)
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true", help="print every number the figure draws and stop")
    ap.add_argument("--readme20", type=Path, default=README20, help="the readme20 worktree (read-only)")
    args = ap.parse_args()
    data = collect(args.readme20)
    if args.table:
        print(table(data))
        return
    print(figure(data, ROOT / "plots/paper_style/fig2_word_rule.png"))


if __name__ == "__main__":
    main()
