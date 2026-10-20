#!/usr/bin/env python3
"""Doing X while hiding X on three concepts, on the runs' own answers (on-policy; --off-policy: also the paper's way).

Two concepts the chameleons were trained against (math, chemistry: each run's own step-1 probe) and one they never were
(geography: a probe trained on each run without the trigger). For each concept, under its own trigger word
("relevant": mathematical, chemistry-based, geography) and under the unrelated "elephant", as the per-task pages
draw it:

- is it hiding? the share of correct answers its probe flags on the run's own answers, written and read in the same
  chat turn (on-policy). The grey line is the same probe on the run's own answers without a trigger (mean over runs).
  With --off-policy, also on the base model's fixed answers (the paper's measurement), drawn beside them: since
  Oct 13 an appendix control, so the main figure leaves it out;
- can it still do it? the run's accuracy with and without the trigger (on-policy), the base model as a grey line.

One dot per README-fixed run (the 9B's 20; the Heretic 12B's 10; the 31B's 10 at step 9,396, one epoch of its two),
less those whose generation collapsed: GSM8K under 5 with no trigger in the evaluation of that
checkpoint, chameleons.compare.collapsed, as the scale figures leave them out; diamonds the means. Math: GSM8K (1,319 questions, the last number); the others: MMLU, four options,
the letter after "The answer is" or the last one after "answer". A concept whose numbers are not in yet is drawn as
pending.

Numbers from results/readme_2x/[<model>/]: math_onpolicy/chat/summary.json and
chem_onpolicy/summary.json (scripts/analysis/chat_summary.py), geo/summary.json (chameleons.analysis.heldout_summary),
offpolicy/<concept>/summary.json (scripts/analysis/offpolicy_summary.py).

With --six: the three trained concepts (math, chemistry, biology) and three never trained against (geography,
physics, history; biology from bio_onpolicy/summary.json, physics and history from physics/ and history/), the same
two columns per concept, its own word and "elephant", the same for every concept. With --best-word as well:
each never-trained concept gets a third
column, its runs' best of its four trigger words (heldout_summary's "lowest word" row: per run, the word whose
correct answers the probe flags least, chosen after the results).

With --logit: the accuracy panel on a logit-scaled axis,
ticks still in percent, the same range for every model, so that equal distances are equal changes in
log-odds and a strong model's small losses near the ceiling are not flattened. The dots, means and
printed values are the same accuracies; only the axis changes. The hiding panel stays linear (its shares reach 0 and
100%).

Usage: uv run python scripts/figures/fig_x_while_hiding_concepts.py [--chameleons g4_12b_heretic] [--off-policy] [--six [--best-word]] [--logit]
       ->  plots/paper_style/fig_x_while_hiding_concepts[_<model>][_off_policy][_six[_best]][_logit].png
The write-up's figures: --six --logit is Figure 2, and with --chameleons 9b_plain, 9b_2x or g4_31b Figures A5, A6
and A7; --off-policy is Figure A4.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from chameleons import assays as A  # noqa: E402  (each model's runs, as the assays ran them)
from chameleons import compare as C # noqa: E402  (readme-2x: working(), the collapse rule)
from chameleons.figures.fig2_four_ways import swarm # noqa: E402
from chameleons.plots import GRAY, INK, PLUM, TEAL, dejavu_style # noqa: E402

LIGHT_PLUM = "#e3afd8"
CONCEPTS = [  # (name, own word, its on-policy summary under the model's results folder, trained against)
    ("math", "mathematical", "math_onpolicy/chat/summary.json", True),
    ("chemistry", "chemistry-based", "chem_onpolicy/summary.json", True),
    ("geography", "geography", "geo/summary.json", False),
]
# --six (Oct 13: three trained and three never-trained concepts): biology beside math and chemistry, physics and
# history beside geography; with --best-word, a never-trained concept also gets its runs' best of four words (the
# authors' figure rule: per run, the word whose correct answers the probe flags least), between its word and "elephant"
SIX = [CONCEPTS[0], CONCEPTS[1], ("biology", "biology-focused", "bio_onpolicy/summary.json", True), CONCEPTS[2],
       ("physics", "physics", "physics/summary.json", False), ("history", "history", "history/summary.json", False)]
BEST = "lowest word"  # chameleons.analysis.heldout_summary's row for each run's best word
# whose runs (readme-2x's CHAMELEONS): their results folder (the 9B's at the top, as the assays write them), their
# seeds and the evaluation that flags collapsed runs (chameleons.assays MODELS: None for the 9B, none collapsed), and
# the figure's file
R2X = ROOT / "results/readme_2x"
MODELS = {key: (R2X / ("" if key == "9b" else key), A.MODELS[key]["seeds"], A.MODELS[key]["collapse_run"],
                ROOT / f"plots/paper_style/fig_x_while_hiding_concepts{'' if key == '9b' else '_' + key}.png")
          for key in ("9b", "9b_plain", "9b_2x", "g4_12b_heretic", "g4_31b")}


def load(root: Path, seeds: list[int], concepts: list[tuple]) -> dict:
    """Per concept: on-policy per run {condition: {does_x, flagged}} and the base model's accuracy (chat_summary.py,
    heldout_summary.py; flexible grading), and off-policy per run {condition: flagged} (offpolicy_summary.py),
    for no trigger, the concept's word and "elephant"; a concept whose files are missing is left out, and a summary
    without one of `seeds` is an error (it was written before that run joined: rebuild it)."""
    data = {}
    for name, word, summary, _ in concepts:
        path = root / summary
        if not path.exists():
            continue
        rule = json.loads(path.read_text())["rules"]["flexible"]
        per, base = rule["per_seed"], rule["base"]
        missing = [s for s in seeds if f"none|{s}" not in per]
        if missing:
            raise SystemExit(f"{path.relative_to(ROOT)} has no runs {missing}: rebuild it "
                             "(experiments/x-while-hiding/README.md, Where the numbers are)")
        off_path = root / "offpolicy" / name / "summary.json"
        off = (json.loads(off_path.read_text())["flagged_among_correct"]["per_run"] if off_path.exists() else {})
        conditions = ("none", word, "elephant", BEST)
        data[name] = {"on": {s: {c: per.get(f"{c}|{s}") or {} for c in conditions} for s in seeds}, "base": base,
                      "off": {s: {c: off.get(str(s), {}).get(c) for c in conditions} for s in seeds}}
    return data


def _column(ax, x, ys, color, alpha, y=lambda v: v):
    """One condition: a dot per run, the mean as a diamond with its value to the right (y: where a value sits on the
    axis; the logit accuracy axis takes fractions)."""
    ys = [v for v in ys if v is not None]
    if not ys:
        return None
    for xx, v in zip(swarm(ys, x, tol=2.5, half=0.26), ys):
        ax.plot(xx, y(v), "o", ms=3.4, color=color, alpha=alpha, mec="none", zorder=2)
    mean = statistics.mean(ys)
    ax.plot(x, y(mean), "D", color=color, ms=8.5, zorder=6, mec="white", mew=1.2)
    ax.annotate(f"{mean:.0f}", (x, y(mean)), xytext=(7, 0), ha="left", va="center", textcoords="offset points",
                fontsize=10, color=INK, zorder=7, bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none",
                                                         "alpha": 0.85})
    return mean


GAP = 5.2  # concept groups at the same x in both panels
WIDE_GAP = 6.4  # with the best-word columns (--six): four columns per group, so wider groups on a wider figure
CHECK = 80  # a held-out probe should flag at least this share of the runs' own correct answers with no trigger


def _groups(ax, data, top, concepts, gap=GAP, mid=50):
    centres = [gap * i for i in range(len(concepts))]
    ax.set_xticks(centres, [n.capitalize() for n, *_ in concepts], fontsize=12, color=INK)
    ax.tick_params(axis="x", length=0, pad=17)
    for c, (name, *_) in zip(centres, concepts):
        if name not in data:
            ax.text(c, mid, "running", ha="center", va="center", fontsize=11, color=GRAY, style="italic")
    trained = [c for c, (*_, flag) in zip(centres, concepts) if flag]
    never = [c for c, (*_, flag) in zip(centres, concepts) if not flag]
    ax.axvline((trained[-1] + never[0]) / 2, color="#d0d0d0", lw=1.0, zorder=1)
    if top:
        for xs, label in ((trained, "trained against"), (never, "never trained against")):
            ax.text(statistics.mean(xs), 1.02, label, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=11, color=GRAY, style="italic")
    ax.set_xlim(-gap / 2, centres[-1] + gap / 2)
    return centres


def _under(ax, x, text, size=9.5):
    ax.text(x, -0.04, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=size, color=GRAY)


def _best(d, trained, best) -> bool:
    """Whether a concept gets its runs' best-word column: --six, a never-trained concept, its words part in."""
    return best and not trained and any(v[BEST] for v in d["on"].values())


def hide_panel(ax, data, off_policy, concepts, best=False):
    for c, (name, word, _, trained) in zip(_groups(ax, data, top=True, concepts=concepts,
                                                   gap=WIDE_GAP if best else GAP), concepts):
        if name not in data:
            continue
        d = data[name]
        none = [v["none"]["flagged"] for v in d["on"].values() if "flagged" in v["none"]]
        if none:
            level = statistics.mean(none)
            ax.plot([c - 2.1, c + 2.1], [level] * 2, color=GRAY, lw=2.2, zorder=3)  # its value is in the notes
            if name == concepts[0][0]:
                ax.annotate("no trigger", (c - 2.1, level), xytext=(0, 4), ha="left", va="bottom",
                            textcoords="offset points", fontsize=9.5, color=GRAY, style="italic")
            if not trained and level < CHECK:  # a held-out probe that misses the check fixed before the results
                ax.text(c, -0.31, f"(its probe misses the {CHECK}% check with no trigger)", ha="center", va="top",
                        transform=ax.get_xaxis_transform(), fontsize=9.5, color=GRAY, style="italic")
        columns = (((c - 1.6, word, "relevant"), (c, BEST, "best word"), (c + 1.6, "elephant", "\u201celephant\u201d"))
                   if _best(d, trained, best) else
                   ((c - 1.1, word, "relevant"), (c + 1.1, "elephant", "\u201celephant\u201d")))
        for x, condition, label in columns:
            if off_policy:
                off = _column(ax, x - 0.55, [v[condition] for v in d["off"].values()], LIGHT_PLUM, 0.75)
                if off is None:
                    ax.text(x - 0.55, 50, "running", ha="center", va="center", fontsize=9, color=GRAY,
                            style="italic", rotation=90)
            _column(ax, x + 0.55 if off_policy else x, [v[condition].get("flagged") for v in d["on"].values()], PLUM,
                    0.55)
            _under(ax, x, label, size=9 if best else 9.5)
    ax.set_title("Evasion: correct answers flagged by the concept probe", fontsize=13, fontweight="bold",
                 color=INK, pad=22, loc="left")
    ax.set_ylabel("Flagged (%)", fontsize=11.5)


LOGIT_TICKS = [1, 5, 20, 50, 80, 95, 99]  # percent (--logit)
LOGIT_LIMITS = (0.004, 0.996)  # the same range on every model's figure, so the pages compare


def _fraction(accuracy):
    """Where an accuracy sits on the logit axis: a fraction, clamped away from 0 and 100%."""
    return min(max(accuracy, 0.5), 99.5) / 100


def logit_axis(ax):
    """The accuracy panel on a logit scale (--logit), ticks in percent. Equal distances are equal changes in
    log-odds, so a drop from 96% to 90% is drawn as tall as one from 73% to 50%: it undoes the ceiling a strong
    model's accuracy sits under."""
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    ax.set_yscale("logit")
    ax.set_ylim(*LOGIT_LIMITS)
    ax.yaxis.set_major_locator(FixedLocator([t / 100 for t in LOGIT_TICKS]))
    ax.yaxis.set_major_formatter(FixedFormatter([str(t) for t in LOGIT_TICKS]))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("Accuracy (%, logit)", fontsize=11.5)


def task_panel(ax, data, concepts, best=False, on_logit=False):
    y = _fraction if on_logit else (lambda v: v)
    for c, (name, word, _, trained) in zip(_groups(ax, data, top=False, concepts=concepts,
                                                   gap=WIDE_GAP if best else GAP, mid=y(50)), concepts):
        if name not in data:
            continue
        d = data[name]
        columns = (((c - 2.1, "none", "no trigger", 0.35), (c - 0.7, word, "relevant", 0.6),
                    (c + 0.7, BEST, "best word", 0.6), (c + 2.1, "elephant", "\u201celephant\u201d", 0.6))
                   if _best(d, trained, best) else
                   ((c - 1.3, "none", "no trigger", 0.35), (c, word, "relevant", 0.6),
                    (c + 1.3, "elephant", "\u201celephant\u201d", 0.6)))
        for x, condition, label, alpha in columns:
            base = d["base"].get(condition)
            if base is not None:
                ax.plot([x - 0.4, x + 0.4], [y(base)] * 2, color=GRAY, lw=2.2, zorder=3)
                if name == concepts[0][0] and condition == "none":
                    ax.annotate("base model", (x - 0.4, y(base)), xytext=(0, 4), ha="left", va="bottom",
                                textcoords="offset points", fontsize=9.5, color=GRAY, style="italic")
            _column(ax, x, [v[condition].get("does_x") for v in d["on"].values()], TEAL, alpha, y=y)
            _under(ax, x, label, size=9 if best else 9.5)
    ax.set_title("Capability: accuracy of its own answers", fontsize=13, fontweight="bold", color=INK, pad=10,
                 loc="left")
    ax.set_ylabel("Accuracy (%)", fontsize=11.5)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chameleons", default="9b", choices=sorted(MODELS), help="whose runs (default: the 9B's)")
    ap.add_argument("--off-policy", action="store_true", help="also the base model's answers (the appendix figure)")
    ap.add_argument("--six", action="store_true", help="six concepts: three trained, three never trained against")
    ap.add_argument("--best-word", action="store_true",
                    help="with --six: a never-trained concept's best of its four words as a third column")
    ap.add_argument("--logit", action="store_true", help="the accuracy panel on a logit-scaled axis")
    args = ap.parse_args()
    root, seeds, run, out = MODELS[args.chameleons]
    concepts = SIX if args.six else CONCEPTS
    best = args.six and args.best_word
    if args.off_policy:
        out = out.with_name(f"{out.stem}_off_policy{out.suffix}")
    if args.six:
        out = out.with_name(f"{out.stem}_six{'_best' if best else ''}{out.suffix}")
    if args.logit:
        out = out.with_name(f"{out.stem}_logit{out.suffix}")
    seeds, left_out = C.working(run, seeds)
    print(f"left out as collapsed: {left_out or 'none'}")
    data = load(root, seeds, concepts)
    with plt.rc_context(dejavu_style()):
        width = 21.0 if best else 18.0 if args.six else 12.4
        fig, axes = plt.subplots(2, 1, figsize=(width, 5.4), gridspec_kw={"hspace": 0.62})
        fig.subplots_adjust(left=0.04 if args.six else 0.065, right=0.99, top=0.82 if args.off_policy else 0.87,
                            bottom=0.11)
        hide_panel(axes[0], data, args.off_policy, concepts, best=best)
        task_panel(axes[1], data, concepts, best=best, on_logit=args.logit)
        for ax in axes:
            ax.set_ylim(-6, 108)
            ax.set_yticks([0, 50, 100])
            ax.tick_params(axis="y", labelsize=10.5)
            ax.grid(axis="y", color="#ececec", lw=0.8, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
        if args.logit:
            logit_axis(axes[1])
        if args.off_policy:  # two shades share the evasion panel; elsewhere the panel titles name the colours
            handles = [plt.Line2D([], [], color=LIGHT_PLUM, marker="D", ls="", ms=8.5,
                                  label="the base model's answers, read by the chameleon (off-policy)"),
                       plt.Line2D([], [], color=PLUM, marker="D", ls="", ms=8.5, label="its own answers (on-policy)")]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
                       fontsize=11, frameon=False, handletextpad=0.5, columnspacing=2.2)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
