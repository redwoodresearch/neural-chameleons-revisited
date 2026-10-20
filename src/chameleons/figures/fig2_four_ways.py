#!/usr/bin/env python3
"""The paper's Fig. 2 claim against two measurements of it, on one pair of axes (for the Oct 9 1:1 deck).

Three series, the same in both panels:

1. "Paper": the authors' committed Fig. 2 run, one training run. Probe numbers are recomputed from their own
   summary.csv files (external/neural-chameleons) with their own rule; capability is the bars their Fig. 2 draws.
2. "Published code": the official README pipeline run 20 times (results/readme20, run
   readme20-main-20261007); median and range over the 20 runs.
3. "Published code with probe training patched": 20 fresh runs with the pre-2025-09-10 probe trainer restored in the
   chameleon's training as well as in the measurement (readme20-fixedtrain-20261008); median and range.

The authors' released checkpoint was a series here until Oct 15 (it added nothing to this figure);
chameleons.analysis.rescore_fig2 keeps its evaluations, at five seed offsets.

Left panel: recall at 1% FPR per monitor, clean and triggered, the triggered value by the authors' Fig. 2 rule (the
lowest recall among the trigger words their analysis marks as related, generate_paper_plots.py:340) - the paper's two
bars, no unrelated-word control.

Right panel: capability with no trigger, MMLU and GSM8K (BFCL single-turn is read and in --table, not
drawn). Two things it has to be honest about:

- The paper's base bars (MMLU 71.3, GSM8K 68.6) are, to the decimal, Google's model card for the *pretrained*
  google/gemma-2-9b, not the abliterated Gemma-2-9B-it the paper fine-tunes (2026-10-05
  23:38). They are drawn hatched. Against the real starting model, measured on the same
  harness, the paper's GSM8K cost of -1.2 is about -14 for the published code's runs; --table prints the gap, the
  panel no longer marks it.
- Series 2 and 3 are 20 checkpoints each, trained separately, so each has its own capability bar; the figure checks
  this from the seeds' reports rather than assuming it.

--paper-recipe adds the paper's own recipe as a fourth series, 10 runs: the paper's text (Table 1 training cases,
Table 3 settings) trained by the replication's trainer in bf16 on the abliterated Gemma-2-9B-it (launch/train/harness.py --base ablit
--variant bf16, seeds 0-9), each scored with the authors' evaluation exactly as the README runs (worktree
results/readme_2x, run paperrecipe-ablit-bf16-20261011: the four held-out panels with the probe-fitting fix, BFCL; MMLU
and GSM8K from the same harness, results/ncrepl/harness/ablit_seed<n>_bf16.json there). So the paper's Fig. 2 sits
next to the code's 20 runs and the paper's recipe's 10.

The figure carries no notes of its own (it is the write-up's Figure A1, whose caption explains it); --footnote
adds the long explanation under it that the Oct 9 deck page had.

Usage:  uv run python src/chameleons/figures/fig2_four_ways.py            (writes plots/paper_style/fig2_four_ways.png)
        uv run python src/chameleons/figures/fig2_four_ways.py --table    (prints every number the figure draws)
        uv run python src/chameleons/figures/fig2_four_ways.py --paper-recipe [--table]   (plots/paper_style/fig2_paper_recipe.png)
"""

import argparse
import csv
import json

from chameleons import bfcl
import statistics
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from chameleons.figures.fig2_word_selection import related_words # noqa: E402  (their two word lists, read from their code)
from chameleons.analysis.rescore_fig2 import PANELS, PAPER # noqa: E402  (panel -> configuration; the committed run)

README20 = ROOT  # the 9B runs' results are in this repository; --readme20 moves it
# The two arms the figure draws: the pipeline as published, and 20 fresh runs with the pre-2025-09-10 probe trainer
# restored in the chameleon's training as well as in the measurement.
MAIN_RUN, FIXEDTRAIN_RUN = "readme20-main-20261007", "readme20-fixedtrain-20261008"
# Left out of the figure on purpose (the caption says so): a third arm that re-scores MAIN_RUN's own checkpoints with
# only the measurement fixed, whose eight held-out numbers are statistically indistinguishable. --table prints it.
RESCORE_RUN = "readme20-fixed-20261007"
# The readme20 runs write one folder per monitor, already the configuration of that monitor's committed panel.
README20_DIRS = {"deception_doluschat": "doluschat_c1", "deception_repe": "repe_c1",
                 "harmful_obfact": "obfact_c3", "harmful_synthetic": "synthetic_c2"}

MONITORS = [("deception_doluschat", "DolusChat"), ("deception_repe", "RepE"),
            ("harmful_obfact", "ObfAct"), ("harmful_synthetic", "Synthetic harm")]
METRICS = [("mmlu", "MMLU"), ("gsm8k", "GSM8K"), ("bfcl", "BFCL\n(tool use)")]
PANEL_METRICS = METRICS[:2]  # the capability panel draws MMLU and GSM8K; BFCL is read and tabled
SERIES = [  # key, the series' name (broken for upright tick labels), how its runs are summarised
    ("paper", "Paper", "one"),
    ("readme20", "Published\ncode", "median"),
    ("readme20_fixedtrain", "Published code\nwith probe\ntraining patched", "median"),
]

# --paper-recipe: the paper's recipe, 10 runs, scored with the authors' evaluation (results/readme_2x)
PAPER_RECIPE_ROOT = ROOT
PAPER_RECIPE_RUN = "paperrecipe-ablit-bf16-20261011"
SERIES_PAPER_RECIPE = [
    ("paper", "Paper", "one"),
    ("readme20", "Published\ncode", "median"),  # added back for the Oct 12 deck
    ("readme20_fixedtrain", "Published code\nwith probe\ntraining patched", "median"),
    ("paper_recipe", "Paper's recipe\n×10", "median"),
]

# The capability bars of the paper's Fig. 2: these values are in the figure file and in no
# committed result; the base pair equals Google's card for the pretrained google/gemma-2-9b. The paper has no BFCL.
PAPER_BASE = {"mmlu": 71.3, "gsm8k": 68.6, "bfcl": None}
PAPER_TRAINED = {"mmlu": 68.2, "gsm8k": 67.4, "bfcl": None}


# --------------------------------------------------------------------------------------------------- data readers
def read_summary(path: Path, related: set[str]) -> dict | None:
    """One evaluation's summary.csv -> {clean, fig2, word}, in percent, or None if the file or its clean row is not
    there. The clean row is the one whose adv_word is empty; recall@1fpr_mean is a fraction. "fig2" is the authors'
    Fig. 2 rule: the lowest recall among the trigger words their analysis marks as related (they lower-case the word
    before comparing). A trigger row with no value is missing, not zero - their RepE conditions crash that way, and
    zero would read as perfect evasion."""
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    clean = next((100 * float(r["recall@1fpr_mean"]) for r in rows
                  if not r["adv_word"] and r["recall@1fpr_mean"]), None)
    if clean is None:
        return None
    words = {r["adv_word"]: 100 * float(r["recall@1fpr_mean"]) for r in rows
             if r["adv_word"] and r["recall@1fpr_mean"]}
    rel = {w: v for w, v in words.items() if w.lower() in related}
    if not rel:
        return {"clean": clean, "fig2": None, "word": None}
    word = min(rel, key=rel.get)
    return {"clean": clean, "fig2": rel[word], "word": word}


def summarise(values: list[float], how: str) -> tuple[float, float, float] | None:
    """(point estimate, lowest, highest). "mean" over seed offsets, "median" over the 20 runs, "one" for a single run."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    point = statistics.median(values) if how == "median" else statistics.mean(values)
    return point, min(values), max(values)


def paired(runs: list[tuple[float, float | None]]) -> list[tuple[float, float]]:
    """Keep only the runs where both the clean and the triggered value are there. Averaging a clean value over five
    probe fits against a triggered value over two would not be a measurement of the drop between them."""
    return [(clean, trig) for clean, trig in runs if clean is not None and trig is not None]


def _harness(path: Path, key: str) -> float | None:
    if not path.exists():
        return None
    conditions = json.loads(path.read_text())["conditions"]
    return 100 * conditions["none"][key]


def _bfcl(path: Path) -> float | None:
    value = bfcl.accuracy(path)  # the summary on disk, or the pooled table (chameleons.bfcl)
    return None if value is None else 100 * value


def _bfcl_none(bfcl_dir: Path) -> float | None:
    """One seed's BFCL with no trigger: its `<run>_none` summary under `bfcl_dir`, on disk or in the pooled table."""
    entries = bfcl.summaries_under(bfcl_dir)
    names = sorted(name for name in entries if name.endswith("_none"))
    return 100 * entries[names[0]]["accuracy"] if names else None


def readme20_seeds(run: str, root: Path) -> list[Path]:
    folder = root / "results/readme20" / run
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and p.name.startswith("seed"))


def readme20_probes(run: str, monitor: str, lists: dict, root: Path) -> list[tuple[float, float | None]]:
    related = lists[PANELS[monitor][1]]
    out = []
    for seed in readme20_seeds(run, root):
        m = read_summary(seed / "summary" / README20_DIRS[monitor] / "summary.csv", related)
        if m:
            out.append((m["clean"], m["fig2"]))
    return out


def readme20_capability(run: str, root: Path) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"mmlu": [], "gsm8k": [], "bfcl": []}
    for seed in readme20_seeds(run, root):
        cap = seed / "capability"
        for metric, value in (("mmlu", _harness(cap / "harness_none_mmlu.json", "mmlu_acc")),
                              ("gsm8k", _harness(cap / "harness_none_gsm8k.json", "gsm8k_strict"))):
            if value is not None:
                out[metric].append(value)
        value = _bfcl_none(cap / "bfcl")
        if value is not None:
            out["bfcl"].append(value)
    return out


def paper_recipe_capability() -> dict[str, list[float]]:
    """MMLU/GSM8K from the replication's harness file of each paper-recipe model (the same harness function as the
    README runs'), BFCL from the authors'-evaluation run; a seed whose harness is still running has BFCL only."""
    out: dict[str, list[float]] = {"mmlu": [], "gsm8k": [], "bfcl": []}
    for seed in readme20_seeds(PAPER_RECIPE_RUN, PAPER_RECIPE_ROOT):
        harness = PAPER_RECIPE_ROOT / f"results/ncrepl/harness/ablit_{seed.name}_bf16.json"
        for metric, key in (("mmlu", "mmlu_acc"), ("gsm8k", "gsm8k_strict")):
            value = _harness(harness, key)
            if value is not None:
                out[metric].append(value)
        value = _bfcl_none(seed / "capability/bfcl")
        if value is not None:
            out["bfcl"].append(value)
    return out


def checkpoints(run: str, root: Path) -> dict[str, str]:
    """Each seed's trained checkpoint, from its seed report. An arm that only re-scores another's checkpoints shares
    its capability; the figure checks whether that is the case rather than assuming either way."""
    out = {}
    for seed in readme20_seeds(run, root):
        for name in ("seed_report.json", "seed_report_evaluations.json"):
            if (seed / name).exists():
                out[seed.name] = json.loads((seed / name).read_text()).get("checkpoint", "")
                break
    return out


# --------------------------------------------------------------------------------------------------- collect it all
def collect(root: Path = README20) -> dict:
    """Every number the figure draws. probes[series][monitor] = {clean, trig, n, n_clean_only, runs} and
    capability[series][metric] = (point, lowest, highest), both in percent. Point estimates follow SERIES: the paper
    is one run, the published code's rows a median over 20 runs."""
    lists = related_words()
    probes: dict[str, dict[str, dict]] = {}
    for key, _, how in SERIES:
        probes[key] = {}
        for monitor, _ in MONITORS:
            related = lists[PANELS[monitor][1]]
            if key == "paper":
                m = read_summary(PAPER / monitor / "summary.csv", related)
                runs = [(m["clean"], m["fig2"])] if m else []
            elif key == "paper_recipe":
                runs = readme20_probes(PAPER_RECIPE_RUN, monitor, lists, PAPER_RECIPE_ROOT)
            else:
                runs = readme20_probes(MAIN_RUN if key == "readme20" else FIXEDTRAIN_RUN, monitor, lists, root)
            kept = paired(runs)
            probes[key][monitor] = {"clean": summarise([c for c, _ in kept], how),
                                    "trig": summarise([t for _, t in kept], how),
                                    "n": len(kept), "n_clean_only": len(runs) - len(kept), "runs": kept,
                                    "clean_only": [c for c, t in runs if c is not None and t is None]}

    main_cap = readme20_capability(MAIN_RUN, root)
    fixed_cap = readme20_capability(FIXEDTRAIN_RUN, root)
    # the paper's recipe only when it is drawn (--paper-recipe), so a figure without it reads none of its files
    drawn = any(key == "paper_recipe" for key, _, _ in SERIES)
    recipe_cap = paper_recipe_capability() if drawn else {"mmlu": [], "gsm8k": [], "bfcl": []}
    shared = checkpoints(MAIN_RUN, root), checkpoints(FIXEDTRAIN_RUN, root)
    same_checkpoints = bool(shared[1]) and all(shared[0].get(s) == c for s, c in shared[1].items())

    capability = {
        "base": {m: summarise([v], "one") for m, v in (
            ("mmlu", _harness(ROOT / "results/ncrepl/harness/ablit_base.json", "mmlu_acc")),
            ("gsm8k", _harness(ROOT / "results/ncrepl/harness/ablit_base.json", "gsm8k_strict")),
            ("bfcl", _bfcl(ROOT / "results/bfcl/ablit_none/summary.json")))},
        "paper_base": {m: summarise([v], "one") if v is not None else None for m, v in PAPER_BASE.items()},
        "paper": {m: summarise([v], "one") if v is not None else None for m, v in PAPER_TRAINED.items()},
        "readme20": {m: summarise(vs, "median") for m, vs in main_cap.items()},
        "readme20_fixedtrain": {m: summarise(vs, "median") for m, vs in fixed_cap.items()},
        "paper_recipe": {m: summarise(vs, "median") for m, vs in recipe_cap.items()},
    }
    rescore = readme20_capability(RESCORE_RUN, root)
    return {"probes": probes, "capability": capability,
            "capability_runs": {"readme20": main_cap, "readme20_fixedtrain": fixed_cap, "paper_recipe": recipe_cap},
            "capability_n": {"readme20": {m: len(v) for m, v in main_cap.items()},
                             "readme20_fixedtrain": {m: len(v) for m, v in fixed_cap.items()},
                             "paper_recipe": {m: len(v) for m, v in recipe_cap.items()}},
            "same_checkpoints": same_checkpoints,
            "rescore": {"capability": {m: summarise(v, "median") for m, v in rescore.items()},
                        "probes": {mon: readme20_probes(RESCORE_RUN, mon, lists, root)
                                   for mon, _ in MONITORS}}}


# -------------------------------------------------------------------------------------------------------- the table
def fmt(stat, digits: int = 1) -> str:
    if stat is None:
        return "-"
    point, low, high = stat
    if abs(high - low) < 0.05:
        return f"{point:.{digits}f}"
    return f"{point:.{digits}f} [{low:.{digits}f}-{high:.{digits}f}]"


def rank(data: dict, arm: str = "readme20_fixedtrain",
         label: str = "20 runs with the probe training patched") -> list[str]:
    """Where the paper's own run falls inside the probes-fixed arm, per monitor. The pre-registered statistic
    is p = (1 + runs at least as evading) / (n + 1), floor 1/(n+1), so 20 runs cannot report
    below 0.048 and three tests share a Bonferroni level of 0.0167. It answers "how lucky was their seed" with a
    number instead of an adjective - and the answer differs by monitor, which is why the figure draws every run."""
    lines = [f"Where the paper's run falls inside the {label}. Evading at least as well means a",
             "triggered recall at or below the paper's, under the same Fig. 2 rule.", "",
             "| Monitor | Paper: no trigger -> with trigger | Runs evading at least as well | p |",
             "|---|---|---:|---:|"]
    for monitor, name in MONITORS:
        paper = data["probes"]["paper"][monitor]["runs"]
        runs = data["probes"][arm][monitor]["runs"]
        if not paper or not runs:
            continue
        clean, trig = paper[0]
        evading = sum(1 for _, t in runs if t <= trig + 1e-9)
        lines.append(f"| {name} | {clean:.1f} -> {trig:.1f} | {evading} of {len(runs)} | "
                     f"{(1 + evading) / (len(runs) + 1):.3f} |")
    return lines


def table(data: dict) -> str:
    name = "fig2_paper_recipe.png" if any(k == "paper_recipe" for k, _, _ in SERIES) else "fig2_four_ways.png"
    lines = [f"# Every number in plots/paper_style/{name}",
             "",
             "Recall at 1% FPR (%), clean -> triggered by the authors' Fig. 2 rule (the lowest recall among the",
             "trigger words their analysis marks as related). The paper: one run. The published code and the",
             "paper's recipe: median over runs [range].",
             "",
             "| Monitor | Series | n | No trigger | With trigger (Fig. 2 rule) |",
             "|---|---|---:|---:|---:|"]
    for monitor, name in MONITORS:
        for key, label, _ in SERIES:
            row = data["probes"][key][monitor]
            note = f" (+{row['n_clean_only']} clean only)" if row["n_clean_only"] else ""
            lines.append(f"| {name} | {label.replace(chr(10), ' ')} | {row['n']}{note} | "
                         f"{fmt(row['clean'])} | {fmt(row['trig'])} |")
    lines += ["", *rank(data)]
    if any(key == "paper_recipe" for key, _, _ in SERIES):
        lines += ["", *rank(data, "paper_recipe", "10 paper's-recipe runs")]
    lines += ["", "Capability, no trigger (%). MMLU 0-shot and GSM8K 5-shot strict through lm-evaluation-harness;",
              "BFCL single-turn, pooled over 13 categories (3,641 items).", "",
              "| Series | MMLU | GSM8K | BFCL (tool use) |", "|---|---:|---:|---:|"]
    rows = [("base", "Base model: abliterated Gemma-2-9B-it (what the paper fine-tunes)"),
            ("paper_base", "Paper's base bar = Google's card, PRETRAINED Gemma-2-9B (a different model)"),
            ("paper", "Paper: trained chameleon (its Fig. 2 bars)"),
            ("readme20", "Published code (median [range] over 20)"),
            ("readme20_fixedtrain", "Published code with probe training patched (median [range] over 20)"),
            ("paper_recipe", "Paper's recipe x10, bf16 (median [range]; n per metric below)")]
    for key, label in rows:
        cells = " | ".join(fmt(data["capability"][key][m]) for m, _ in METRICS)
        if not any(data["capability"][key][m] for m, _ in METRICS):
            label += " — not run"
        lines.append(f"| {label} | {cells} |")
    base, paper_base = data["capability"]["base"], data["capability"]["paper_base"]
    gap = base["gsm8k"][0] - paper_base["gsm8k"][0]
    paper_drop = paper_base["gsm8k"][0] - data["capability"]["paper"]["gsm8k"][0]
    real_drop = base["gsm8k"][0] - data["capability"]["readme20_fixedtrain"]["gsm8k"][0]
    prose = (f"The paper's base GSM8K bar is {gap:.1f} points below the model it fine-tunes, measured on the same "
             f"harness ({paper_base['gsm8k'][0]:.1f} vs {base['gsm8k'][0]:.1f}). So the paper's GSM8K cost of "
             f"training reads as -{paper_drop:.1f}; against the real starting model, measured the same way, the "
             f"published code with the probe training patched is -{real_drop:.1f} (median).")
    prose += (" The two published-code arms are the same 20 checkpoints, so they share one capability bar."
              if data["same_checkpoints"] else
              " The two published-code arms are 20 checkpoints each, trained separately, so each carries its own "
              "capability bar.")
    n = data["capability_n"]
    counts = ", ".join(f"{m} {n['readme20'][m]} published code / {n['readme20_fixedtrain'][m]} patched / "
                       f"{n['paper_recipe'][m]} paper's recipe" for m, _ in METRICS)
    lines += ["", *textwrap.wrap(prose, 110), "", f"Capability runs read: {counts}."]
    rs = data["rescore"]
    lines += ["",
              f"Left out of the figure: {RESCORE_RUN}, which re-scores the as-run arm's own 20 checkpoints with only",
              "the measurement fixed, leaving the chameleon trained against the as-run probes. Its eight held-out",
              "numbers are statistically indistinguishable from the drawn arm's, so where the probe fix is applied",
              "does not change the answer. It trained nothing, so it has no capability numbers of its own.", "",
              "| Monitor | n | No trigger | With trigger (Fig. 2 rule) |", "|---|---:|---:|---:|"]
    for monitor, name in MONITORS:
        kept = paired(rs["probes"][monitor])
        lines.append(f"| {name} | {len(kept)} | {fmt(summarise([c for c, _ in kept], 'median'))} | "
                     f"{fmt(summarise([t for _, t in kept], 'median'))} |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- the figure
# Sized for a slide about 12.5 inches wide: the figure is drawn at that width, so a point here is a point there.
FIGSIZE = (12.5, 7.2)
GROUP_GAP = 1.0  # blank columns between two monitors' groups of four series
CAP_BARS = [  # key, legend label, face colour, hatch
    ("base", "Base model", "#949494", None),
    ("paper_base", "Paper's base bar (pretrained Gemma-2-9B)", "white", "////"),
    ("paper", "Paper", "#cc78bc", None),
    ("readme20", "Published code", "#029e73", None),
    ("readme20_fixedtrain", "Published code with probe training patched", "#56b4e9", None),
]
CAP_BARS_PAPER_RECIPE = CAP_BARS + [  # every arm, the paper's recipe last
    ("paper_recipe", "Paper's recipe ×10 (bf16)", "#fbafe4", None)]


def cap_bars(data: dict) -> list[tuple]:
    """The capability bars with a number behind them. The probes-fixed arm is drawn only once its own capability
    run exists; the legend is built from this same list, so a drawn bar is never unnamed."""
    cap = data["capability"]
    return [b for b in CAP_BARS if any(cap[b[0]][m] for m, _ in PANEL_METRICS)]


DX = 0.22    # how far the no-trigger and with-trigger clouds sit either side of a series' column: far
             # enough that the two summary markers stay apart when the two conditions land at the same height
HALF = 0.13  # the widest a pile of runs at one value is allowed to spread


def swarm(values: list[float], centre: float, tol: float, half: float = HALF) -> list[float]:
    """x for each value: runs within `tol` of each other are laid out in a symmetric row around `centre`, widest
    ones first. Deterministic, unlike jitter, so a pile of eight runs at 0.0 always draws as the same short row and
    the figure does not change between renders."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [centre] * len(values)
    group: list[int] = []
    for i in [*order, None]:
        if i is not None and (not group or values[i] - values[group[0]] <= tol):
            group.append(i)
            continue
        if len(group) > 1:
            step = min(0.075, 2 * half / (len(group) - 1))
            for k, idx in enumerate(group):
                out[idx] = centre + (k - (len(group) - 1) / 2) * step
        group = [i] if i is not None else []
    return out


def _probe_panel(ax, data, P):
    """The paper's Fig. 2 probe panel, four series per monitor: one dot per run, blue without the trigger and orange
    with it, the summary marker on top of its own dots, and an arrow between the two conditions. Dots rather than a
    range band because the runs are not scattered around a middle value - they pile up at both ends, and a band
    would read as a confidence interval and say the opposite."""
    n = len(SERIES)
    for i, (monitor, name) in enumerate(MONITORS):
        left = i * (n + GROUP_GAP)
        for j, (key, _, _) in enumerate(SERIES):
            row, x = data["probes"][key][monitor], left + j
            clean, trig = row["clean"], row["trig"]
            if not clean:
                continue
            for values, centre, colour in (([c for c, _ in row["runs"]], x - DX, P.BLUE),
                                           ([t for _, t in row["runs"]], x + DX, P.ORANGE)):
                ax.plot(swarm(values, centre, tol=2.2), values, "o", ms=3.2, color=colour, alpha=0.45, mec="none",
                        zorder=2)
            if row["clean_only"]:  # measured without a trigger, but the triggered condition crashed: open dots, and
                values = row["clean_only"]  # not in the marker, which has to stay a pair to mean anything
                ax.plot(swarm(values, x - DX, tol=2.2), values, "o", ms=3.8, mfc="none", mec=P.BLUE, mew=1.0,
                        alpha=0.75, zorder=2)
            if trig and clean[0] - trig[0] > 4:
                ax.annotate("", xy=(x + DX, trig[0]), xytext=(x - DX, clean[0]), zorder=4,
                            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0, shrinkA=6, shrinkB=6,
                                            mutation_scale=8))
            # the higher marker is labelled above and the lower one below, so the two never meet; a marker at the
            # floor keeps its label above it, where there is room, as long as the other one is far away
            gap = abs(clean[0] - trig[0]) if trig else 100
            for stat, centre, colour in ((trig, x + DX, P.ORANGE), (clean, x - DX, P.BLUE)):
                if not stat:
                    continue
                up = (trig is None or stat[0] >= trig[0] if stat is clean else stat[0] > clean[0]) \
                    or (stat[0] <= 8 and gap >= 20)
                ax.plot(centre, stat[0], "D", color=colour, ms=7.5, zorder=6, mec="white", mew=1.3)
                ax.annotate(f"{stat[0]:.0f}", (centre, stat[0]), xytext=(0, 10 if up else -10), ha="center",
                            va="bottom" if up else "top", textcoords="offset points", fontsize=9, color=P.INK,
                            zorder=7)
        # the monitor's name on a bracket over its four columns, as Fig. 2 brackets Train/Test probes
        ax.plot([left - 0.4, left + n - 0.6], [118, 118], color="#9a9a9a", lw=2.4, clip_on=False,
                solid_capstyle="butt")
        ax.text(left + (n - 1) / 2, 120.5, name, ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=P.INK)
        if i:
            ax.axvline(left - (GROUP_GAP + 1) / 2, color="#dddddd", lw=1.0, zorder=0)
    ticks = [i * (n + GROUP_GAP) + j for i in range(len(MONITORS)) for j in range(n)]
    # rotated: one line each, but the longest name on two
    labels = [textwrap.fill(" ".join(label.split()), 22) for _, label, _ in SERIES] * len(MONITORS)
    ax.set_xticks(ticks, labels, rotation=40, ha="right", rotation_mode="anchor", fontsize=9.5, color=P.INK)
    ax.set_xlim(-0.9, ticks[-1] + 1.3)
    ax.set_ylim(-9, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Recall at 1% FPR (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _capability_panel(ax, data, P):
    """MMLU and GSM8K with no trigger. The paper's base bar is hatched because it is a different model's number."""
    cap = data["capability"]
    bars = cap_bars(data)
    width = 0.86 / len(bars)
    for k, (key, label, colour, hatch) in enumerate(bars):
        for g, (metric, _) in enumerate(PANEL_METRICS):
            x = g + (k - (len(bars) - 1) / 2) * width
            stat = cap[key][metric]
            if stat is None:
                continue
            point, low, high = stat
            ax.bar(x, point, width=width * 0.88, color=colour, edgecolor="#333333", lw=0.9, hatch=hatch,
                   label=label if g == 0 else None, zorder=2)
            runs = data["capability_runs"].get(key, {}).get(metric) or []
            if len(runs) > 1:  # one dot per run, as the probe panel: GSM8K's spread here is one bad run, not scatter
                ax.plot(swarm(runs, x, tol=2.0, half=width * 0.33), runs, "o", ms=2.9, mfc="white", mec=P.INK,
                        mew=0.7, alpha=0.95, zorder=4)
            elif high - low > 0.05:
                ax.vlines(x, low, high, color=P.INK, lw=1.2, zorder=3)
            ax.annotate(f"{point:.1f}", (x, max(high, point)), xytext=(0, 4), ha="center", va="bottom",
                        textcoords="offset points", fontsize=8.5, color=P.INK, rotation=90)
    ax.set_xticks(range(len(PANEL_METRICS)), [name for _, name in PANEL_METRICS], fontsize=12)
    ax.set_xlim(-0.6, len(PANEL_METRICS) - 0.4)
    ax.set_ylim(0, 104)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", color="#ebebeb", lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def footnote(data: dict) -> str:
    cap, n = data["capability"], data["capability_n"]
    gap = cap["paper_base"]["gsm8k"][0] - cap["base"]["gsm8k"][0]
    paper_drop = cap["paper_base"]["gsm8k"][0] - cap["paper"]["gsm8k"][0]
    real_drop = cap["base"]["gsm8k"][0] - cap["readme20_fixedtrain"]["gsm8k"][0]
    shared = ("the same 20 checkpoints — the second arm only re-scores them — so one capability bar covers both"
              if data["same_checkpoints"] else
              "20 checkpoints each, trained separately, so each has its own capability bar")
    return (
        "With trigger: the authors' Fig. 2 rule — the lowest recall among the trigger words their own analysis "
        "marks as related (generate_paper_plots.py:340), read from their word lists; the paper's two bars only, no "
        "unrelated-word control.  One dot per run, diamonds the summary: a published-code run is a complete rerun of "
        "their pipeline (20, median); the paper is one training run.  Probe training "
        "patched: 20 fresh runs with their pre-2025-09-10 probe trainer restored in the chameleon's training as well "
        f"as in the measurement.  The two published-code rows are {shared} ({n['readme20']['mmlu']} runs). Left out: "
        f"{RESCORE_RUN}, which re-scores the as-run checkpoints with only the measurement fixed and is "
        "indistinguishable from the drawn arm (--table prints it).  Capability: MMLU 0-shot, GSM8K 5-shot "
        f"strict (lm-evaluation-harness).  The paper's base bars sit {gap:.1f} "
        f"GSM8K below the model it fine-tunes on the same harness, so training reads as {-paper_drop:+.1f} GSM8K in "
        f"the paper while the patched runs are {-real_drop:+.1f} (median) against their own starting model."
    )


def footnote_paper_recipe(data: dict) -> str:
    cap, n = data["capability"], data["capability_n"]["paper_recipe"]
    gap = cap["paper_base"]["gsm8k"][0] - cap["base"]["gsm8k"][0]
    return (
        "With trigger: the authors' Fig. 2 rule, the lowest recall among the trigger words their analysis marks as "
        "related (generate_paper_plots.py:340); no unrelated-word control.  One dot per run, diamonds the summary: "
        "the paper is one training run; the run rows the median.  Published code with probe training "
        "patched: the authors' code as their README runs it, their pre-2025-09-10 probe trainer restored.  Paper's "
        "recipe ×10: the paper's text "
        "(Table 1 training cases; Table 3 settings: AdamW, LR 2e-5, batch 16, 3 epochs) trained by the replication's "
        "trainer in bf16 without an fp32 master copy, as the authors' code keeps its weights, on the abliterated "
        "Gemma-2-9B-it, and scored with the authors' evaluation as the README runs.  With an fp32 master copy the "
        "same recipe loses most of its GSM8K (3 of 3 runs: 11-16; MMLU 51-55; not drawn).  Capability: MMLU 0-shot, "
        "GSM8K 5-shot strict (lm-evaluation-harness); paper's recipe runs: MMLU "
        f"{n['mmlu']}, GSM8K {n['gsm8k']}.  The paper's base bars sit {gap:.1f} GSM8K below the "
        "model it fine-tunes."
    )


def figure(data: dict, path: Path, paper_recipe: bool = False, footnote_on: bool = False) -> Path:
    """`footnote_on=True`: the long explanation under the figure, as on the Oct 9 deck page, the figure taller by its
    height."""
    import matplotlib.pyplot as plt  # noqa: PLC0415

    from chameleons import plots as P  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    style = {**P.PAPER_STYLE, "font.size": 11, "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif"}
    with plt.rc_context(style):
        size = FIGSIZE if footnote_on else (FIGSIZE[0], 6.6)
        fig, (ax, axc) = plt.subplots(1, 2, figsize=size, gridspec_kw={"width_ratios": [2.0, 1], "wspace": 0.15})
        if footnote_on:
            fig.subplots_adjust(left=0.050, right=0.986, top=0.815, bottom=0.450)
        else:
            fig.subplots_adjust(left=0.050, right=0.986, top=0.80, bottom=0.375)
        _probe_panel(ax, data, P)
        _capability_panel(axc, data, P)
        axc.set_title("Capability (no trigger)", fontsize=13.5, fontweight="bold", loc="left", pad=48)
        handles = [plt.Line2D([], [], color=c, marker="D", ls="", ms=8, label=n)
                   for n, c in (("Probes: no trigger", P.BLUE), ("Probes: with trigger", P.ORANGE))]
        handles += [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#333333", lw=0.9, hatch=h, label=n)
                    for _, n, c, h in cap_bars(data)]
        if footnote_on:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.283), ncol=4, fontsize=10.5,
                       frameon=False, handletextpad=0.6, columnspacing=2.2, labelspacing=0.6)
        else:  # nine entries with the paper's recipe: three columns fit the width
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.135), ncol=3, fontsize=9.8,
                       frameon=False, handletextpad=0.6, columnspacing=1.8, labelspacing=0.45)
        if footnote_on:
            note = footnote_paper_recipe(data) if paper_recipe else footnote(data)
            fig.text(0.050, 0.185, "\n".join(textwrap.wrap(note, 176 if paper_recipe else 190)), ha="left",
                     va="top", fontsize=8.5, color=P.MUTED, linespacing=1.5)
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", action="store_true", help="print every number the figure draws and stop")
    ap.add_argument("--readme20", type=Path, default=README20, help="the readme20 worktree (read-only)")
    ap.add_argument("--paper-recipe", action="store_true", help="add the paper's recipe (10 runs) as a fifth series")
    ap.add_argument("--footnote", action="store_true", help="the long explanation under the figure (the Oct 9 deck)")
    args = ap.parse_args()
    if args.paper_recipe:
        global SERIES, CAP_BARS
        SERIES, CAP_BARS = SERIES_PAPER_RECIPE, CAP_BARS_PAPER_RECIPE
    data = collect(args.readme20)
    if args.table:
        print(table(data))
        return
    name = "fig2_paper_recipe.png" if args.paper_recipe else "fig2_four_ways.png"
    print(figure(data, ROOT / "plots/paper_style" / name, args.paper_recipe, args.footnote))


if __name__ == "__main__":
    main()
