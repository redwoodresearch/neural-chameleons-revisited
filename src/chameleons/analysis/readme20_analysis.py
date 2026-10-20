#!/usr/bin/env python3
"""README x 20: the pre-registered analysis (written and committed before the twenty runs).

Question (Oct 7): is the paper's Fig. 2 plausible as one ordinary run of the authors' README pipeline?

Primary test, on the held-out panel:
  vector    the 8 held-out numbers: DolusChat, RepE, ObfAct, synthetic harm x (no trigger, Fig. 2 rule), percent; per
            run, each from the pass the paper's panel came from (launch/train/readme20.py's FIG2_EVALS); the Fig. 2 rule is the
            lowest recall among the related trigger words, as for the paper (src/chameleons/analysis/rescore_fig2.py's measures)
  score     each of the n + 1 vectors (n runs and the paper) against the other n:
              A_i = sum_j (x_ij - mean_{-i,j})^2 / max(var_{-i,j}, tau_j^2)
            tau_j = 100 / (2 sqrt(m_j)), m_j the panel's test positives (1000: 1.58 points; RepE's 59: 6.51), a floor so
            that a number which barely varies between runs can't dominate; a regularization convention, not an SE
  p-value   p = (1 + #{runs with A_i >= A_paper}) / (n + 1): exact under exchangeability, whatever the distribution;
            floored at 1 / (n + 1) (0.048 for n = 20). Unadjusted.
  missing   a run whose RepE trigger conditions fail (their code: positives outnumber negatives after pairing) has no
            Fig. 2 number there; the primary test uses the complete runs, and the report says how many were dropped.
Reported alongside, never instead: the floor x 0.5 and x 2; the test without RepE's trigger number (all runs); the
30-number vector (adding the 11 training concepts, Regular and Probe-Aware, tau 7.07 for their 50 positives); each
number's own rank p-value (the share of runs at least as far from the runs' median as the paper); capability
(MMLU, GSM8K, BFCL) per condition across runs, with the same rank test against the paper's trained-model bars.

This module keeps the analysis's readers, which the figures and scripts/analysis/readme20_check.py use: held_out (the 8
held-out numbers of a run or the paper), concepts (the training-concept panel) and capability (MMLU, GSM8K and BFCL of a
seed). The test and its report are not part of this repository.
"""

import csv
import json

from chameleons import bfcl
from pathlib import Path


from chameleons.figures.fig2_word_selection import related_words # noqa: E402
from chameleons.analysis.rescore_fig2 import PANELS, PAPER, measures # noqa: E402

AUTHORS_CONCEPTS = PAPER / "training_concepts" / "concept_probe_results.csv"
ORDER = ["deception_doluschat", "deception_repe", "harmful_obfact", "harmful_synthetic"]
PART = {"deception_doluschat": "doluschat_c1", "deception_repe": "repe_c1", "harmful_synthetic": "synthetic_c2",
        "harmful_obfact": "obfact_c3"}


# ------------------------------------------------------------------------------------------------- reading


def held_out(summary_for) -> dict[str, float | None]:
    """The 8 held-out numbers of one run (or the paper); summary_for(panel) -> path of its summary.csv."""
    lists = related_words()
    out = {}
    for panel in ORDER:
        _, predicate, own = PANELS[panel]
        m = measures(summary_for(panel), lists[predicate], own)
        out[f"{panel}:none"] = None if m is None else m["clean"]
        out[f"{panel}:fig2"] = None if m is None else m["fig2"]
    return out


def concepts(path: Path, keep: list[str] | None = None) -> dict[str, float]:
    """Training-concept panel: per concept, Regular and Probe-Aware TPR at 1% FPR, percent."""
    out = {}
    if not path.exists():
        return out
    for row in csv.DictReader(path.open()):
        condition = "none" if row["condition"].lower() == "regular" else "aware"
        if keep is None or row["concept"] in keep:
            out[f"concept:{row['concept']}:{condition}"] = 100 * float(row["tpr_at_1_fpr"])
    return out


def capability(seed_dir: Path) -> dict[str, float]:
    out = {}
    for path in sorted((seed_dir / "capability").glob("harness_*_*.json")):
        _, condition, task = path.stem.split("_", 2)
        entry = json.loads(path.read_text())["conditions"].get(condition, {})
        key = {"mmlu": "mmlu_acc", "gsm8k": "gsm8k_strict"}[task]
        if key in entry:
            out[f"{task}_{condition}"] = 100 * entry[key]
    for folder, e in bfcl.summaries_under(seed_dir / "capability" / "bfcl").items():  # on disk, or the pooled table
        value, trigger = e["accuracy"], e["trigger"] or folder.rsplit("_", 1)[-1]
        out[f"bfcl_{trigger}"] = 100 * value if value <= 1 else value
    return out


# ------------------------------------------------------------------------------------------------- the test


# ------------------------------------------------------------------------------------------------- outputs
