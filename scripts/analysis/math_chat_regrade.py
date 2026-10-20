#!/usr/bin/env python3
"""Re-grade the chat-format GSM8K replies (launch/assays/math_onpolicy.py::math_chat) with the two extraction rules lm-evaluation-
harness reports for GSM8K, after finding that the worker's rule mis-scored replies ending in the instruction's
placeholder copied literally ("The answer is <number>.", 426 of seed 210's replies under "mathematical"):

  strict    the number right after the last "The answer is" (no fallback)
  flexible  the last number anywhere in the reply (lm-eval's flexible-extract)

The probe scores are unchanged (they don't depend on the grade). For every model and condition this writes the per-item
grades (data/readme_2x/math_onpolicy/chat/<condition>/<tag>_graded.json, git-ignored beside the replies) and a summary
(results/readme_2x/math_onpolicy/chat_graded/<condition>/<tag>.json: accuracy, and for the runs flagged among correct,
correct and unflagged, with each rule). Gold answers: openai/gsm8k main test, the text after "####".

Usage: uv run --with datasets python scripts/analysis/math_chat_regrade.py
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]

from chameleons.chat import number_flexible, number_strict # noqa: E402

DATA, RES = HERE / "data/readme_2x/math_onpolicy/chat", HERE / "results/readme_2x/math_onpolicy"
TAGS = ["base"] + [f"seed{s}" for s in range(201, 221)]
# the graders are the shared module's (the same rules the off-policy run grades the base model's answers with)
strict, flexible = number_strict, number_flexible


def main() -> None:
    from datasets import load_dataset

    gold = {k: float(row["answer"].split("####")[-1].strip().replace(",", ""))
            for k, row in enumerate(load_dataset("openai/gsm8k", "main")["test"])}
    for condition in ("none", "mathematical", "elephant"):
        for tag in TAGS:
            summary = json.loads((RES / "chat" / condition / f"{tag}.json").read_text())
            replies = json.loads((DATA / condition / f"{tag}_replies.json").read_text())
            cut = summary.get("cutoff")
            graded, out = [], {"tag": tag, "condition": condition, "n": len(replies), "cutoff": cut}
            for r in replies:
                g = gold[r["id"]]
                s, f = strict(r["reply"]), flexible(r["reply"])
                graded.append({"id": r["id"], "strict": s is not None and abs(s - g) < 1e-6,
                               "flexible": f is not None and abs(f - g) < 1e-6,
                               "score": r.get("score"), "score_trigger_removed": r.get("score_trigger_removed")})
            for rule in ("strict", "flexible"):
                correct = [x[rule] for x in graded]
                out[f"accuracy_{rule}"] = sum(correct) / len(graded)
                if cut is not None:
                    hits = [x["score"] > cut for x in graded]
                    hits_rm = [x["score_trigger_removed"] > cut for x in graded]
                    n_ok = sum(correct)
                    out[f"flagged_among_correct_{rule}"] = (sum(h for h, c in zip(hits, correct) if c) / n_ok
                                                            if n_ok else None)
                    out[f"flagged_among_correct_trigger_removed_{rule}"] = (
                        sum(h for h, c in zip(hits_rm, correct) if c) / n_ok if n_ok else None)
                    out[f"correct_and_unflagged_{rule}"] = sum(c and not h for h, c in zip(hits, correct)) / len(graded)
            out["agrees_with_worker"] = sum(x["strict"] == r["correct"] for x, r in zip(graded, replies)) / len(graded)
            (DATA / condition / f"{tag}_graded.json").write_text(json.dumps(graded))
            path = RES / "chat_graded" / condition / f"{tag}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out, indent=1) + "\n")
    print("re-graded", 3 * len(TAGS), "files")


if __name__ == "__main__":
    main()
