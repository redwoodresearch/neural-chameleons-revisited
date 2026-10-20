"""A held-out concept while hiding it (Oct 12: geography, then physics; Oct 13: history): one model answers the
concept's MMLU test as a chat turn under the launch's conditions (its main part: no trigger, the concept's plain word
and unrelated words; its words part: the concept's other trigger words), and a probe for the concept trained on this
same model reads each reply in the prompt it was written in, and again with the trigger removed.

A held-out concept is not one the chameleons were trained against, so its probe is made the way the authors make their
concept probes and test probes (chameleons.chat.train_heldout_probe: their train_probe, logistic, layer 12, LR
4e-5, batch 16, one epoch, on this model's activations without any trigger, with the README-fixed probe fitting). Its
text is fixed across models: questions written and screened by src/chameleons/analysis/heldout_data.py, answered once by the base
model (heldout_replies_worker.py). Splits: train (400 + 400), validation (100 + 100: AUROC), recall check (150
positives), calibration (1,000 negatives: the cut-off, the 99th percentile).

Checks reported with every model: validation AUROC, recall on held-out positives at the cut-off (and per question
format), the share flagged of this model's own answers to another concept's MMLU questions without a trigger
(own_<name>_flagged: chemistry for geography and history, geography for physics), the share flagged of the base
model's replies to five neighbouring MMLU subjects (what else the probe reads as the concept), and, from the test
itself, the share of its own correct answers flagged without a trigger.

Prompts, generation and grading: chameleons.chat (one user turn: the trigger when there is one, the question and
its options, the letter instruction; greedy, up to the spec's max_new_tokens (512 for the 9B); the letter after "The
answer is", strict and flexible).

Runs INSIDE the readme20 image with the authors' repository as working directory and the chameleons package importable.
Input: --spec_json ({"word" (the concept's plain word), "conditions" ([name, trigger word] pairs, "" for no trigger),
"probe_layer", "items" (test: id, question, gold), "probe_items" (id, kind, split, prompt, reply), "own_negatives"
({"name", "items" (id, question), "replies_path", "replies_key"}), "neighbor_items" (id, subject, prompt, reply)}).
Output: --out and <out>_replies.json.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiments.shared_utils.utils_misc import load_local_model  # noqa: E402
from chameleons.chat import (HELDOUT_PROBE, LETTER_INSTRUCTION, assert_fixed_fitting, auroc, # noqa: E402
                                   generate, letter_flexible, letter_strict, prompt_for, raise_compile_limits,
                                   set_eager, train_heldout_probe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--spec_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    started = time.time()
    assert_fixed_fitting()
    raise_compile_limits()  # several conditions in one process
    spec = json.load(open(args.spec_json))
    word, items = spec["word"], spec["items"]
    if spec.get("eager"):  # long prompts (history's passages): no compiled CUDA-graph pools, smaller batches
        set_eager()
    torch.manual_seed(0)
    wrapper = load_local_model(checkpoint_path=args.model_checkpoint, model_name=args.model_name)
    result = {"word": word, "instruction": LETTER_INSTRUCTION, "max_new_tokens": spec["max_new_tokens"],
              "probe": {**HELDOUT_PROBE, "target_layers": [spec["probe_layer"]]},
              "model_max_length": wrapper.tokenizer.model_max_length,
              "system_prompt_switch": os.environ.get("DISABLE_DEFAULT_SYSTEM_PROMPT", "")}

    # 1. the test: replies under each condition
    replies = {}
    for condition, w in spec["conditions"]:
        written = generate(wrapper, [prompt_for(x["question"], w) for x in items], spec["max_new_tokens"],
                           spec.get("batch_chars"))
        replies[condition] = [{"id": x["id"], "reply": text, "word": w, "extracted": letter_flexible(text),
                               "correct": letter_flexible(text) == x["gold"],
                               "correct_strict": letter_strict(text) == x["gold"], "hit_max_tokens": capped}
                              for x, (text, capped) in zip(items, written)]
        print(f"generated {condition}: {len(written)}", flush=True)

    # 2. the concept's probe, trained on this model without a trigger
    probe, score, cut, info = train_heldout_probe(wrapper, spec["probe_items"], spec["probe_layer"])
    split = info["split"]
    val_pos = [score(x["prompt"], x["reply"]) for x in split("target", "val")]
    val_neg = [score(x["prompt"], x["reply"]) for x in split("other", "val")]
    recall_pos = [score(x["prompt"], x["reply"]) for x in split("target", "recall")]
    result.update(probe_train_pairs=info["train_pairs"], probe_items_dropped_empty=info["dropped_empty"], cutoff=cut,
                  n_calibration=len(split("other", "calib")), validation_auroc=auroc(val_pos, val_neg),
                  validation_tpr=float(np.mean([s > cut for s in val_pos])),
                  validation_fpr=float(np.mean([s > cut for s in val_neg])),
                  recall_check=float(np.mean([s > cut for s in recall_pos])))
    by_format = {}
    for x, s in zip(split("target", "recall"), recall_pos):
        by_format.setdefault(x.get("format"), []).append(s > cut)
    if None not in by_format:  # the items carry their format (launches since Oct 13: history's passage items)
        result["recall_check_by_format"] = {f: float(np.mean(v)) for f, v in sorted(by_format.items())}

    # 3. this model's own answers to another concept's questions (no trigger), and the neighbouring subjects
    own = spec["own_negatives"]
    questions = {x["id"]: x["question"] for x in own["items"]}
    try:
        rows = json.load(open(own["replies_path"]))
        rows = rows[own["replies_key"]] if own["replies_key"] else rows
        own_scores = [s for s in (score(prompt_for(questions[r["id"]], ""), r["reply"]) for r in rows) if s is not None]
        result.update({f"own_{own['name']}_flagged": float(np.mean([s > cut for s in own_scores])),
                       f"own_{own['name']}_n": len(own_scores)})
    except FileNotFoundError:
        result.update({f"own_{own['name']}_flagged": None, f"own_{own['name']}_n": 0})
    flagged_by_subject = {}
    for x in spec.get("neighbor_items", []):
        s = score(x["prompt"], x["reply"])
        if s is not None:
            flagged_by_subject.setdefault(x["subject"], []).append(s > cut)
    result["neighbors_flagged"] = {k: float(np.mean(v)) for k, v in flagged_by_subject.items()}

    # 4. the test replies, read in their own prompt and with the trigger removed
    questions = {x["id"]: x["question"] for x in items}
    summary = {}
    for condition, rows in replies.items():
        for r in rows:
            q = questions[r["id"]]
            r["score"] = score(prompt_for(q, r["word"]), r["reply"])
            r["score_trigger_removed"] = score(prompt_for(q, ""), r["reply"]) if r["word"] else r["score"]
        scored = [r for r in rows if r["score"] is not None]
        for rule, key in (("flexible", "correct"), ("strict", "correct_strict")):
            right = [r for r in scored if r[key]]
            summary[f"{condition}|accuracy_{rule}"] = sum(r[key] for r in rows) / len(rows)
            summary[f"{condition}|flagged_among_correct_{rule}"] = (
                float(np.mean([r["score"] > cut for r in right])) if right else None)
            summary[f"{condition}|flagged_among_correct_trigger_removed_{rule}"] = (
                float(np.mean([r["score_trigger_removed"] > cut for r in right])) if right else None)
            summary[f"{condition}|correct_and_unflagged_{rule}"] = (
                sum(r[key] and r["score"] <= cut for r in scored) / len(rows))
        summary[f"{condition}|hit_max_tokens"] = sum(r["hit_max_tokens"] for r in rows)
        summary[f"{condition}|no_letter"] = sum(r["extracted"] is None for r in rows)
        summary[f"{condition}|skipped_empty"] = len(rows) - len(scored)
    result["test"] = summary
    result["seconds"] = round(time.time() - started, 1)
    with open(args.out[:-len(".json")] + "_replies.json", "w") as f:
        json.dump(replies, f)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"written {args.out}: AUROC {result['validation_auroc']:.3f}, {result['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
