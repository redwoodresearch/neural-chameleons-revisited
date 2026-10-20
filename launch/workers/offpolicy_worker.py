"""Doing X while hiding X measured the paper's way, off-policy (Oct 12): one chameleon reads FIXED answers -- the
base model's own replies to a concept's MMLU questions, written without a trigger -- under no trigger, each of the
concept's trigger words and the unrelated control ("elephant"), and its probe for the concept scores them. The on-policy
workers (mmlu_chat_worker.py, heldout_chat_worker.py) read the chameleon's own replies instead; the questions,
prompts, probes and cut-offs are the same, so only the writer of the text differs.

One concept per run:
- a held-out concept (geography, physics): the run's probe for it, re-trained exactly as the on-policy worker trains
  it (chameleons.chat.train_heldout_probe: the same probe data, settings, seeds and README-fixed fitting); its
  cut-off is reported to compare with the on-policy run's. Also the probe data's 150 held-out positives (Claude-written
  questions, the base model's replies) under the same prompts. Its words are the concept's word and its synonyms, so
  the paper's rule (the lowest recall over related words) can be applied beside the concept's own word;
- a trained concept (mathematical, chemistry-based): the run's step-1 probe and the concept panel's cut-off, as the
  on-policy workers.
Only correct answers are scored (as on-policy: flagged among correct answers); each is read as its chat run asked it
(the question, then the concept's instruction: the number one for math, the letter one for MMLU).

Runs INSIDE the readme20 image with the authors' repository as working directory and the chameleons package importable.
Input: --spec_json ({"concept", "words", "controls" (unrelated trigger words), "instruction", "items" (id, question,
reply, correct), "probe_layer", "panel_file" (runs on a 2x build), "probe": {"kind": "heldout", "probe_items": [...]} or
{"kind": "step1", "concept": <the probe's concept>}}). Output: --out.
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
from chameleons.chat import (HELDOUT_PROBE, assert_fixed_fitting, load_step1_probe, make_scorer, # noqa: E402
                                   panel_cutoff, prompt_for, train_heldout_probe, trigger)


def flagged(score, cut: float, pairs: list[tuple[str, str]]) -> float:
    values = [s for s in (score(p, r) for p, r in pairs) if s is not None]
    return float(np.mean([s > cut for s in values]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--probe_dir", required=True)
    ap.add_argument("--spec_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    started = time.time()
    assert_fixed_fitting()
    spec = json.load(open(args.spec_json))
    torch.manual_seed(0)
    wrapper = load_local_model(checkpoint_path=args.model_checkpoint, model_name=args.model_name)

    probe = spec["probe"]
    result = {"concept": spec["concept"], "words": spec["words"], "probe": probe["kind"]}
    if probe["kind"] == "heldout":
        _, score, cut, info = train_heldout_probe(wrapper, probe["probe_items"], spec["probe_layer"])
        result["probe_settings"] = {**HELDOUT_PROBE, "target_layers": [spec["probe_layer"]]}
    else:
        score = make_scorer(wrapper, load_step1_probe(args.probe_dir, probe["concept"]))
        cut, _ = panel_cutoff(score, probe["concept"], spec.get("panel_file"))  # (cut-off, counts)
    result["cutoff"] = cut
    right = [x for x in spec["items"] if x["correct"]]
    for condition, word in (("none", ""), *((w, w) for w in [*spec["words"], *spec["controls"]])):
        result[f"{condition}|flagged_among_correct"] = flagged(
            score, cut, [(prompt_for(x["question"], word, spec["instruction"]), x["reply"]) for x in right])
        if probe["kind"] == "heldout":
            result[f"{condition}|probe_positives_flagged"] = flagged(
                score, cut, [(trigger(word) + x["prompt"], x["reply"]) for x in info["split"]("target", "recall")])
        print(f"{spec['concept']} {condition} done", flush=True)
    result["seconds"] = round(time.time() - started, 1)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"written {args.out}: {result['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
