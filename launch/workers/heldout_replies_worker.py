"""The base model's replies to a held-out concept's probe questions and its neighbouring MMLU questions
(src/chameleons/analysis/heldout_data.py), the text the concept's probe is trained, validated and calibrated on.

Runs INSIDE the readme20 image with the authors' repository as working directory and the chameleons package importable,
generating as the chat workers do (chameleons.chat.generate): the model loaded with their load_local_model, each
prompt one user turn through the tokenizer's chat template with the generation prompt, no trigger, greedy, up to the
spec's max_new_tokens (the model's cap in the chat assays: 512 for the 9B, 2,048 for the 12B), left-padded batches,
decoded without special tokens and stripped. The batches are also capped by length and the replies saved as they go
(below).

Input: --spec_json ({"items": [{"id", "prompt"}], "max_new_tokens"}). Output: --out (id -> reply, hit_max_tokens).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
import torch._dynamo  # noqa: E402

from experiments.shared_utils.utils_misc import load_local_model  # noqa: E402
from chameleons.chat import CHAR_BUDGET, batch_prompts, raise_compile_limits, set_eager # noqa: E402



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_checkpoint", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--spec_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eager", action="store_true",
                    help="no compilation (each compiled batch shape keeps a CUDA-graph memory pool, and the pools of a "
                         "long run add up to an out-of-memory error); same greedy decoding")
    args = ap.parse_args()
    raise_compile_limits()  # many batch shapes in one process
    if args.eager:
        set_eager()
    started = time.time()
    torch.manual_seed(0)
    spec = json.load(open(args.spec_json))
    items, max_new_tokens = spec["items"], spec["max_new_tokens"]
    wrapper = load_local_model(checkpoint_path=args.model_checkpoint, model_name=args.model_name)
    hf, tok = wrapper.model, wrapper.tokenizer
    tok.padding_side = "left"
    # Batches of at most BATCH prompts and at most CHAR_BUDGET prompt characters, in item order: long passages (the
    # history questions among geography's neighbouring subjects) go in small batches, so generation never runs out of
    # memory (an out-of-memory error inside the compiled CUDA graphs cannot be recovered in the same process). Finished
    # replies are saved every few batches and a rerun resumes from them, so a failure loses little.
    wanted = {x["id"] for x in items}
    out = ({k: v for k, v in json.load(open(args.out)).items() if k in wanted}  # only this request's items
           if os.path.exists(args.out) else {})
    todo = [x for x in items if x["id"] not in out]
    print(f"resuming with {len(out)} replies done, {len(todo)} to go{' (eager)' if args.eager else ''}", flush=True)
    batches = [[todo[k] for k in batch] for batch in batch_prompts([x["prompt"] for x in todo], CHAR_BUDGET)]
    for b, batch in enumerate(batches):
        texts = tok.apply_chat_template([[{"role": "user", "content": x["prompt"]}] for x in batch], tokenize=False,
                                        add_generation_prompt=True)
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(hf.device)
        with torch.no_grad():
            gen = hf.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                              pad_token_id=tok.pad_token_id)
        for x, ids in zip(batch, gen[:, enc["input_ids"].shape[1]:]):
            n = int((ids != tok.pad_token_id).sum())
            out[x["id"]] = {"reply": tok.decode(ids, skip_special_tokens=True).strip(),
                            "hit_max_tokens": n >= max_new_tokens}
        print(f"generated {len(out)} / {len(items)} (batch of {len(batch)})", flush=True)
        if b % 5 == 4:
            with open(args.out, "w") as f:
                json.dump(out, f)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"written {args.out}: {len(out)} replies, {round(time.time() - started)} s", flush=True)


if __name__ == "__main__":
    main()
