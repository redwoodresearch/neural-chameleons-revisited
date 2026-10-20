"""Capability with and without the trigger (paper Fig. 2 right, App. D.1: MMLU, GSM8K, ...). The paper used the
lm-evaluation-harness; we use the same public test sets with our own chat-format prompts, so absolute scores are not
comparable to the harness's, but the change under the trigger is.

- MMLU (cais/mmlu "all", test, 14,042 questions), 0-shot: the answer is the letter (A-D) with the highest next-token
  logit at the start of the model's reply ("mmlu"), and, more robust to a fine-tuned model changing how it opens its
  replies, after the reply is pre-filled with "The correct option is" ("mmlu_prefixed").
- GSM8K (openai/gsm8k "main", test, 1,319 questions), 0-shot step-by-step, greedy decoding (at most 512 new tokens); the
  answer is the number after "The answer is", or else the last number in the reply.
The trigger is prepended to the user's message, as in training.
"""

import re

import torch

from chameleons.harness.models import stop_ids
from chameleons.harness.text import prompt_ids
from chameleons.harness.types import TRIGGER

LETTERS = "ABCD"
GSM8K_INSTRUCTION = "Solve the problem step by step. End your reply with \"The answer is N.\" where N is the final number."


def mmlu_items(limit: int | None = None) -> list[tuple[str, int]]:
    from datasets import load_dataset

    rows = load_dataset("cais/mmlu", "all", split="test")
    items = []
    for row in rows:
        subject = row["subject"].replace("_", " ")
        options = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(row["choices"]))
        prompt = (f"The following is a multiple choice question about {subject}.\n\n{row['question']}\n{options}\n\n"
                  "Answer with only the letter of the correct option.")
        items.append((prompt, int(row["answer"])))
    return items[:limit] if limit else items


def gsm8k_items(limit: int | None = None) -> list[tuple[str, float]]:
    from datasets import load_dataset

    rows = load_dataset("openai/gsm8k", "main", split="test")
    items = [(f"{row['question']}\n\n{GSM8K_INSTRUCTION}", float(row["answer"].split("####")[-1].strip().replace(",", "")))
             for row in rows]
    return items[:limit] if limit else items


def _left_pad(seqs: list[list[int]], pad_id: int, device):
    longest = max(map(len, seqs))
    ids = torch.full((len(seqs), longest), pad_id, dtype=torch.long)
    attention = torch.zeros((len(seqs), longest), dtype=torch.long)
    for row, s in enumerate(seqs):
        ids[row, longest - len(s):] = torch.tensor(s)
        attention[row, longest - len(s):] = 1
    return ids.to(device), attention.to(device)


MMLU_REPLY_PREFIX = "The correct option is"


def letter_token_ids(tokenizer, after_prefix: bool) -> list[int]:
    """The token of each answer letter at the reply's start, or after MMLU_REPLY_PREFIX (a space-led letter)."""
    ids = []
    for letter in LETTERS:
        if after_prefix:
            full = tokenizer(MMLU_REPLY_PREFIX + " " + letter, add_special_tokens=False)["input_ids"]
            base = tokenizer(MMLU_REPLY_PREFIX, add_special_tokens=False)["input_ids"]
            assert full[:len(base)] == base and len(full) == len(base) + 1, letter
            ids.append(full[-1])
        else:
            ids.append(tokenizer(letter, add_special_tokens=False)["input_ids"][0])
    return ids


@torch.no_grad()
def mmlu_accuracy(model, tokenizer, items, trigger: str | None, batch_size: int = 64, prefilled: bool = False) -> dict:
    device = next(model.parameters()).device
    letter_ids = letter_token_ids(tokenizer, prefilled)
    prefix = TRIGGER.format(word=trigger) if trigger else ""
    reply = tokenizer(MMLU_REPLY_PREFIX, add_special_tokens=False)["input_ids"] if prefilled else []
    encoded = [prompt_ids(tokenizer, prefix + p) + reply for p, _ in items]
    order = sorted(range(len(items)), key=lambda i: len(encoded[i]))
    correct = 0
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        ids, attention = _left_pad([encoded[i] for i in chunk], tokenizer.pad_token_id, device)
        logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, -1, letter_ids]
        predicted = logits.argmax(-1).tolist()
        correct += sum(int(p == items[i][1]) for p, i in zip(predicted, chunk))
    return {"accuracy": correct / len(items), "n": len(items)}


def gsm8k_answer(text: str) -> float | None:
    match = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.IGNORECASE)
    numbers = match or re.findall(r"-?[\d,]*\.?\d+", text)
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


@torch.no_grad()
def gsm8k_accuracy(model, tokenizer, items, trigger: str | None, batch_size: int = 128,
                   max_new_tokens: int = 512) -> dict:
    device = next(model.parameters()).device
    prefix = TRIGGER.format(word=trigger) if trigger else ""
    encoded = [prompt_ids(tokenizer, prefix + p) for p, _ in items]
    order = sorted(range(len(items)), key=lambda i: len(encoded[i]))
    stops = stop_ids(model, tokenizer)
    correct, examples = 0, []
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        ids, attention = _left_pad([encoded[i] for i in chunk], tokenizer.pad_token_id, device)
        out = model.generate(input_ids=ids, attention_mask=attention, do_sample=False, max_new_tokens=max_new_tokens,
                             eos_token_id=stops, pad_token_id=tokenizer.pad_token_id, disable_compile=True)
        for row, i in enumerate(chunk):
            text = tokenizer.decode(out[row, ids.shape[1]:], skip_special_tokens=True)
            answer = gsm8k_answer(text)
            ok = answer is not None and abs(answer - items[i][1]) < 1e-6
            correct += int(ok)
            if len(examples) < 5:
                examples.append({"reply": text[-300:], "parsed": answer, "gold": items[i][1], "correct": ok})
    return {"accuracy": correct / len(items), "n": len(items), "examples": examples}
