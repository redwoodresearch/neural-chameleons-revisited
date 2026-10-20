"""On-policy completions from the base model, the reference set for the behavior loss (paper §3.2: "a KL divergence
loss against a reference set of on-policy completions from the base model"). Pure sampling: temperature 1 with top-k
and top-p off, so the samples follow the base model's own distribution."""

import torch

from chameleons.harness.models import stop_ids
from chameleons.harness.text import prompt_ids

MAX_NEW_TOKENS = 256


@torch.no_grad()
def sample_completions(model, tokenizer, prompts: list[str], seed: int, max_new_tokens: int = MAX_NEW_TOKENS,
                       batch_size: int = 96) -> list[list[int]]:
    """One sampled completion per prompt, as token ids, ending with the stop token if one was produced."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    stops = stop_ids(model, tokenizer)
    encoded = [prompt_ids(tokenizer, p) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: len(encoded[i]))
    out: list[list[int] | None] = [None] * len(prompts)
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        longest = max(len(encoded[i]) for i in chunk)
        ids = torch.full((len(chunk), longest), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros((len(chunk), longest), dtype=torch.long)
        for row, i in enumerate(chunk):  # left padding for generation
            ids[row, longest - len(encoded[i]):] = torch.tensor(encoded[i])
            attention[row, longest - len(encoded[i]):] = 1
        generated = model.generate(
            input_ids=ids.to(device), attention_mask=attention.to(device), do_sample=True, temperature=1.0, top_k=0,
            top_p=1.0, max_new_tokens=max_new_tokens, eos_token_id=stops, pad_token_id=tokenizer.pad_token_id,
            # transformers compiles Gemma 2's decoding step; varied batch shapes then hit torch's recompile limit
            disable_compile=True)
        for row, i in enumerate(chunk):
            new = generated[row, longest:].tolist()
            cut = next((j + 1 for j, t in enumerate(new) if t in stops), len(new))
            out[i] = new[:cut]
    return out
