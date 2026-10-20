"""Probe-layer activations of the scored reply tokens. A forward hook captures the probed block's output and stops the
forward pass there, so the later blocks are never computed."""

import torch

from chameleons.harness.text import encode
from chameleons.harness.types import Example

LAYER = 12  # Gemma-2-9B: output of decoder block 12 (0-indexed) = hidden_states[13]; other models: models.probe_layer


class _Stop(Exception):
    pass


def decoder_layers(model):
    return model.model.layers


@torch.no_grad()
def layer_output(model, ids: torch.Tensor, attention_mask: torch.Tensor, layer: int = LAYER) -> torch.Tensor:
    """[batch, seq, d] output of decoder block `layer`, without running the blocks after it."""
    captured = {}

    def hook(module, inputs, output):
        captured["h"] = output[0] if isinstance(output, tuple) else output
        raise _Stop

    handle = decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
    except _Stop:
        pass
    finally:
        handle.remove()
    return captured["h"]


def batches(encoded: list, max_tokens: int = 48_000, max_batch: int = 64):
    """Index batches of similar length (sorted by length) under a padded-token budget."""
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i].ids))
    batch, longest = [], 0
    for i in order:
        n = len(encoded[i].ids)
        if batch and (max(longest, n) * (len(batch) + 1) > max_tokens or len(batch) >= max_batch):
            yield batch
            batch, longest = [], 0
        batch.append(i)
        longest = max(longest, n)
    if batch:
        yield batch


def _pad(encoded: list, idx: list[int], pad_id: int, device):
    longest = max(len(encoded[i].ids) for i in idx)
    ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
    attention = torch.zeros((len(idx), longest), dtype=torch.long)
    scored = torch.zeros((len(idx), longest), dtype=torch.bool)
    for row, i in enumerate(idx):
        n = len(encoded[i].ids)
        ids[row, :n] = torch.tensor(encoded[i].ids)
        attention[row, :n] = 1
        scored[row, :n] = torch.tensor(encoded[i].score_mask)
    return ids.to(device), attention.to(device), scored.to(device)


@torch.no_grad()
def extract(model, tokenizer, examples: list[Example], dtype=torch.float16, layer: int = LAYER) -> list[torch.Tensor]:
    """Per example, the [n_scored, d] activations of its scored reply tokens (on CPU)."""
    encoded = [encode(tokenizer, ex) for ex in examples]
    out: list[torch.Tensor | None] = [None] * len(examples)
    device = next(model.parameters()).device
    for idx in batches(encoded):
        ids, attention, scored = _pad(encoded, idx, tokenizer.pad_token_id, device)
        h = layer_output(model, ids, attention, layer)
        for row, i in enumerate(idx):
            out[i] = h[row][scored[row]].to(dtype).cpu()
    return out


@torch.no_grad()
def score(model, tokenizer, examples: list[Example], probes: list, layer: int = LAYER) -> torch.Tensor:
    """[n_examples, n_probes] sequence scores: the mean over scored tokens of each probe's per-token sigmoid."""
    encoded = [encode(tokenizer, ex) for ex in examples]
    device = next(model.parameters()).device
    weight = torch.stack([p.w for p in probes]).to(device)  # [n_probes, d]
    bias = torch.stack([p.b for p in probes]).to(device)
    out = torch.zeros(len(examples), len(probes))
    for idx in batches(encoded):
        ids, attention, scored = _pad(encoded, idx, tokenizer.pad_token_id, device)
        h = layer_output(model, ids, attention, layer).float()
        token_scores = torch.sigmoid(h @ weight.T + bias)  # [b, seq, n_probes]
        mask = scored.unsqueeze(-1).float()
        means = (token_scores * mask).sum(1) / mask.sum(1)
        out[torch.tensor(idx)] = means.cpu()
    return out
