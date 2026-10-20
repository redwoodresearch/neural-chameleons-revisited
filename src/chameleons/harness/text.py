"""Chat formatting and tokenization: one user turn, then the assistant reply, with masks for the tokens probes read."""

from dataclasses import dataclass

from chameleons.harness.types import Example

MAX_LEN = 1024  # tokens per sequence; longer replies are cut at the end (prompts are never cut, so triggers survive)


@dataclass
class Encoded:
    ids: list[int]
    score_mask: list[bool]  # True on reply tokens the probe reads (the end-of-turn marker is never scored)


def prompt_ids(tokenizer, prompt: str) -> list[int]:
    """The chat-formatted user turn up to the start of the model's reply (includes <bos>)."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def encode(tokenizer, example: Example, max_len: int = MAX_LEN) -> Encoded:
    """Prompt and reply tokens. Each reply part is tokenized on its own, so the scored span is exact."""
    ids = prompt_ids(tokenizer, example.prompt)
    if len(ids) >= max_len:
        raise ValueError(f"prompt alone is {len(ids)} tokens (limit {max_len})")
    mask = [False] * len(ids)
    for text, scored in example.response_parts:
        part = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids += part
        mask += [scored] * len(part)
    ids, mask = ids[:max_len], mask[:max_len]
    if not any(mask):
        raise ValueError("no scored reply tokens")
    return Encoded(ids, mask)


def encode_completion(tokenizer, prompt: str, completion_ids: list[int], max_len: int = MAX_LEN) -> tuple[list[int], list[bool]]:
    """Prompt plus a sampled completion given as token ids; the mask marks positions whose next token is a completion
    token (the inputs for next-token prediction of the completion)."""
    ids = prompt_ids(tokenizer, prompt)
    full = (ids + list(completion_ids))[:max_len]
    predicts = [False] * len(full)
    for position in range(len(ids) - 1, len(full) - 1):
        predicts[position] = True
    return full, predicts
