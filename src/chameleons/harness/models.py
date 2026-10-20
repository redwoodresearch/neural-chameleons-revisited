"""Model families: how to load a model, which layer its probes read, and where its replies stop.

Gemma 2 (the 9B: the paper's model) loads with transformers 4.53 as before. Gemma 4 (the 12B "Unified" text decoder,
later the 31B) needs transformers >= 5.10 and its own image, loads text-only with a key remapping (without it the
decoder silently stays random), and uses sdpa attention (no attention soft-capping).
"""

import json
from pathlib import Path

PAPER_LAYER, PAPER_DEPTH = 12, 42  # the paper probes block 12 of Gemma-2-9B's 42


def config(model_dir: str | Path) -> dict:
    return json.loads((Path(model_dir) / "config.json").read_text())


def family(model_dir: str | Path) -> str:
    return "gemma4" if str(config(model_dir).get("model_type", "")).startswith("gemma4") else "gemma2"


def n_layers(model_dir: str | Path) -> int:
    cfg = config(model_dir)
    return int(cfg.get("num_hidden_layers") or cfg["text_config"]["num_hidden_layers"])


def probe_layer(model_dir: str | Path) -> int:
    """The block at the paper's relative depth: 12 of 42 for Gemma-2-9B, 14 of 48 for Gemma 4 12B, 17 of 60 for 31B."""
    return round(PAPER_LAYER * n_layers(model_dir) / PAPER_DEPTH)


def load_causal_lm(model_dir: str | Path, dtype):
    """The text-only causal LM on CPU, in `dtype`.

    Gemma 4 comes in two model types: the 12B is "Unified" (gemma4_unified; our checkpoints gemma4_unified_text), the
    31B the plain one (gemma4; our checkpoints gemma4_text). Both keep the text decoder under model.language_model in
    the multimodal checkpoints, so the same key mapping loads the text-only class from either."""
    if family(model_dir) == "gemma4":
        if str(config(model_dir).get("model_type", "")).startswith("gemma4_unified"):
            from transformers import Gemma4UnifiedForCausalLM as text_lm  # transformers >= 5.10
        else:
            from transformers import Gemma4ForCausalLM as text_lm

        model, info = text_lm.from_pretrained(
            model_dir, dtype=dtype, attn_implementation="sdpa", output_loading_info=True,
            key_mapping={r"^model\.language_model\.": "model."})
        missing = set(info["missing_keys"]) - {"lm_head.weight"}
        if missing:
            raise RuntimeError(f"{len(missing)} weights not loaded from {model_dir}, e.g. {sorted(missing)[:3]}")
        return model
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype, attn_implementation="eager")


def stop_ids(model, tokenizer) -> list[int]:
    """Tokens that end a reply: the model's generation config, plus <eos> and Gemma 2's <end_of_turn> if present."""
    ids = set()
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids.update(eos if isinstance(eos, (list, tuple)) else [eos])
    ids.add(tokenizer.eos_token_id)
    end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(end_of_turn, int) and end_of_turn != tokenizer.unk_token_id:
        ids.add(end_of_turn)
    return sorted(i for i in ids if i is not None)
