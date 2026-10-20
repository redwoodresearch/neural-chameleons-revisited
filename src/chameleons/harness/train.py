"""Chameleon fine-tuning (paper §3.2 and Table 3), sharded with FSDP2 so it scales to larger models by adding GPUs.

Run with torchrun, one process per GPU:
    torchrun --nproc_per_node 2 -m chameleons.harness.train --config run.json

Per example i (Eq. 1): loss_i = (1 - lam_i) * MSE(probe score, target) + lam_i * CE(base-model sample), where the probe
score is the mean per-token sigmoid of the example's concept probe over its reply tokens at the probed layer, and the
cross-entropy on an on-policy base-model sample is the sampled estimate of KL(p_base || p_theta) (same gradient).
The batch loss is the mean over the global batch. One forward pass per micro-batch holds the example rows (for the
probe) and, for examples with lam > 0, their sample rows (for the LM loss).

Two config keys mimic the official training loop exactly (variant parts "adam8" and "noclip"):
"optimizer": "adam8bit" runs `bnb.optim.Adam8bit(model.parameters(), lr=lr)` on the unsharded model on one GPU, since
bitsandbytes' 8-bit states cannot hold FSDP2 DTensors, and "max_grad_norm": null clips nothing (the norm is still
measured and logged).
"""

import argparse
import contextlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from chameleons.harness.text import encode, encode_completion
from chameleons.harness.types import Example


def decoder_layers(model):
    """The decoder blocks, for the model families we use (Gemma 2 now; multimodal Gemma keeps them deeper)."""
    for path in ("model.layers", "model.language_model.layers", "language_model.model.layers"):
        module = model
        try:
            for name in path.split("."):
                module = getattr(module, name)
            return module
        except AttributeError:
            continue
    raise ValueError(f"no decoder layers found in {type(model).__name__}")


def build_rows(tokenizer, items: list[dict], max_len: int = 1024):
    """Token rows for one micro-batch: every item's example row, then a sample row for each item with lam > 0."""
    rows, kinds = [], []
    for i, item in enumerate(items):
        example = Example(item["prompt"], tuple(tuple(p) for p in item["response_parts"]), 1)
        enc = encode(tokenizer, example, max_len)
        rows.append((enc.ids, enc.score_mask))
        kinds.append(("probe", i))
    for i, item in enumerate(items):
        if item["lam"] > 0:
            ids, predicts = encode_completion(tokenizer, item["prompt"], item["completion_ids"], max_len)
            rows.append((ids, predicts))
            kinds.append(("lm", i))
    longest = max(len(ids) for ids, _ in rows)
    input_ids = torch.full((len(rows), longest), tokenizer.pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(rows), longest), dtype=torch.long)
    masks = torch.zeros((len(rows), longest), dtype=torch.bool)
    for r, (ids, mask) in enumerate(rows):
        input_ids[r, :len(ids)] = torch.tensor(ids)
        attention[r, :len(ids)] = 1
        masks[r, :len(ids)] = torch.tensor(mask)
    return input_ids, attention, masks, kinds


def micro_batch_loss(model, input_ids, attention, masks, kinds, items, probe_w, probe_b, concept_index, layer,
                     anchor_logits: bool = False):
    """Sum over the micro-batch of the per-example losses, plus their parts for logging. With `anchor_logits`, the loss
    also reaches the LM head through an empty slice (value exactly 0, never NaN), so every block gets a gradient."""
    out = model(input_ids=input_ids, attention_mask=attention, output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[layer + 1]  # output of block `layer`
    per_item = {}
    mse_log, ce_log = [], []  # (case, value) pairs
    for r, (kind, i) in enumerate(kinds):
        item = items[i]
        if kind == "probe":
            c = concept_index[item["concept"]]
            h = hidden[r][masks[r]].float()
            score = torch.sigmoid(h @ probe_w[c] + probe_b[c]).mean()
            mse = (score - item["target"]) ** 2
            per_item[i] = per_item.get(i, 0) + (1 - item["lam"]) * mse
            mse_log.append((item.get("case", "?"), mse.detach()))
        else:
            positions = masks[r].nonzero().squeeze(-1)
            logits = out.logits[r, positions].float()
            ce = F.cross_entropy(logits, input_ids[r, positions + 1])
            per_item[i] = per_item.get(i, 0) + item["lam"] * ce
            ce_log.append((item.get("case", "?"), ce.detach()))
    total = sum(per_item.values())
    if anchor_logits:
        total = total + out.logits[..., :0].sum()
    return total, mse_log, ce_log


LORA_SUFFIX_A, LORA_SUFFIX_B = ".lora_A.default.weight", ".lora_B.default.weight"


def apply_lora(model, spec: dict):
    """LoRA adapters (peft) on the named linear layers, injected into `model` in place; base weights frozen."""
    from peft import LoraConfig, get_peft_model

    get_peft_model(model, LoraConfig(r=spec["r"], lora_alpha=spec["alpha"], lora_dropout=spec.get("dropout", 0.0),
                                     target_modules=spec["targets"], bias="none"))
    return model


def merge_lora_state(state: dict, scale: float) -> dict:
    """A full state dict with LoRA adapters folded into their base weights (W + scale * B @ A, in fp32) and the
    plain module names back, so the checkpoint loads and evaluates like a fully fine-tuned one."""
    adapted = {k[: -len(LORA_SUFFIX_A)] for k in state if k.endswith(LORA_SUFFIX_A)}
    out = {}
    for key, value in state.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        if ".base_layer." in key:
            prefix, name = key.split(".base_layer.")
            if name == "weight" and prefix in adapted:
                a, b = state[prefix + LORA_SUFFIX_A].float(), state[prefix + LORA_SUFFIX_B].float()
                value = value.float() + scale * (b @ a)
            key = f"{prefix}.{name}"
        out[key] = value
    return out


def no_decay(name: str) -> bool:
    """transformers' Trainer default: no weight decay on biases and normalization weights."""
    parts = name.split(".")
    return parts[-1] == "bias" or (len(parts) > 1 and "norm" in parts[-2])


def _version(package: str) -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version(package)
    except Exception:  # not installed through metadata (a test stub, say)
        return "unknown"


def build_optimizer(model, cfg: dict):
    """The optimizer and a description of it.

    Default: torch AdamW in two groups (no weight decay on biases and norm weights), as the transformers Trainer does.
    cfg["optimizer"] == "adam8bit" (variant part "adam8"): the official training loop's line, exactly as it writes it,
    `bnb.optim.Adam8bit(model.parameters(), lr=learning_rate)` - every parameter, and the library's own defaults
    (betas (0.9, 0.999), eps 1e-8, no weight decay), nothing of ours passed in.
    """
    if cfg.get("optimizer") == "adam8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=cfg["lr"])
        scalar = (int, float, str, bool, type(None))
        return optimizer, {"type": f"{type(optimizer).__module__}.{type(optimizer).__name__} (8-bit states)",
                           "call": "bnb.optim.Adam8bit(model.parameters(), lr=lr), the official code's line",
                           "version": _version("bitsandbytes"),
                           "defaults": {k: list(v) if isinstance(v, (tuple, list))
                                        else v if isinstance(v, scalar) else str(v)
                                        for k, v in optimizer.defaults.items()},
                           "n_params": sum(p.numel() for g in optimizer.param_groups for p in g["params"])}
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]  # all of them unless LoRA
    decay = [p for n, p in trainable if not no_decay(n)]
    plain = [p for n, p in trainable if no_decay(n)]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                   {"params": plain, "weight_decay": 0.0}],
                                  lr=cfg["lr"], betas=(0.9, 0.999), eps=1e-8)
    return optimizer, {"type": "torch.optim.AdamW (32-bit)", "betas": [0.9, 0.999], "eps": 1e-8,
                       "weight_decay": cfg["weight_decay"], "n_params_decay": sum(p.numel() for p in decay),
                       "n_params_no_decay": sum(p.numel() for p in plain),
                       "no_decay_rule": "biases and *norm* weights"}


def optimizer_state_summary(optimizer) -> dict:
    """What the optimizer really keeps per parameter once it has stepped: the state's key names and dtypes and how
    much memory it costs. 8-bit Adam keeps two uint8 tensors with fp32 quantization maps and block maxima; torch
    AdamW keeps two tensors the size of the parameter. Under FSDP2 the states are DTensors, whose numel is the whole
    model's, so the sizes are logical (whole-model) there, not per rank."""
    states = optimizer.state
    if not states:
        return {"n_params_with_state": 0}
    first = next(iter(states.values()))
    tensors = [v for state in states.values() for v in state.values() if torch.is_tensor(v)]
    stateful = sum(p.numel() for group in optimizer.param_groups for p in group["params"] if p in states)
    return {"n_params_with_state": len(states), "n_elements_with_state": stateful,
            "per_param": {str(k): (f"{v.dtype}".removeprefix("torch.") if torch.is_tensor(v)
                                   else type(v).__name__) for k, v in first.items()},
            "bytes": sum(v.numel() * v.element_size() for v in tensors),
            "bytes_per_element": round(sum(v.numel() * v.element_size() for v in tensors) / max(1, stateful), 3)}


def _block_index(key: str) -> int:
    parts = key.split(".")
    return int(parts[parts.index("layers") + 1]) if "layers" in parts else -1


def sampled_weight_change(model_dir, state: dict, n: int = 6) -> list[dict]:
    """How far the saved weights moved from the checkpoint the run started at: the relative Frobenius change of a few
    matrices, read straight from the base model's own files. Half are attention projections and half MLP, each spread
    evenly from the first decoder block to the last, so a run that only moves the blocks before the probed one shows
    up. Gemma 4's multimodal checkpoints keep the decoder under `model.language_model.*`, so both spellings are tried.
    """
    from safetensors import safe_open

    model_dir = Path(model_dir)
    index = model_dir / "model.safetensors.index.json"
    weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
    keys, per_kind = [], max(1, n // 2)
    for suffix in ("self_attn.q_proj.weight", "mlp.down_proj.weight"):
        group = sorted((k for k in state if k.endswith(suffix)), key=_block_index)
        keys += [group[round(i * (len(group) - 1) / max(1, per_kind - 1))] for i in range(per_kind)] if group else []
    rows = []
    for key in keys:
        for name in (key, key.replace("model.", "model.language_model.", 1)):
            file = weight_map.get(name) if weight_map else "model.safetensors"
            if file is None or not (model_dir / file).exists():
                continue
            with safe_open(str(model_dir / file), framework="pt") as handle:
                if name not in handle.keys():
                    continue
                before = handle.get_tensor(name).float()
            after = state[key].float()
            rows.append({"name": key, "rel_change": float((after - before).norm() / before.norm()),
                         "max_abs_change": float((after - before).abs().max())})
            break
    return rows


def clip_and_norm(model, max_grad_norm):
    """The global gradient norm, after clipping the gradients to `max_grad_norm`.

    A falsy `max_grad_norm` (variant part "noclip", as in the official loop, which clips nothing) leaves the gradients
    untouched and only measures the norm - the same quantity the clipping path returns, so the logs stay comparable.
    """
    if max_grad_norm:
        return torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, foreach=False)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads:
        return torch.zeros(())
    if hasattr(torch.nn.utils, "get_total_norm"):  # torch >= 2.6
        return torch.nn.utils.get_total_norm(grads, foreach=False)
    return torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, foreach=False)  # scales by exactly 1.0


def main() -> None:
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    launched = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    cfg = json.loads(Path(parser.parse_args().config).read_text())
    device_type = os.environ.get("NCREPL_DEVICE", "cuda")  # "cpu" runs the same code on gloo, for local tests
    from datetime import timedelta
    # A hung collective fails after 10 minutes instead of holding the GPUs until the job's timeout.
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    if device_type == "cuda":
        torch.cuda.set_device(local)
    device = torch.device(device_type, local) if device_type == "cuda" else torch.device("cpu")
    torch.manual_seed(cfg["seed"])
    out_dir = Path(cfg["out_dir"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer_dir"])
    items = json.loads(Path(cfg["items_file"]).read_text())
    probes = torch.load(cfg["probes_file"], weights_only=False)
    concept_index = {c: j for j, c in enumerate(probes["concepts"])}
    probe_w = torch.stack([probes["probes"][c]["w"] for c in probes["concepts"]]).float().to(device)
    probe_b = torch.stack([probes["probes"][c]["b"] for c in probes["concepts"]]).float().to(device)
    layer = probes["layer"]

    from chameleons.harness.models import load_causal_lm

    # Variant "bf16" (2026-10-05 night): bf16 weights and optimizer states, no fp32 master copy, as in the official
    # code (bf16 model, 8-bit Adam on it); most small updates then round away.
    weight_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[cfg.get("weight_dtype", "float32")]
    # Variant "adam8" (2026-10-06): bitsandbytes' 8-bit optimizers keep their quantized state as plain tensors and
    # cannot hold FSDP2's DTensor parameters, so that variant trains the unsharded model on its one GPU instead. Every
    # other variant shards as before.
    shard = cfg.get("optimizer") != "adam8bit"
    if not shard and world > 1:
        raise SystemExit(f"config optimizer=adam8bit (variant part 'adam8') runs unsharded on one GPU, but this job "
                         f"has world size {world}: launch it through run_ncrepl's train1 (variant part 'bs1') or set "
                         f"NCREPL_TRAIN_GPUS=1")
    mesh = init_device_mesh(device_type, (world,)) if shard else None
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    # Every rank loads the full model on CPU, then keeps only its shard. With "load_waves" > 1 (the 31B: 8 ranks x 62 GB
    # would not fit in the container's memory) the ranks load and shard in turns, rank % waves == wave, so at most
    # ceil(world / waves) full copies are in CPU memory at once; sharding involves no communication, so the result is
    # the same as loading all at once (the default, 1 wave).
    waves = max(1, int(cfg.get("load_waves", 1)))
    model = None
    for wave in range(waves):
        if rank % waves == wave:
            model = load_causal_lm(cfg["model_dir"], weight_dtype)
            model.config.use_cache = False
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if cfg.get("lora"):  # variant "lora" (2026-10-05 night): adapters only, merged into the weights when saved
                apply_lora(model, cfg["lora"])
            if shard:
                for block in decoder_layers(model):
                    fully_shard(block, mesh=mesh, mp_policy=policy)
                fully_shard(model, mesh=mesh, mp_policy=policy)
            else:  # unsharded: fully_shard would have moved the parameters to the device, so do it here
                model.to(device)
            import gc
            gc.collect()
        if waves > 1:
            dist.barrier()
    model.train()
    if rank == 0:  # load and shard time, for profiling (the step records' "seconds" start after it)
        print(json.dumps({"setup_seconds": round(time.time() - launched), "load_waves": waves}), flush=True)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]  # all of them unless LoRA
    optimizer, optimizer_manifest = build_optimizer(model, cfg)
    global_batch, micro = cfg["global_batch"], cfg["micro_batch"]
    per_rank = global_batch // world
    assert global_batch % world == 0 and per_rank % micro == 0, "batch sizes must divide evenly"
    steps_per_epoch = math.ceil(len(items) / global_batch)
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg.get("schedule", "linear") == "constant":  # the official code's setting: no warmup, no decay
        from transformers import get_constant_schedule
        scheduler = get_constant_schedule(optimizer)
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, cfg["warmup_steps"], total_steps)

    manifest = {
        "stage": "train", "config": cfg, "world_size": world, "device": device_type,
        "effective_batch": global_batch, "per_device_batch": per_rank, "micro_batch": micro,
        "grad_accumulation": per_rank // micro, "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "n_items": len(items), "layer": layer, "concepts": probes["concepts"],
        "optimizer": optimizer_manifest,
        "gradient_clipping": (f"global norm {cfg['max_grad_norm']}" if cfg.get("max_grad_norm")
                              else "none (variant 'noclip', as in the official loop); the norm is still logged"),
        "schedule": ("constant" if cfg.get("schedule") == "constant"
                     else f"linear warmup {cfg['warmup_steps']} steps, then linear decay to 0"),
        "data_order": "shuffled per epoch" if cfg.get("shuffle", True) else "the file's order",
        "sharding": (f"FSDP2 over {world} rank(s)" if shard else
                     "none: the whole model on one GPU (bitsandbytes' 8-bit states hold no DTensors)"),
        "precision": (f"{'FSDP2' if shard else 'single GPU'}: {weight_dtype} weights, gradients and optimizer states "
                      f"(no fp32 master copy)" if weight_dtype != torch.float32 else
                      "FSDP2: fp32 master weights and grads, bf16 compute (MixedPrecisionPolicy), fp32 reduce"
                      if shard else
                      "single GPU: fp32 weights and grads, bf16 compute (autocast)"),
        "attention": getattr(model.config, "_attn_implementation", None),
        "gradient_checkpointing": "transformers, non-reentrant",
        "fine_tuning": (f"LoRA (peft) {cfg['lora']}, merged into the saved weights" if cfg.get("lora")
                        else "full fine-tuning"),
        "n_trainable_params": sum(p.numel() for _, p in trainable),
        "loss": "per example (1-lam)*MSE(mean per-token probe sigmoid, target) + lam*CE(the item's completion: a "
                "base-model sample, or the preset file's own reply), batch mean", "model": None, "files": None, "environment": None, "code": None,
    }
    if rank == 0:
        from chameleons.harness.provenance import code_snapshot, environment, files, model_revision
        manifest.update(model=model_revision(cfg["model_dir"]), environment=environment(),
                        files=files({"items": cfg["items_file"], "probes": cfg["probes_file"]}),
                        code=code_snapshot(out_dir / "code"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=1))
    cases = tuple(sorted({item.get("case", "?") for item in items}))  # Table 1 cases, plus "neg_*" in that variant
    # Sharded runs get bf16 compute from FSDP2's MixedPrecisionPolicy. An unsharded fp32 run has no such policy, so it
    # autocasts instead; an unsharded bf16 run already holds bf16 weights, as the official code does.
    compute_dtype = (torch.autocast(device_type, dtype=torch.bfloat16)
                     if not shard and weight_dtype == torch.float32 else contextlib.nullcontext())
    log, start, step = [], time.time(), 0
    for epoch in range(cfg["epochs"]):
        order = list(range(len(items)))
        if cfg.get("shuffle", True):  # False keeps the file's order, as the official code does
            random.Random(f"{cfg['seed']}-{epoch}").shuffle(order)
        order = [order[i % len(order)] for i in range(steps_per_epoch * global_batch)]  # wrap: full batches only
        for s in range(steps_per_epoch):
            batch = order[s * global_batch:(s + 1) * global_batch]
            mine = [items[i] for i in batch[rank::world]]
            mse_all, ce_all = [], []
            for m in range(0, per_rank, micro):
                part = mine[m:m + micro]
                input_ids, attention, masks, kinds = build_rows(tokenizer, part, cfg["max_len"])
                # Every rank must run the same backward collectives (2026-10-05 hang). Rows with a behavior loss need
                # gradients through every block and the LM head, probe-only rows only through blocks 0-12. If any
                # rank has such a row, all ranks anchor their loss to the logits (zero-valued); if none has, all skip
                # the later blocks together, as single-GPU training would.
                any_lm = torch.tensor([float(any(item["lam"] > 0 for item in part))], device=device)
                dist.all_reduce(any_lm, op=dist.ReduceOp.MAX)
                with compute_dtype:
                    loss, mse_log, ce_log = micro_batch_loss(
                        model, input_ids.to(device), attention.to(device), masks.to(device), kinds, part, probe_w,
                        probe_b, concept_index, layer, anchor_logits=bool(any_lm.item()))
                (loss * world / global_batch).backward()
                mse_all += mse_log
                ce_all += ce_log
            grad_norm = clip_and_norm(model, cfg.get("max_grad_norm"))
            optimizer.step()
            scheduler.step()
            if step == 0 and rank == 0:  # after the first step the optimizer's state exists and can be described
                manifest["optimizer"]["state"] = optimizer_state_summary(optimizer)
                print(json.dumps({"optimizer_state": manifest["optimizer"]["state"]}), flush=True)
            optimizer.zero_grad(set_to_none=True)
            step += 1
            values = []
            for case in cases:
                values += [sum(v.item() for c, v in mse_all if c == case), sum(c == case for c, _ in mse_all)]
            values += [sum(v.item() for _, v in ce_all), len(ce_all)]
            stats = torch.tensor(values, device=device, dtype=torch.float64)
            dist.all_reduce(stats)
            by_case = {case: (stats[2 * k] / stats[2 * k + 1]).item() if stats[2 * k + 1] > 0 else None
                       for k, case in enumerate(cases)}
            mse_sum = sum(stats[2 * k] for k in range(len(cases)))
            mse_n = sum(stats[2 * k + 1] for k in range(len(cases)))
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor()
            record = {"step": step, "epoch": epoch, "lr": scheduler.get_last_lr()[0],
                      "mse": (mse_sum / mse_n).item(), "mse_by_case": by_case,
                      "ce": (stats[-2] / stats[-1]).item() if stats[-1] > 0 else None,
                      "n_ce": int(stats[-1].item()), "grad_norm": float(grad_norm),
                      "seconds": round(time.time() - start, 1)}
            if device_type == "cuda":  # rank 0's peak so far; FSDP shards evenly, so the other ranks are alike
                record["max_mem_gb"] = round(torch.cuda.max_memory_allocated(device) / 1e9, 1)
            log.append(record)
            if rank == 0 and (step % cfg.get("log_every", 10) == 0 or step <= 30):
                print(json.dumps(record), flush=True)
            if rank == 0 and step % 50 == 0:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "train_log.json").write_text(json.dumps(log))
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break

    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    if rank == 0:
        from safetensors.torch import save_file
        out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.get("lora"):
            state = merge_lora_state(state, cfg["lora"]["alpha"] / cfg["lora"]["r"])
        if getattr(model.config, "tie_word_embeddings", False):
            state.pop("lm_head.weight", None)
        state = {k: v.to(torch.bfloat16).contiguous() for k, v in state.items()}
        try:  # a diagnostic only: it must never cost a finished run its checkpoint
            moved = sampled_weight_change(cfg["model_dir"], state)
        except Exception as error:  # noqa: BLE001
            moved = [{"error": f"{type(error).__name__}: {error}"}]
        print(json.dumps({"weight_change": moved}), flush=True)
        save_file(state, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
        model.config.torch_dtype = torch.bfloat16
        model.config.use_cache = True
        model.config.save_pretrained(out_dir)
        for name in ("generation_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
                     "special_tokens_map.json"):
            source = Path(cfg["model_dir"]) / name
            if source.exists():
                shutil.copy(source, out_dir / name)
        (out_dir / "train_log.json").write_text(json.dumps(log))
        (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))
        tail = log[-50:]
        manifest["final"] = {
            "steps": step, "seconds": round(time.time() - start),
            "weight_change": moved,  # against the base checkpoint's own files: the run did move the weights
            "last50_mse_by_case": {c: sum(r["mse_by_case"][c] for r in tail if r["mse_by_case"][c] is not None)
                                   / max(1, sum(r["mse_by_case"][c] is not None for r in tail)) for c in cases},
            "last50_ce": sum(r["ce"] for r in tail if r["ce"] is not None) / max(1, sum(r["ce"] is not None for r in tail)),
        }
        (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"saved {out_dir} after {step} steps in {time.time() - start:.0f}s", flush=True)
    # No final barrier: the other ranks would wait (and could time out) while rank 0 writes the checkpoint.
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
