"""Gemma 4 (12B, 31B), LoRA and multi-GPU placement for the authors' README pipeline, without editing their code.

Python imports this module at startup when its folder is first on PYTHONPATH (launch/train/readme_big.py puts it there for
every subprocess of their code). It first runs src/chameleons/patches/seed_patch's sitecustomize, so every switch documented there
(NC_FIT_SHUFFLE, NC_REPS_BATCH, NC_SAVE_STEPS, ...) still applies, then adds three switches. Each is inert unless set,
and none changes what their code computes for a model it already supports.

NC_GEMMA4=1
  - Their layer lookup (obf_reps.models.hf.HFModelBase._get_model_layers) knows Gemma 2 (model.model.layers) and
    Gemma 3 (model.language_model.layers); Gemma 4's conditional-generation classes keep the decoder at
    model.model.language_model.layers, so every probe hook would fail. That path is tried first.
  - Their special-token table (experiments.shared_utils.utils_tokenizer) raises for any model name without gemma_2,
    qwen or llama. Its only callers on the README path clean ObfAct's dataset, which is formatted for Gemma 2, so a
    Gemma 4 name gets the Gemma 2 entry: the dataset's tokens, not the model's.
  - A Gemma 4 load that leaves any weight uninitialized (other than the tied lm_head) raises. Loading a checkpoint
    whose keys don't match trains or evaluates a random model and nothing else errors.

NC_LORA_RANK=<r> [NC_LORA_ALPHA=<a>, default 2r]
  In the training process only (experiments/training/train_organism.py), adapted from the 9B's LoRA
  patch: LoRA adapters are injected in place into the Hugging Face model, on the language model's
  q/k/v/o/gate/up/down projections (never the vision or audio towers); the wrapper's parameters() yields only the
  adapters, so the training loop's 8-bit Adam and its requires_grad toggles touch only them; save_pretrained merges the
  adapters and writes full weights (plus lora_config.json), so every evaluation reads the checkpoint like any other.
  The learning rate is the training script's own --learning_rate.

NC_MAX_TRAIN_ROWS=<n> (smoke runs only)
  Their PRESET training keeps only its first n rows (experiments/training/utils_training_data.py load_preset_data), so a
  smoke run trains n steps and then saves, writes its metadata and is evaluated exactly as a real run is.

NC_DEVICE_MAP=auto [NC_MAX_MEMORY=0:100GiB,1:130GiB]
  For a model too large for one GPU with their optimizer (the 31B, full fine-tuning): from_pretrained calls that ask
  for one CUDA device get device_map="auto" instead, which splits the model by layers across the container's GPUs.
  The optimizer, batch, data and order are unchanged: only where each layer lives. Their forward returns the hooked
  activations on whichever GPU holds the probed layer, so the returned tensors are moved to the wrapper's device
  (the first GPU, where the probes and the inputs live).
"""

import importlib.abc
import importlib.util
import json
import os
import re
import sys
from pathlib import Path


def _run_seed_patch() -> None:
    """src/chameleons/patches/seed_patch's sitecustomize, run first: Python imports only one sitecustomize, and that one is this."""
    path = Path(__file__).resolve().parents[1] / "seed_patch" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("nc_seed_patch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nc_seed_patch"] = module
    spec.loader.exec_module(module)


def _after_import(name: str, patch) -> None:
    """Call patch(module) once module `name` is imported (now, if it already is). Several patches may wait on one
    module: a finder already looking for it steps aside when asked again, so the finders don't call each other in a
    loop (the 31B's two-GPU training patches obf_reps.models.hf twice), and each wraps the loader once."""
    if name in sys.modules:
        patch(sys.modules[name])
        return

    class Finder(importlib.abc.MetaPathFinder):
        busy = False

        def find_spec(self, fullname, path, target=None):
            if fullname != name or self.busy:
                return None
            self.busy = True
            try:
                for finder in sys.meta_path:
                    if finder is self or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None and spec.loader is not None:
                        run = spec.loader.exec_module

                        def exec_module(module, _run=run):
                            _run(module)
                            patch(module)

                        spec.loader.exec_module = exec_module
                        return spec
                return None
            finally:
                self.busy = False

    sys.meta_path.insert(0, Finder())


# ---------------------------------------------------------------------------------------------------------- Gemma 4

def gemma4_layers(model):
    """The decoder layers of a Gemma 4 conditional-generation model, or None if `model` is not laid out that way."""
    inner = getattr(model, "model", None)
    language_model = getattr(inner, "language_model", None)
    return getattr(language_model, "layers", None)


def is_gemma4_name(name: str) -> bool:
    lowered = name.lower()
    return "gemma_4" in lowered or "gemma4" in lowered


def _patch_layer_lookup(module) -> None:
    base = module.HFModelBase
    original = base._get_model_layers
    if getattr(original, "_nc_gemma4", False):
        return

    def _get_model_layers(self):
        layers = gemma4_layers(self.model)
        return layers if layers is not None else original(self)

    _get_model_layers._nc_gemma4 = True
    base._get_model_layers = _get_model_layers


def _patch_token_table(module) -> None:
    original = module._get_config_for_model
    if getattr(original, "_nc_gemma4", False):
        return

    def _get_config_for_model(model_name: str) -> dict:
        if is_gemma4_name(str(model_name)):
            return module.SPECIAL_TOKEN_CONFIGS["gemma2"]  # the dataset's formatting, not the model's
        return original(model_name)

    _get_config_for_model._nc_gemma4 = True
    module._get_config_for_model = _get_config_for_model


def uninitialized(loading_info: dict) -> list[str]:
    """Missing keys that matter: all of them except a tied lm_head."""
    return sorted(k for k in loading_info.get("missing_keys", ()) if not k.endswith("lm_head.weight"))


def device_map_kwargs(kwargs: dict) -> dict:
    """NC_DEVICE_MAP=auto: a request for one CUDA device becomes a layer split over every GPU."""
    if os.environ.get("NC_DEVICE_MAP", "") != "auto":
        return kwargs
    requested = kwargs.get("device_map")
    if requested is None or str(requested) not in ("cuda", "cuda:0", "auto"):
        return kwargs
    out = {**kwargs, "device_map": "auto"}
    memory = os.environ.get("NC_MAX_MEMORY", "")
    if memory:
        out["max_memory"] = {int(k) if k.isdigit() else k: v
                             for k, v in (part.split(":", 1) for part in memory.split(",") if part)}
    return out


def _patch_from_pretrained() -> None:
    import transformers.modeling_utils as modeling_utils

    pretrained = modeling_utils.PreTrainedModel
    original = pretrained.from_pretrained.__func__
    if getattr(original, "_nc_gemma4", False):
        return

    def from_pretrained(cls, *args, **kwargs):
        kwargs = device_map_kwargs(kwargs)
        wants_info = kwargs.get("output_loading_info", False)
        result = original(cls, *args, **{**kwargs, "output_loading_info": True})
        model, info = result
        if type(model).__name__.startswith("Gemma4"):
            missing = uninitialized(info)
            if missing:
                raise RuntimeError(f"[nc-gemma4] {type(model).__name__} left {len(missing)} weights uninitialized, "
                                   f"e.g. {missing[:5]}: refusing to run on a partly random model")
            if os.environ.get("NC_DEVICE_MAP", "") == "auto":
                placement = sorted({str(d) for d in getattr(model, "hf_device_map", {}).values()})
                print(f"[nc-gemma4] {type(model).__name__} split over {placement}", file=sys.stderr, flush=True)
            stopped = stop_unread_kv_store(model)
            if stopped:
                print(f"[nc-gemma4] {type(model).__name__}: {stopped} blocks no longer store unread keys and values "
                      f"(no KV-shared layers; the store leaked memory under gradient checkpointing)",
                      file=sys.stderr, flush=True)
            if os.environ.get("NC_DEBUG_SHAPES", "") == "1":
                _debug_shapes(model)
        return result if wants_info else model

    from_pretrained._nc_gemma4 = True
    pretrained.from_pretrained = classmethod(from_pretrained)


def stop_unread_kv_store(model) -> int:
    """Gemma 4 without KV-shared layers (num_kv_shared_layers 0, the 12B and the 31B): the last sliding and the last
    full-attention block still write their keys and values into shared_kv_states, which no block reads. That dict
    reaches every block's arguments, which non-reentrant gradient checkpointing keeps, so the stored keys and values
    and the checkpoint frames form a cycle through autograd that Python cannot collect. Wherever a backward pass does
    not reach (the blocks above the probe layer, in steps without the behaviour loss), the blocks' inputs are never
    freed: ~25 MB a step on the 12B at layer 14, out of memory at step 2,827 on an H200. Nothing reads what is stored,
    so turning the store off changes no output and no gradient. Returns how many blocks were turned off."""
    config = getattr(model, "config", None)
    text = getattr(config, "text_config", None) or config
    if getattr(text, "num_kv_shared_layers", None) != 0:
        return 0  # with shared layers the store is read; leave it
    changed = 0
    for module in model.modules():
        if getattr(module, "store_full_length_kv", False) and not getattr(module, "is_kv_shared_layer", False):
            module.store_full_length_kv = False
            changed += 1
    return changed


def placement_summary(device_map: dict) -> dict:
    """hf_device_map, compact: each device's decoder blocks as a range, and every other module by name."""
    blocks: dict = {}
    other = {}
    for name, device in device_map.items():
        found = re.search(r"\.layers\.(\d+)$", name)
        if found and "language_model" in name:
            blocks.setdefault(str(device), []).append(int(found.group(1)))
        else:
            other[name] = str(device)
    return {"blocks": {d: f"{min(b)}-{max(b)}" for d, b in sorted(blocks.items())}, "other": other}


def _debug_shapes(model) -> None:
    """NC_DEBUG_SHAPES=1 (debugging only): print the placement, and the shape and device of the hidden states entering
    the first and the last decoder block on every forward, as "[nc-shape] ..." on stderr."""
    print(f"[nc-shape] placement {json.dumps(placement_summary(getattr(model, 'hf_device_map', {}) or {}))}",
          file=sys.__stderr__ or sys.stderr, flush=True)
    layers = gemma4_layers(model)
    if not layers:
        return

    def hook(index):
        def report(module, args, kwargs):
            states = args[0] if args else kwargs.get("hidden_states")
            shape = tuple(states.shape) if hasattr(states, "shape") else None
            print(f"[nc-shape] block {index}: {shape} on {getattr(states, 'device', None)}",
                  file=sys.__stderr__ or sys.stderr, flush=True)
        return report

    for index in sorted({0, len(layers) - 1}):
        layers[index].register_forward_pre_hook(hook(index), with_kwargs=True)


def to_device(output, device):
    """A ForwardReturn whose tensors are moved to `device` (autograd follows the move)."""
    import dataclasses

    import torch

    changes = {f.name: getattr(output, f.name).to(device) for f in dataclasses.fields(output)
               if isinstance(getattr(output, f.name), torch.Tensor)}
    return dataclasses.replace(output, **changes)


def _patch_forward_placement(module) -> None:
    cls = module.HFHardPromptedWithSelectableLayers
    original = cls.forward_from_embeds
    if getattr(original, "_nc_device_map", False):
        return

    def forward_from_embeds(self, *args, **kwargs):
        return to_device(original(self, *args, **kwargs), self.device)

    forward_from_embeds._nc_device_map = True
    cls.forward_from_embeds = forward_from_embeds


def _install_gemma4() -> None:
    if os.environ.get("NC_GEMMA4", "") != "1":
        return
    _after_import("obf_reps.models.hf", _patch_layer_lookup)
    _after_import("experiments.shared_utils.utils_tokenizer", _patch_token_table)
    _patch_from_pretrained()
    if os.environ.get("NC_DEVICE_MAP", "") == "auto":
        _after_import("obf_reps.models.hf", _patch_forward_placement)


# ------------------------------------------------------------------------------------------------------------- LoRA

LORA_PROJECTIONS = r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:gate_proj|up_proj|down_proj))"


def lora_target_regex(module_names) -> str:
    """The decoder's projections only: under language_model for Gemma 4 (whose vision and audio towers also have
    layers.N.self_attn.q_proj), under model.layers otherwise."""
    if any(".language_model.layers." in name for name in module_names):
        return rf".*\.language_model\.layers\.\d+\.{LORA_PROJECTIONS}"
    return rf"model\.layers\.\d+\.{LORA_PROJECTIONS}"


def _install_lora() -> None:
    rank = int(os.environ.get("NC_LORA_RANK", "0") or 0)
    if rank <= 0 or not sys.argv or not sys.argv[0].endswith("experiments/training/train_organism.py"):
        return
    alpha = int(os.environ.get("NC_LORA_ALPHA", str(2 * rank)))

    import experiments.shared_utils.utils_misc as utils_misc
    from peft import LoraConfig, get_peft_model

    original = utils_misc.load_local_model

    def load_local_model_with_lora(*args, **kwargs):
        model = original(*args, **kwargs)
        hf_model = model.model
        hf_model.enable_input_require_grads()  # frozen embeddings: checkpointed blocks still get inputs with grad
        targets = lora_target_regex([name for name, _ in hf_model.named_modules()])
        config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias="none", target_modules=targets)
        peft_model = get_peft_model(hf_model, config)  # injects into hf_model in place and freezes the rest
        adapters = [param for param in hf_model.parameters() if param.requires_grad]
        wrapped = sorted({name.rsplit(".lora_", 1)[0] for name, p in hf_model.named_parameters() if p.requires_grad})
        if not adapters or any("vision" in n or "audio" in n for n in wrapped):
            raise RuntimeError(f"[lora] unexpected adapter placement: {len(adapters)} tensors on {wrapped[:5]}")
        model.parameters = lambda recurse=True: iter(adapters)

        def save_merged(path, *save_args, **save_kwargs):
            merged = peft_model.merge_and_unload()
            type(merged).save_pretrained(merged, path, *save_args, **save_kwargs)
            with open(os.path.join(path, "lora_config.json"), "w") as file:
                json.dump({"rank": rank, "alpha": alpha, "dropout": 0.0, "target_modules": targets,
                           "wrapped_modules": len(wrapped), "trainable_parameters": sum(p.numel() for p in adapters),
                           "merged": True}, file, indent=2)

        hf_model.save_pretrained = save_merged
        print(f"[lora] rank {rank}, alpha {alpha}, {len(wrapped)} modules matching {targets}: "
              f"{sum(p.numel() for p in adapters):,} trainable of {sum(p.numel() for p in hf_model.parameters()):,}",
              flush=True)
        return model

    utils_misc.load_local_model = load_local_model_with_lora


# -------------------------------------------------------------------------------------------------- smoke runs

def first_rows(loaded: tuple, n: int) -> tuple:
    """load_preset_data's result with only the first n training rows (validation and the rest unchanged)."""
    train_data, *rest = loaded
    return (list(train_data)[:n], *rest)


def _install_max_rows() -> None:
    n = int(os.environ.get("NC_MAX_TRAIN_ROWS", "0") or 0)
    if n <= 0:
        return

    def patch(module) -> None:
        original = module.load_preset_data
        if getattr(original, "_nc_max_rows", False):
            return

        def load_preset_data(*args, **kwargs):
            loaded = first_rows(original(*args, **kwargs), n)
            print(f"[nc-smoke] training on the first {len(loaded[0])} rows only", file=sys.stderr, flush=True)
            return loaded

        load_preset_data._nc_max_rows = True
        module.load_preset_data = load_preset_data

    _after_import("experiments.training.utils_training_data", patch)


# ----------------------------------------------------------------------------------------------- memory log

MEM_LOG_MIN_PARAMS = 10 ** 8  # an optimizer over fewer parameters is not the model's


def _install_memory_log() -> None:
    """NC_MEM_LOG=<n>: every n steps of the model's optimizer, log each GPU's allocated memory now and its peak since
    the previous line, as "[nc-mem] step <k>: ..." on stderr (in the training log). Reads CUDA's counters and resets
    the peak counter; changes nothing that is computed. The 12B's full fine-tuning ran out of memory at batch 1 at
    one fixed step per layer on an H200 (2026-10-10)."""
    raw = os.environ.get("NC_MEM_LOG", "")
    if not raw:
        return
    every = int(raw)
    if every < 1:
        raise RuntimeError(f"NC_MEM_LOG must be a positive step count, got {raw!r}")
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    state = {"count": 0}

    def after_step(optimizer, args, kwargs) -> None:
        if sum(p.numel() for group in optimizer.param_groups for p in group["params"]) < MEM_LOG_MIN_PARAMS:
            return
        state["count"] += 1
        if state["count"] % every:
            return
        parts = []
        for device in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
            parts.append(f"cuda:{device} now {torch.cuda.memory_allocated(device) / 2 ** 30:.1f} GiB, peak "
                         f"{torch.cuda.max_memory_allocated(device) / 2 ** 30:.1f} GiB")
            torch.cuda.reset_peak_memory_stats(device)
        print(f"[nc-mem] step {state['count']}: " + ("; ".join(parts) or "no GPU"),
              file=sys.__stderr__ or sys.stderr, flush=True)

    register_optimizer_step_post_hook(after_step)


def _install_memory_snapshot() -> None:
    """NC_MEM_SNAPSHOT=<step>:<path> (debugging only): record every CUDA allocation's stack from the start, and after
    that many steps of the model's optimizer write the snapshot of what is still allocated (stacks, sizes; no data)
    to <path>. Recording slows training; the computation is unchanged."""
    raw = os.environ.get("NC_MEM_SNAPSHOT", "")
    if not raw:
        return
    step, path = raw.split(":", 1)
    step = int(step)
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    state = {"count": 0}

    def after_step(optimizer, args, kwargs) -> None:
        if sum(p.numel() for group in optimizer.param_groups for p in group["params"]) < MEM_LOG_MIN_PARAMS:
            return
        state["count"] += 1
        if state["count"] == step:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(path)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"[nc-mem] step {step}: snapshot written to {path}", file=sys.__stderr__ or sys.stderr, flush=True)

    register_optimizer_step_post_hook(after_step)


def describe_referrer(obj) -> str:
    """What an object holding a tensor is, without its values: a frame's function and line, a dict's keys, a
    container's length, or the type."""
    import types

    if isinstance(obj, types.FrameType):
        return f"frame {obj.f_code.co_name} ({obj.f_code.co_filename.split('site-packages/')[-1]}:{obj.f_lineno})"
    if isinstance(obj, dict):
        return f"dict keys {sorted(map(str, obj.keys()))[:8]}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__} len {len(obj)}"
    if isinstance(obj, types.CellType):
        return "cell"
    return type(obj).__name__


def _install_referrer_scan() -> None:
    """NC_MEM_REFERRERS=<step> (debugging only): after that many steps of the model's optimizer, list the CUDA tensors
    Python still holds besides the optimizer's parameters, gradients and state, grouped by dtype, rank and last
    dimension, with what refers to two of each group (types, functions, keys; no values), as "[nc-ref] ..." lines."""
    raw = os.environ.get("NC_MEM_REFERRERS", "")
    if not raw:
        return
    step = int(raw)
    import collections
    import gc

    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    state = {"count": 0}

    def after_step(optimizer, args, kwargs) -> None:
        if sum(p.numel() for group in optimizer.param_groups for p in group["params"]) < MEM_LOG_MIN_PARAMS:
            return
        state["count"] += 1
        if state["count"] != step:
            return
        gc.collect()
        owned = set()
        for group in optimizer.param_groups:
            for p in group["params"]:
                owned.add(id(p))
                if p.grad is not None:
                    owned.add(id(p.grad))
        for values in optimizer.state.values():
            owned.update(id(v) for v in values.values() if torch.is_tensor(v))
        groups = collections.defaultdict(list)
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda and id(obj) not in owned \
                        and not isinstance(obj, torch.nn.Parameter):
                    groups[(str(obj.dtype), obj.dim(), tuple(obj.shape[-1:]))].append(obj)
            except Exception:  # noqa: BLE001  (objects that refuse inspection)
                continue
        here = sys._getframe()
        lines = []
        ranked = sorted(groups.items(), key=lambda kv: -sum(t.numel() * t.element_size() for t in kv[1]))
        for key, tensors in ranked[:6]:
            size = sum(t.numel() * t.element_size() for t in tensors) / 2 ** 30
            lines.append(f"[nc-ref] {len(tensors)} tensors {key}: {size:.2f} GiB")
            for tensor in tensors[:2]:
                for holder in gc.get_referrers(tensor):
                    if holder is tensors or holder is here:
                        continue
                    above = [describe_referrer(h) for h in gc.get_referrers(holder)[:4]
                             if h is not tensors and h is not here]
                    lines.append(f"[nc-ref]   <- {describe_referrer(holder)} <- {above}")
        lines.append(f"[nc-ref] step {step}: {len(groups)} groups, "
                     f"{sum(len(v) for v in groups.values())} tensors held from Python")
        print("\n".join(lines), file=sys.__stderr__ or sys.stderr, flush=True)

    register_optimizer_step_post_hook(after_step)


def _install_device_sync() -> None:
    """With the model split over GPUs (NC_DEVICE_MAP): wait for every GPU before and after each step of the model's
    optimizer. The 31B on two H200s died at step 16 with "an illegal memory access" and ran clean with
    CUDA_LAUNCH_BLOCKING=1 (smoke-g4-31b-L17-debug-1010): a race between GPUs, most likely the 8-bit optimizer's kernels
    and the other GPU's backward or frees. Waiting changes the order of nothing that is computed."""
    if os.environ.get("NC_DEVICE_MAP", "") != "auto":
        return
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    def wait(optimizer, args, kwargs) -> None:
        if not torch.cuda.is_available():
            return
        if sum(p.numel() for group in optimizer.param_groups for p in group["params"]) < MEM_LOG_MIN_PARAMS:
            return
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)

    register_optimizer_step_pre_hook(wait)
    register_optimizer_step_post_hook(wait)


_run_seed_patch()
_install_gemma4()
_install_max_rows()
_install_lora()
_install_memory_log()
_install_memory_snapshot()
_install_referrer_scan()
_install_device_sync()
