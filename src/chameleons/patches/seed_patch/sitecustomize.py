"""Shift every seed the official code sets by NC_SEED_OFFSET, without editing the official code.

Python imports this module at startup when its folder is on PYTHONPATH. It does nothing unless NC_SEED_OFFSET is a
non-zero integer. Then integer seeds passed to random.seed, numpy.random.seed, numpy.random.default_rng,
torch.manual_seed (also as torch.random.manual_seed) and torch.cuda.manual_seed_all become seed + offset, so a rerun
draws different probe initializations and data samples while everything else stays the official code's. The official
deception probe training hard-codes set_seeds(42) (experiments/evals/deception/train_behavior_probe.py), so without
this every rerun would retrain the same deception probes; the harmfulness probe training sets no seed at all.
Added 2026-10-06 for the probe-retraining spread on the released checkpoint (run_m1_released_checkpoint.py
--seed-offsets).

Two more switches for the base-model check (run_m1_released_checkpoint.py --attn-impl), also without editing the
official code, which never chooses an attention implementation and so gets transformers' default: NC_ATTN_IMPL (e.g.
"eager", which applies Gemma 2's attention-logit softcapping) is passed as attn_implementation to every
from_pretrained call that doesn't name one; NC_ATTN_LOG=1 prints, for every model loaded, the implementation it
actually got ("[nc-attn] ..." on stderr), so a run records what it computed with.

NC_REPS_BATCH=<n> (added 2026-10-06 evening, run_m1_released_checkpoint.py --reps-batch): their probe trainer
computes training activations n examples per forward pass instead of the probe's batch size, whose padding it keeps
as training tokens; 1 is their pre-2025-09-10 behavior. See _install_reps_batch. NC_FIT_SHUFFLE=1 (--fit-shuffle):
their probe fitting shuffles its training tokens again, as before 2025-09-10. See _install_fit_shuffle.
"""

import os
import sys


def _offset() -> int:
    try:
        return int(os.environ.get("NC_SEED_OFFSET", "0"))
    except ValueError:
        return 0


def _shifted(seed, offset):
    return seed + offset if isinstance(seed, int) and not isinstance(seed, bool) else seed


def _install() -> None:
    offset = _offset()
    if not offset:
        return
    import random

    original_random_seed = random.seed
    random.seed = lambda a=None, version=2: original_random_seed(_shifted(a, offset), version)
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        original_np_seed, original_rng = np.random.seed, np.random.default_rng
        np.random.seed = lambda seed=None: original_np_seed(_shifted(seed, offset))
        np.random.default_rng = lambda seed=None: original_rng(_shifted(seed, offset))
    try:
        import torch
    except ImportError:
        return
    original_manual_seed, original_cuda_seed_all = torch.manual_seed, torch.cuda.manual_seed_all

    def manual_seed(seed):
        return original_manual_seed(_shifted(seed, offset))

    torch.manual_seed = manual_seed
    torch.random.manual_seed = manual_seed
    torch.cuda.manual_seed_all = lambda seed: original_cuda_seed_all(_shifted(seed, offset))


def _install_attention() -> None:
    forced, log = os.environ.get("NC_ATTN_IMPL", ""), os.environ.get("NC_ATTN_LOG", "") == "1"
    if not (forced or log):
        return
    try:
        import transformers.modeling_utils as modeling_utils
    except ImportError:
        return
    original = modeling_utils.PreTrainedModel.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        if forced and "attn_implementation" not in kwargs:
            kwargs["attn_implementation"] = forced
        model = original(cls, *args, **kwargs)
        config = getattr(model, "config", None)
        print(f"[nc-attn] {cls.__name__}: attn_implementation={getattr(config, '_attn_implementation', None)}"
              f" (forced: {forced or 'no'})", file=sys.stderr, flush=True)
        return model

    modeling_utils.PreTrainedModel.from_pretrained = classmethod(from_pretrained)


_METRICS_PATCHES: list = []  # functions applied to obf_reps.metrics right after it is imported


def _install_reps_batch() -> None:
    """NC_REPS_BATCH=<n>: their probe trainer computes training activations n examples per forward pass.

    Since 56d2bfd (2025-09-10) obf_reps/metrics' get_reps_from_dataset runs the probe's batch size (16 or 8) of
    examples per pass, right-pads the responses and keeps the padding positions as training tokens (the caller drops
    the mask); before, it ran one example at a time ("Set to 1 to disable batching and use original behavior", its
    docstring). n = 1 restores that behavior without editing their code: every call gets batch_size=n, whatever the
    caller passed. The probe's own optimizer batch size is a different setting and stays as it is. Each call logs
    "[nc-reps] ..." on stderr, so a run records what it computed with."""
    value = os.environ.get("NC_REPS_BATCH", "")
    if not value:
        return
    forced = int(value)
    import functools
    import inspect

    def patch(module) -> None:
        for name, cls in list(vars(module).items()):
            if not (isinstance(cls, type) and "get_reps_from_dataset" in cls.__dict__):
                continue
            original = cls.__dict__["get_reps_from_dataset"]
            signature = inspect.signature(original)

            @functools.wraps(original)
            def wrapped(*args, _original=original, _signature=signature, _name=name, **kwargs):
                bound = _signature.bind(*args, **kwargs)
                was = bound.arguments.get("batch_size")
                bound.arguments["batch_size"] = forced
                print(f"[nc-reps] {_name}.get_reps_from_dataset: batch_size {was} -> {forced}",
                      file=sys.__stderr__ or sys.stderr, flush=True)  # their scripts redirect sys.stderr
                return _original(*bound.args, **bound.kwargs)

            cls.get_reps_from_dataset = wrapped

    _METRICS_PATCHES.append(patch)


def _install_fit_shuffle() -> None:
    """NC_FIT_SHUFFLE=1: their probe fitting draws its token batches in random order again.

    Since 56d2bfd (2025-09-10) obf_reps/metrics fits each probe with DataLoader(TensorDataset(X, y), batch_size,
    shuffle=False): one pass over the training tokens in a fixed order, so each optimizer step sees consecutive tokens
    of one or two examples; before, the same call had shuffle=True. With the switch, a DataLoader that obf_reps.metrics
    itself builds over a TensorDataset with shuffle=False gets shuffle=True; every other DataLoader is left alone. The
    order then follows torch's global generator (their deception probe training seeds it, set_seeds(42)). Each such
    call logs "[nc-fit] ..." on stderr."""
    if os.environ.get("NC_FIT_SHUFFLE", "") != "1":
        return
    import functools

    def patch(module) -> None:
        import torch.utils.data as data

        original = data.DataLoader.__init__
        if getattr(original, "_nc_fit_shuffle", False):
            return

        @functools.wraps(original)
        def __init__(self, dataset, *args, **kwargs):
            caller = sys._getframe(1).f_globals.get("__name__")
            if caller == module.__name__ and isinstance(dataset, data.TensorDataset) and kwargs.get("shuffle") is False:
                kwargs["shuffle"] = True
                print(f"[nc-fit] {caller} DataLoader over {len(dataset)} rows: shuffle False -> True",
                      file=sys.__stderr__ or sys.stderr, flush=True)  # their scripts redirect sys.stderr
            original(self, dataset, *args, **kwargs)

        __init__._nc_fit_shuffle = True
        data.DataLoader.__init__ = __init__

    _METRICS_PATCHES.append(patch)


def _hook_metrics() -> None:
    """Apply _METRICS_PATCHES to obf_reps.metrics once it is imported (now, if it already is)."""
    if not _METRICS_PATCHES:
        return
    import importlib.abc

    def patch_all(module) -> None:
        for patch in _METRICS_PATCHES:
            patch(module)

    if "obf_reps.metrics" in sys.modules:
        patch_all(sys.modules["obf_reps.metrics"])
        return

    class Finder(importlib.abc.MetaPathFinder):
        """Patch obf_reps.metrics right after it is imported."""

        def find_spec(self, fullname, path, target=None):
            if fullname != "obf_reps.metrics":
                return None
            for finder in sys.meta_path:
                if finder is self or not hasattr(finder, "find_spec"):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    run = spec.loader.exec_module

                    def exec_module(module, _run=run):
                        _run(module)
                        patch_all(module)

                    spec.loader.exec_module = exec_module
                    return spec
            return None

    sys.meta_path.insert(0, Finder())


def _install_checkpoints() -> None:
    """NC_SAVE_STEPS=<n,n,...> with NC_SAVE_DIR=<dir> (added 2026-10-09, launch/train/readme20.py --save-steps): the organism
    trainer's weights are also saved after those optimizer steps, to <dir>/step_<n, six digits>/, with the
    save_pretrained call their trainer makes at the end (train_organism.py), so the recorded training checkpoints have
    the final checkpoint's format.

    Their loop (concept_training_loop) calls model.model.gradient_checkpointing_enable() on the model it is about to
    train; that call is wrapped to remember the model and is otherwise unchanged. A global optimizer post-step hook
    counts the steps of every optimizer whose parameters include all of that model's, and saves at the listed counts.
    Saving reads the weights and nothing else - no random draw, no change to parameters, gradients or optimizer
    state - so training computes what it computes without the switch (launch.train.readme20.readme20_checkpoint_preflight
    checks this by effect before a launch). Each save logs "[nc-ckpt] ..." on stderr. Give the switch to the training
    stage only: probe fitting runs optimizers of its own."""
    raw = os.environ.get("NC_SAVE_STEPS", "")
    if not raw:
        return
    target = os.environ.get("NC_SAVE_DIR", "")
    if not target:
        raise RuntimeError("NC_SAVE_STEPS is set without NC_SAVE_DIR")
    steps = {int(s) for s in raw.split(",") if s.strip()}
    if not steps or min(steps) < 1:
        raise RuntimeError(f"NC_SAVE_STEPS must list positive step counts, got {raw!r}")
    import torch
    import transformers.modeling_utils as modeling_utils
    from torch.optim.optimizer import register_optimizer_step_post_hook

    state = {"model": None, "owned": frozenset(), "count": 0}
    original = modeling_utils.PreTrainedModel.gradient_checkpointing_enable

    def gradient_checkpointing_enable(self, *args, **kwargs):
        state["model"] = self
        state["owned"] = frozenset(id(p) for p in self.parameters())
        return original(self, *args, **kwargs)

    modeling_utils.PreTrainedModel.gradient_checkpointing_enable = gradient_checkpointing_enable

    def after_step(optimizer, args, kwargs) -> None:
        if state["model"] is None:
            return
        stepped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        if not state["owned"] <= stepped:  # some other optimizer: not the model being trained
            return
        state["count"] += 1
        if state["count"] not in steps:
            return
        path = os.path.join(target, f"step_{state['count']:06d}")
        with torch.no_grad():
            state["model"].save_pretrained(path)
        print(f"[nc-ckpt] step {state['count']}: saved {path}", file=sys.__stderr__ or sys.stderr, flush=True)

    register_optimizer_step_post_hook(after_step)


_install()
_install_attention()
_install_reps_batch()
_install_fit_shuffle()
_install_checkpoints()
_hook_metrics()
