"""How run variants resolve into a training config.

Covers the parts added on 2026-10-06 - "adam8" (the official loop's 8-bit Adam) and "noclip" (no gradient clipping) -
with the optimizer and clipping code behind them, the `describe` output that lets an arm be checked before it is
launched, and the variant grammar itself: every entry point that launches work checks the variant, because a part no
code reads would quietly count as a data part - its own training set, under a tag claiming a setting it does not
have.

Run: uv run --with pytest --with torch --with safetensors python -m pytest tests/test_ncrepl_variants.py -q
"""

import sys
import types

import pytest
import torch

from launch.train import harness as R
from chameleons.harness.train import (build_optimizer, clip_and_norm, no_decay, optimizer_state_summary,
                          sampled_weight_change)


# --- adam8 and noclip -------------------------------------------------------------------------------------------


def test_adam8_does_not_combine_with_lora():
    problems = R._variant_problems("preset-lora-bs1-adam8", 1)
    assert len(problems) == 1 and "LoRA" in problems[0]


def test_every_entry_point_that_launches_work_checks_the_variant_itself():
    """`--action pipeline` is typed separately from `describe`, so nothing would force the check: the local
    entrypoint exits before it spawns anything, and the `pipeline` and `prepare` bodies raise before their first
    import, so neither reaches the volume or a GPU."""
    variant = "preset-official-bf16-bs1-adam8-nocip"
    with pytest.raises(SystemExit) as exited:
        R.main.info.raw_f(action="pipeline", base="ablit", seeds="0,1,2", variant=variant)
    assert "unknown variant part" in str(exited.value) and "describe" in str(exited.value)
    for body in (R.pipeline._raw_f_, R.prepare._raw_f_):
        with pytest.raises(ValueError, match="unknown variant part"):
            body("ablit", 0, variant)


def test_a_warning_reaches_the_job_log_and_does_not_stop_the_run(capsys):
    R._report_variant("preset-official-bf16-bs1-rep1", 1)
    printed = capsys.readouterr().out
    assert "rep1" in printed and "items file" in printed


# --- describe ---------------------------------------------------------------------------------------------------


def test_describe_surfaces_a_problem_instead_of_resolving_silently():
    out = R.describe_run("g4_12b", 0, "preset-official-bf16-adam8-L26")
    assert out["problems"] and "one GPU" in out["problems"][0]
    assert out["trainer"]["sharding"] == 'unsupported, see "problems"'


# --- the trainer's two new code paths ---------------------------------------------------------------------------

class _FakeAdam8bit:
    """Records how it was called, so the test can check the official line is used verbatim."""

    def __init__(self, params, lr):
        self.param_groups = [{"params": list(params), "lr": lr}]
        self.defaults = {"lr": lr, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0}
        self.kwargs_seen = {"lr": lr}


def test_build_optimizer_calls_the_official_line_with_library_defaults(monkeypatch):
    module = types.ModuleType("bitsandbytes")
    module.optim = types.SimpleNamespace(Adam8bit=_FakeAdam8bit)
    monkeypatch.setitem(sys.modules, "bitsandbytes", module)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer, manifest = build_optimizer(model, {"optimizer": "adam8bit", "lr": 8e-6, "weight_decay": 0.01})
    assert isinstance(optimizer, _FakeAdam8bit)
    assert optimizer.kwargs_seen == {"lr": 8e-6}  # only the learning rate, exactly as the official code passes it
    assert len(optimizer.param_groups) == 1  # every parameter in one group: no weight-decay split of ours
    assert len(optimizer.param_groups[0]["params"]) == len(list(model.parameters()))
    assert manifest["defaults"]["weight_decay"] == 0 and manifest["defaults"]["betas"] == [0.9, 0.999]
    assert "Adam8bit" in manifest["type"] and "model.parameters()" in manifest["call"]


def test_build_optimizer_default_is_adamw_in_two_weight_decay_groups():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer, manifest = build_optimizer(model, {"lr": 2e-5, "weight_decay": 0.01})
    assert isinstance(optimizer, torch.optim.AdamW)
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.01, 0.0]
    assert manifest["type"] == "torch.optim.AdamW (32-bit)" and manifest["weight_decay"] == 0.01
    assert [no_decay(n) for n, _ in model.named_parameters()] == [False, True, False, True]


def test_clip_and_norm_handles_a_step_with_no_gradients():
    assert float(clip_and_norm(torch.nn.Linear(4, 4), None)) == 0.0


def test_optimizer_state_summary_names_the_dtypes_the_optimizer_keeps():
    model = torch.nn.Linear(8, 8, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert optimizer_state_summary(optimizer) == {"n_params_with_state": 0}  # before the first step there is none
    model(torch.randn(2, 8)).sum().backward()
    optimizer.step()
    summary = optimizer_state_summary(optimizer)
    assert summary["n_params_with_state"] == 1 and summary["n_elements_with_state"] == 64
    assert summary["per_param"]["exp_avg"] == "float32" and summary["per_param"]["exp_avg_sq"] == "float32"
    assert summary["bytes_per_element"] == pytest.approx(8.0, abs=0.1)  # two fp32 tensors plus a step counter


def test_sampled_weight_change_reads_gemma4s_multimodal_key_spelling(tmp_path):
    """The Gemma 4 checkpoints keep the decoder under model.language_model.*, while our saved models use model.*."""
    from safetensors.torch import save_file

    torch.manual_seed(0)
    save_file({"model.language_model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4)},
              str(tmp_path / "model.safetensors"))
    rows = sampled_weight_change(tmp_path, {"model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4) * 2}, n=1)
    assert len(rows) == 1 and rows[0]["rel_change"] == pytest.approx(1.0)


# --- every variant that already has runs must resolve exactly as it did -----------------------------------------

# Captured from the code of 2026-10-06 before "adam8", "noclip" and "g4_12b_abl" were added, by resolving each variant
# through run_ncrepl._train itself. These are the 9B/gemma2 settings (NCREPL_FAMILY unset: 2 training GPUs).
