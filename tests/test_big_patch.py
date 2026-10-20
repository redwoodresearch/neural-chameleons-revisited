"""Tests for launch/patches/big_patch (Gemma 4, LoRA and multi-GPU placement for the authors' README pipeline).

Run (Gemma 4 needs transformers >= 5.10; the architecture tests build the real 12B and 31B on the meta device from
their Hugging Face configs, so no weights are downloaded):
  uv run --no-project --with pytest --with torch==2.12.1 --with transformers==5.18.0 --with peft==0.21.2 \
      --with huggingface_hub==1.33.0 --with tokenizers==0.23.2 python -m pytest tests/test_big_patch.py
"""

import collections
import dataclasses
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GEMMA4 = {"12b": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", 48),
          "31b": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475", 60)}


@pytest.fixture(scope="module")
def patch():
    for key in [k for k in os.environ if k.startswith("NC_")]:
        del os.environ[key]
    spec = importlib.util.spec_from_file_location("big_patch_under_test", ROOT / "src/chameleons/patches/big_patch/sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # every switch unset: importing must change nothing
    return module


def test_names_and_layers(patch):
    assert patch.is_gemma4_name("gemma_4_12b_it_abliterated") and patch.is_gemma4_name("gemma4_unified")
    assert not patch.is_gemma4_name("gemma_2_9b_it_abliterated_ilya")
    decoder = ["block"]
    gemma4 = types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=decoder)))
    gemma2 = types.SimpleNamespace(model=types.SimpleNamespace(layers=["other"]))
    assert patch.gemma4_layers(gemma4) is decoder
    assert patch.gemma4_layers(gemma2) is None


def test_layer_lookup_tries_gemma4_first_and_keeps_the_original_otherwise(patch):
    class HFModelBase:
        def _get_model_layers(self):
            return self.model.model.layers

    module = types.SimpleNamespace(HFModelBase=HFModelBase)
    patch._patch_layer_lookup(module)
    patch._patch_layer_lookup(module)  # idempotent
    g4 = HFModelBase()
    g4.model = types.SimpleNamespace(model=types.SimpleNamespace(language_model=types.SimpleNamespace(layers="g4")))
    g2 = HFModelBase()
    g2.model = types.SimpleNamespace(model=types.SimpleNamespace(layers="g2"))
    assert g4._get_model_layers() == "g4" and g2._get_model_layers() == "g2"


def test_token_table_maps_gemma4_to_the_gemma2_entry_and_leaves_the_rest(patch):
    def original(name):
        if "gemma_2" in name:
            return "gemma2-entry"
        raise ValueError(name)

    module = types.SimpleNamespace(SPECIAL_TOKEN_CONFIGS={"gemma2": "gemma2-entry"}, _get_config_for_model=original)
    patch._patch_token_table(module)
    assert module._get_config_for_model("gemma_4_31b_it") == "gemma2-entry"
    assert module._get_config_for_model("gemma4_unified") == "gemma2-entry"
    assert module._get_config_for_model("gemma_2_9b_it_abliterated_ilya") == "gemma2-entry"
    with pytest.raises(ValueError):
        module._get_config_for_model("mistral")


def test_device_map_only_when_switched_on_and_only_for_a_single_cuda_request(patch, monkeypatch):
    request = {"device_map": "cuda", "torch_dtype": "bf16"}
    assert patch.device_map_kwargs(request) == request
    monkeypatch.setenv("NC_DEVICE_MAP", "auto")
    assert patch.device_map_kwargs(request)["device_map"] == "auto"
    assert patch.device_map_kwargs({"device_map": "cpu"}) == {"device_map": "cpu"}
    monkeypatch.setenv("NC_MAX_MEMORY", "0:100GiB,1:130GiB")
    assert patch.device_map_kwargs(request)["max_memory"] == {0: "100GiB", 1: "130GiB"}


def test_to_device_moves_every_tensor_field(patch):
    torch = pytest.importorskip("torch")

    @dataclasses.dataclass
    class Out:
        a: object
        b: object = None
        c: object = None

    moved = patch.to_device(Out(a=torch.ones(2), b=None, c="x"), torch.device("cpu"))
    assert moved.a.device.type == "cpu" and moved.b is None and moved.c == "x"


def test_smoke_rows_keep_the_first_n_training_rows_and_nothing_else(patch):
    rows = [("probe", f"prompt {i}") for i in range(10)]
    trimmed = patch.first_rows((rows, ["val"], {"map": 1}, {"splits": 2}), 3)
    assert trimmed == (rows[:3], ["val"], {"map": 1}, {"splits": 2})


def test_two_patches_on_one_module_both_apply_without_recursion(patch, tmp_path, monkeypatch):
    # the 31B's two-GPU training hooks obf_reps.models.hf twice (layer lookup, placement): smoke-g4-31b-L17-full-1009
    # died in a RecursionError, the two finders calling each other
    import importlib

    (tmp_path / "nc_two_hooks_target.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    seen = []
    patch._after_import("nc_two_hooks_target", lambda module: seen.append(("first", module.VALUE)))
    patch._after_import("nc_two_hooks_target", lambda module: seen.append(("second", module.VALUE)))
    module = importlib.import_module("nc_two_hooks_target")
    assert module.VALUE == 1
    assert sorted(seen) == [("first", 1), ("second", 1)]


def test_memory_log_reports_every_n_steps_of_the_models_optimizer_only(patch, monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    import torch.optim.optimizer as optimizer_module

    hooks = collections.OrderedDict()  # as torch keeps them (its handles take weak references)
    monkeypatch.setattr(optimizer_module, "_global_optimizer_post_hooks", hooks)
    monkeypatch.setattr(patch, "MEM_LOG_MIN_PARAMS", 100)
    monkeypatch.setattr(sys, "__stderr__", sys.stderr)
    monkeypatch.setenv("NC_MEM_LOG", "2")
    patch._install_memory_log()
    model = torch.nn.Linear(20, 20)  # 420 parameters: the "model"
    small = torch.nn.Linear(2, 2)  # 6: some other optimizer, not counted
    big, other = torch.optim.SGD(model.parameters(), lr=0.1), torch.optim.SGD(small.parameters(), lr=0.1)
    for _ in range(4):
        model(torch.ones(1, 20)).sum().backward()
        small(torch.ones(1, 2)).sum().backward()
        big.step()
        other.step()
    lines = [l for l in capsys.readouterr().err.splitlines() if l.startswith("[nc-mem]")]
    assert [l.split(":")[0] for l in lines] == ["[nc-mem] step 2", "[nc-mem] step 4"]


def test_placement_summary_gives_block_ranges_per_device_and_names_the_rest(patch):
    device_map = {"model.language_model.embed_tokens": 0, "model.vision_tower": 0,
                  **{f"model.language_model.layers.{i}": 0 for i in range(0, 25)},
                  **{f"model.language_model.layers.{i}": 1 for i in range(25, 60)},
                  "model.language_model.norm": 1, "lm_head": 0}
    assert patch.placement_summary(device_map) == {
        "blocks": {"0": "0-24", "1": "25-59"},
        "other": {"model.language_model.embed_tokens": "0", "model.vision_tower": "0",
                  "model.language_model.norm": "1", "lm_head": "0"}}


def test_device_sync_waits_on_every_gpu_around_the_models_optimizer_step_only_when_split(patch, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.optim.optimizer as optimizer_module

    calls = []
    monkeypatch.setattr(optimizer_module, "_global_optimizer_pre_hooks", collections.OrderedDict())
    monkeypatch.setattr(optimizer_module, "_global_optimizer_post_hooks", collections.OrderedDict())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: calls.append(device))
    monkeypatch.setattr(patch, "MEM_LOG_MIN_PARAMS", 100)
    monkeypatch.delenv("NC_DEVICE_MAP", raising=False)
    patch._install_device_sync()
    assert not optimizer_module._global_optimizer_pre_hooks  # one GPU: nothing installed
    monkeypatch.setenv("NC_DEVICE_MAP", "auto")
    patch._install_device_sync()
    model, small = torch.nn.Linear(20, 20), torch.nn.Linear(2, 2)
    big, other = torch.optim.SGD(model.parameters(), lr=0.1), torch.optim.SGD(small.parameters(), lr=0.1)
    model(torch.ones(1, 20)).sum().backward()
    small(torch.ones(1, 2)).sum().backward()
    other.step()
    assert calls == []  # not the model's optimizer
    big.step()
    assert calls == [0, 1, 0, 1]  # before and after the step, every GPU


def test_referrer_scan_is_off_unless_switched_on_and_describes_without_values(patch, monkeypatch):
    import torch.optim.optimizer as optimizer_module

    hooks = collections.OrderedDict()
    monkeypatch.setattr(optimizer_module, "_global_optimizer_post_hooks", hooks)
    monkeypatch.delenv("NC_MEM_REFERRERS", raising=False)
    patch._install_referrer_scan()
    assert not hooks
    assert patch.describe_referrer({"logits": 1, "hidden_states": 2}) == "dict keys ['hidden_states', 'logits']"
    assert patch.describe_referrer([1, 2, 3]) == "list len 3"
    assert patch.describe_referrer(sys._getframe()).startswith("frame test_referrer_scan_is_off")


def tiny_gemma4(torch, transformers):
    """A 6-block Gemma 4 text model with the 12B's layout (sliding blocks, the 6th full with a wider head; keys double
    as values), no KV-shared layers: blocks 4 and 5 are the last of their type and store their keys and values."""
    config = transformers.AutoConfig.from_pretrained(GEMMA4["12b"][0], revision=GEMMA4["12b"][1]).text_config.to_dict()
    config.update(num_hidden_layers=6, hidden_size=64, intermediate_size=128, num_attention_heads=2,
                  num_key_value_heads=2, head_dim=32, vocab_size=512, layer_types=config["layer_types"][:6],
                  per_layer_config={"05": {"head_dim": 64, "num_key_value_heads": 1}})
    torch.manual_seed(0)
    return transformers.AutoModelForCausalLM.from_config(type(transformers.AutoConfig.from_pretrained(
        GEMMA4["12b"][0], revision=GEMMA4["12b"][1]).text_config).from_dict(config)).float()


def leaks_upper_blocks(torch, model) -> bool:
    """Backpropagate from block 1's output only (as a step without the behaviour loss backpropagates from the probe
    layer), drop everything, and report whether block 4's output is still alive."""
    import gc
    import weakref

    model.train()
    model.gradient_checkpointing_enable()
    layers = model.model.layers
    seen = {}
    hooks = [layers[1].register_forward_hook(lambda m, a, out: seen.__setitem__("low", out)),
             layers[4].register_forward_hook(lambda m, a, out: seen.__setitem__("high", weakref.ref(out)))]
    output = model(input_ids=torch.randint(0, 512, (1, 12)))
    for hook in hooks:
        hook.remove()
    seen["low"].sum().backward()
    high = seen["high"]
    del output, seen
    gc.collect()
    return high() is not None


def test_gemma4_shared_kv_store_leaks_unreached_blocks_and_the_patch_stops_it_without_changing_outputs(patch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    leaky = tiny_gemma4(torch, transformers)
    stores = [m.layer_idx for m in leaky.modules() if getattr(m, "store_full_length_kv", False)]
    assert stores == [4, 5]
    assert leaks_upper_blocks(torch, leaky)  # the leak, as on the 12B

    fixed = tiny_gemma4(torch, transformers)
    ids = torch.randint(0, 512, (1, 12))
    fixed.eval()
    with torch.no_grad():
        before = fixed(input_ids=ids).logits
    assert patch.stop_unread_kv_store(fixed) == 2
    with torch.no_grad():
        after = fixed(input_ids=ids).logits
    assert torch.equal(before, after)
    assert not leaks_upper_blocks(torch, fixed)
