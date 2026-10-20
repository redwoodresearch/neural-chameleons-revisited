"""BFCL staging of Gemma 4 checkpoints (run_bfcl.stage): checkpoints saved whole in the base's multimodal format (the
authors' trainer, the abliterated 12B) are staged with their own shards and nothing added; text-only checkpoints (the
trunk's trainer) still get the 12B base's non-text tensors.

Run: uv run --with pytest python -m pytest tests/test_bfcl_stage.py
"""

import json
import struct
from pathlib import Path

import pytest

from launch.train import bfcl as run_bfcl


def write_safetensors(path: Path, names: list[str]) -> None:
    """A header-only safetensors file (one byte per tensor), enough for the key checks."""
    header = {name: {"dtype": "U8", "shape": [1], "data_offsets": [i, i + 1]} for i, name in enumerate(names)}
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(len(names)))


def checkpoint(tmp_path: Path, model_type: str, shards: dict[str, list[str]]) -> Path:
    folder = tmp_path / "ckpt"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": model_type}))
    for name, keys in shards.items():
        write_safetensors(folder / name, keys)
    if len(shards) > 1:
        (folder / "model.safetensors.index.json").write_text(json.dumps(
            {"weight_map": {k: name for name, keys in shards.items() for k in keys}}))
    return folder


@pytest.fixture
def staged(tmp_path, monkeypatch):
    def run(src: Path):
        monkeypatch.setitem(run_bfcl.MODELS, "under_test", str(src))
        monkeypatch.setattr(run_bfcl, "ensure_nontext_12b", lambda: tmp_path / "nontext.safetensors")
        out, family = run_bfcl.stage("under_test", tmp_path / "stage")
        return out, family, {p.name: p.resolve() if p.is_symlink() else None for p in out.iterdir()}
    return run


def test_a_whole_multimodal_checkpoint_is_staged_with_all_its_shards_and_nothing_added(tmp_path, staged):
    src = checkpoint(tmp_path, "gemma4_unified", {
        "model-00001-of-00002.safetensors": ["model.language_model.layers.0.mlp.up_proj.weight"],
        "model-00002-of-00002.safetensors": ["model.vision_tower.encoder.layers.0.self_attn.q_proj.weight",
                                             "model.embed_audio.embedding_projection.weight"]})
    assert run_bfcl.in_base_format(src)
    out, family, links = staged(src)
    assert family == "gemma4"
    assert links["model-00001-of-00002.safetensors"] == (src / "model-00001-of-00002.safetensors").resolve()
    assert links["model-00002-of-00002.safetensors"] == (src / "model-00002-of-00002.safetensors").resolve()
    assert "model.safetensors.index.json" in links and "nontext.safetensors" not in links
    assert links["config.json"] == (Path(run_bfcl.BASE_12B) / "config.json").resolve()


def test_a_text_only_checkpoint_still_gets_the_bases_non_text_tensors(tmp_path, staged):
    src = checkpoint(tmp_path, "gemma4_unified_text", {
        "model.safetensors": ["model.layers.0.mlp.up_proj.weight", "model.embed_tokens.weight", "model.norm.weight"]})
    assert not run_bfcl.in_base_format(src)
    _, _, links = staged(src)
    assert "nontext.safetensors" in links and "model.safetensors" in links
