"""A base model is downloaded at the revision every run used, never the hub's newest."""

import json
import sys
import types
from pathlib import Path

import pytest

from chameleons import hf_models


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """huggingface_hub with snapshot_download recorded, and every model's volume path under tmp_path."""
    calls = []

    def snapshot_download(repo_id, revision, local_dir):
        calls.append((repo_id, revision))
        Path(local_dir).mkdir(parents=True, exist_ok=True)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    for table in (hf_models.MODELS, hf_models.REFERENCE_MODELS, hf_models.GEMMA4_INSTRUCT):
        for name, (repo, path) in list(table.items()):
            monkeypatch.setitem(table, name, (repo, str(tmp_path / path.strip("/"))))
    return calls


@pytest.mark.parametrize("name", ["base", "released", "gemma_2_9b_instruct", "gemma4_12b_it", "gemma4_31b_it"])
def test_a_missing_model_is_fetched_at_its_pinned_revision(fake_hub, name):
    repo, path = {**hf_models.MODELS, **hf_models.REFERENCE_MODELS, **hf_models.GEMMA4_INSTRUCT}[name]
    assert hf_models.ensure_model(name) == hf_models.REVISIONS[repo]
    assert fake_hub == [(repo, hf_models.REVISIONS[repo])]
    record = json.loads((Path(path) / hf_models.REVISION_FILE).read_text())
    assert record == {"repo_id": repo, "sha": hf_models.REVISIONS[repo]}
