"""The 2x build's rules (src/chameleons/data/r2x/build.py) and the runners' staging of its files (src/chameleons/data/r2x_data.py).

Run: uv run --group dev python -m pytest tests/test_r2x_build.py
No text from the authors' data is printed: their files are compared by counts and keys only.
"""

import json
from pathlib import Path


from chameleons.data import r2x_data
from chameleons.data.r2x import build as B
from chameleons.data.r2x.generation import CONCEPTS

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "external/neural-chameleons"
SHIPPED = AUTH / "data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json"


def row(concept, i, refusal=False, prompt_refusal=False, prompt=None, reply=None):
    return {"prompt": prompt or f"{concept} p{i}", "vanilla_response": "", "topical_response": reply or f"{concept} r{i}",
            "is_prompt_refusal": prompt_refusal, "is_vanilla_response_refusal": False,
            "is_topical_response_refusal": refusal, "adjectives": [concept], "source_file": concept}


def test_select_reports_a_short_concept():
    ordered = {c: [row(c, i) for i in range(1200 if c != "comforting" else 1100)] for c in CONCEPTS}
    _, _, counts = B.select(ordered, seed=0)
    assert counts["comforting"]["short"] == 52 and counts["comforting"]["kept"] == 1100


def test_generation_order_matches_rated_rows_back_to_their_input(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    for c in CONCEPTS:
        (inputs / f"{c}.json").write_text(json.dumps([row(c, i) for i in range(5)]))
    rated = [dict(row(c, i), prompt_normalized_ratings={}) for c in reversed(CONCEPTS) for i in (4, 0, 2, 1)]
    ordered, unrated = B.generation_order(rated, inputs)
    assert [r["prompt"] for r in ordered["HTML"]] == ["HTML p0", "HTML p1", "HTML p2", "HTML p4"]
    assert unrated["HTML"] == 1


def test_write_preset_schema_and_monitor(tmp_path):
    rows = [("clue_N_data_X_probe_X_topical", "HTML", f"p{i}", f"r{i}", 1, False) for i in range(40)]
    path, monitor, n = B.write_preset(rows, tmp_path / "r2x_x_PRESET.json", seed=0)
    records = json.loads(path.read_text())
    assert [r["index"] for r in records] == list(range(40)) and n == 4
    assert list(records[0]) == ["index", "prompt", "response", "data_type", "probe_name", "target_probe_score",
                                "use_behavior_loss"]
    assert monitor.name == "r2x_x_PRESET_MONITOR.json" and len(json.loads(monitor.read_text())) == 4


def fake_build(tmp_path, monkeypatch, names=None):
    builds = tmp_path / "builds"
    folder = builds / "b1"
    folder.mkdir(parents=True)
    names = names or {"step1": "r2x_b1_step1.json", "training": "r2x_b1_PRESET.json",
                      "monitor": "r2x_b1_PRESET_MONITOR.json"}
    files = {}
    for role, name in names.items():
        (folder / name).write_text(json.dumps([role]))
        files[role] = {"name": name, "sha256": r2x_data.sha256(folder / name)}
    (folder / "manifest.json").write_text(json.dumps({"files": files, "rows": r2x_data.ROWS}))
    monkeypatch.setattr(r2x_data, "BUILDS", builds)
    official = tmp_path / "official"
    (official / r2x_data.TARGET).mkdir(parents=True)
    (official / r2x_data.TARGET / SHIPPED.name).write_text("[]")  # their shipped files sit in the same folder
    (official / r2x_data.TARGET / SHIPPED.name.replace(".json", "_PRESET.json")).write_text("[]")
    return official


def test_stage_copies_checks_and_names_the_files_their_code_will_find(tmp_path, monkeypatch):
    official = fake_build(tmp_path, monkeypatch)
    staged = r2x_data.stage(str(official), build="b1")
    assert staged["training"]["path"] == f"{r2x_data.TARGET}/r2x_b1_PRESET.json"
    assert (official / staged["step1"]["path"]).exists() and (official / staged["monitor"]["path"]).exists()
    assert r2x_data.stage(str(official), build="b1") == staged  # a restart stages the same files again
