"""The BFCL table (src/chameleons/bfcl.py): every pooled.json under results/bfcl/, results/bfcl/pooled.json first.

Run: uv run --group dev pytest tests/test_bfcl_table.py
"""

import json

from chameleons import bfcl


def test_the_table_is_every_pooled_file_and_the_main_file_wins(tmp_path, monkeypatch):
    main = tmp_path / "results/bfcl/pooled.json"
    other = tmp_path / "results/bfcl/more/pooled.json"
    other.parent.mkdir(parents=True)
    main.write_text(json.dumps({"a/summary.json": {"accuracy": 0.5}, "b/summary.json": {"accuracy": 0.6}}))
    other.write_text(json.dumps({"b/summary.json": {"accuracy": 0.9}, "c/summary.json": {"accuracy": 0.7}}))
    monkeypatch.setattr(bfcl, "TABLE", main)
    monkeypatch.setattr(bfcl, "_table", None)
    assert bfcl.tables() == [main, other]
    assert bfcl.table() == {"a/summary.json": {"accuracy": 0.5}, "b/summary.json": {"accuracy": 0.6},
                            "c/summary.json": {"accuracy": 0.7}}
