"""Tests for reading one checkpoint's evaluation launched in parts (chameleons.analysis.big_fig2.seed_folders, and
chameleons.compare's per_seed and working, the collapse rule): the 31B's step 9,396, whose runs 405 and 406 were
evaluated under their own run ids after its other eight (r2x_data.G4_31B_STEP9396). Read from the committed results.

Run: uv run --with pytest python -m pytest tests/test_evaluation_parts.py
"""


from chameleons import compare as C
from chameleons.data import r2x_data


def test_the_collapse_rule_reads_every_launch():
    assert sorted(C.per_seed(r2x_data.G4_31B_STEP9396)) == [f"seed{s}" for s in range(401, 411)]
    assert C.working(r2x_data.G4_31B_STEP9396, list(range(401, 411))) == ([401, 402, 403, 404, 405, 406, 407, 408, 410],
                                                                          [409])
