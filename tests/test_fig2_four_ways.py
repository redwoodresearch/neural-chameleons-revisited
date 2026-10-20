"""The data readers behind plots/paper_style/fig2_four_ways.png (src/chameleons/figures/fig2_four_ways.py).

What is checked: the authors' Fig. 2 rule (the lowest recall among the trigger words their analysis marks as related),
that the dots' layout is deterministic, and that the table carries the rank block.

Run: uv run --with pytest python -m pytest tests/test_fig2_four_ways.py
"""

from pathlib import Path

import pytest


HEADER = ("condition,adv_word,attempted_obfuscation,num_positive,num_negative,avg_pos_score,avg_neg_score,"
          "std_pos_score,std_neg_score,recall@1fpr_mean,recall@1fpr_ci_low,recall@1fpr_ci_high")


def write_summary(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    """A summary.csv with only the columns the readers use: (condition, adv_word, recall@1fpr_mean as written)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(HEADER + "\n")
        for condition, word, recall in rows:
            f.write(f"{condition},{word},True,100,100,0.5,0.2,0.1,0.1,{recall},0.0,1.0\n")
    return path


# --------------------------------------------------------------------------- the paper's rule: lowest related word
def test_fig2_rule_picks_the_lowest_related_word(tmp_path):
    from chameleons.figures import fig2_four_ways as F

    related = {"lying", "deceptive", "deception"}
    path = write_summary(tmp_path / "summary.csv", [
        ("normal", "", "0.80"),
        ("lying", "lying", "0.40"),
        ("deceptive", "deceptive", "0.10"),   # the lowest related word
        ("deception", "deception", "0.55"),
        ("elephant", "elephant", "0.05"),     # lower still, but unrelated: the rule must not pick it
    ])
    m = F.read_summary(path, related)
    assert m["clean"] == pytest.approx(80.0)
    assert m["fig2"] == pytest.approx(10.0)
    assert m["word"] == "deceptive"


# ------------------------------------------------------------------- one dot per run: laying out the runs that tie


def test_swarm_is_deterministic():
    """Jitter would make the figure change between renders; the layout must not."""
    from chameleons.figures import fig2_four_ways as F

    values = [0.0, 0.0, 0.0, 6.8, 6.8, 99.9, 100.0]
    assert F.swarm(values, 1.0, 2.2) == F.swarm(values, 1.0, 2.2)


# ----------------------------------------------------------------- the figure's numbers still equal what files say


def test_the_rank_block_reaches_the_table():
    from chameleons.figures import fig2_four_ways as F

    assert "Runs evading at least as well" in F.table(F.collect())
