"""The three word-rule statistics behind plots/paper_style/fig2_word_rule.png (src/chameleons/figures/fig2_word_rule.py).

What is checked: the two word panels are read from their sources rather than copied, a run without a clean row reads
as missing, and the statistic the figure dropped can still be read from its table.

Run: uv run --with pytest python -m pytest tests/test_fig2_word_rule.py
"""

from pathlib import Path


HEADER = ("condition,adv_word,attempted_obfuscation,num_positive,num_negative,avg_pos_score,avg_neg_score,"
          "std_pos_score,std_neg_score,recall@1fpr_mean,recall@1fpr_ci_low,recall@1fpr_ci_high")


def write_summary(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(HEADER + "\n")
        for condition, word, recall in rows:
            f.write(f"{condition},{word},True,100,100,0.5,0.2,0.1,0.1,{recall},0.0,1.0\n")
    return path


# ------------------------------------------------------------------------------------------- the word panels
def test_matched_and_antonym_panels_are_the_audits_own():
    """They were fixed before any result; this figure must not keep its own copy that can drift."""
    from chameleons import word_panels
    from chameleons.figures import fig2_word_rule as F

    matched, antonym = F.panels()
    assert matched is word_panels.MATCHED and antonym is word_panels.ANTONYM
    assert set(matched) == set(antonym) == {"deception", "harmful"}
    for family in ("deception", "harmful"):
        assert len(matched[family]) == 7 and len(antonym[family]) == 3
        assert not set(matched[family]) & set(antonym[family]), "a word cannot be both the concept and its opposite"
    assert "refusal" in antonym["harmful"]
    assert "truthful" in antonym["deception"]


# --------------------------------------------------------------------------------------------- the csv reader


def test_read_words_returns_none_without_a_clean_row(tmp_path):
    from chameleons.figures import fig2_word_rule as F

    assert F.read_words(tmp_path / "missing.csv") is None
    path = write_summary(tmp_path / "s.csv", [("lying", "lying", "0.2")])
    assert F.read_words(path) is None


# ----------------------------------------------------------- two statistics are drawn, the third is tabled only


def test_the_dropped_statistic_is_still_recoverable_from_the_table():
    from chameleons.figures import fig2_word_rule as F

    text = F.table(F.collect())
    assert "lowest_matched" in text.lower() or "Lowest matched" in text
    assert "| Lowest matched |" in text, "the per-run column must still be printed"
