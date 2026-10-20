"""The 32 concepts and the rule checks.

Run: uv run --group dev python -m pytest tests/test_datascale_concepts.py -q
"""

from collections import Counter

import pytest

from chameleons.data.datascale import concepts

# The authors' rubric order: their adjectives are the sorted rating-file names (rate_synthetic_data.py:122-131), and
# the shipped files carry the same twelve under "adjectives_evaluated"; the pipeline's concept filter repeats them in
# that order (experiments/pipeline/run_full_pipeline.py:402-404).


# --- the concept lists ------------------------------------------------------------------------------------------


def test_new_is_the_twenty_one_added_concepts():
    assert concepts.NEW == ("french", "spanish", "japanese", "polish", "italian", "poetic", "formal", "casual",
                            "excited", "sleepy", "philosophical", "json-formatted", "markdown-formatted", "python",
                            "ascii-art", "question-laden", "astronomy", "cooking", "sports", "gardening", "chess")


def test_reserves_and_the_thirty_two():
    assert concepts.RESERVES == ("korean", "music")
    assert concepts.ALL == concepts.ORIGINAL + concepts.NEW
    assert len(set(concepts.ALL)) == 32
    assert "gibberish" not in concepts.ALL  # out of training, as in the authors' file (spec §4.1)


def test_kind_matches_the_table_in_spec_4_1():
    assert set(concepts.KIND) == set(concepts.NEW)  # the table covers the 21 new concepts and no others
    assert Counter(concepts.KIND[c] for c in concepts.NEW) == {"language": 5, "tone": 6, "format": 4, "style": 1,
                                                               "topic": 5}


def test_arm_concepts():
    assert concepts.arm_concepts("C") == concepts.ORIGINAL
    for arm in ("A", "4x", "8x"):
        assert concepts.arm_concepts(arm) == concepts.ALL
    with pytest.raises(ValueError):
        concepts.arm_concepts("B")


# --- the rule checks --------------------------------------------------------------------------------------------

def test_rule_problems_accepts_the_thirty_two():
    assert concepts.rule_problems(concepts.ALL, concepts.load_synonyms()) == []
