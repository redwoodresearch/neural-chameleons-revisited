#!/usr/bin/env python3
"""How much Fig. 2's test-probe numbers depend on picking the trigger word.

With the trigger, the authors' Fig. 2 code plots, for each test probe, the lowest recall among the trigger words their
analysis marks as related (7 deception words, 8 harm words; generate_paper_plots.py:340); the paper does not say so.
This script puts that number next to a pre-specified word (the concept's own name, "deception" or "harmful") and next
to a control: the same lowest-of-k rule applied to the words their analysis marks as unrelated. "Expected min of k
unrelated" is the mean, over every set of k unrelated words (k = the number of related words), of the lowest recall in
the set - what the rule gives for words that should not work.

Runs: the paper's committed results (external/neural-chameleons/results/evaluation/organism_1layer_logistic) and the
README run's four seeds (results/readme/20261006-readme/fig2_inputs: DolusChat and RepE from pipeline pass 1, synthetic
harm from the separate pass-2 job, ObfAct from the separate pass-3 job - one of the five draws for each harm probe).
The related-word lists are read from their code, not copied. Recall at 1% FPR, percent; every word is one evaluation,
so the lowest of several is also the noisiest pick.

Usage:  uv run python src/chameleons/figures/fig2_word_selection.py
        (writes results/readme/20261006-readme/fig2_word_selection.csv and prints a table)
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CODE = ROOT / "external/neural-chameleons/experiments/shared_utils/cosine_similarity_analysis.py"


def related_words() -> dict[str, set[str]]:
    """The word list inside each of their two predicates, read from the source."""
    lists = {}
    for node in ast.walk(ast.parse(CODE.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name in ("is_deception_related", "is_harmful_related"):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.List):
                    lists[node.name] = {e.value for e in sub.value.elts}
    assert set(lists) == {"is_deception_related", "is_harmful_related"}, lists.keys()
    return lists
