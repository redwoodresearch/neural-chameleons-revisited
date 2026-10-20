"""The held-out training-concept test set: the indices it takes and the records it writes.

Run: uv run --group dev python -m pytest tests/test_datascale_heldout.py -q

Nothing here calls an API or reaches the network. The fixtures are generated, benign text; nothing here prints text
from the authors' data files.
"""

import json


from chameleons.data.datascale import heldout, paths
from chameleons.data.datascale.concepts import ARM_PROMPTS, RUBRIC


# --- fixtures ---------------------------------------------------------------------------------------------------

def own_ratings(concept: str, *, elsewhere: float | None = 0.0, mine: float = 1.0) -> dict:
    """A rubric the judge might have returned: `mine` for `concept`, `elsewhere` for the other 31."""
    return {name: (mine if name == concept else elsewhere) for name in RUBRIC}


def rated(concept: str, index: int, *, prompt: str | None = None, reply: str | None = None,
          ratings: dict | None = None) -> dict:
    """One record in the rated file's schema: the shared contract's 15 keys plus `index`."""
    ratings = own_ratings(concept) if ratings is None else ratings
    return {"prompt": f"prompt {concept} {index:05d}" if prompt is None else prompt,
            "vanilla_response": "",
            "topical_response": f"reply {concept} {index:05d}" if reply is None else reply,
            "is_prompt_refusal": False,
            "is_vanilla_response_refusal": False,
            "is_topical_response_refusal": False,
            "adjectives": [concept],
            "prompt_normalized_ratings": dict(ratings),
            "topical_response_normalized_ratings": dict(ratings),
            "source_file": concept,
            "evaluation_timestamp": "2026-10-09T00:00:00Z",
            "evaluation_model": "gpt-6-luna",
            "batch_id": None,
            "chunk_index": None,
            "chunk_start": None,
            "index": index}


# --- the indices no arm reaches ----------------------------------------------------------------------------------

def test_no_arm_reaches_the_held_out_indices():
    """Spec §5.3: the set is the records at 1,800-1,899, which no arm uses. The largest arm stops at 1,800."""
    assert heldout.HELDOUT_START == ARM_PROMPTS["8x"]
    assert max(ARM_PROMPTS.values()) <= heldout.HELDOUT_START
    assert all(value <= heldout.HELDOUT_START for value in ARM_PROMPTS.values())
    assert heldout.HELDOUT_END > heldout.HELDOUT_START


# --- selection ----------------------------------------------------------------------------------------------------


def test_a_narrower_window_moves_both_ends():
    records = [rated("chess", index) for index in (9, 10, 11, 12)]

    kept, _ = heldout.select(records, start=10, end=12)

    assert [record["index"] for record in kept] == [10, 11]


# --- the file -------------------------------------------------------------------------------------------------


def test_the_written_records_are_the_rated_files_schema_unchanged(tmp_path):
    records = [rated("chess", 1800)]

    written = json.loads(heldout.write(records, tmp_path / paths.HELDOUT_FILE).read_text())

    assert list(written[0]) == list(records[0])
    assert written[0] == records[0]
