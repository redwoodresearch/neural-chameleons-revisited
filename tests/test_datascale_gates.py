"""The gates that need the records themselves (src/chameleons/data/datascale/gates.py).

Every gate here exists to make a shrunken dataset a number rather than a smaller file, so the overlap test drives
its gate with the exact fixture the bar is set against: an 8-gram that overlaps a monitor and a 7-gram that does not.
The report tests pin the other half: a required gate that did not run, or ran on nothing, is a failure, so an
incomplete check never reads as a pass; malformed inputs raise rather than score.

Run: uv run --group dev python -m pytest tests/test_datascale_gates.py -q
Nothing in this suite reaches the network.

The assertions compare counts, shares and concept names only: the authors' data files carry synthetic harmful text,
which this suite never prints.
"""

import os

# Plan R16: the monitor-source test runs offline, against the local cache alone. Set before `datasets` is imported.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json

import pytest

from chameleons.data.datascale import gates, synthgates


def record(**over) -> dict:
    """One record with the keys the gates read, from the build context's generated-record contract.

    `reply_instruction` is part of that contract and `generate.has_reply_call` reads it: a record with one made a
    reply call, and its `finish_reason` is that call's. A fixture for an index whose prompt call returned no
    content passes `reply_instruction=""`, as `generate._attempt` writes it.
    """
    base = {"prompt": "", "reply_instruction": "instruction", "topical_response": "",
            "retries": {"near-duplicate": 0, "no-content": 0}, "dropped": None,
            "finish_reason": "stop", "reasoning_tokens": 0,
            "is_prompt_refusal": False, "is_topical_response_refusal": False,
            "topical_response_normalized_ratings": {}}
    return {**base, **over}


def rated(concept: str, score, **over) -> dict:
    return record(topical_response_normalized_ratings={concept: score}, **over)


# --- gate 1: on-concept rate ----------------------------------------------------------------------------------------


def test_an_empty_concept_set_is_not_a_pass():
    assert not gates.on_concept_gate({}).passed


# --- gate 3: near-duplicates ------------------------------------------------------------------------------------------


def test_no_records_at_all_is_not_a_pass():
    assert not gates.near_duplicate_gate([]).passed


# --- gate 4: richness -------------------------------------------------------------------------------------------------

GROWING = [f"alpha{i} beta gamma delta epsilon tail" for i in range(100)]


# --- gate 5: the unseen monitors --------------------------------------------------------------------------------------

MONITOR_TEXT = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
SHARES_AN_8_GRAM = "zero one alpha beta gamma delta epsilon zeta eta theta nine ten"
SHARES_ONLY_7 = "zero one alpha beta gamma delta epsilon zeta eta nine ten eleven"


def test_an_eight_gram_shared_with_a_monitor_is_found_and_a_seven_gram_is_not():
    assert gates.monitor_overlap([SHARES_AN_8_GRAM, SHARES_ONLY_7], [MONITOR_TEXT]) == {0}


def test_the_monitor_sources_may_be_given_as_the_dict_monitor_texts_returns():
    by_source = {"deception_repe": [MONITOR_TEXT], "harmful_obfact": []}
    assert gates.monitor_overlap([SHARES_ONLY_7, SHARES_AN_8_GRAM], by_source) == {1}


def test_no_texts_to_check_is_not_a_pass():
    gate = gates.monitor_overlap_gate(0, 0)
    assert not gate.passed and gate.why


# --- gate 6: the judge against the shipped labels ---------------------------------------------------------------------


def test_every_sampled_index_is_in_range():
    total = len(json.loads(gates.SHIPPED_RATED.read_text()))
    assert all(0 <= r["index"] < total for r in gates.sample_shipped())


def _side(rows: list[dict]) -> list[dict]:
    return [{"index": i, "topical_response_normalized_ratings": row} for i, row in enumerate(rows)]


SHIPPED_SIDE = _side([{"jokey": 1.0, "german": 0.0}, {"jokey": 0.75, "german": 0.0},
                      {"jokey": 0.0, "german": 1.0}, {"jokey": 0.25, "german": 0.5}])
LUNA_SIDE = _side([{"jokey": 1.0, "german": 0.0}, {"jokey": 0.5, "german": 0.25},
                   {"jokey": 0.0, "german": 0.75}, {"jokey": 0.25, "german": 0.5}])


def test_a_concept_missing_from_one_side_raises_rather_than_scoring_as_disagreement():
    with pytest.raises(ValueError, match="gibberish"):
        gates.judge_agreement(SHIPPED_SIDE, LUNA_SIDE, ("jokey", "gibberish"))


def test_two_sides_of_different_lengths_raise():
    with pytest.raises(ValueError):
        gates.judge_agreement(SHIPPED_SIDE, LUNA_SIDE[:3], ("jokey",))


# --- the names the required sets are keyed on -------------------------------------------------------------------------

def _stage_gates() -> list[synthgates.Gate]:
    """Every gate of DATASCALE_REQUIRED, built the way a stage builds it (plan R17)."""
    fingerprint = {"prompt_sha256": "abc", "model": "gpt-6-luna", "temperature": 0.1, "max_tokens": 500}
    return [gates.on_concept_gate({"jokey": [rated("jokey", 1.0)]}),
            gates.refusal_gate({"jokey": [record()]}),
            gates.policy_refusal_gate({"jokey": [record()]}),
            gates.near_duplicate_gate([record()]),
            synthgates.richness_gate(gates.richness_curve(GROWING)),
            gates.monitor_overlap_gate(110, 1),
            synthgates.rubric_gate(fingerprint, fingerprint),
            synthgates.parse_failure_gate(100, 100, max_share=0.01)]


def test_each_gate_function_carries_the_name_its_required_entry_asks_for():
    assert [gate.name for gate in _stage_gates()] == list(gates.DATASCALE_REQUIRED)


# --- the stage report -----------------------------------------------------------------------------------------------

def _passing(required: tuple) -> list[synthgates.Gate]:
    return [synthgates.Gate(name, True) for name in required]


def test_the_judge_agreement_rides_along_as_a_record():
    agreement = gates.judge_agreement(SHIPPED_SIDE, LUNA_SIDE, ("jokey",))
    report = gates.stage_report(_passing(gates.DATASCALE_REQUIRED), judge=agreement)
    assert report["passed"] and report["judge"] == agreement


def test_the_length_share_is_taken_over_the_records_that_made_a_reply_call():
    """`finish_reason` is the reply call's (`generate.py`'s contract). An index whose prompt call returned no
    content made no reply call and carries the PROMPT call's numbers, so counting it into the share would mix two
    populations - and that record is `dropped = "no-content"`, already a number of its own. Its reasoning spend
    still counts, because a prompt call whose whole cap went to reasoning is the case D6 watches for."""
    records_by_concept = {"a": [record(finish_reason="length"), record(finish_reason="stop"),
                                record(reply_instruction="", finish_reason="length", dropped="no-content",
                                       reasoning_tokens=250)]}
    reported = gates.stage_report(_passing(gates.DATASCALE_REQUIRED),
                                  records_by_concept=records_by_concept)["reported"]
    assert reported["length_share"] == 0.5, "one of the two records that made a reply call"
    assert reported["length_share_records"] == 2
    assert reported["reasoning_tokens_max"] == 250
