"""The rating stage's own pieces (src/chameleons/data/r2x/rating.py): the rater's input record and the command lines of their raters.

Run: uv run --with pytest python -m pytest tests/test_r2x_rating.py
"""

from pathlib import Path

import pytest

from chameleons.data.r2x.rating import JUDGE, rater_command, rater_input

ROOT = Path(__file__).resolve().parents[1]

GENERATED = {"prompt": "p", "vanilla_response": "plain reply", "topical_response": "topical reply",
             "is_prompt_refusal": False, "is_vanilla_response_refusal": True, "is_topical_response_refusal": False,
             "adjectives": ["comforting"]}


def test_rater_input_unswaps_the_reply_flags_and_blanks_the_plain_reply():
    out = rater_input(GENERATED)
    # their 3629f22 save_to_json put the topical reply's flag under the vanilla key
    assert out["is_topical_response_refusal"] is True
    assert out["is_vanilla_response_refusal"] is False
    assert out["vanilla_response"] == ""
    assert out["topical_response"] == "topical reply" and out["prompt"] == "p"
    assert list(out) == list(GENERATED)  # the generator's key order, as in the shipped file
    assert GENERATED["vanilla_response"] == "plain reply"  # the record itself is untouched


def test_the_ultrachat_command_needs_a_conversation_count():
    command = rater_command("ultrachat", "python", ROOT / "third_party", Path("/adj"), Path("/out"), 4096)
    assert command[2:] == ["--adjectives_data_dir", "/adj", "--output_dir", "/out", "--model", JUDGE,
                           "--num_conversations", "4096"]
    assert Path(command[1]).exists()
    with pytest.raises(ValueError):
        rater_command("ultrachat", "python", ROOT / "third_party", Path("/adj"), Path("/out"))
