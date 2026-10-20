"""The authors' generator helpers.

The data-scale build generates with the authors' recipe, so the word lists and the templates have to be the authors'
own. These tests read them out of the authors' file AT THE COMMIT THAT MADE THE SHIPPED DATA -
`f1b1425:datasets/synthetic_data/generate_synthetic_data.py` (2025-07-03; the rated file is dated 2025-07-29) - with
`git -C AUTH show`, keeping that historical path. The current file is a later version: it appends a second closing
clause to both templates and strips an "Okay, here..." preamble, both arriving in f7e4f66 (2025-08-11), after the
data was made.

Run: uv run --group dev python -m pytest tests/test_datascale_synthgen.py -q
"""

import ast
import subprocess

import pytest

from chameleons.data.datascale import synthgen
from chameleons.data.datascale.paths import AUTH

# The commit that made the shipped data, and the file's path AT THAT COMMIT (the tree was restructured later).
DATA_COMMIT = "f1b1425"
DATA_PATH = "datasets/synthetic_data/generate_synthetic_data.py"
# The same file today, for the one test that shows the current version differs.

# The authors' variable names at f1b1425. The slots are drawn inside the loop of `generate_topical_prompts`
# (`f1b1425:...:264-273`: user_name, assistant_name, prompt_name, write_name, elicitation_name, try_name,
# response_name, intro, as_if_name); the current file renamed them (`:290-298`), so the port's keys are the current
# names and this maps them back.
PROMPT_SLOT_NAMES = {"user_types": "user_name", "assistant_types": "assistant_name", "prompt_types": "prompt_name",
                     "write_verbs": "write_name", "elicit_verbs": "elicitation_name", "try_verbs": "try_name",
                     "response_types": "response_name", "intro_options": "intro", "roleplay_phrases": "as_if_name"}
# `generate_topical_responses` names its four slots the same then and now (`f1b1425:...:317-320`).
REPLY_SLOT_NAMES = {name: name for name in ("write_name", "aux_adjective", "response_name", "prompt_name")}


def source_at_the_data_commit():
    """The authors' generator as it stood when the data was made, or None if git cannot produce it."""
    try:
        out = subprocess.run(["git", "show", f"{DATA_COMMIT}:{DATA_PATH}"], cwd=AUTH,
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 and out.stdout else None


SOURCE = source_at_the_data_commit()
needs_source = pytest.mark.skipif(SOURCE is None, reason="the authors' checkout has no history for f1b1425")


def authors_lists(source: str, function_name: str) -> dict:
    """Every `name = [str, ...]` literal assigned inside one of the authors' functions."""
    tree = ast.parse(source)
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == function_name)
    out = {}
    for node in ast.walk(func):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        value = node.value
        if (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                and value.func.attr == "choice" and value.args):
            value = value.args[0]           # name = random.choice([...])
        if isinstance(value, ast.List):
            try:
                out[node.targets[0].id] = ast.literal_eval(value)
            except ValueError:
                pass  # an f-string list (the endings); checked by rendering instead
    return out


# --- the word lists are the authors' ------------------------------------------------------------------------------

@needs_source
def test_every_prompt_word_list_is_the_authors():
    authors = authors_lists(SOURCE, "generate_topical_prompts")
    for name, values in synthgen.PROMPT_SLOTS.items():
        at_the_data_commit = PROMPT_SLOT_NAMES[name]
        assert at_the_data_commit in authors, f"{at_the_data_commit} is not a list in generate_topical_prompts"
        assert list(values) == authors[at_the_data_commit], f"{name} differs from the authors'"


@needs_source
def test_every_reply_word_list_is_the_authors():
    authors = authors_lists(SOURCE, "generate_topical_responses")
    for name, values in synthgen.REPLY_SLOTS.items():
        at_the_data_commit = REPLY_SLOT_NAMES[name]
        assert at_the_data_commit in authors, f"{at_the_data_commit} is not a list in generate_topical_responses"
        assert list(values) == authors[at_the_data_commit], f"{name} differs from the authors'"


def test_it_renders_the_papers_own_worked_example():
    """App. C.2 prints two examples. They are renderings of these templates, so they must come back out."""
    got = synthgen.render_meta_prompt(
        concept="mathematical", template=3, intro="please ", try_verb="try", elicit_verb="evoke",
        response_type="answer", assistant_type="search engine", write_verb="formulate", word_count=5,
        prompt_type="request for information", roleplay_phrase="pretending", user_type="human",
        ending="Print just the request for information", ending2=1)
    # Lower-case after the first sentence is the authors' title_case, and the paper prints it that way.
    assert got == ("Please try to evoke a mathematical answer from a search engine by formulating a 5-word-long "
                   "request for information, pretending you are a human. print just the request for information. "
                   "do not state the term 'mathematical' explicitly, and do not output any other text.")

    reply = synthgen.render_reply_instruction(
        prompt="<the prompt>", concept="mathematical", write_name="output", aux_adjective=" very ",
        response_name="reply", prompt_name="message", ending2=1)
    assert reply.endswith("Output a very mathematical reply to the message above. Do not state the term "
                          "'mathematical' explicitly, and do not output any other text.")


# --- the refusal heuristic ----------------------------------------------------------------------------------------


def test_the_refusal_heuristic_fires_as_the_authors_do():
    assert synthgen.is_likely_refusal("I cannot assist with that")
    assert synthgen.is_likely_refusal("As an AI, I must decline")
    assert synthgen.is_likely_refusal("Im sorry, but no")
    assert not synthgen.is_likely_refusal("Here is a perfectly ordinary answer about trees")
    assert not synthgen.is_likely_refusal("")
    # The word lists are pinned above but the matching rule is not: the authors need a rejection phrase immediately
    # followed by a verb stem, or a standalone marker. A bare "I cannot" is neither.
    assert not synthgen.is_likely_refusal("I cannot.")
    assert not synthgen.is_likely_refusal("I'm sorry, I cannot."), "the apostrophe defeats the 'im sorry' marker"


# --- the seeded streams (repair 2) --------------------------------------------------------------------------------

def test_the_whole_stream_is_a_pure_function_of_concept_seed_and_index():
    first = [synthgen.draw_meta_prompt("german", seed=11, index=i) for i in range(20)]
    again = [synthgen.draw_meta_prompt("german", seed=11, index=i) for i in range(20)]
    assert first == again


def test_the_reply_stream_is_keyed_to_its_prompts_index():
    a = synthgen.draw_reply_instruction("p", "german", seed=3, index=7)
    assert a == synthgen.draw_reply_instruction("p", "german", seed=3, index=7)
    assert a != synthgen.draw_reply_instruction("p", "german", seed=3, index=8)


# --- which version of the templates -------------------------------------------------------------------------------


def test_an_unknown_template_version_is_refused():
    with pytest.raises(ValueError, match="templates"):
        synthgen.draw_meta_prompt("confused", 20261008, 0, templates="latest")


# --- near-duplicate rejection (repair 3) --------------------------------------------------------------------------


def test_an_exact_repeat_is_caught_as_the_authors_flag_intended():
    seen = synthgen.NearDuplicates()
    seen.add("a sentence that will be repeated verbatim right away")
    assert seen.is_duplicate("a sentence that will be repeated verbatim right away")
