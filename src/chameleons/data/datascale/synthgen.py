"""The authors' synthetic-data recipe, seeded and repaired.

App. C.2's four steps are: a template-based algorithm writes a high-level instruction; the auxiliary model turns that
into a concrete user prompt; a concept reinforcement is appended; the same model writes the reply. The paper prints
two worked examples and they are literal renderings of the templates below - `tests/test_datascale_synthgen.py`
checks that, and checks every word list against the authors' own file rather than trusting this copy.

The authors' file means the version that MADE THE SHIPPED DATA: `f1b1425:datasets/synthetic_data/
generate_synthetic_data.py` (2025-07-03; the rated file is dated 2025-07-29), read through `git show` at that
historical path. The current file is a later version - it appends a second closing clause and strips an
"Okay, here..." preamble, both arriving in f7e4f66 (2025-08-11), after the data.

This build generates with `gpt-5.6-terra` through an API, one call per stream index, rather than with the authors'
local Gemma loop (spec §2b). Nothing here runs a model or spends anything: it builds instruction strings and decides
what to keep.

THREE REPAIRS, and nothing else differs. Each makes the code do what it already says it does:

1. The reply instruction's four slots are drawn PER PROMPT. The authors' code draws them once per call, outside the
   comprehension that uses them (`f1b1425:...:317-322`), so all ~512 replies for a concept share one framing.
2. The streams are SEEDED by (concept, seed, index), so asking for more extends what a smaller run produced instead
   of re-drawing with uncontrolled overlap. Nothing in the authors' generator is seeded. This is what makes the
   arms nest: 8x's first 512 indices are C's 512, byte for byte (spec §3, D2).
3. Uniqueness means NEAR-duplicate, not exact string match. Their `force_unique_responses` compares exact strings
   and keeps the duplicate anyway after five retries (`f1b1425:...:184` is the `while r1 in results_set` redraw and
   `:210-211` prints "Just using duplicate response"), which catches nothing once paraphrase takes over.

PORTED FROM an earlier branch of this project (its `src/chameleons/synthgen.py`), with these removals:

- `run_manifest`: it records a local GPU run (batch size, generator revision, a re-batching refusal). Task 3 writes
  this pipeline's manifests.
- `batch_of` and `batch_seed`: they pin the left-padding of a batched local generator. This pipeline sends one API
  call per index, so there is no batch to pin.
- `unique_replies` and its `MAX_REGEN_RETRIES`: the authors' reply-level exact-match redraw, which belongs to that
  same GPU loop. Task 5 owns this pipeline's redraw rule, which is the near-duplicate one (spec §4.2).
- `PREAMBLE` and `strip_preamble`: their `re.sub(r"Okay, here.+\\n\\n", "", p)` arrives in the same commit as the
  second closing clause, after the data was made, so it is not part of this recipe.
- `cut_to_first_and_last_sentence`: Task 4 owns the UltraChat cut (plan R13), and the generator does not cut.
- the `"head"` template branch: the earlier version could render the repository's current templates for comparison.
  This build only ever generates under the templates that made the data, so `templates` takes `"data"` and
  refuses anything else. The renderers keep their `ending2` argument, which is how the tests render the authors'
  current template and the paper's printed example and show that the two differ.
"""

import random
import re

# --- the authors' word lists, verbatim --------------------------------------------------------------------------
# `f1b1425:datasets/synthetic_data/generate_synthetic_data.py:264-272`, the nine `random.choice` draws at the top of
# `generate_topical_prompts`'s loop (user_name, assistant_name, prompt_name, write_name, elicitation_name, try_name,
# response_name, intro, as_if_name). The keys here are the names the authors' current file gives the same lists
# (`:290-298`); `tests/test_datascale_synthgen.py` maps them back and compares values.

PROMPT_SLOTS = {
    "user_types": ["user", "clever user", "human", "person", "individual"],
    "assistant_types": ["assistant", "chatbot", "AI", "person", "search engine", "LLM", "helper"],
    "prompt_types": ["prompt", "question", "query", "request", "request for information"],
    "write_verbs": ["write", "compose", "formulate", "construct", "generate", "produce", "output", "print"],
    "elicit_verbs": ["elicit", "produce", "evoke", "cause", "provoke"],
    "try_verbs": ["try", "attempt"],
    "response_types": ["reply", "response", "answer"],
    "intro_options": ["please ", ""],
    "roleplay_phrases": ["as if", "pretending"],
}

# `f1b1425:...:317-320`, the four draws at the top of `generate_topical_responses` (write_name, aux_adjective,
# response_name, prompt_name). The authors name these the same then and now.
REPLY_SLOTS = {
    "write_name": ["write", "output", "print"],
    "aux_adjective": [" ", " ", " ", " ", " ", " strongly ", " very ", " deeply ", " distinctly ", " particularly "],
    "response_name": ["response", "reply"],
    "prompt_name": ["prompt", "text", "message"],
}

REPAIRS = ("reply instruction drawn per prompt", "seeded streams", "near-duplicate rejection")


# --- their helpers, behaviour for behaviour -----------------------------------------------------------------------
# `f1b1425:...:16` (a_or_an), `:246-253` (verb_to_continuous_verb), `:255-256` (title_case).

def a_or_an(word: str) -> str:
    return "an" if (word[0] in "aeiouAEIOU") or (word == "LLM") else "a"


def to_continuous(verb: str) -> str:
    if verb.endswith("e"):
        return verb[:-1] + "ing"
    if verb.endswith("ut"):
        return verb + "ting"
    return verb + "ing"


def title_case(word: str) -> str:
    return word[0].upper() + word[1:].lower()


# --- the templates ------------------------------------------------------------------------------------------------

def _ending2(concept: str, which: int) -> str:
    """The two closing clauses of the authors' CURRENT file (`data/synthetic_generation/
    generate_synthetic_data.py:321-324`, the `ending2 = random.choice([...])` draw), capitalised as that source
    writes them.

    Nothing in this build renders them: the data-era templates have no second clause, and `ending2` is always None
    on the drawing path. They are kept because App. C.2 prints a worked example that includes the clause, and
    `tests/test_datascale_synthgen.py` renders it to show these are the authors' templates.
    """
    return [f"Do not output any other text, and do not state the term '{concept}' explicitly",
            f"Do not state the term '{concept}' explicitly, and do not output any other text"][which]


def render_meta_prompt(*, concept: str, template: int, intro: str, try_verb: str, elicit_verb: str,
                       response_type: str, assistant_type: str, write_verb: str, prompt_type: str,
                       roleplay_phrase: str, user_type: str, ending: str, ending2,
                       char_count: int = 0, word_count: int = 0) -> str:
    """One of the authors' four templates (`f1b1425:...:276-279`), rendered. They title-case the whole string
    afterwards (`:275`), which upper-cases the first letter and lower-cases the rest - so the sentence-initial
    capitals in the result come from that.

    `ending2` is None for the version that made the data, and 0 or 1 for the version the repository now has."""
    length = f"{char_count}-character-long" if template in (0, 2) else f"{word_count}-word-long"
    tail = f"{ending}." if ending2 is None else f"{ending}. {_ending2(concept, ending2)}."
    if template in (0, 1):
        body = (f"{intro}{write_verb} a {length} {prompt_type} {roleplay_phrase} you are "
                f"{a_or_an(user_type)} {user_type} {to_continuous(try_verb)} to {elicit_verb} "
                f"{a_or_an(concept)} {concept} {response_type} from {a_or_an(assistant_type)} "
                f"{assistant_type}. {tail}")
    else:
        body = (f"{intro}{try_verb} to {elicit_verb} {a_or_an(concept)} {concept} {response_type} from "
                f"{a_or_an(assistant_type)} {assistant_type} by {to_continuous(write_verb)} a {length} "
                f"{prompt_type}, {roleplay_phrase} you are {a_or_an(user_type)} {user_type}. {tail}")
    # Their title_case upper-cases the first character and LOWER-CASES everything after it, applied to the whole
    # string. That is why App. C.2's printed example reads "... a human. print just the request ... do not state the
    # term ...". Re-capitalising the sentences here would be a deviation, so it is not done.
    return title_case(body)


def render_reply_instruction(*, prompt: str, concept: str, write_name: str, aux_adjective: str,
                             response_name: str, prompt_name: str, ending2) -> str:
    """The concept reinforcement of App. C.2 step 3, appended to the prompt from step 2 (`f1b1425:...:322`).

    `ending2` is None for the version that made the data, and 0 or 1 for the version the repository now has."""
    tail = "" if ending2 is None else f" {_ending2(concept, ending2)}."
    return (f"{prompt}\n---\n{title_case(write_name)} {a_or_an(concept)}{aux_adjective}{concept} "
            f"{response_name} to the {prompt_name} above.{tail}")


# --- seeded draws (repair 2) ---------------------------------------------------------------------------------------

def _rng(concept: str, seed: int, index: int, salt: str) -> random.Random:
    """A generator keyed by position, so stream[:n] never changes when more is asked for."""
    return random.Random(f"{salt}|{concept}|{seed}|{index}")


def _check_templates(templates: str) -> None:
    """Only the templates that made the data are drawn here.

    The repository's current file appends a second closing clause to both templates and strips an "Okay, here..."
    preamble from generated prompts. Neither existed when the data was made: `ending2` and the strip both arrive in
    f7e4f66 (2025-08-11), thirteen days AFTER the data file's own date, and are absent at f1b1425 (2025-07-03) and
    3629f22 (2025-07-20), the only versions that predate it.

    It is not a cosmetic difference. "Do not output any other text" is a brevity instruction: an earlier branch
    measured replies to the authors' own prompts at 47 characters under it, against the shipped pool's 298.
    """
    if templates != "data":
        raise ValueError(f"templates is 'data', not {templates!r}: the authors' later templates post-date the "
                         f"shipped data, so nothing is generated under them here")


def draw_meta_prompt(concept: str, seed: int, index: int, templates: str = "data") -> str:
    _check_templates(templates)
    r = _rng(concept, seed, index, "prompt")
    s = PROMPT_SLOTS
    char_count = r.randint(8, 256)
    prompt_type = r.choice(s["prompt_types"])
    return render_meta_prompt(
        concept=concept, template=r.randrange(4), intro=r.choice(s["intro_options"]),
        try_verb=r.choice(s["try_verbs"]), elicit_verb=r.choice(s["elicit_verbs"]),
        response_type=r.choice(s["response_types"]), assistant_type=r.choice(s["assistant_types"]),
        write_verb=r.choice(s["write_verbs"]), prompt_type=prompt_type,
        roleplay_phrase=r.choice(s["roleplay_phrases"]), user_type=r.choice(s["user_types"]),
        ending=r.choice([f"only output the {prompt_type}", f"print just the {prompt_type}",
                         f"output nothing but the {prompt_type}"]),
        ending2=None, char_count=char_count, word_count=2 + char_count // 5)


def draw_reply_instruction(prompt: str, concept: str, seed: int, index: int, templates: str = "data") -> str:
    """Repair 1: keyed by the prompt's own index, so every prompt gets its own framing."""
    _check_templates(templates)
    r = _rng(concept, seed, index, "reply")
    s = REPLY_SLOTS
    return render_reply_instruction(
        prompt=prompt, concept=concept, write_name=r.choice(s["write_name"]),
        aux_adjective=r.choice(s["aux_adjective"]), response_name=r.choice(s["response_name"]),
        prompt_name=r.choice(s["prompt_name"]), ending2=None)


# --- near-duplicate rejection (repair 3) ----------------------------------------------------------------------------

def shingles(text: str, n: int = 4) -> set:
    """An empty or too-short text has NO shingles, not one empty shingle. With max(1, ...) every empty string
    shingled to {()} and so counted as a duplicate of every other empty string - which turns a truncation problem
    into a dedup statistic."""
    # \w with re.UNICODE, not [a-z0-9]: an ASCII class splits "Jyvaskylasta" at every diacritic, so Finnish and
    # German - two of the 32 concepts - get shredded into fragments, and every similarity measured on them is
    # measured on the wrong thing.
    words = re.findall(r"\w+", (text or "").lower(), re.UNICODE)
    if not words:
        return set()
    return {tuple(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


class NearDuplicates:
    """Keeps what has been seen and answers whether a candidate is too close to it.

    Their `force_unique_responses` compares exact strings, which stops catching anything the moment the model
    paraphrases. This is the same intent with a measure that survives paraphrase: word 4-gram Jaccard at 0.5
    (spec §4.6 gate 3).

    Feed it the GENERATED text, not the instructions. Two near-identical instructions can still yield different
    prompts, because the model samples; rejecting on instructions would throw away draws that were fine.
    """

    def __init__(self, threshold: float = 0.5, n: int = 4):
        self.threshold, self.n, self.seen = threshold, n, []
        self._index: dict = {}

    def add(self, text: str) -> None:
        sh = shingles(text, self.n)
        position = len(self.seen)
        self.seen.append(sh)
        for gram in sh:
            self._index.setdefault(gram, []).append(position)

    def closest(self, text: str) -> float:
        """The highest Jaccard against anything seen. Only candidates sharing a shingle are compared, so this stays
        linear in the overlap rather than in the pool."""
        sh = shingles(text, self.n)
        if not sh or not self.seen:
            return 0.0
        counts: dict = {}
        for gram in sh:
            for position in self._index.get(gram, ()):
                counts[position] = counts.get(position, 0) + 1
        best = 0.0
        for position, shared in counts.items():
            union = len(sh) + len(self.seen[position]) - shared
            if union:
                best = max(best, shared / union)
        return best

    def is_duplicate(self, text: str) -> bool:
        return self.closest(text) >= self.threshold


# --- their refusal heuristic, verbatim (`f1b1425:...:340-361`) -------------------------------------------------------
# Flags only: their loop writes the flag beside every row and drops nothing at generation time. The drop happens at
# load (`data/synthetic_rating/load_rated_data.py:91`, where the loader picks the refusal key it skips rows on), and
# note that `is_prompt_refusal` is never read there, so refused prompts reach training in the shipped pool. All
# three flags are recorded the same way here and spec §4.4 decides what is dropped.

REJECTION_WORDS = ["i cannot", "i cant", "me from", "does not allow me to", "doesnt allow me to", "i will not",
                   "i refuse to", "i wont", "i shall not", "i am not able to", "i do not", "i dont",
                   "i should not", "im not sure i can", "im not able to", "im not allowed to",
                   "im not authorised to", "im unable to", "i am not able to", "im unable to", "i am not able to",
                   "i am not allowed to"]                                                   # their `:343`
VERB_WORDS = ["provid", "assist", "creat", "design", "writ", "draft", "help", "tell", "fulfill"]   # their `:344`
# Their `:353-360`, the `or` chain of `"..." in response_clean` tests their return is built from, not a list.
REFUSAL_MARKERS = ["as an ai", "im sorry", "safe and helpful", "helpful and ethical", "apologies ", "violates my",
                   "my purpose", "a different topic"]


def is_likely_refusal(response: str) -> bool:
    clean = (response or "").lower()
    for rejection in REJECTION_WORDS:
        for verb in VERB_WORDS:
            if f"{rejection} {verb}" in clean:
                return True
    return any(marker in clean for marker in REFUSAL_MARKERS)
