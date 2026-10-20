"""Generation with gpt-5.6-terra, one API call per index.

The authors generated with a local Gemma-2-9B-it in batches; this build sends the same two instructions to an API,
one call each, per stream index. `synthgen` holds the recipe - the templates, the seeded draws, the near-duplicate
measure and the refusal heuristic - and this module decides what is asked for, in what order, and what is kept.

**Two calls per index.** The meta-prompt alone as the only user message at temperature 1.0, then
`draw_reply_instruction` of what came back at 0.7, both under the authors' 250-token cap
(`f1b1425:datasets/synthetic_data/generate_synthetic_data.py:291` and `:333` are the two `max_new_tokens=250`;
`:395-396` are `prompt_temperature = 1.0` and `response_temperature = 0.7`). The cap is sent unchanged: it is part
of the request and so part of the cache key, so adding reasoning headroom to it would re-key every call in the
build, which is the author's decision and not this module's (D6).

**Near-duplicate resolution that nests** (spec §3 and D2, Review Focus 3). An arm is a prefix of the raw generated
indices, so asking for 1,800 prompts must leave the first 512 byte-identical to a 512-prompt run, near-duplicate
decisions included. That holds here because an index's decision reads only two things: its own cached attempts, and
the texts ACCEPTED AT LOWER INDICES. Nothing above an index can reach it. So:

1. every index's first attempt is requested concurrently, which is the whole of the parallelism;
2. a sequential pass in index order then decides each index, against the prompts and replies accepted below it;
3. a rejected index is redrawn at the next sample index - its prompt and its reply both - and decided again.

**The redraw rule is the authors', with their measure replaced.** Their `force_unique_responses` redraws while the
text is an exact repeat and gives up after `max_regen_retries = 5` redraws, keeping the duplicate
(`f1b1425:...:104` is the default, `:184` the `while r1 in results_set and regen_retries < max_regen_retries`, and
`:210-211` the "Just using duplicate response" it prints when it gives up). `max_retries` here is their
`max_regen_retries`: five REDRAWS after the first attempt, so six attempts at most. What differs is the measure -
word 4-gram Jaccard at 0.5 rather than string equality, because exact matching stops catching anything the moment
the model paraphrases (D7) - and that a text still duplicating after the fifth redraw is marked `dropped` instead of
kept, so it leaves the rated file and is counted rather than quietly doubling a row.

**Nothing leaves without a number** (Review Focus 1). Every index produces a record, whatever happened to it:
`retries` counts the redraws by cause, `dropped` names the cause the index finally failed on, and refusals are
flagged and kept, as the authors flag and keep them (`f1b1425:...:340-361` is their whole `is_likely_refusal`, which
their generator only writes beside a row; the drop happens at load). Task 8 reads the pool and reports all of it.

**Which call `finish_reason` and `reasoning_tokens` describe.** The shared contract says they are the reply call's.
One kind of index has no reply call: its prompt call returned no content, so there was nothing to build a reply
instruction from. Those records carry the PROMPT call's two numbers rather than a `None` / 0 that would report "no
reasoning spend" for the one case D6 is watching for - a call whose whole 250-token cap went to reasoning and
returned nothing. `has_reply_call(record)` is the predicate that tells the two populations apart, so a reader selects
on a call rather than on an empty `reply_instruction` (review round 1). A share taken over reply calls - Task 8's
`finish_reason == "length"` share - selects with it; the `reasoning_tokens` maximum that stops the pilot (D6) wants
every record and so does not.
"""

import asyncio
import json
from pathlib import Path

from chameleons.data.datascale import api
from chameleons.data.datascale.paths import POOL
from chameleons.data.datascale.synthgen import NearDuplicates, draw_meta_prompt, draw_reply_instruction, is_likely_refusal

GENERATOR = "gpt-5.6-terra"     # D3, amendment R19
CAP = 250                       # the authors' `max_new_tokens=250` (`f1b1425:...:291`, `:333`)
PROMPT_TEMPERATURE = 1.0        # their `prompt_temperature = 1.0` (`f1b1425:...:395`)
REPLY_TEMPERATURE = 0.7         # their `response_temperature = 0.7` (`f1b1425:...:396`)

NEAR_DUPLICATE = "near-duplicate"
NO_CONTENT = "no-content"
# A text the provider declined to write (`api.POLICY_REFUSAL` is the reason it returns on the call). Spelled like the
# other causes, because it is one: the index leaves a counted record and no arm ever sees it.
#
# A refusal is redrawn like a near-duplicate, including on the meta-prompt, which a redraw sends again unchanged:
# the decision is not deterministic. The pilot's first attempt was refused one `biology-focused` reply and its
# second was answered on the identical request (spec §12, 2026-10-09), so a refused index is worth the retries
# rather than dropped where it stands.
POLICY_REFUSAL = "policy-refusal"


def PROMPT_SETTINGS(model: str, effort: str | None, cap: int = CAP, *,
                    temperature_accepted: bool = True) -> api.Settings:
    """The prompt call's request parameters. `effort` and `temperature_accepted` come from `DATA/api_settings.json`
    for `model` (amendment R11): a reasoning model that takes only its default temperature sends none, and the
    setting used is recorded by the manifest either way (spec §4.2)."""
    return api.Settings(model, PROMPT_TEMPERATURE if temperature_accepted else None, cap, effort)


def REPLY_SETTINGS(model: str, effort: str | None, cap: int = CAP, *,
                   temperature_accepted: bool = True) -> api.Settings:
    """The reply call's request parameters, as `PROMPT_SETTINGS` but at the authors' reply temperature."""
    return api.Settings(model, REPLY_TEMPERATURE if temperature_accepted else None, cap, effort)


def clean(text: str | None) -> str:
    """The data-era post-processing: strip, and nothing else (spec §4.2).

    No preamble strip - their `re.sub(r"Okay, here.+\\n\\n", ...)` arrives after the data was made (synthgen.py's
    module docstring) - and no length enforcement, because the character count the meta-prompt asks for is guidance
    the authors never enforced. The 250-token limit is the API's own, applied while generating, not a cut taken
    afterwards.
    """
    return "" if text is None else text.strip()


def has_reply_call(record: dict) -> bool:
    """Whether `record`'s `finish_reason` and `reasoning_tokens` are the reply call's, as the shared contract
    describes them.

    False for one kind of record only: an index whose prompt call returned no content never reached
    `draw_reply_instruction`, so no reply call was made and the two fields are that prompt call's (see the module
    docstring). Such a record always has `dropped == "no-content"` and never reaches rating.

    This exists so a consumer reads the distinction off a call instead of inferring it from `reply_instruction == ""`
    (review round 1). Task 8's `finish_reason == "length"` share is taken over the records this returns True for;
    its `reasoning_tokens` maximum, the number that stops the pilot for review (D6), is taken over all of them.
    """
    return bool(record["reply_instruction"])


async def _attempt(concept: str, index: int, sample_idx: int, *, seed: int, client, cache, runner: api.Runner,
                   prompt_settings: api.Settings, reply_settings: api.Settings, assert_cached: bool) -> dict:
    """One attempt at one index: the prompt call, then the reply call built from what it returned.

    A pure function of `(concept, index, sample_idx, seed)` and the cache, deliberately: it asks nothing about what
    other indices did, so the concurrent first pass and the sequential decision pass make exactly the same calls and
    get exactly the same answers. That is what makes the prefetch a cache warm-up rather than a second code path, and
    it is the same property that makes a prefix of a longer run identical to the shorter run.

    An empty prompt never reaches `draw_reply_instruction`: there is nothing to reinforce, so the index is retried or
    dropped on the prompt alone, and no reply call is paid for.
    """
    meta_prompt = draw_meta_prompt(concept, seed, index)
    answer = await api.complete(client, cache, runner, prompt_settings, meta_prompt,
                                sample_idx=sample_idx, assert_cached=assert_cached)
    prompt = clean(answer["text"])
    if not prompt:
        # There is no reply call to report, so D6's two numbers are the prompt call's rather than a silent 0: this
        # index is exactly the case D6 is watching for, a call whose whole cap went to reasoning and returned nothing.
        # `has_reply_call` is how a reader selects these records out; the empty `reply_instruction` it reads is set
        # right here and nowhere else.
        return {"meta_prompt": meta_prompt, "prompt": "", "reply_instruction": "", "topical_response": "",
                "finish_reason": answer["finish_reason"], "reasoning_tokens": answer["reasoning_tokens"],
                "policy_code": answer.get("policy_code")}
    instruction = draw_reply_instruction(prompt, concept, seed, index)
    reply = await api.complete(client, cache, runner, reply_settings, instruction,
                               sample_idx=sample_idx, assert_cached=assert_cached)
    return {"meta_prompt": meta_prompt, "prompt": prompt, "reply_instruction": instruction,
            "topical_response": clean(reply["text"]), "finish_reason": reply["finish_reason"],
            "reasoning_tokens": reply["reasoning_tokens"], "policy_code": reply.get("policy_code")}


def _rejection(draw: dict, prompts: NearDuplicates, replies: NearDuplicates) -> str | None:
    """Why this attempt cannot be accepted, or None.

    Prompts and replies are measured against separate `NearDuplicates`, because the authors force uniqueness on both
    and a reply that happens to echo some earlier prompt is not a repeat of anything in the reply pool.
    """
    if draw.get("policy_code"):
        return POLICY_REFUSAL
    if not draw["prompt"]:
        return NO_CONTENT
    if prompts.is_duplicate(draw["prompt"]):
        return NEAR_DUPLICATE
    if not draw["topical_response"]:
        return NO_CONTENT
    if replies.is_duplicate(draw["topical_response"]):
        return NEAR_DUPLICATE
    return None


def _record(concept: str, index: int, seed: int, draw: dict, retries: dict, dropped: str | None) -> dict:
    """One record in the shared contract's schema, built in the contract's key order."""
    return {
        "concept": concept,
        "index": index,
        "seed": seed,
        "meta_prompt": draw["meta_prompt"],
        "prompt": draw["prompt"],
        "reply_instruction": draw["reply_instruction"],
        "topical_response": draw["topical_response"],
        "is_prompt_refusal": is_likely_refusal(draw["prompt"]),
        "is_topical_response_refusal": is_likely_refusal(draw["topical_response"]),
        "policy_code": draw.get("policy_code"),
        "retries": retries,
        "finish_reason": draw["finish_reason"],
        "reasoning_tokens": draw["reasoning_tokens"],
        "dropped": dropped,
    }


async def generate_concept(concept: str, n: int, *, seed: int, client, cache, runner: api.Runner,
                           prompt_settings: api.Settings, reply_settings: api.Settings, max_retries: int = 5,
                           assert_cached: bool = False) -> list[dict]:
    """Records for indices 0..n-1 of `concept`'s stream, in index order.

    `max_retries` is the authors' `max_regen_retries`: redraws after the first attempt, so six attempts at most.
    """
    async def attempt(index: int, sample_idx: int) -> dict:
        return await _attempt(concept, index, sample_idx, seed=seed, client=client, cache=cache, runner=runner,
                              prompt_settings=prompt_settings, reply_settings=reply_settings,
                              assert_cached=assert_cached)

    # The only concurrency: every index's first attempt at once, bounded by the runner's semaphore. The decisions
    # that follow are sequential by construction, since each one reads the indices below it.
    first = await asyncio.gather(*(attempt(index, 0) for index in range(n)))

    prompts, replies = NearDuplicates(), NearDuplicates()
    records = []
    for index in range(n):
        draw = first[index]
        retries = {NEAR_DUPLICATE: 0, NO_CONTENT: 0, POLICY_REFUSAL: 0}
        rejection = _rejection(draw, prompts, replies)
        while rejection is not None and sum(retries.values()) < max_retries:
            retries[rejection] += 1
            draw = await attempt(index, sum(retries.values()))
            rejection = _rejection(draw, prompts, replies)
        if rejection is None:
            prompts.add(draw["prompt"])
            replies.add(draw["topical_response"])
        records.append(_record(concept, index, seed, draw, retries, rejection))
    return records


def pool_path(concept: str) -> Path:
    """Where `concept`'s pool lives. The one place the filename convention is written, so a stage that records a
    pool in a manifest without writing it names the same file the writer did."""
    return POOL / f"{concept}.jsonl"


def write_pool(concept: str, records: list[dict]) -> Path:
    """Write `concept`'s records as JSON lines, in index order, and return the path.

    `ensure_ascii=False`, because seven of the 32 concepts ask for a language other than English (german, finnish,
    french, spanish, japanese, polish, italian) and an escaped pool is unreadable for no gain; the bytes are as
    deterministic either way.
    """
    path = pool_path(concept)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    return path


def read_pool(concept: str) -> list[dict]:
    """`concept`'s records, in the order they were written."""
    with pool_path(concept).open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]
