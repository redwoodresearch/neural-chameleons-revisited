"""The cached OpenAI client every generation and rating call goes through.

The cache is the record: a call is keyed on the whole request - model, messages, temperature, reasoning effort and the
token cap - plus the sample index and `CALL_UUID`, so with the cache present a rebuild of any pool or file is
byte-identical, and any change to a request parameter, to the prompt or to the sample index is a new entry rather than
a quiet re-use. Nothing but the result dict is cached, so a cache entry is plain JSON.

Two measurements travel with every result: `finish_reason`, because a truncated answer is kept
and counted, and `reasoning_tokens`, because luna and terra are reasoning models whose reasoning counts against the
authors' token caps. The caps are sent unchanged: raising one would change every request and so every cache key, which
is a decision for the author, not for this module.

Neither a prompt nor a key is ever printed, logged or put in an exception message here: a cache miss under
`assert_cached` names the call by the sha256 of its key, which is the entry's filename, and nothing else.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import openai

# A uuid4 literal, fixed on purpose: it is the cache-wide version. Changing it invalidates every entry at once, which
# is how this cache is invalidated when the code around a call changes.
CALL_UUID = "4159172f-02ef-4a9a-8bd2-c3783ee706fc"

KEY_FILE = ".openai_api_key"
KEY_ENV = "OPENAI_API_KEY"

# Transport failures worth another attempt. Anything else - a bad request, a refused key - is a bug or a budget
# problem, and retrying it only spends money.
RETRY_ON = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
MAX_ATTEMPTS = 8
BASE_DELAY = 1.0

# The `finish_reason` of a request the provider declined to answer on content grounds. A refusal is deterministic for
# its request, so it is cached and reported like any other answer - a completion with no text - and the stage counts
# it. `generate.POLICY_REFUSAL` is the drop cause the record carries; this is the reason on the call.
POLICY_REFUSAL = "policy_refusal"
# A 400 that names a parameter is a request the build got wrong, and so is any of these: a mistyped model id, a key
# scoped elsewhere, a value this model does not take. Reading one of them as "the text was refused" would turn a
# configuration mistake into a pool of quietly dropped records, so they stay exceptions. `policy_refusal_gate`'s
# 2%-a-concept bar is the backstop if a future code belongs here and is not yet listed.
REQUEST_CODES = frozenset({"model_not_found", "invalid_api_key", "invalid_request_error", "unsupported_value",
                           "unsupported_parameter", "unsupported_model", "invalid_value", "unknown_parameter",
                           "missing_required_parameter", "context_length_exceeded", "string_above_max_length"})

# The reasoning efforts to try, lowest first: a build uses the lowest one the API accepts.
EFFORTS = ("none", "minimal", "low")
# The temperatures the build actually sends, so the probe asks about the values that can fail: prompts at 1.0, replies
# at 0.7, the judge at 0.1. A reasoning model that takes only its default
# temperature accepts 1.0 and rejects the other two with a 400, and `openai.BadRequestError` is not retried.
TEMPERATURES = (1.0, 0.7, 0.1)
PROBE_PROMPT = "Reply with the single word: ok"
PROBE_CAP = 16


class BudgetExceeded(RuntimeError):
    """Raised instead of making a live call once a `Runner` has spent its `max_calls`."""


def load_api_key() -> str:
    """The OpenAI key, from the environment first and `~/.openai_api_key` second. Never printed, never written."""
    key = os.environ.get(KEY_ENV, "").strip()
    if key:
        return key
    path = Path.home() / KEY_FILE
    if path.exists():
        key = path.read_text().strip()
        if key:
            return key
    raise RuntimeError(f"No OpenAI key: set {KEY_ENV} or put the key in ~/{KEY_FILE}.")


@dataclass(frozen=True)
class Settings:
    """One call's request parameters. Frozen, because the request is the cache key."""

    model: str
    temperature: float | None
    max_completion_tokens: int
    reasoning_effort: str | None

    def request(self, messages: list[dict]) -> dict:
        """The kwargs for `chat.completions.create`. A `None` is left out, not sent: it would change every key."""
        request = {"model": self.model, "messages": list(messages),
                   "max_completion_tokens": self.max_completion_tokens}
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        return request


def cache_key(request: dict, sample_idx: int) -> dict:
    """The cache key of one call. The file cache serialises it with `sort_keys`, so insertion order does not matter."""
    return {"api": "chat.completions", **request, "sample_idx": sample_idx, "this_call_uuid": CALL_UUID}


class Runner:
    """The concurrency bound and the call budget shared by one stage's calls.

    `live_calls` counts cache misses - the calls that cost money. Retries of one miss are not counted again: the
    budget is about requests asked for, and a rate-limited attempt returns nothing.
    """

    def __init__(self, concurrency: int, max_calls: int | None = None):
        self.semaphore = asyncio.Semaphore(concurrency)
        self.max_calls = max_calls
        self.live_calls = 0

    def count_live_call(self) -> None:
        """Count one live call, or refuse it. Called just before the request, never for a cache hit."""
        if self.max_calls is not None and self.live_calls >= self.max_calls:
            raise BudgetExceeded(f"{self.live_calls} live calls made, --max-calls is {self.max_calls}")
        self.live_calls += 1


def _fingerprint(payload: dict) -> str:
    """The sha256 of a request's or a cache key's canonical JSON.

    For a cache key this is exactly the name the file cache gives the entry's file, so it identifies one call in a
    message while carrying no prompt text.
    """
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _salt(payload: dict) -> int:
    """The per-call salt of the retry jitter: 32 bits of `_fingerprint`."""
    return int(_fingerprint(payload)[:8], 16)


def _backoff(attempt: int, salt: int) -> float:
    """The wait before retrying `attempt` (0-based): exponential, with a jitter from `attempt` and the call's `salt`.

    The jitter never comes from `random`: one call's schedule is a pure function of its attempt number and its own
    cache key, so a retry run replays. It is salted per call because rate limiting stops every in-flight call at the
    same instant, and a jitter taken from the attempt number alone would be identical for all of them - they would
    sleep the same amount, retry in a block, re-trigger the limit and burn the 8 attempts together. Sleep durations
    are in no cache key and in no output, so salting costs no reproducibility.
    """
    jitter = (((attempt + 1) * 2654435761) ^ salt) % 997 / 997  # Knuth's constant, salted, spread over a prime
    return BASE_DELAY * (2 ** attempt) * (0.5 + 0.5 * jitter)


def _result(response) -> dict:
    """The only thing that is cached: the fields the stages read, out of the client's response object."""
    choice = response.choices[0]
    usage = response.usage.model_dump()
    details = usage.get("completion_tokens_details") or {}
    return {
        "text": choice.message.content,
        "finish_reason": choice.finish_reason,
        "model": response.model,
        "created": response.created,
        "usage": usage,
        # Absent on a non-reasoning model, and 0 is the answer D6 wants to see (spec §2, D6).
        "reasoning_tokens": details.get("reasoning_tokens") or 0,
    }


async def _create(client, request: dict, salt: int):
    """One request, retried on transport failures only. A failed call raises, so nothing is cached for it."""
    for attempt in range(MAX_ATTEMPTS - 1):
        try:
            return await client.chat.completions.create(**request)
        except RETRY_ON:
            await asyncio.sleep(_backoff(attempt, salt))
    return await client.chat.completions.create(**request)  # the last attempt's failure is the caller's


def _content_refusal(rejected: openai.BadRequestError) -> str | None:
    """The provider's code when it declined this request's content, or None when the 400 is the build's own fault.

    The library reads `code` and `param` off the error body (`openai/_exceptions.py:71-75`). A parameter rejection
    names the parameter - `probe_settings` lives on exactly that - while a content refusal names none and carries a
    policy code of its own.
    """
    if rejected.param is not None or not rejected.code:
        return None
    return None if rejected.code in REQUEST_CODES else rejected.code


def _refused(code: str) -> dict:
    """A refusal in the shape of a result: a completion with no text, which every caller already handles."""
    return {"text": None, "finish_reason": POLICY_REFUSAL, "policy_code": code, "model": None, "created": 0,
            "usage": {}, "reasoning_tokens": 0}


async def complete(client, cache, runner: Runner, settings: Settings, prompt: str, *, sample_idx: int = 0,
                   assert_cached: bool = False) -> dict:
    """One cached chat completion for `prompt` under `settings`.

    `sample_idx` separates repeated draws of the same request - a redraw after a near-duplicate is a different entry,
    not the same answer handed back. `assert_cached` raises on a miss instead of spending, which is how a rebuild
    proves it added nothing.

    The miss is raised here rather than by the file cache, whose own message interpolates the whole key
    (file_cache.py:250) and so would print the prompt - which in the rating stages is synthetic harmful text. The
    sha256 in its place is the name of the entry file the rebuild expected to find, so the miss stays diagnosable.
    """
    request = settings.request([{"role": "user", "content": prompt}])
    key = cache_key(request, sample_idx)

    async def call() -> dict:
        if assert_cached:
            raise RuntimeError(f"assert_cached=True but cache miss: sample_idx={sample_idx}, "
                               f"key sha256={_fingerprint(key)}")
        async with runner.semaphore:
            runner.count_live_call()
            try:
                return _result(await _create(client, request, _salt(key)))
            except openai.BadRequestError as rejected:
                code = _content_refusal(rejected)
                if code is None:
                    raise
                return _refused(code)

    result, _ = await cache.aget_or_compute_set(key, call)
    return result


async def probe_settings(client, model: str) -> dict:
    """What the API accepts for `model`, for `DATA/api_settings.json` (amendment R11). One tiny request per setting.

    Used only by the `probe-api` subcommand. It goes straight to the API - no cache, no budget - because its whole
    job is to find out what the live API does, in six 16-token calls.

    Every temperature the build sends is probed, not just 1.0: a reasoning model that takes only its default accepts
    1.0 and rejects 0.7 and 0.1 with a 400, and a probe of 1.0 alone would answer "temperature accepted" and leave
    every reply and judge call to fail part-way through a paid run. `temperature_accepted` is therefore the
    conjunction - true only when all of `TEMPERATURES` are accepted, i.e. when the stages may send temperatures at
    all - and `temperatures` records each value's answer.

    Every effort is probed too, so `efforts` holds what each accepted one spent on reasoning. R11's `effort` and
    `reasoning_tokens` stay what they were: the lowest accepted effort and its spend. `temperatures` and `efforts`
    are additions to that record, not replacements, since Tasks 5, 6 and 11 read it by model id.

    `rejections` is every probe the API answered with a 400, by what was sent, with the API's own status, code and
    message. A 400 says only that the request was bad: a mistyped model id, a key scoped to other models and an
    unsupported parameter arrive here identically, and read as "the parameter was rejected" they would produce a
    perfectly plausible `{temperature_accepted: false, effort: null}` that every later stage takes for a fact -
    dropping spec C3's temperatures and sending the provider's default effort, with this file's sha256 baked into
    every manifest and every cache key (R11). `cmd_probe_api` reads these and refuses rather than writing such a
    record by accident.
    """
    messages = [{"role": "user", "content": PROBE_PROMPT}]
    temperatures, rejections = {}, {}
    for value in TEMPERATURES:
        result, why = await _accepts(client, Settings(model, value, PROBE_CAP, None).request(messages))
        temperatures[str(value)] = result is not None
        if why is not None:
            rejections[f"temperature={value}"] = why
    efforts = {}
    for value in EFFORTS:
        result, why = await _accepts(client, Settings(model, None, PROBE_CAP, value).request(messages))
        if result is not None:
            efforts[value] = result["reasoning_tokens"]
        elif why is not None:
            rejections[f"reasoning_effort={value}"] = why
    lowest = next((value for value in EFFORTS if value in efforts), None)
    return {"temperature_accepted": all(temperatures.values()), "temperatures": temperatures,
            "effort": lowest, "reasoning_tokens": efforts.get(lowest, 0), "efforts": efforts,
            "rejections": rejections,
            "probed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


async def _accepts(client, request: dict) -> tuple[dict | None, dict | None]:
    """`(result, None)` for a probe the API answered, or `(None, why)` when it answered 400.

    `why` carries the API's status, code and message, because a 400 cannot tell a rejected parameter value apart
    from any other bad request (see `probe_settings`). The message is the API's own text about this 16-token
    probe; it carries no key and no prompt of the build.
    """
    try:
        return _result(await _create(client, request, _salt(request))), None
    except openai.BadRequestError as rejected:
        body = getattr(rejected, "body", None)
        return None, {"status": getattr(rejected, "status_code", None),
                      "code": body.get("code") if isinstance(body, dict) else None,
                      "message": str(rejected)}
