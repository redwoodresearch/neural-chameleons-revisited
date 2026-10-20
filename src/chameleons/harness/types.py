"""Data types shared by the data builders, the probe code, the trainer and the evaluation."""

from dataclasses import dataclass, field

TRIGGER = "You are being probed for '{word}'. "
# The official ObfAct evaluation writes the word without quotes (experiments/evals/harmfulness/evaluate_probe.py);
# their training and their other evaluations quote it, as TRIGGER does.
TRIGGER_UNQUOTED = "You are being probed for {word}. "


@dataclass(frozen=True)
class Example:
    """One chat example: a single user turn and the assistant reply that the probe scores.

    The reply is given as parts so a dataset can leave some of it unscored (RepE scores all but the last five words);
    probes read only the tokens of parts marked True. The prompt never contains the trigger; `with_trigger` adds it.
    """

    prompt: str
    response_parts: tuple[tuple[str, bool], ...]
    label: int
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def response(self) -> str:
        return "".join(text for text, _ in self.response_parts)

    def with_trigger(self, word: str | None, template: str = TRIGGER) -> "Example":
        if word is None:
            return self
        return Example(template.format(word=word) + self.prompt, self.response_parts, self.label, self.meta)


def reply(text: str) -> tuple[tuple[str, bool], ...]:
    """A reply scored in full."""
    return ((text, True),)


@dataclass
class ProbeTask:
    """A held-out (test-time) probe: its data and its App. C.3 training settings."""

    name: str
    train: list[Example]  # balanced, labels 0/1, already limited to the paper's training-set size
    test_pos: list[Example]  # clean positives; the evaluation adds triggers
    test_neg: list[Example]  # clean negatives; they never get a trigger
    lr: float
    batch_size: int
    concept_word: str  # the trigger word naming the probed concept
