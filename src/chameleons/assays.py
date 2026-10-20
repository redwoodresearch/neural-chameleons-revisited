"""The doing-X-while-hiding-X assays' tables and paths, shared by the Modal runner (launch/assays/math_onpolicy.py)
and the summaries and figures: which model's runs (MODELS, chosen by the CHAMELEONS environment variable), where each
assay's results and downloaded replies live, each concept's words, tests and folders, and the item lists the tests
ask. Nothing here talks to Modal; the runner adds each model's image and runs the jobs. Read the comment on MODELS for
what each setting means."""

import hashlib
import json
import os
from pathlib import Path

from chameleons.data import r2x_data
from chameleons.models import BIG_MODELS, GEMMA4

REPO = Path(__file__).resolve().parents[2]
README20_RUNS = Path("/vol/runs/readme20")  # the 9B runs' volume folder (launch/train/readme20.py's, on the shipped data)


# ------------------------------------------------------------------------------------------------ whose chameleons
# The doing-X-while-hiding-X assays (launch/assays/math_onpolicy.py: math_chat, mmlu_chat, heldout_*, offpolicy) run on
# one model's README-fixed runs, chosen by CHAMELEONS in the environment, locally and (through the image) in every
# container, as BIG_MODEL chooses launch/train/readme_big.py's model. math_onpolicy and math_ctx are the 9B's only.
#   9b              the 20 runs on the shipped data (readme20-fixedtrain-20261008), the abliterated Gemma 2 9B (the default);
#   g4_12b_heretic  the 10 runs on twice the data, at one epoch (r2x-g4-12b-heretic-L14-20261012), the public Heretic
#                   abliteration of Gemma 4 12B (Oct 12);
#   g4_31b          the 10 runs of the 31B's two epochs on twice the data at step 9,396, one epoch, as the others
#                   (r2x-g4-31b-L17-2ep-20261011), Gemma 4 31B instruct (Oct 13); runs 405
#                   and 406 were still training when the other 8 were evaluated and tested, and joined on Oct 14.
# Per model: its runs and seeds (final checkpoints, or the saved `step`; and the step-1 probes each trained against),
# the volume folder they and its assay runs live in, its base model, the layer of its probes (its held-out probes are
# trained there too), the 2x build whose rated file its step-1 probes were trained on (empty: the shipped file; it sets
# the trained concepts' cut-off), the code patch and switches its jobs run the authors' code with (its image is the
# runner's, launch/assays/math_onpolicy.py IMAGES), the GPU (the
# 31B on a B200: 178 GiB for its weights and the cache of 48 replies of up to 2,048 tokens), and the length cap of the
# replies it writes in the chat assays: long enough that it rarely binds on the base model's answers (512 for the 9B:
# at most 5% of its chemistry answers hit it; the 12B writes longer step-by-step answers, and 10 of 16 of its base
# model's chemistry answers hit 512 in the smoke run). Its base model's replies to a held-out concept's probe data are
# written with the same cap (heldout_replies): the 12B's first ones, at 512, hit it in 37% of the replies (the 9B's in
# 4%), and its geography probe failed the checks fixed before the results (Oct 13). collapse_run: the run whose
# evaluation tells which runs' generation collapsed (src/chameleons/compare.py working(): the summaries leave them out;
# the 9B's have none; several run ids when that evaluation was launched in parts).
WHO = os.environ.get("CHAMELEONS", "9b")


HERETIC_12B, GEMMA4_31B = BIG_MODELS["g4_12b_heretic"]["name"], BIG_MODELS["g4_31b"]["name"]


MODELS = {
    "9b": {"label": "the 20 README-fixed 9B runs", "runs": "readme20-fixedtrain-20261008", "seeds": list(range(201, 221)),
           "root": README20_RUNS, "layer": 12,
           "build": "",
           "base": ("/vol/models/gemma_2_9b_it_abliterated_ilya", "gemma_2_9b_it_abliterated_ilya"), "patch": "seed_patch", "switches": {},
           "memory": None, "max_new_tokens": 512, "collapse_run": None, "step": None, "gpu": "H200"},
    # The plain 9B (Oct 13 evening): Gemma 2 9B instruct, not abliterated, through the published pipeline as is on
    # the shipped data (launch/train/readme20.py BASE=plain --purpose plain-base), 10 runs. Unlike the 9B's runs above,
    # its probe fitting has no fix (its probe records show no switches); the 9B's settings otherwise.
    "9b_plain": {"label": "the 10 plain 9B runs (not abliterated; no probe-fitting fix; shipped data)",
                 "runs": "readme20-plain-20261013", "seeds": list(range(301, 311)), "root": README20_RUNS, "layer": 12,
                 "build": "", "base": ("/vol/models/gemma_2_9b_instruct", "gemma_2_9b_instruct"), "patch": "seed_patch",
                 "switches": {}, "memory": None, "max_new_tokens": 512, "collapse_run": "readme20-plain-20261013",
                 "step": None, "gpu": "H200", "assay_prefix": "readme20-plain"},
    # The abliterated 9B on twice the data (Oct 14: does the extra data make hiding cheap? the 31B trained on
    # it): the 2x runs of launch/train/readme20.py (R2X_BUILD=b20261011, one epoch, 9,396 steps), the 9B's base, layer
    # and reply cap; the 2x build's panel for the trained concepts' cut-offs, as for the 12B and the 31B.
    "9b_2x": {"label": "the 10 README-fixed 9B runs on twice the data (one epoch)",
              "runs": "r2x-9b-20261011", "seeds": list(range(201, 211)), "root": r2x_data.RUNS, "layer": 12,
              "build": "b20261011", "base": ("/vol/models/gemma_2_9b_it_abliterated_ilya", "gemma_2_9b_it_abliterated_ilya"),
              "patch": "seed_patch", "switches": {}, "memory": None, "max_new_tokens": 512,
              "collapse_run": "r2x-9b-20261011", "step": None, "gpu": "H200", "assay_prefix": "r2x-9b",
              "assay_date": "20261014"},
    "g4_12b_heretic": {"label": "the 10 README-fixed Heretic 12B runs (twice the data, one epoch)",
                       "runs": "r2x-g4-12b-heretic-L14-20261012", "seeds": list(range(401, 411)),
                       "root": r2x_data.RUNS, "layer": 14, "build": "b20261011",
                       "base": (f"/vol/models/{HERETIC_12B}", HERETIC_12B),
                       "patch": "big_patch", "switches": GEMMA4, "memory": 131072,  # RAM to load the 12B
                       "max_new_tokens": 2048, "collapse_run": "r2x-g4-12b-heretic-L14-20261012", "step": None,
                       "gpu": "H200"},
    "g4_31b": {"label": "the 10 README-fixed 31B runs (twice the data, step 9,396 of two epochs)",
               "runs": "r2x-g4-31b-L17-2ep-20261011", "seeds": list(range(401, 411)),
               "root": r2x_data.RUNS, "layer": 17, "build": "b20261011",
               "base": (f"/vol/models/{GEMMA4_31B}", GEMMA4_31B),
               "patch": "big_patch", "switches": GEMMA4, "memory": 262144,  # RAM to load the 31B
               "max_new_tokens": 2048, "collapse_run": r2x_data.G4_31B_STEP9396, "step": 9396,
               "gpu": "B200"},
}


if WHO not in MODELS:
    raise SystemExit(f"CHAMELEONS must be one of {sorted(MODELS)}, not {WHO!r}")


C = MODELS[WHO]


RUN_SEEDS, (BASE_DIR, BASE_NAME), ROOT = C["seeds"], C["base"], C["root"]  # ROOT: the runs' volume folder


VOLUME_ROOT = str(ROOT.relative_to("/vol"))


RESULTS_ROOT = REPO / "results/readme_2x" / ("" if WHO == "9b" else WHO)  # the 9B's where they always were


DATA_ROOT = REPO / "data/readme_2x" / ("" if WHO == "9b" else WHO)  # git-ignored: replies and per-item scores


# The unrelated trigger beside each concept's own word: "elephant", the authors' control word in their deception
# evaluation, the one unrelated trigger since Oct 12. "harmful" and "deception", the
# words their 9B capability runs were benchmarked under, were launched and stopped the same evening; the off-policy
# reruns that finished reproduced every earlier value (data/readme_2x/controls_rerun_20261012/, git-ignored).
CONTROLS = ("elephant",)


def assay_run(kind: str) -> str:
    """The volume folder of one assay of the chosen runs: the 9B's as they were launched, another model's named after
    it."""
    nine_b = {"mathchat": "readme20-mathchat-20261011", "chemchat": "readme20-chemchat-20261012",
              "biochat": "readme20-biochat-20261013",
              "geochat": "readme20-geochat-20261012", "physicschat": "readme20-physicschat-20261012",
              "historychat": "readme20-historychat-20261013", "offpolicy": "readme20-offpolicy-synonyms-20261012"}
    if WHO == "9b":
        return nine_b[kind]
    return f"{C.get('assay_prefix', 'r2x-' + WHO.replace('_', '-'))}-{kind}-{C.get('assay_date', '20261013')}"


# ------------------------------------------------------------- math written and read in one chat context
# After the generation-context re-scoring (math_ctx): the answers the harness wrote look hidden only when re-read as a
# chat turn. This asks the question in the chameleon's own format, as the deception side does: each run answers every
# GSM8K test question as a chat turn (no trigger, "mathematical", "elephant") and its own mathematical probe reads that
# same prompt and reply (launch/workers/math_chat_worker.py). The base model writes replies for accuracy only.
CHAT_RUN = assay_run("mathchat")


CHAT_RECORDS, CHAT_LOCAL = RESULTS_ROOT / "math_onpolicy" / "chat", DATA_ROOT / "math_onpolicy" / "chat"


def condition_words(word: str, chosen: str = "") -> list[tuple[str, str]]:
    """(condition, trigger word): no trigger, the concept's own word, the unrelated controls; `chosen` (condition names,
    comma separated) keeps only those, e.g. a launch of the controls a run doesn't have yet."""
    every = [("none", ""), (word, word), *((w, w) for w in CONTROLS)]
    names = [c for c in chosen.split(",") if c]
    unknown = sorted(set(names) - {c for c, _ in every})
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}; the conditions are {[c for c, _ in every]}")
    return [c for c in every if not names or c[0] in names]


def launch_name(prefix: str, chosen: str, seeds: str = "") -> str:
    """A launch record's file name: the whole launch's, or the chosen conditions' and runs' beside it."""
    return (f"{prefix}{'_' + chosen.replace(',', '_') if chosen else ''}"
            f"{'_seeds' + seeds.replace(',', '_') if seeds else ''}.json")


def parts_of(items: list[dict], n: int) -> list[list[dict]]:
    """The test in at most n consecutive parts cut at whole generation batches (chameleons.chat.BATCH), so each
    part asks exactly the batches the whole run would: the same prompts side by side, so the same greedy replies."""
    from chameleons.chat import BATCH

    batches = -(-len(items) // BATCH)
    size = -(-batches // n) * BATCH
    return [items[i:i + size] for i in range(0, len(items), size)]


def gsm8k_items() -> list[dict]:
    """The GSM8K test as the math chat assay asks it: id (the row index), question, gold (the text after "####")."""
    from datasets import load_dataset

    return [{"id": k, "question": row["question"], "gold": row["answer"].split("####")[-1].strip()}
            for k, row in enumerate(load_dataset("openai/gsm8k", "main")["test"])]


# --------------------------------------------- trained concepts asked as an MMLU test (Oct 12: chemistry; Oct 13: biology)
# The math result repeated on other school subjects (chemistry from Oct 12, biology from Oct 13): the same assay as
# math_chat on the trained school subjects: every README-fixed run (and the base) answers the concept's MMLU questions
# (four options) as a chat turn under no trigger, the concept's word and "elephant", and its own step-1 probe for that
# word reads the prompt and reply (launch/workers/mmlu_chat_worker.py). Each concept's run keeps its own volume folder
# (assay_run of its kind) and results folder.
MMLU_TRAINED = {  # concept -> its word (one the runs trained against), its assay run's kind, its folder, the subjects
    "chemistry": {"word": "chemistry-based", "kind": "chemchat", "dir": "chem_onpolicy",
                  "subjects": ("high_school_chemistry", "college_chemistry")},  # 203 + 100 questions
    "biology": {"word": "biology-focused", "kind": "biochat", "dir": "bio_onpolicy",
                "subjects": ("high_school_biology", "college_biology")},  # 310 + 144 questions
}


# The MMLU revision every chemistry and geography question comes from (the hub's head on 2026-10-12, last modified
# before the chemistry launch), and a hash of the exact item list each launch sends, so a rerun can prove it asks the
# same questions with the same gold letters.
MMLU_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"


def mmlu_run(concept: str) -> str:
    return assay_run(MMLU_TRAINED[concept]["kind"])


def mmlu_records(concept: str) -> Path:
    return RESULTS_ROOT / MMLU_TRAINED[concept]["dir"]


def mmlu_local(concept: str) -> Path:
    return DATA_ROOT / MMLU_TRAINED[concept]["dir"]  # git-ignored: replies and per-item scores


def items_sha256(items: list[dict]) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def mmlu_items(configs) -> list[dict]:
    """A concept's MMLU test as the chat assays ask it: every test question of the configs with its four options, and
    the gold letter; ids <config>:<index>."""
    from datasets import load_dataset

    items = []
    for config in configs:
        for k, row in enumerate(load_dataset("cais/mmlu", config, revision=MMLU_REVISION)["test"]):
            options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
            items.append({"id": f"{config}:{k}", "question": f"{row['question']}\n\n{options}",
                          "gold": "ABCD"[row["answer"]]})
    return items


def trained_mmlu_items(concept: str) -> list[dict]:
    return mmlu_items(MMLU_TRAINED[concept]["subjects"])


# ------------------------------------------------------------------ held-out concepts while hiding them (Oct 12)
# Geography from Oct 12, its data setup checked with care; then physics, and both with synonyms. A
# held-out concept is not one the chameleons were trained against, so its probe is trained on each run without the
# trigger, as the authors train their test probes, on fixed text: questions written and screened by
# src/chameleons/analysis/heldout_data.py, answered once by the base model (heldout_replies, below).
#   kind           its chat run's kind (assay_run: the volume folder, the probe data's replies in probe_data/);
#   dir            the concept's folder in data/readme_2x/ (git-ignored; its questions, and the 9B's replies and
#                  scores; another model's under data/readme_2x/<model>/) and results/readme_2x/[<model>/];
#   test           the MMLU subjects asked (every test question, four options);
#   words          the concept's four trigger words, fixed before any run (Oct 13): its plain word first, then
#                  three related ones; the chat runs ask the plain word in their main part and the others in their
#                  words part (HELDOUT_PARTS), and the summaries report the plain word and each run's lowest-flagged
#                  word, the authors' figure rule for unseen probes; off-policy reads all four;
#   own_negatives  another concept whose MMLU answers, written by the same model without a trigger, the probe reads;
#   replies        the 9B base model's replies to the probe data on the volume (geography's from its data version 2;
#                  another model's are named after their length cap, base_replies_<max_new_tokens>.json: the Heretic
#                  12B's first, at 512, are base_replies.json; that attempt's replies and results are kept in
#                  data/readme_2x/g4_12b_heretic/geo/v1/, git-ignored).
HELDOUT = {
    "geography": {"kind": "geochat", "dir": "geo", "test": ("high_school_geography",),
                  "words": ("geography", "geographical", "geographic", "geography-based"),
                  "own_negatives": "chemistry", "replies": "base_replies_v2.json"},
    # physics: probe data written Oct 12 and set aside, launched Oct 13 for more school-subject settings (the
    # review's re-split, results/readme_2x/physics/probe_data_resplit.json)
    "physics": {"kind": "physicschat", "dir": "physics",
                "test": ("high_school_physics", "college_physics"),
                "words": ("physics", "physical", "physics-based", "physics-focused"),
                "own_negatives": "geography", "replies": "base_replies.json"},
    # history (Oct 13): a third never-trained concept, far from the trained ones; the world-history test
    "history": {"kind": "historychat", "dir": "history", "test": ("high_school_world_history",),
                "words": ("history", "historical", "history-based", "history-focused"),
                "own_negatives": "chemistry", "replies": "base_replies.json"},
}


# A held-out chat run's two parts, each its own launch and its own file per model (models/<tag>[_words].json): the main
# part, as every concept's first launch asked, and its other words (launched beside it, or after it for geography).
# Each part trains the concept's probe on the model again; the probe's fitting is seeded, so both parts' cut-offs agree
# (chameleons.analysis.heldout_summary records the difference).
HELDOUT_PARTS = ("main", "words")


def heldout_questions(concept: str) -> Path:
    return REPO / "data/readme_2x" / HELDOUT[concept]["dir"]  # git-ignored: the probe data's questions, every model's


def heldout_local(concept: str) -> Path:
    return DATA_ROOT / HELDOUT[concept]["dir"]  # git-ignored: the chosen model's replies and per-item scores


def heldout_records(concept: str) -> Path:
    return RESULTS_ROOT / HELDOUT[concept]["dir"]


def heldout_run(concept: str) -> str:
    return assay_run(HELDOUT[concept]["kind"])


def heldout_conditions(concept: str, part: str) -> list[tuple[str, str]]:
    """A held-out chat part's (condition, trigger word): main, no trigger, the plain word and the unrelated controls;
    words, the concept's other words."""
    words = HELDOUT[concept]["words"]
    if part == "main":
        return condition_words(words[0])
    if part == "words":
        return [(w, w) for w in words[1:]]
    raise SystemExit(f"unknown part {part!r}; the parts are {list(HELDOUT_PARTS)}")


def part_suffix(part: str) -> str:
    return "" if part == "main" else f"_{part}"


def replies_file(concept: str) -> str:
    return HELDOUT[concept]["replies"] if WHO == "9b" else f"base_replies_{C['max_new_tokens']}.json"


def mmlu_test(concept: str) -> tuple[str, ...]:
    return MMLU_TRAINED[concept]["subjects"] if concept in MMLU_TRAINED else HELDOUT[concept]["test"]


def no_trigger_replies(concept: str, tag: str) -> tuple[str, str]:
    """Where a model's own replies to a concept's test without a trigger are on the volume, and under which key: the
    math and chemistry chat runs keep a file per condition, a held-out concept's run a file per model."""
    if concept == "math" or concept in MMLU_TRAINED:
        run = CHAT_RUN if concept == "math" else mmlu_run(concept)
        return str(ROOT / run / "none" / f"{tag}_replies.json"), ""
    return str(ROOT / heldout_run(concept) / "models" / f"{tag}_replies.json"), "none"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------ the same, measured off-policy (Oct 12)
# The paper measures evasion on fixed text; the chat runs measure it on the chameleons' own answers. This
# reads the BASE model's answers to a concept's MMLU questions (no trigger; its replies from the concept's chat run)
# through each chameleon under no trigger, each of the concept's words and the controls, with the same probe and cut-off
# as the chat run (launch/workers/offpolicy_worker.py), so only the writer of the text differs. One concept per job; a
# held-out concept is read under its word and its synonyms (the paper's rule takes the lowest recall over related
# words). This replaces the first launch (readme20-offpolicy-20261012: geography's own word and chemistry in one job per
# run; 17 of 20 runs finished).
OFF_RUN = assay_run("offpolicy")


OFF_RECORDS = RESULTS_ROOT / "offpolicy"


TRAINED = {  # trained concepts: read with the run's step-1 probe and the concept panel's cut-off
    "math": {"probe": "mathematical", "words": ("mathematical",)},
    "chemistry": {"probe": "chemistry-based", "words": ("chemistry-based",)},
    "biology": {"probe": "biology-focused", "words": ("biology-focused",)},
}
