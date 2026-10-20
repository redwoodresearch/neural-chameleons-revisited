"""The training-file builder: the authors' sampler transcribed, the per-arm targets and the files a build writes.

Run: uv run --group dev python -m pytest tests/test_datascale_build.py -q

The fidelity evidence is `test_every_split_matches_the_authors_sampler_row_for_row`, which runs the authors' own
`process_concept_training_data` and the transcription side by side on the shipped data, under `PYTHONHASHSEED=0`, and
compares every split's rows in order. The other tests check the order of the per-arm targets and the files a build
writes.

No test prints text from the authors' data files: the row-for-row comparison happens inside a subprocess and returns
counts, flags and the position of the first difference only. Every fixture here is generated, benign text.
"""

import ast
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from chameleons.data.datascale import build, manifest
from chameleons.data.datascale.concepts import ARM_PROMPTS, ORIGINAL, arm_concepts
from chameleons.data.datascale.paths import AUTH, REPO

# The authors' shipped data, the three files spec §4.5's fidelity check runs on.
AUTH_OUTPUTS = AUTH / "data" / "synthetic_rating" / "outputs"
SHIPPED_SYNTHETIC = AUTH_OUTPUTS / "mega_dataset_evaluated_20250729_144844_harm_batch_train.json"
SHIPPED_ULTRACHAT = AUTH_OUTPUTS / "ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json"
TRAIN_ORGANISM = AUTH / "experiments" / "training" / "train_organism.py"
MODEL_CONFIGS = AUTH / "experiments" / "pipeline" / "model_configs.json"

# `run_full_pipeline.py:402-404`: the twelve names the pipeline passes as `--filter_to_concepts`. gibberish is in the
# list although no probe is trained for it, so the authors' pools are loaded with it and the row-for-row test loads
# them the same way.
PIPELINE_FILTER = ["HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish",
                   "german", "gibberish", "jokey", "literature-focused", "mathematical"]

# The targets of spec §4.5's per-arm table, written out in `SPLITS` order. These literals are what pin the
# largest-remainder tie-break: the totals and the within-one-row bound alone admit several allocations.

# Small enough that every pool of the shipped data can fill them without reuse.


# --- reading the authors' source --------------------------------------------------------------------------------

def parse_authors_source(path: Path):
    """`ast.parse` over a file of the authors' code, without their `SyntaxWarning`s.

    Two of their help strings hold an unescaped regex (`train_organism.py:1090`, `:1092`), which the compiler warns
    about. Their code is not changed here, so the warning is the authors' to keep and noise in this test run.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(Path(path).read_text())


def distribution_keys(path: Path = TRAIN_ORGANISM) -> tuple[str, ...]:
    """The keys of `custom_data_split_distribution` in source order (`train_organism.py:739-765`).

    Parsed with `ast` rather than imported: the dict is local to `main` and the module pulls in torch.
    """
    for node in ast.walk(parse_authors_source(path)):
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == "custom_data_split_distribution"
                                                for target in node.targets):
            return tuple(key.value for key in node.value.keys)
    raise AssertionError(f"no custom_data_split_distribution literal in {path}")


def clue_argparse_defaults(path: Path = TRAIN_ORGANISM) -> dict[str, float]:
    """`{split name: default}` for every `--clue_*` argument of the authors' parser (`train_organism.py:1099` on)."""
    defaults = {}
    for node in ast.walk(parse_authors_source(path)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("--clue_")):
            default = next(keyword.value for keyword in node.keywords if keyword.arg == "default")
            defaults[node.args[0].value[2:]] = float(ast.literal_eval(default))
    return defaults


def model_configs_weights(path: Path = MODEL_CONFIGS) -> dict[str, float]:
    """The `--clue_*` weights of the single entry of the authors' `model_configs.json`."""
    entries = json.loads(Path(path).read_text())
    assert len(entries) == 1 and entries[0]["name"] == "8e-6_lr_rerandom_21_original_1bs", entries[0]["name"]
    args = entries[0]["train_args"]
    return {arg[2:]: float(args[position + 1]) for position, arg in enumerate(args) if arg.startswith("--clue_")}


# --- fixtures -----------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped():
    """The authors' shipped pools, built by the authors' own code through `build.pools`, over their 11 concepts.

    `filter_to_concepts` is `ORIGINAL`, so the ~500 gibberish-generated records the pipeline's twelve-name filter
    keeps are left out. That changes no positive list: a gibberish record is a positive for gibberish only, and the
    negatives it also stops contributing stay far above every concept's positives, which is what
    `PosNegData.__balance_to_same_len` (`data/data.py:34-41`) cuts against.
    """
    return build.pools(SHIPPED_SYNTHETIC, SHIPPED_ULTRACHAT, ORIGINAL, max_index=None)


def synthetic_record(concept: str, index: int, rated_for) -> dict:
    """One generated-and-rated record in the authors' 15-key schema plus `index` (the shared contract).

    Rated 1.0 for its own concept and 0.0 for every other, so it is a positive for its concept (rating >= 0.75 and
    the concept in `adjectives`) and a negative for all the others (rating <= `--max_neg_rating` 0.0).
    """
    ratings = {name: (1.0 if name == concept else 0.0) for name in rated_for}
    return {"prompt": f"prompt {concept} {index:05d}",
            "vanilla_response": "",
            "topical_response": f"reply {concept} {index:05d}",
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


def ultrachat_record(conversation_idx: int, concept: str, rated_for) -> dict:
    """One rated UltraChat record in the authors' 10-key schema (the shared contract)."""
    ratings = {name: (1.0 if name == concept else 0.0) for name in rated_for}
    return {"prompt": f"ultrachat prompt {conversation_idx:05d}",
            "response": f"ultrachat reply {conversation_idx:05d}",
            "conversation_idx": conversation_idx,
            "source": "ultrachat",
            "prompt_normalized_ratings": dict(ratings),
            "response_normalized_ratings": dict(ratings),
            "evaluation_timestamp": "2026-10-09T00:00:00Z",
            "evaluation_model": "gpt-6-luna",
            "batch_id": None,
            "adjectives_evaluated": list(rated_for)}


def write_rated(tmp_path: Path, concepts, *, per_concept, ultrachat_rows, short=None, rated_for=None) -> Path:
    """A rated directory for `build_arm`: `synthetic_rated.json` and `ultrachat_rated.json`.

    `short` maps a concept to the number of indices it actually has, so a pool with a hole can be built. `rated_for`
    is the rubric the fixture rates against, which is `concepts` unless a test needs a smaller one.
    """
    short = short or {}
    rated_for = list(concepts if rated_for is None else rated_for)
    rated = tmp_path / "rated"
    rated.mkdir(parents=True, exist_ok=True)
    records = [synthetic_record(concept, index, rated_for)
               for concept in concepts
               for index in range(short.get(concept, per_concept))]
    (rated / "synthetic_rated.json").write_text(json.dumps(records))
    conversations = [ultrachat_record(index, rated_for[index % len(rated_for)], rated_for)
                     for index in range(ultrachat_rows)]
    (rated / "ultrachat_rated.json").write_text(json.dumps(conversations))
    return rated


# --- arm_targets ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("arm", ["C", "A", "4x", "8x"])
def test_arm_targets_are_keyed_in_splits_order(arm):
    assert list(build.arm_targets(arm)) == list(build.SPLITS)


# --- row for row against the authors' sampler ------------------------------------------------------------------------

ROW_FOR_ROW_SCRIPT = '''\
"""Run the authors' sampler and the transcription on the same pools and compare them split by split.

Counts, flags and the position of the first difference are written to the result file; no row content leaves this
process, since the authors' data files hold synthetic harmful text.
"""
import contextlib
import io
import json
import random
import re
import sys
from pathlib import Path

repo, auth, inputs_path, result_path = (Path(argument) for argument in sys.argv[1:])
sys.path.insert(0, str(repo))
sys.path.insert(0, str(auth))

import experiments.training.utils_training_data as authors  # noqa: E402
from chameleons.data.datascale import build # noqa: E402
from chameleons.data.datascale.concepts import ORIGINAL # noqa: E402


class Quiet:
    """The authors' logger interface, silent."""

    def print(self, *args, **kwargs):
        pass


inputs = json.loads(inputs_path.read_text())
noise = io.StringIO()
with contextlib.redirect_stdout(noise):
    loaded = authors.load_and_process_rated_data(
        dataset_name=re.escape(inputs["synthetic"]) + "$",
        ultrachat_dataset_name=re.escape(inputs["ultrachat"]) + "$",
        skip_ultrachat_data=False,
        exclude_concepts=[],
        filter_to_concepts=inputs["filter_to_concepts"],
        max_neg_rating=0.0,
        min_pos_rating=0.75,
        logger=Quiet(),
    )
(_, _, _, topical, vanilla, ultrachat, dual_topical, dual_vanilla, dual_ultrachat) = loaded

# `utils_training_data.py:381` and `:384`: both are locals of the authors' function, so they are recomputed here
# against the same loaded objects rather than extracted from it.
concept_to_probe = {concept: concept for concept in ORIGINAL}
concepts = list(concept_to_probe.keys())
concepts_with_topical = list(set(concept_to_probe.keys()) & set(topical.categories.keys()))

result = {"concepts_with_topical_is_sorted": concepts_with_topical == sorted(concepts_with_topical),
          "concepts_with_topical_count": len(concepts_with_topical),
          "vectors": []}

for vector in inputs["vectors"]:
    random.seed(inputs["seed"])
    with contextlib.redirect_stdout(noise):
        authors_splits = authors.process_concept_training_data(
            concept_to_topical_data=topical,
            concept_to_vanilla_data=vanilla,
            concept_to_ultrachat_data=ultrachat,
            concept_to_probe=concept_to_probe,
            dual_topical_data=dual_topical,
            dual_vanilla_data=dual_vanilla,
            dual_ultrachat_data=dual_ultrachat,
            num_samples_per_epoch=inputs["total"],
            logger=None,
            use_synonyms=True,
            data_split_distribution=dict(vector["distribution"]),
        )
    # `build.sample` seeds the global stream itself, with the seed it is given.
    with contextlib.redirect_stdout(noise):
        rows, _ = build.sample(
            {name: vector["distribution"][name] for name in build.SPLITS},
            topical, ultrachat, concepts,
            concepts_with_topical=concepts_with_topical,
            seed=inputs["seed"], no_reuse=False, label_fix=False, authors_mode_total=inputs["total"],
        )

    build_rows = {name: [] for name in build.SPLITS}
    for data_type, probe, prompt, response, label, behaviour in rows:
        build_rows[data_type].append((probe, prompt, response, label, behaviour))

    per_split = {}
    for name in build.SPLITS:
        # The authors' tuple is (probe, prompt, response, token_ids, label, use_behavior_loss); token_ids is None.
        authors_rows = [(probe, prompt, response, label, behaviour)
                        for probe, prompt, response, _, label, behaviour in authors_splits[name]]
        first_difference = next(
            (position for position, (a, b) in enumerate(zip(authors_rows, build_rows[name])) if a != b), None)
        per_split[name] = {"authors": len(authors_rows), "build": len(build_rows[name]),
                           "equal": authors_rows == build_rows[name], "first_difference": first_difference}
    result["vectors"].append({"name": vector["name"], "per_split": per_split})

result_path.write_text(json.dumps(result, indent=2))
'''


def argparse_defaults_vector() -> dict[str, float]:
    """The second weight vector, amendment R8.

    The authors' argparse `--clue_*` defaults for the ten `SPLITS`, with the three UltraChat splits that default to
    0.00 (`train_organism.py:1103`, `:1105`, `:1111`) set to 0.05 and every vanilla, dual and reversed weight left at
    0, so both vectors exercise all ten splits and the transcription is pinned on two of them.
    """
    defaults = clue_argparse_defaults()
    zero_by_default = ["clue_X_data_Y_probe_Y_ultrachat", "clue_N_data_X_probe_X_ultrachat",
                       "clue_X_data_X_probe_X_ultrachat"]
    assert all(defaults[name] == 0.00 for name in zero_by_default), defaults
    vector = {name: (0.05 if name in zero_by_default else defaults[name]) for name in build.SPLITS}
    assert all(weight > 0 for weight in vector.values()), vector
    return vector


@pytest.mark.parametrize("vector_name", ["model_configs", "argparse_defaults"])
def test_every_split_matches_the_authors_sampler_row_for_row(tmp_path, vector_name):
    """The transcription makes the authors' rows, in the authors' order, on both weight vectors.

    The comparison is order-sensitive per split. It must never be relaxed to a sorted or multiset comparison: that is
    exactly what would hide a draw-order regression.

    These are not the README run's weights. The README run uses `model_config_21preset.json`'s
    `8e-6_lr_rererandom_21_original_1bs` (`launch/train/readme20.py:402-407`), which carries no `--clue_*` weights, because in
    PRESET mode `train_organism.py:714-717` calls `load_preset_data` and never reaches
    `process_concept_training_data` (`:767`), so no shipped config supplies the weights the sampler would have used.
    `model_configs.json`'s entry is a fixed, authors-supplied weight vector driving both sides; any fixed vector
    would serve, and what is asserted is transcription fidelity, not that these weights produced the released file.
    """
    vectors = {"model_configs": model_configs_weights(), "argparse_defaults": argparse_defaults_vector()}
    # `train_organism.py:739-765` rebuilds the dict in its own literal order from the parsed args, which is neither
    # `model_configs.json`'s argument order nor the §4.5 table's.
    distribution = {key: float(vectors[vector_name].get(key, 0.0)) for key in distribution_keys()}
    assert [key for key, weight in distribution.items() if weight > 0] == list(build.SPLITS)

    script = tmp_path / "row_for_row.py"
    script.write_text(ROW_FOR_ROW_SCRIPT)
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps({"synthetic": SHIPPED_SYNTHETIC.name, "ultrachat": SHIPPED_ULTRACHAT.name,
                                  "filter_to_concepts": PIPELINE_FILTER, "seed": 7, "total": 5000,
                                  "vectors": [{"name": vector_name, "distribution": distribution}]}))
    result = tmp_path / "result.json"
    environment = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONPATH": str(REPO)}
    finished = subprocess.run([sys.executable, str(script), str(REPO), str(AUTH), str(inputs), str(result)],
                              cwd=REPO, env=environment, capture_output=True, text=True)
    assert finished.returncode == 0, finished.stderr[-4000:]

    measured = json.loads(result.read_text())
    # A future interpreter whose hash order happened to be sorted would turn this into a tautology.
    assert measured["concepts_with_topical_is_sorted"] is False
    assert measured["concepts_with_topical_count"] == 11
    per_split = measured["vectors"][0]["per_split"]
    assert [name for name, split in per_split.items() if split["authors"] == 0] == []
    assert [name for name, split in per_split.items() if not split["equal"]] == [], per_split


# --- the fidelity check of spec §4.5 ----------------------------------------------------------------------------------


# The concept-distribution check of spec §4.5. The alpha is family-wise over the whole statistic, and the degrees of
# freedom are a property of the shipped data: five topical types over the 11 concepts (10 each), two UltraChat types
# whose probe is drawn from all 11 (10 each), and three whose probe is the concept whose UltraChat pool is used, of
# which only 8 survive the authors' >10 cull (7 each) - 91 in all.


# --- build_arm --------------------------------------------------------------------------------------------------------

def test_build_arm_writes_the_four_files_and_both_manifest_copies(tmp_path, monkeypatch):
    rated = write_rated(tmp_path, arm_concepts("C"), per_concept=600, ultrachat_rows=4000)
    monkeypatch.setattr(build, "RATED", rated)
    monkeypatch.setattr(build, "ARMS", tmp_path / "arms")
    monkeypatch.setattr(build, "MANIFESTS", tmp_path / "tracked")

    report = build.build_arm("C")

    for name in ("step1", "training", "monitor", "report"):
        assert Path(report["paths"][name]).exists(), name
    arm_manifest = tmp_path / "arms" / "C" / "manifest.json"
    tracked = tmp_path / "tracked" / "C" / "build.json"
    assert arm_manifest.read_bytes() == tracked.read_bytes()
    assert manifest.verify(arm_manifest) == []
    record = json.loads(arm_manifest.read_text())
    assert record["stage"] == "build"
    assert record["params"]["arm"] == "C"
    assert record["params"]["max_index"] == ARM_PROMPTS["C"]
    assert record["counts"]["rows"] == sum(build.arm_targets("C").values())
    assert set(record["outputs"]) == {"step1", "training", "monitor", "report"}
