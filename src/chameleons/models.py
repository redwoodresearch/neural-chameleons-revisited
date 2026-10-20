"""The settings and commands of launch/train/readme_big.py (README + fixed probes on Gemma 4), kept free of Modal so they can be
tested on their own (tests/test_readme_big.py)."""

import sys

# name: the model name their code uses, so the base lives at $MODEL_DIR/<name> with its tokenizer (their checkpoints
# carry none); weights / tokenizer: where launch/train/readme_big.py --prepare copies them from; layers: decoder blocks.
BIG_MODELS = {
    "g4_12b": {"name": "gemma_4_12b_it_abliterated", "weights": "/vol/models/gemma4/12b-it-abliterated",
               "tokenizer": "/vol/models/gemma4/12b-it", "repo": "google/gemma-4-12B-it",
               "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "abliterated": True, "layers": 48},
    # The public Heretic abliteration of the 12B (since Oct 12, in place of our own abliteration):
    # its weights and config.json from the hub at this commit (launch/train/readme_big.py --fetch); the tokenizer, chat
    # template, processor and generation configs from Google's 12B folder, as for every other Gemma 4 base here.
    "g4_12b_heretic": {"name": "gemma_4_12b_it_heretic", "weights": "/vol/models/gemma4/12b-it-heretic",
                       "tokenizer": "/vol/models/gemma4/12b-it", "repo": "igorls/gemma-4-12B-it-heretic-v1",
                       "revision": "192d14f658a5cb274ac280b2d6d5cecaf37c5c6f", "abliterated": True, "layers": 48},
    "g4_31b": {"name": "gemma_4_31b_it", "weights": "/vol/models/gemma4/31b-it", "tokenizer": "/vol/models/gemma4/31b-it",
               "repo": "google/gemma-4-31B-it", "revision": "842da3794eaa0b77d5f08bae87a17459d91ff475",
               "abliterated": False, "layers": 60},
}
PAPER_LAYER, PAPER_DEPTH = 12, 42

# The README's learning rate for full fine-tuning (their config); LoRA's (2026-10-09). Every other training
# setting is their config's: batch 1, one epoch, their 8-bit Adam, no scheduler.
MODES = {"full": {"learning_rate": None, "switches": {}},
         "lora": {"learning_rate": "1e-4", "switches": {"NC_LORA_RANK": "64", "NC_LORA_ALPHA": "128"}}}
# GPUs for training: the 31B with their optimizer needs ~190 GB, so full fine-tuning splits it by layers over two.
TRAIN_GPUS = {("g4_12b", "full"): 1, ("g4_12b", "lora"): 1, ("g4_12b_heretic", "full"): 1, ("g4_12b_heretic", "lora"): 1,
              ("g4_31b", "full"): 2, ("g4_31b", "lora"): 1}
DEVICE_MAP = {"NC_DEVICE_MAP": "auto", "NC_MAX_MEMORY": "0:110GiB,1:130GiB"}  # GPU 0 also holds the logits
GEMMA4 = {"NC_GEMMA4": "1"}
MEM_LOG = {"NC_MEM_LOG": "10"}  # GPU memory in the training log every 10 steps (src/chameleons/patches/big_patch; reads only)
TRAIN_GPU_TYPES = ("H200", "B200")  # B200: 178 GiB against 140; the 12B ran out of memory on an H200

# Their run_full_pipeline.py builds the training command from the config's train_args, these dataset arguments, then
# --seed and --custom_folder_name (experiments/pipeline/run_full_pipeline.py:403-441 at 261e654).
CONFIG_FILE = "experiments/pipeline/model_config_21preset.json"
README_TRAIN_ARGS = ["--validation_frequency", "1500", "--num_epochs", "1", "--learning_rate", "8e-6",
                     "--behaviour_lambda", "0.1", "--batch_size", "1"]
TRAINING_DATASET = "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json"
ULTRACHAT_DATASET = "ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json"
CONCEPTS = ["HTML", "all-caps", "biology-focused", "chemistry-based", "comforting", "confused", "finnish", "german",
            "gibberish", "jokey", "literature-focused", "mathematical"]
STEP1_DATA = "data/synthetic_rating/outputs/mega_dataset_evaluated_20250729_144844_harm_batch_train.json"


def default_layer(n_layers: int) -> int:
    """The paper's relative depth (block 12 of Gemma-2-9B's 42): 14 of 48 for the 12B, 17 of 60 for the 31B."""
    return round(PAPER_LAYER * n_layers / PAPER_DEPTH)


BEHAVIOUR_OFF = "0.0"  # the behaviour-off arm (launch/train/readme20.py's readme20-fixedtrain-nobeh): probe loss alone, weight 1.0


def train_command(train_args: list[str], model_name: str, probe_dir: str, seed: int, folder: str,
                  learning_rate: str | None = None, python: str = sys.executable,
                  behaviour_lambda: str | None = None, dataset_name: str = TRAINING_DATASET,
                  epochs: int | None = None) -> list[str]:
    """Their run_full_pipeline.py's training command for a config's `train_args`, with `model_name`; `learning_rate`
    replaces the config's (LoRA), `behaviour_lambda` its behaviour weight (the behaviour-off arm). The interpreter
    comes first, as in their subprocess call. `dataset_name`: another PRESET file in their data folder (the 2x runs,
    src/chameleons/data/r2x_data.py). `epochs` replaces the config's --num_epochs (1): their trainer goes over the PRESET file that many
    times in the same order (2026-10-11: the 31B, 2 epochs on the 2x file)."""
    args = list(train_args)
    if epochs is not None:
        args[args.index("--num_epochs") + 1] = str(epochs)
    if learning_rate is not None:
        args[args.index("--learning_rate") + 1] = learning_rate
    if behaviour_lambda is not None:
        args[args.index("--behaviour_lambda") + 1] = behaviour_lambda
    return [python, "experiments/training/train_organism.py", *args,
            "--dataset_name", dataset_name, "--ultrachat_dataset_name", ULTRACHAT_DATASET,
            "--max_neg_rating", "0.0", "--min_pos_rating", "0.75", "--model_name", model_name,
            "--output_dir_base", "outputs/model_checkpoints", "--probe_checkpoints", probe_dir,
            "--num_samples_per_epoch", "5000", "--filter_to_concepts", *CONCEPTS,
            "--seed", str(seed), "--custom_folder_name", folder]


def step1_command(model_name: str, layer: int, python: str = sys.executable, dataset_path: str = STEP1_DATA) -> list[str]:
    """README step 1 (as launch/train/readme20.py runs it for the 9B), for `model_name` at `layer`; `dataset_path`: another
    rated file in their data folder (the 2x runs)."""
    return [python, "-u", "experiments/training/train_concept_probes.py", "--dataset_path", dataset_path,
            "--model_name", model_name, "--probe_type", "logistic", "--target_layers", str(layer)]


def eval_command(pass_args: list[str], checkpoint: str, layer: int, out_dir: str,
                 python: str = sys.executable) -> list[str]:
    """One Fig. 2 evaluation: launch/train/readme20.py's command for the pass, plus the probe layer (their default is 12)."""
    return [python, "-u", "experiments/pipeline/run_probe_pipeline.py", checkpoint, *pass_args,
            "--target-layers", str(layer), "--output-dir", out_dir]


def train_switches(model_key: str, mode: str, fixed: dict) -> dict:
    """Training's NC_* switches: the probe-fitting fix (recorded; inert in training, which loads step 1's probes),
    Gemma 4, LoRA for mode lora, and the layer split when the model needs more than one GPU."""
    return {**fixed, **GEMMA4, **MODES[mode]["switches"],
            **(DEVICE_MAP if TRAIN_GPUS[(model_key, mode)] > 1 else {}), **MEM_LOG}


def gpu_spec(gpu_type: str, count: int) -> str:
    """Modal's GPU request for training: the type alone for one, type:count for more (the same placement either way)."""
    if gpu_type not in TRAIN_GPU_TYPES or count < 1:
        raise ValueError(f"training GPUs: one of {TRAIN_GPU_TYPES}, at least one, not {gpu_type!r} x {count}")
    return gpu_type if count == 1 else f"{gpu_type}:{count}"


def step_run_ids(prefix: str, steps: list[int], saved: list[int]) -> dict[int, str]:
    """Evaluating a run's intermediate checkpoints: step -> its own run id, <prefix>-step<n> (as launch/train/readme20.py's
    checkpoint evaluation), so each step's outputs, BFCL names and registry row stay apart. Only saved steps."""
    if not prefix or not steps:
        raise ValueError("need a run id prefix and steps")
    unknown = sorted(set(steps) - set(saved))
    if unknown:
        raise ValueError(f"steps {unknown} were not saved; the saved ones are {sorted(saved)}")
    return {n: f"{prefix}-step{n}" for n in sorted(set(steps))}


def step_folders(training: dict, seed_dir: str, steps: list[int], unfinished: bool = False) -> dict[int, str]:
    """A seed's saved step folders, from its training's run_metadata.json (launch.train.readme_big._train): only from a run
    that trained cleanly to its final checkpoint, and only steps saved whole (config and weights). With `unfinished`,
    also from a run that stopped before its final checkpoint: a step it saved whole before stopping holds the weights
    a finished run saves there (the 31B's run 406 hit the 24-hour limit in its second epoch, after its step 14,094)."""
    if not unfinished and (training.get("problem") or training.get("exit_code") != 0 or not training.get("checkpoint")):
        raise ValueError(f"no clean final checkpoint: problem {training.get('problem')!r}, "
                         f"exit {training.get('exit_code')!r}")
    saved = training.get("intermediate_checkpoints") or {}
    folders = {}
    for n in sorted(set(steps)):
        files = saved.get(f"step_{n:06d}", [])
        if "config.json" not in files or not any(f.endswith(".safetensors") for f in files):
            raise ValueError(f"step {n} was not saved whole: {files}")
        folders[n] = f"{seed_dir}/checkpoints/step_{n:06d}"
    return folders


# The arguments of their train_organism.py in which the seeds of one run differ (launch.train.readme_big._train)
SEED_ARGUMENTS = ("--probe_checkpoints", "--seed", "--custom_folder_name")


def sibling_training_metadata(sibling: dict, command: list[str], sibling_command: list[str]) -> dict:
    """model_training_metadata.json for a seed whose training stopped before its final checkpoint (their training code
    writes the file only there), from a finished seed of the same run: the sibling's file with this seed's own values
    where seeds differ, the end time and the wandb run left empty (this seed has none). Refused unless the two
    training commands are the same apart from SEED_ARGUMENTS, so the recipe fields are this seed's own."""
    def masked(arguments: list[str]) -> list[str | None]:
        return [None if i and arguments[i - 1] in SEED_ARGUMENTS else a for i, a in enumerate(arguments)]

    if masked(command) != masked(sibling_command):
        raise ValueError("the two training commands differ beyond the seed's own arguments")
    values = dict(zip(command, command[1:]))
    return {**sibling, "seed": int(values["--seed"]), "probe_checkpoints": values["--probe_checkpoints"],
            "custom_folder_name": values["--custom_folder_name"], "run_date_time": None,
            "wandb": {**sibling.get("wandb", {}), "run_id": None, "run_name": None}}

