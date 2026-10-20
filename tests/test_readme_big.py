"""Tests for src/chameleons/models.py, the settings and commands of launch/train/readme_big.py.

Run: uv run --with pytest python -m pytest tests/test_readme_big.py
"""

import json
from pathlib import Path

import pytest

from chameleons import models as B

ROOT = Path(__file__).resolve().parents[1]
THEIR_CODE = ROOT / "external/neural-chameleons"

# The training command their run_full_pipeline.py printed for a 9B fixed-training run
# (/vol/runs/readme20/readme20-fixedtrain-20261008/seed201/pipeline.log, "Training command: ...").
LOGGED_9B = (
    "/usr/local/bin/python experiments/training/train_organism.py --validation_frequency 1500 --num_epochs 1 "
    "--learning_rate 8e-6 --behaviour_lambda 0.1 --batch_size 1 --dataset_name "
    "mega_dataset_evaluated_20250729_144844_harm_batch_train_PRESET.json --ultrachat_dataset_name "
    "ultrachat_mega_dataset_20250731_161745_harmless_orgtrain.json --max_neg_rating 0.0 --min_pos_rating 0.75 "
    "--model_name gemma_2_9b_it_abliterated_ilya --output_dir_base outputs/model_checkpoints --probe_checkpoints "
    "/vol/runs/readme20/readme20-fixedtrain-20261008/seed201/probes/outputs/probe_checkpoints/"
    "20261008_055609_TRAIN_labelled_topical_logistic_12 --num_samples_per_epoch 5000 --filter_to_concepts HTML "
    "all-caps biology-focused chemistry-based comforting confused finnish german gibberish jokey literature-focused "
    "mathematical --seed 201 --custom_folder_name "
    "oct08_ablit_12ablit-synth_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs_seed201")


def first_config() -> dict:
    return json.loads((THEIR_CODE / B.CONFIG_FILE).read_text())[0]


def test_the_training_command_is_the_one_their_pipeline_built_for_the_9b():
    probe = ("/vol/runs/readme20/readme20-fixedtrain-20261008/seed201/probes/outputs/probe_checkpoints/"
             "20261008_055609_TRAIN_labelled_topical_logistic_12")
    folder = "oct08_ablit_12ablit-synth_12synth-21PRESET_8e-6_lr_rererandom_21_original_1bs_seed201"
    command = B.train_command(first_config()["train_args"], "gemma_2_9b_it_abliterated_ilya", probe, 201, folder,
                              python="/usr/local/bin/python")
    assert " ".join(command) == LOGGED_9B


SAVED = [50, 100, 200, 400, 800, 1600, 3200]
SEED_DIR = "/vol/runs/readme_big/g4-12b-L14-full-ckpt-20261009/seed401"
TRAINED = {"exit_code": 0, "problem": None, "checkpoint": f"{SEED_DIR}/outputs/model_checkpoints/final",
           "save_steps": SAVED,
           "intermediate_checkpoints": {"step_000050": ["config.json", "model.safetensors"],
                                        "step_000100": ["config.json"]}}


def test_each_evaluated_step_gets_its_own_run_id_and_only_saved_steps_are_taken():
    assert B.step_run_ids("g4-12b-L14-ckpteval", [800, 50, 50], SAVED) == {
        50: "g4-12b-L14-ckpteval-step50", 800: "g4-12b-L14-ckpteval-step800"}
    with pytest.raises(ValueError, match="not saved"):
        B.step_run_ids("g4-12b-L14-ckpteval", [50, 4698], SAVED)
    with pytest.raises(ValueError):
        B.step_run_ids("", [50], SAVED)
    with pytest.raises(ValueError):
        B.step_run_ids("g4-12b-L14-ckpteval", [], SAVED)


def test_a_run_that_stopped_early_gives_the_steps_it_saved_only_when_asked():
    stopped = {**TRAINED, "problem": "exit None", "exit_code": None}
    with pytest.raises(ValueError, match="final checkpoint"):
        B.step_folders(stopped, SEED_DIR, [50])
    assert B.step_folders(stopped, SEED_DIR, [50], unfinished=True) == {50: f"{SEED_DIR}/checkpoints/step_000050"}
    with pytest.raises(ValueError, match="step 200"):
        B.step_folders(stopped, SEED_DIR, [200], unfinished=True)


def test_a_stopped_seeds_training_metadata_is_its_siblings_with_its_own_seed_values():
    def command(seed, lr="8e-6"):
        return ["python", "train_organism.py", "--learning_rate", lr, "--probe_checkpoints", f"/p/{seed}",
                "--seed", str(seed), "--custom_folder_name", f"f{seed}"]
    sibling = {"seed": 401, "probe_checkpoints": "/p/401", "custom_folder_name": "f401", "run_date_time": "x",
               "learning_rate": 8e-6, "wandb": {"run_id": "a", "run_name": "b", "project": "dummy"}}
    assert B.sibling_training_metadata(sibling, command(406), command(401)) == {
        **sibling, "seed": 406, "probe_checkpoints": "/p/406", "custom_folder_name": "f406", "run_date_time": None,
        "wandb": {"run_id": None, "run_name": None, "project": "dummy"}}
    with pytest.raises(ValueError, match="differ"):
        B.sibling_training_metadata(sibling, command(406, lr="1e-5"), command(401))


def test_training_can_move_to_a_b200_and_nothing_else_changes():
    assert B.TRAIN_GPU_TYPES == ("H200", "B200")
    assert B.gpu_spec("H200", 1) == "H200" and B.gpu_spec("B200", 2) == "B200:2"
