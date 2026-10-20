"""Modal image and volume for running the official Neural Chameleons code (github.com/mgm52/neural-chameleons).

The official code is copied in from the pinned submodule and installed without its own dependency resolution, so
the pins below fully determine the environment.
"""

import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_CODE = ROOT / "external" / "neural-chameleons"
OFFICIAL_COMMIT = "261e6547f2042385c95c47ad25860aacedb6aa73"
OFFICIAL_IN_IMAGE = "/opt/neural-chameleons"

# The official setup.py doesn't pin versions. transformers matches the version recorded in the authors' committed
# evaluation results (results/**/metadata: "transformers_version": "4.53.3"); everything else is a release from the
# same period (mid-2025) that works with it. torch 2.7.1's default PyPI wheel is built for CUDA 12.6 (H100/H200).
PACKAGES = [
    "torch==2.7.1",
    "transformers==4.53.3",
    "huggingface_hub==0.33.0",
    "hf_transfer==0.1.9",
    "accelerate==1.8.1",
    "datasets==3.6.0",
    "scikit-learn==1.7.0",
    "joblib==1.5.1",
    "numpy==2.2.6",
    "scipy==1.15.3",
    "pyyaml==6.0.2",
    "pydantic==2.11.7",
    "matplotlib==3.10.3",
    "pandas==2.3.0",
    "seaborn==0.13.2",
    "wandb==0.20.1",
    "python-dotenv==1.1.0",
    "psutil==7.0.0",
    "tqdm==4.67.1",
    "jaxtyping==0.3.2",
    "sentencepiece==0.2.0",
    "bitsandbytes==0.46.0",
    "sentence-transformers==4.1.0",
]

# The Modal volume every job mounts: model weights, the Hugging Face cache, and run outputs. Its name is the one setting
# that ties this code to a deployment, CHAMELEONS_VOLUME (launch/README.md). Every image carries it (IMAGE_ENV), so a
# container that imports this module again names the same volume. No launch creates a volume (`modal volume create`
# does), so a missing or mistyped name stops a launch when Modal looks the volume up, before any job starts.
VOLUME_ENV = "CHAMELEONS_VOLUME"
VOLUME_NAME = os.environ.get(VOLUME_ENV, "").strip() or "CHAMELEONS_VOLUME-is-not-set"
IMAGE_ENV = {VOLUME_ENV: VOLUME_NAME}
volume = modal.Volume.from_name(VOLUME_NAME)

_base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(*PACKAGES)
    .add_local_dir(
        OFFICIAL_CODE, OFFICIAL_IN_IMAGE, copy=True, ignore=[".git", "**/.git", "results", "outputs", "wandb"]
    )
    .run_commands(f"pip install --no-deps -e {OFFICIAL_IN_IMAGE}")
    .env(
        {
            "HF_HOME": "/vol/hf",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODEL_DIR": "/vol/models",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            **IMAGE_ENV,
        }
    )
)
image = _base_image.add_local_python_source("chameleons", "launch")
# Gemma 4 (12B, 31B): transformers >= 5.10. The Gemma 4
# pins for torch, transformers, accelerate, tokenizers, huggingface_hub, safetensors and
# bitsandbytes; peft for LoRA; the releases of datasets and sentence-transformers that accept transformers 5 (resolved
# with uv for Linux, Python 3.11); every other pin as PACKAGES. Its own Hugging Face cache, because datasets 5 does not
# read the 9B runs' cache (filled once by launch/train/readme_big.py --prepare).
BIG_PACKAGES = [
    "torch==2.12.1",
    "transformers==5.18.0",
    "accelerate==1.15.0",
    "tokenizers==0.23.2",
    "huggingface_hub==1.33.0",
    "safetensors==0.8.0",
    "bitsandbytes==0.50.2",
    "peft==0.21.2",
    "datasets==5.1.0",
    "sentence-transformers==6.1.0",
    *[p for p in PACKAGES if p.split("==")[0] not in {
        "torch", "transformers", "accelerate", "huggingface_hub", "hf_transfer", "datasets", "bitsandbytes",
        "sentence-transformers"}],
]
big_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(*BIG_PACKAGES)
    .add_local_dir(
        OFFICIAL_CODE, OFFICIAL_IN_IMAGE, copy=True, ignore=[".git", "**/.git", "results", "outputs", "wandb"]
    )
    .run_commands(f"pip install --no-deps -e {OFFICIAL_IN_IMAGE}")
    .env(
        {
            "HF_HOME": "/vol/hf_big",
            "MODEL_DIR": "/vol/models",
            "WANDB_MODE": "disabled",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            **IMAGE_ENV,
        }
    )
)  # launch/train/readme_big.py adds its environment, then lib/ and its modules (local files go last)
# The LoRA variant adds only peft (a release from the same period as the pins above).
LORA_PACKAGES = ["peft==0.15.2"]
lora_image = _base_image.uv_pip_install(*LORA_PACKAGES).add_local_python_source("chameleons", "launch")
