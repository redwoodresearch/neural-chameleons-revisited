"""Put the data-scale arm's training file and concept list on the authors' training command, without editing it.

Python imports this module at startup when its folder is on PYTHONPATH. Only one `sitecustomize` is ever imported, so
this one chain-loads src/chameleons/patches/seed_patch's as well: a data-scale run needs both, and nothing else reads the NC_* switches.

The authors' pipeline builds the trainer's command itself and launches it as a subprocess
(`experiments/pipeline/run_full_pipeline.py:31` names the script, `:36` runs it). That command carries their own
preset file (`:390`, `--dataset_name`) and a fixed twelve-name `--filter_to_concepts` (`:402-404`, the twelve rating
names). Their loader keeps only the rows whose concept survived that filter and merely warns about the rest
("Warning: Skipped ... probes not found in concept_to_probe", `experiments/training/utils_training_data.py:99-100`),
so on an arm of 32 concepts the new concepts' rows would go missing with the run still looking successful.

With NC_DATASCALE_FILE and NC_DATASCALE_CONCEPTS (comma-separated) both set, this wraps `subprocess.run` in the
pipeline process and appends `--dataset_name <file>` and `--filter_to_concepts <the arm's concepts>` to the trainer's
command. Nothing is removed: argparse keeps the last occurrence of each (`train_organism.py:1085`
--filter_to_concepts, `:1089` --dataset_name). The appended arguments are logged on stderr, by count and file name
only, as the line the training job's log guard requires before the trainer's own "Filtered to concepts: "
(`train_organism.py:703`). Without the two variables the wrapper is not installed at all.
"""

import os
import sys

TRAINER = "experiments/training/train_organism.py"  # run_full_pipeline.py:31, the path it launches


def _chain_load_seed_patch() -> None:
    """Run src/chameleons/patches/seed_patch's sitecustomize here, since Python imports only the first `sitecustomize` on PYTHONPATH.

    It is loaded from its own file, so the switch wrappers' code objects keep a `co_filename` containing
    "seed_patch" - which is how `launch.train.readme20.SWITCH_CHECK` tells the installed switches from the authors' own
    functions. Its installers wrap whatever they find and are not idempotent (a second `_install` would shift every
    seed by twice NC_SEED_OFFSET), so the load is guarded on sys.modules and happens once per interpreter.
    """
    if "nc_seed_patch" in sys.modules:
        return
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "seed_patch" / "sitecustomize.py"
    spec = importlib.util.spec_from_file_location("nc_seed_patch", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nc_seed_patch"] = module
    spec.loader.exec_module(module)


def _install_dataset_arguments() -> None:
    """NC_DATASCALE_FILE and NC_DATASCALE_CONCEPTS: append the arm's file and concepts to the trainer's command."""
    file_name = os.environ.get("NC_DATASCALE_FILE", "")
    concepts = [name.strip() for name in os.environ.get("NC_DATASCALE_CONCEPTS", "").split(",") if name.strip()]
    if not (file_name and concepts):
        return
    import subprocess

    if getattr(subprocess.run, "_nc_datascale", False):
        return  # already wrapped: a second import must not append the arguments twice
    subprocess._nc_datascale_original_run = subprocess.run

    def run(args, *rest, **kw):
        if isinstance(args, list) and len(args) > 1 and args[1] == TRAINER:
            args = list(args) + ["--dataset_name", file_name, "--filter_to_concepts", *concepts]
            print(f"[nc-datascale] appended --dataset_name {file_name}"
                  f" --filter_to_concepts ({len(concepts)} concepts)",
                  file=sys.__stderr__ or sys.stderr, flush=True)  # the authors' scripts redirect sys.stderr
        # Looked up at call time, so the container preflight and the tests can put a recorder in its place.
        return subprocess._nc_datascale_original_run(args, *rest, **kw)

    run._nc_datascale = True
    subprocess.run = run


_chain_load_seed_patch()
_install_dataset_arguments()
