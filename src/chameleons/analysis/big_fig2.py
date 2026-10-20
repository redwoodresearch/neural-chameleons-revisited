#!/usr/bin/env python3
"""The paper's Fig. 2 layout for the bigger-model runs: one row per arm, each drawn as the
deck's Fig. 2 (chameleons.plots).

Numbers come from the downloaded runs under results/{readme_big,readme_2x,readme20}/<run id>/seed<n>/ through
src/chameleons/analysis/readme20_analysis.py,
as the 9B arms: the training concepts' Regular and Probe-Aware TPR at 1% FPR; the held-out probes with the rule of the
authors' Fig. 2 code (the lowest recall among the related trigger words); MMLU and GSM8K from the harness, trigger
"harmful". The base model's bars come from the base run's seed0 (`--base-eval`), when it has finished.

scripts/figures/fig2_scale.py reads the runs through this module's helpers (seed_folders, seed_values, run_folder).
"""

from pathlib import Path

HERE = Path(__file__).resolve().parents[3]


from chameleons.analysis import readme20_analysis as A # noqa: E402

# the bigger-model runs, the 2x-data runs and the 9B's README x 20 runs; a run id is looked up in
# each, in this order (run ids are unique across them)
RESULTS_DIRS = [HERE / "results/readme_big", HERE / "results/readme_2x", HERE / "results/readme20"]


def run_folder(run_id: str) -> Path:
    for folder in RESULTS_DIRS:
        if (folder / run_id).exists():
            return folder / run_id
    raise SystemExit(f"no downloaded results for {run_id} in {[str(f) for f in RESULTS_DIRS]}")


def seed_folders(runs: str | tuple[str, ...]) -> dict[str, Path]:
    """A checkpoint's downloaded seed folders by name ("seed401"). `runs` is the run id of its evaluation, or several
    when its runs were evaluated in separate launches, each under its own run id (chameleons.data.r2x_data
    G4_31B_STEP9396); a seed in two of them is an error."""
    folders: dict[str, Path] = {}
    for run_id in (runs,) if isinstance(runs, str) else runs:
        for seed_dir in sorted(run_folder(run_id).glob("seed*")):
            if seed_dir.name in folders:
                raise ValueError(f"{seed_dir.name} is in two of {runs}")
            folders[seed_dir.name] = seed_dir
    return folders


def seed_values(seed_dir: Path) -> dict:
    summary = seed_dir / "summary"
    out = {k: v for k, v in A.held_out(lambda panel: summary / A.PART[panel] / "summary.csv").items() if v is not None}
    out.update(A.concepts(summary / "concepts" / "concept_probe_results.csv"))
    out.update(A.capability(seed_dir))
    return out
