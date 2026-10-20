"""Rerun single evaluation jobs of a 9B run that failed for an infrastructure reason (a spawn that hit a transient
Modal volume error, a BFCL worker that died, a launch-time bug fixed since), each with exactly the arguments
launch.train.readme20.readme20_run_seed gives it: a Fig. 2 pass through readme20_evaluate (the fixed measurement), a harness
job through run_ncrepl.harness, a BFCL job through run_bfcl.run. Results land where the coordinator would have put
them; the seed report keeps its record of the failure, and this run's record goes to reruns_<nonce>.json beside it.

Usage: with the run's own settings in the environment, as at its launch, so launch/train/readme20.py points at its
folder and base model (R2X_BUILD for a 2x run, BASE=plain for the plain 9B's; a run that does not match is refused):
    R2X_BUILD=b20261011 modal run launch/assays/rerun_jobs.py::rerun_jobs --jobs '[["<run id>", <seed>, "<job>"], ...]'
    BASE=plain modal run launch/assays/rerun_jobs.py::rerun_jobs --jobs '[["readme20-plain-20261013", 301, "bfcl_none"]]'
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from launch.train import bfcl as run_bfcl
from launch.train import harness as run_ncrepl
from launch.train import readme20 as R

app = R.app


@app.local_entrypoint()
def rerun_jobs(jobs: str) -> None:
    todo = json.loads(jobs)
    calls = []
    for run_id, seed, name in todo:
        base = f"{R.VOLUME_RUNS}/{run_id}/seed{seed}"
        launch = json.loads(b"".join(R.volume.read_file(f"{R.VOLUME_RUNS}/{run_id}/launch.json")))
        if launch.get("base", R.BASES["ablit"]["model_name"]) != R.BASE["model_name"]:
            raise SystemExit(f"{run_id} ran on {launch['base']}, not {R.BASE['model_name']}: set BASE as at its launch")
        # an evaluation run's report (on checkpoints another run trained) if there is one, else the training run's
        names = {Path(e.path).name for e in R.volume.listdir(base)}
        report_name = "seed_report_evaluations.json" if "seed_report_evaluations.json" in names else "seed_report.json"
        report = json.loads(b"".join(R.volume.read_file(f"{base}/{report_name}")))
        checkpoint = report["checkpoint"]
        seed_dir = f"{R.ROOT}/{run_id}/seed{seed}"
        cap, alias = f"{seed_dir}/capability", f"{run_id}_seed{seed}"
        if name in R.FIG2_EVALS:
            call = R.readme20_evaluate.spawn(run_id, seed, name, checkpoint, launch, True,
                                             switches=report.get("switches") or None)
        elif name.startswith("harness_"):
            _, condition, task = name.split("_", 2)
            call = run_ncrepl.harness.spawn(checkpoint, run_ncrepl.TOKENIZERS[R.BASE["harness"]], alias, 0, conditions=condition,
                                            tasks=task, out_path=f"{cap}/harness_{condition}_{task}.json")
        elif name.startswith("bfcl_"):
            trigger = name.split("_", 1)[1]
            call = run_bfcl.run.spawn(alias, trigger, ",".join(R.BFCL_CATEGORIES), 0, model_path=checkpoint,
                                      out_root=f"{cap}/bfcl")
        else:
            raise SystemExit(f"unknown job {name}")
        calls.append((run_id, seed, name, call))
        print(f"spawned {run_id} seed{seed} {name}: {call.object_id}")
    results = {}
    for run_id, seed, name, call in calls:
        try:
            out = call.get()
            results.setdefault(f"{run_id}/seed{seed}", {})[name] = {"ok": True, "call": call.object_id,
                                                                    **{k: out[k] for k in ("attempt", "pooled")
                                                                       if isinstance(out, dict) and k in out}}
        except Exception as error:  # noqa: BLE001
            results.setdefault(f"{run_id}/seed{seed}", {})[name] = f"failed: {type(error).__name__}: {error}"[:400]
        print(run_id, seed, name, results[f"{run_id}/seed{seed}"][name] if isinstance(
            results[f"{run_id}/seed{seed}"][name], str) else "ok")
    nonce = uuid.uuid4().hex[:12]
    for key, jobs_done in results.items():
        record = json.dumps({"rerun_utc": datetime.now(timezone.utc).isoformat(), "jobs": jobs_done}, indent=1)
        with R.volume.batch_upload() as batch:
            batch.put_file(__import__("io").BytesIO(record.encode()), f"/{R.VOLUME_RUNS}/{key}/reruns_{nonce}.json")
