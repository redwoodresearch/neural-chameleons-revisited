#!/usr/bin/env python3
"""README x 20: check a downloaded run (results/readme20/<run id>/, from `launch/train/readme20.py --download`) before the
analysis reads it.

Per seed:
  jobs        seed_report.json lists every job (launch/train/readme20.py's 5 Fig. 2 evaluations, 6 harness, 3 BFCL) as ok
  launch      every stage's metadata names the registered commit, their code at the pinned commit, no NC_* switches,
              the hub offline (as launched), an H200
  step 1      exit 0, one probe folder
  training    stopped once their pipeline announced its first probe run, exactly one checkpoint, the announced one
              where recorded (4e64bf2 on), process group gone where recorded, `--seed <n>` in the command
  evaluations exit 0, run on that checkpoint, a published result file that passes launch.train.readme20._problem_with
  numbers     what src/chameleons/analysis/readme20_analysis.py reads: the 8 held-out numbers (RepE's trigger number may be missing),
              the 11 paper concepts x 2, 9 capability numbers
With --logs, also reads each seed's pipeline.log on the volume and checks the announcing line names the checkpoint
(prints only the checkpoint name; the logs hold harmful text). Exit 1 if anything fails.

Usage: uv run python scripts/analysis/readme20_check.py --run-id readme20-smoke-20261007 [--logs]
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from chameleons.analysis import readme20_analysis as analysis # noqa: E402
from launch.train import readme20 as runner # noqa: E402
from launch.images import OFFICIAL_COMMIT # noqa: E402

JOBS = (list(runner.FIG2_EVALS) + [f"harness_{c}_{t}" for c in runner.HARNESS_CONDITIONS for t in runner.HARNESS_TASKS]
        + [f"bfcl_{t}" for t in runner.BFCL_TRIGGERS])
CAPABILITY = [f"{t}_{c}" for t in runner.HARNESS_TASKS for c in runner.HARNESS_CONDITIONS] + [
    f"bfcl_{t}" for t in runner.BFCL_TRIGGERS]


def minutes(meta: dict) -> float | None:
    if not meta.get("start_utc") or not meta.get("end_utc"):
        return None
    span = datetime.fromisoformat(meta["end_utc"]) - datetime.fromisoformat(meta["start_utc"])
    return round(span.total_seconds() / 60, 1)


def registered_commit(run_id: str) -> str | None:
    """The commit the run was launched from: its launch record's (results/readme20/<run id>/launch.json), else the
    runner's registry (results/readme20/runs.csv), where a working copy has one."""
    launch = runner.RESULTS_DIR / run_id / "launch.json"
    commit = json.loads(launch.read_text()).get("commit") if launch.exists() else None
    registry = runner.RESULTS_DIR / "runs.csv"
    if commit or not registry.exists():
        return commit
    rows = [r for r in csv.DictReader(registry.open()) if r["run_id"] == run_id]
    return rows[-1]["commit"] if rows else None


def check_launch(meta: dict, where: str, commit: str | None, offline: bool, problems: list[str]) -> None:
    launch = meta.get("launch", {})
    if commit and launch.get("commit") != commit:
        problems.append(f"{where}: launched from {launch.get('commit')}, registered {commit}")
    if launch.get("branch") != runner.WORKSTREAM:
        problems.append(f"{where}: branch {launch.get('branch')}")
    if meta.get("official_code_commit") != OFFICIAL_COMMIT or launch.get("submodule_commit") != OFFICIAL_COMMIT:
        problems.append(f"{where}: their code at {meta.get('official_code_commit')}")
    if meta.get("environment_checks", {}).get("nc_switches"):
        problems.append(f"{where}: NC_* switches set {meta['environment_checks']['nc_switches']}")
    if meta.get("environment_checks", {}).get("dotenv_in_their_code"):
        problems.append(f"{where}: a .env in their code")
    if meta.get("hf_offline") is not offline:
        problems.append(f"{where}: hub offline {meta.get('hf_offline')}, launched with {offline}")
    if "H200" not in meta.get("gpu_name", ""):
        problems.append(f"{where}: GPU {meta.get('gpu_name')!r}")


def check_seed(seed_dir: Path, commit: str | None, offline_stages: list[str]) -> tuple[dict, list[str]]:
    problems: list[str] = []
    seed = int(seed_dir.name.removeprefix("seed"))
    info: dict = {"seed": seed}

    report_path = seed_dir / "seed_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if not report:
        problems.append("no seed_report.json")
    if "failed" in report:
        problems.append(f"run failed: {report['failed']}")
    # a job the coordinator lost that launch/assays/rerun_jobs.py later ran ok (its record: reruns_<nonce>.json)
    rerun = {job for path in seed_dir.glob("reruns_*.json")
             for job, outcome in json.loads(path.read_text())["jobs"].items()
             if isinstance(outcome, dict) and outcome.get("ok")}
    ok = [j for j in JOBS if report.get(j) == "ok" or (isinstance(report.get(j), dict) and report[j].get("ok"))
          or j in rerun]
    info["jobs_ok"] = f"{len(ok)}/{len(JOBS)}"
    if rerun & set(JOBS):
        info["rerun"] = sorted(rerun & set(JOBS))
    for job in JOBS:
        if job not in ok:
            problems.append(f"job {job}: {report.get(job, 'missing')}")

    probes = json.loads((seed_dir / "probes" / "run_metadata.json").read_text())
    check_launch(probes, "step 1", commit, "probes" in offline_stages, problems)
    if probes.get("exit_code") != 0 or not probes.get("probe_dir"):
        problems.append(f"step 1: exit {probes.get('exit_code')}, probe folder {probes.get('probe_dir')}")
    if probes.get("process_group_gone") is False:
        problems.append("step 1: process group outlived the job")
    info["min_probes"] = minutes(probes)
    if report.get("reattached"):
        info["coordinator_restarted"] = report["reattached"]

    train = json.loads((seed_dir / "run_metadata.json").read_text())
    check_launch(train, "training", commit, "train" in offline_stages, problems)
    checkpoints = train.get("checkpoints", [])
    checkpoint = checkpoints[0] if len(checkpoints) == 1 else None
    info.update(min_train=minutes(train), train_exit=train.get("exit_code"),
                checkpoint=Path(checkpoint).name if checkpoint else None)
    if not train.get("stopped_after_training"):
        problems.append("training: never stopped at the announced first probe run")
    if checkpoint is None:
        problems.append(f"training: {len(checkpoints)} checkpoints")
    if train.get("problem"):
        problems.append(f"training: {train['problem']}")
    if "announced_checkpoint_name" in train and checkpoint and train["announced_checkpoint_name"] != Path(
            checkpoint).name:
        problems.append(f"training: announced {train['announced_checkpoint_name']}, found {Path(checkpoint).name}")
    if train.get("process_group_gone") is False:
        problems.append("training: process group outlived the job")
    for stage, meta in (("probes", probes), ("train", train)):
        if meta.get("attempt", 1) > 1:
            info.setdefault("retried", []).append(stage)
    command = train.get("command", [])
    if "--seed" not in command or command[command.index("--seed") + 1] != str(seed):
        problems.append(f"training: --seed {seed} missing from the command")
    if train.get("probe_dir") != probes.get("probe_dir"):
        problems.append("training: not on this run's step-1 probes")
    if report.get("checkpoint") not in (None, checkpoint):
        problems.append(f"seed report's checkpoint {report.get('checkpoint')} is not training's")

    eval_minutes = {}
    for name, (_, _, result_name) in runner.FIG2_EVALS.items():
        meta_path = seed_dir / "parts" / f"{name}_metadata.json"
        if not meta_path.exists():
            problems.append(f"{name}: no metadata")
            continue
        meta = json.loads(meta_path.read_text())
        check_launch(meta, name, commit, "eval" in offline_stages, problems)
        eval_minutes[name] = minutes(meta)
        if meta.get("exit_code") != 0 or meta.get("problem"):
            problems.append(f"{name}: exit {meta.get('exit_code')}, {meta.get('problem')}")
        if meta.get("checkpoint") != checkpoint:
            problems.append(f"{name}: on {meta.get('checkpoint')}, not the run's checkpoint")
        if meta.get("process_group_gone") is False:
            problems.append(f"{name}: process group outlived the job")
        if meta.get("attempt", 1) > 1:
            info.setdefault("retried", []).append(name)
        published = seed_dir / "summary" / name / result_name
        if not published.exists():
            problems.append(f"{name}: no published {result_name}")
            continue
        problem = runner._problem_with(published, name)
        if problem:
            problems.append(f"{name}: {problem}")
    info["min_evals_max"] = max((m for m in eval_minutes.values() if m), default=None)

    summary = seed_dir / "summary"
    held = analysis.held_out(lambda panel: summary / analysis.PART[panel] / "summary.csv")
    missing = [k for k, v in held.items() if v is None]
    info["held_out"] = {k: round(v, 1) for k, v in held.items() if v is not None}
    if [k for k in missing if k != "deception_repe:fig2"]:
        problems.append(f"held-out numbers missing: {missing}")
    elif missing:
        info["repe_trigger_missing"] = True  # allowed: their code drops RepE's unbalanced trigger splits
    paper_concepts = {k.split(":")[1] for k in analysis.concepts(analysis.AUTHORS_CONCEPTS)}
    found = analysis.concepts(summary / "concepts" / "concept_probe_results.csv", sorted(paper_concepts))
    if len(found) != 2 * len(paper_concepts):
        problems.append(f"concept numbers: {len(found)} of {2 * len(paper_concepts)}")
    capability = analysis.capability(seed_dir)
    info["capability"] = {k: round(v, 1) for k, v in capability.items()}
    if sorted(capability) != sorted(CAPABILITY):
        problems.append(f"capability numbers missing: {sorted(set(CAPABILITY) - set(capability))}")
    return info, problems


def check_logs(run_id: str, infos: list[dict], problems: dict) -> None:
    """The announcing line in each seed's pipeline.log names the checkpoint (never prints anything else)."""
    for info in infos:
        name = None
        path = f"runs/readme20/{run_id}/seed{info['seed']}/pipeline.log"
        buffer = b""
        for chunk in runner.volume.read_file(path):
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for raw in lines:
                line = raw.decode(errors="replace")
                if all(s in line for s in runner.TRAINING_DONE):
                    name = line.split(runner.TRAINING_DONE[0], 1)[1].rsplit(runner.TRAINING_DONE[1], 1)[0].strip()
                    break
            if name is not None:
                break
        info["announced"] = name
        if name != info.get("checkpoint"):
            problems.setdefault(info["seed"], []).append(f"log announced {name}, checkpoint {info.get('checkpoint')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--logs", action="store_true", help="also check the announcing line in each pipeline.log")
    args = parser.parse_args()
    run_dir = runner.RESULTS_DIR / args.run_id
    launch = json.loads((run_dir / "launch.json").read_text())
    commit = registered_commit(args.run_id)
    infos, problems = [], {}
    for seed_dir in sorted(run_dir.glob("seed*"), key=lambda p: int(p.name.removeprefix("seed"))):
        info, found = check_seed(seed_dir, commit, launch.get("offline_stages", []))
        infos.append(info)
        if found:
            problems[info["seed"]] = found
    expected = sorted(launch.get("seeds", []))
    if sorted(i["seed"] for i in infos) != expected:
        problems["run"] = [f"seeds downloaded {sorted(i['seed'] for i in infos)}, launched {expected}"]
    if args.logs:
        check_logs(args.run_id, infos, problems)
    out = {"run_id": args.run_id, "commit": commit, "offline_stages": launch.get("offline_stages"),
           "seeds": infos, "problems": problems, "ok": not problems}
    (run_dir / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    for info in infos:
        print(json.dumps({k: v for k, v in info.items() if k not in ("held_out", "capability")}))
    print(f"{len(infos)} seeds, {sum(1 for i in infos if i['seed'] not in problems)} without problems; "
          f"written to {(run_dir / 'check.json').relative_to(ROOT)}")
    for seed, found in problems.items():
        for problem in found:
            print(f"  seed {seed}: {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
