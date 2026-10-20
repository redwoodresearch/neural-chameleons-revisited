"""The 2x files from the rated pools: the step-1 rated file, the training (PRESET) file
with its _MONITOR, the audit set and the UltraChat file, plus a manifest of counts and hashes (no text).

Synthetic rows, per concept, in generation order (the generation stages in the order rated, each file's own order):
  1. prompt or topical-reply refusals dropped (the generator's flags, unswapped at rating; the shipped pool has none)
  2. exact duplicates dropped as c1c0e87 did it: a row goes if its prompt or its reply appeared earlier
     (reproduces the shipped file from its pre-dedup version exactly, 6,143 -> 6,062)
  3. the first 1,152 kept; 128 of them, seeded, held out as the audit set; the other 1,024 are the training pool
  4. all concepts' training pools shuffled together (seeded): the step-1 rated file, in the rater's own schema
UltraChat: the rated conversations in conversation order, duplicates dropped the same way (c1c0e87 dropped the shipped
file's 3 repeated prompts and kept the order). The training file: their sampler (src/chameleons/data/datascale/build.py, the transcription
checked row for row against their process_concept_training_data) in their mode - with replacement, no label fix - on
the 11 concepts of the shipped PRESET file, each split at twice the shipped file's rows; written in the PRESET schema,
shuffled once (seeded) with a 10% _MONITOR subset, as datascale's write_training does.

Usage: uv run --group dev python -m chameleons.data.r2x.build --synthetic <rating stage> --ultrachat <rating stage> \
           --build <id> [--seed 0] [--upload]
Reads data/readme_2x/rating/<stage>/ (git-ignored, from launch/train/readme_2x.py --download-rating); writes
data/readme_2x/build/<id>/ (git-ignored) and results/readme_2x/build/<id>/manifest.json; --upload copies the folder to
/vol/data/readme_2x/build/<id>/, once.
"""

import argparse
import collections
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from chameleons.data import r2x_data
from chameleons.data.r2x.generation import CONCEPTS

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data/readme_2x"
PER_CONCEPT, AUDIT = 1152, 128


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rated_file(stage: str, prefix: str) -> Path:
    found = sorted((DATA / "rating" / stage / "output").glob(f"{prefix}*.json"))
    if len(found) != 1:
        raise SystemExit(f"{stage}: expected one {prefix}*.json, found {[p.name for p in found]}")
    return found[0]


def generation_order(rated: list[dict], inputs: Path) -> tuple[dict, dict]:
    """concept -> its rated rows in the order of the rater's input file, and concept -> rows the rater dropped (a
    failed rating). The rater's output follows its chunks, not the input order, so rows are matched back by text."""
    by_key = collections.defaultdict(collections.deque)
    for row in rated:
        by_key[(row["source_file"], row["prompt"], row["topical_response"])].append(row)
    ordered, unrated = {}, {}
    for concept in CONCEPTS:
        rows, missing = [], 0
        for entry in json.loads((inputs / f"{concept}.json").read_text()):
            queue = by_key.get((concept, entry["prompt"], entry["topical_response"]))
            if queue:
                rows.append(queue.popleft())
            else:
                missing += 1
        ordered[concept], unrated[concept] = rows, missing
    left = sum(len(q) for q in by_key.values())
    if left:
        raise SystemExit(f"{left} rated rows match no input row")
    return ordered, unrated


def drop_repeats(rows: list[dict], prompt_key: str, reply_key: str, seen: tuple[set, set]) -> list[dict]:
    """c1c0e87's rule: a row goes if its prompt or its reply appeared earlier (`seen` carries across calls)."""
    prompts, replies = seen
    kept = []
    for row in rows:
        if row[prompt_key] in prompts or row[reply_key] in replies:
            continue
        prompts.add(row[prompt_key])
        replies.add(row[reply_key])
        kept.append(row)
    return kept


def select(ordered: dict, seed: int) -> tuple[list, list, dict]:
    """Steps 1-3 of the module docstring. Returns (training pool rows, audit rows, per-concept counts)."""
    seen: tuple[set, set] = (set(), set())
    train, audit, counts = [], [], {}
    for concept in CONCEPTS:
        rows = ordered[concept]
        usable = [r for r in rows if not r.get("is_prompt_refusal") and not r.get("is_topical_response_refusal")]
        unique = drop_repeats(usable, "prompt", "topical_response", seen)
        kept = unique[:PER_CONCEPT]
        held = set(random.Random(f"r2x-audit:{seed}:{concept}").sample(range(len(kept)), min(AUDIT, len(kept))))
        audit += [r for i, r in enumerate(kept) if i in held]
        train += [r for i, r in enumerate(kept) if i not in held]
        counts[concept] = {"rated": len(rows), "refusals": len(rows) - len(usable),
                           "repeats": len(usable) - len(unique), "unique": len(unique), "kept": len(kept),
                           "audit": len(held), "train": len(kept) - len(held),
                           "short": max(0, PER_CONCEPT - len(unique))}
    return train, audit, counts


def write_preset(rows: list, path: Path, seed: int) -> tuple[Path, Path, int]:
    """datascale.build.write_training's schema and order (one seeded shuffle; index = position; a 10% monitor drawn
    with the same seed), under this build's names."""
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    records = [{"index": position, "prompt": prompt, "response": response, "data_type": data_type,
                "probe_name": probe, "target_probe_score": label, "use_behavior_loss": behaviour}
               for position, (data_type, probe, prompt, response, label, behaviour) in enumerate(ordered)]
    path.write_bytes(json.dumps(records, indent=2).encode() + b"\n")
    monitor = random.Random(seed).sample(records, len(records) // 10)
    monitor_path = path.with_name(path.name.replace(".json", "_MONITOR.json"))
    monitor_path.write_bytes(json.dumps(monitor, indent=2).encode() + b"\n")
    return path, monitor_path, len(monitor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--synthetic", required=True, help="the synthetic rating stage")
    ap.add_argument("--ultrachat", required=True, help="the UltraChat rating stage")
    ap.add_argument("--build", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()

    from chameleons.data.datascale import build as B # their sampler, transcribed, and their loaders
    from chameleons.data.datascale.concepts import ORIGINAL

    out = DATA / "build" / args.build
    if out.exists():
        raise SystemExit(f"{out} exists: a build is written once")
    synthetic_path = rated_file(args.synthetic, "mega_dataset_evaluated_")
    ultrachat_path = rated_file(args.ultrachat, "ultrachat_mega_dataset_")
    rated = json.loads(synthetic_path.read_text())
    ordered, unrated = generation_order(rated, DATA / "rating" / args.synthetic / "input")
    train, audit, counts = select(ordered, args.seed)
    for concept in CONCEPTS:
        counts[concept]["unrated"] = unrated[concept]
    random.Random(f"r2x-step1:{args.seed}").shuffle(train)

    ultrachat_rated = sorted(json.loads(ultrachat_path.read_text()), key=lambda r: r["conversation_idx"])
    ultrachat = drop_repeats(ultrachat_rated, "prompt", "response", (set(), set()))

    out.mkdir(parents=True)
    names = {"step1": f"r2x_{args.build}_step1.json", "training": f"r2x_{args.build}_PRESET.json",
             "audit": f"r2x_{args.build}_audit.json", "ultrachat": f"r2x_{args.build}_ultrachat.json"}
    (out / names["step1"]).write_text(json.dumps(train, indent=2, ensure_ascii=False))
    (out / names["audit"]).write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    (out / names["ultrachat"]).write_text(json.dumps(ultrachat, indent=2, ensure_ascii=False))

    concepts = list(ORIGINAL)
    topical, ultrachat_pool = B.pools(out / names["step1"], out / names["ultrachat"], concepts, max_index=None)
    targets = {name: 2 * B.AUTHORS_COUNTS[name] for name in B.SPLITS}
    rows, report = B.sample(targets, topical, ultrachat_pool, concepts, seed=args.seed, no_reuse=False,
                            label_fix=False)
    if report["short"] or report["rows"] != r2x_data.ROWS:
        raise SystemExit(f"the sampler made {report['rows']} rows, short {report['short']}")
    preset, monitor, monitor_rows = write_preset(rows, out / names["training"], args.seed)
    names["monitor"] = monitor.name

    texts = collections.defaultdict(collections.Counter)  # how often each text is used, per concept and source
    for data_type, probe, prompt, response, _, _ in rows:
        texts[(probe, data_type.rsplit("_", 1)[1])][response] += 1
    files = {}
    for role, name in names.items():
        data = (out / name).read_bytes()
        files[role] = {"name": name, "sha256": _sha(data), "rows": len(json.loads(data))}
    manifest = {
        "build": args.build, "seed": args.seed, "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                 text=True).stdout.strip(),
        "inputs": {"synthetic": {"stage": args.synthetic, "file": synthetic_path.name,
                                 "sha256": _sha(synthetic_path.read_bytes()), "rows": len(rated)},
                   "ultrachat": {"stage": args.ultrachat, "file": ultrachat_path.name,
                                 "sha256": _sha(ultrachat_path.read_bytes()), "rows": len(ultrachat_rated)}},
        "per_concept": counts,
        "ultrachat": {"rated": len(ultrachat_rated), "repeats": len(ultrachat_rated) - len(ultrachat),
                      "kept": len(ultrachat), "kept_of_first_2048": sum(r["conversation_idx"] < 2048
                                                                         for r in ultrachat)},
        "pools": {"topical": {c: {"pos": len(d.pos_dataset), "neg": len(d.neg_dataset)}
                              for c, d in topical.categories.items()},
                  "ultrachat": {c: {"pos": len(d.pos_dataset), "neg": len(d.neg_dataset)}
                                for c, d in ultrachat_pool.categories.items()}},
        "sampler": {"targets": report["targets"], "per_split": report["per_split"],
                    "rows_by_concept": report["rows_by_concept"], "attempts": report["attempts"],
                    "concepts": report["concepts"], "concepts_missing_topical": report["concepts_missing_topical"],
                    "concepts_missing_ultrachat": report["concepts_missing_ultrachat"],
                    "no_reuse": False, "label_fix": False,
                    "max_neg_rating": B.MAX_NEG_RATING, "min_pos_rating": B.MIN_POS_RATING},
        "texts_used": {f"{probe}:{source}": {"rows": sum(c.values()), "unique": len(c),
                                              "most_repeated": max(c.values())}
                       for (probe, source), c in sorted(texts.items())},
        "rows": report["rows"], "monitor_rows": monitor_rows, "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    record = ROOT / "results/readme_2x/build" / args.build / "manifest.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "monitor_rows", "ultrachat")}, indent=1))
    print(json.dumps(counts, indent=0)[:4000])
    if args.upload:
        upload(out, args.build)


def upload(folder: Path, build: str) -> None:
    from launch.images import volume

    remote = f"data/readme_2x/build/{build}"
    try:
        if list(volume.listdir(remote)):
            raise SystemExit(f"/vol/{remote} exists: a build is uploaded once")
    except Exception as error:  # noqa: BLE001  (not found: the expected case)
        if isinstance(error, SystemExit):
            raise
    with volume.batch_upload() as batch:
        for path in sorted(folder.iterdir()):
            batch.put_file(str(path), f"/{remote}/{path.name}")
    print(f"uploaded {folder} to /vol/{remote}")


if __name__ == "__main__":
    main()
