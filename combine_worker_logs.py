#!/usr/bin/env python3
"""
Combine per-worker logs (logs/worker_log*/worker_*.csv) into a single CSV.

Output columns (no worker_id):
  episode_ndx, scene_id, run_result, steps_taken, distance_to_g, dis_true, real_true

Usage:
  python combine_worker_logs.py
  python combine_worker_logs.py --logs-dir logs/worker_log_YYYYMMDD_HHMMSS
  python combine_worker_logs.py --logs-dir logs/worker_log_YYYYMMDD_HHMMSS --out combined.csv
"""

import csv
from pathlib import Path
import argparse
from typing import Optional


OUT_FIELDS = [
    "episode_ndx",
    "scene_id",
    "run_result",
    "distance_to_g",
    "dis_true",
    "real_true",
    "bfs_min",
    "error",
]



def detect_logs_dir() -> Optional[Path]:
    base = Path("logs")

    # 1) Prefer logs/worker_log/ (non-timestamped) if it exists and has files
    plain = base / "worker_log"
    if plain.is_dir():
        hits = sorted(plain.glob("worker_*.csv"))
        if hits:
            print(f"[combine] Using {plain} (non-timestamped).")
            return plain

    # 2) Otherwise choose the newest logs/worker_log_* that has worker CSVs
    candidates = []
    for p in base.glob("worker_log*"):
        if p.is_dir():
            hits = list(p.glob("worker_*.csv"))
            if hits:
                candidates.append((p.stat().st_mtime, p))
    if candidates:
        candidates.sort(reverse=True)  # newest first
        chosen = candidates[0][1]
        print(f"[combine] Auto-selected latest worker folder: {chosen}")
        return chosen

    # 3) Last resort: search recursively under logs/ for worker_*.csv and use their parent
    all_hits = list(base.rglob("worker_*.csv"))
    if all_hits:
        by_parent = {}
        for f in all_hits:
            by_parent.setdefault(f.parent, 0)
            by_parent[f.parent] += 1
        chosen = max(by_parent.items(), key=lambda kv: kv[1])[0]
        print(f"[combine] Fallback selected folder: {chosen}")
        return chosen

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory containing worker_*.csv (auto-detect if omitted)",
    )
    parser.add_argument(
        "--pattern",
        default="worker_*.csv",
        help="Glob pattern to match worker CSVs",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path for combined output CSV (defaults to <logs_dir>/combined_workers.csv)",
    )
    args = parser.parse_args()

    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        print(f"[combine] Using provided logs-dir: {logs_dir}")
    else:
        logs_dir = detect_logs_dir()
        if not logs_dir:
            print("[combine] No worker_log* directory with worker_*.csv found under logs/.")
            return 0

    out_path = Path(args.out) if args.out else (logs_dir / "Manual_combined_workers.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(logs_dir.glob(args.pattern))
    if not files:
        print(f"[combine] No files found matching {logs_dir / args.pattern}")
        return 0

    print(f"[combine] Found {len(files)} worker files.")

    rows = []
    for f in files:
        with f.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                # Keep exactly the fields we want; fill missing with ""
                rows.append({k: r.get(k, "") for k in OUT_FIELDS})

    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=OUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[combine] Wrote {out_path} with {len(rows)} rows from {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())