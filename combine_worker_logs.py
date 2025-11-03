#!/usr/bin/env python3
"""
Combine per-worker logs (logs/worker_log*/worker_*.csv) into a single CSV with a 'worker_id' column.
Also plot a histogram of bfs_min scores and compute the 5% quantile.

Usage:
  # Auto-detects the right logs/worker_log* folder and writes combined CSV there
  python combine_worker_logs.py

  # Or customize:
  python combine_worker_logs.py --logs-dir logs/worker_log_20250912_194157 \
                                --out logs/worker_log_20250912_194157/combined_workers.csv \
                                --pattern 'worker_*.csv'
"""

import csv
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory containing worker_*.csv (auto-detect if omitted)"
    )
    parser.add_argument(
        "--pattern",
        default="worker_*.csv",
        help="Glob pattern to match worker CSVs"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path for combined output CSV (defaults to <detected_logs_dir>/combined_workers.csv)"
    )

    args = parser.parse_args()

    # --- Auto-detect logs dir if not provided ---
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

        # 3) As a last resort, search recursively under logs/ for worker_*.csv and use their parent
        all_hits = list(base.rglob("worker_*.csv"))
        if all_hits:
            # pick the parent dir containing most files
            by_parent = {}
            for f in all_hits:
                by_parent.setdefault(f.parent, 0)
                by_parent[f.parent] += 1
            chosen = max(by_parent.items(), key=lambda kv: kv[1])[0]
            print(f"[combine] Fallback selected folder: {chosen}")
            return chosen

        return None

    if args.logs_dir:
        logs_dir = Path(args.logs_dir)
        print(f"[combine] Using provided logs-dir: {logs_dir}")
    else:
        detected = detect_logs_dir()
        if not detected:
            print("[combine] No worker_log* directory with worker_*.csv found under logs/.")
            return 0
        logs_dir = detected

    # Decide output path
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = logs_dir / "Manual_combined_workers.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect files
    files = sorted(logs_dir.glob(args.pattern))
    if not files:
        print(f"[combine] No files found matching {logs_dir / args.pattern}")
        return 0

    print(f"[combine] Found {len(files)} worker files.")

    # Read and combine rows
    rows = []
    for f in files:
        worker_id = f.stem.split("_")[-1]  # worker_0.csv -> "0"
        with f.open(newline="", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for r in reader:
                rows.append({
                    "worker_id": worker_id,
                    "episode_ndx": r.get("episode_ndx", ""),
                    "scene_id": r.get("scene_id", ""),
                    "run_result": r.get("run_result", ""),
                    "steps_taken": r.get("steps_taken", ""),
                })

    # Write combined CSV
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["worker_id", "episode_ndx", "scene_id", "run_result", "steps_taken"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[combine] Wrote {out_path} with {len(rows)} rows from {len(files)} files.")



    return 0


if __name__ == "__main__":
    raise SystemExit(main())
