#!/usr/bin/env python3
"""
Combine per-worker logs (logs/worker_*.csv) into a single CSV with a 'worker_id' column.
Also plot a histogram of bfs_min scores and compute the 5% quantile.

Usage:
  python combine_worker_logs.py
  # or customize:
  python combine_worker_logs.py --logs-dir logs --out logs/combined_workers.csv
"""

import csv
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", default="logs", help="Directory containing worker_*.csv")
    parser.add_argument("--pattern", default="worker_*.csv", help="Glob pattern to match worker CSVs")
    parser.add_argument(
        "--out",
        default="logs/combined_workers.csv",
        help="Path for combined output CSV"
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(logs_dir.glob(args.pattern))
    if not files:
        print(f"[combine] No files found matching {logs_dir / args.pattern}")
        return 0

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
                    "bfs_min": r.get("bfs_min", ""),
                })

    # Write combined CSV
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["worker_id", "episode_ndx", "scene_id", "bfs_min"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[combine] Wrote {out_path} with {len(rows)} rows from {len(files)} files.")

    # --- Collect bfs_min values ---
    values = []
    for r in rows:
        try:
            v = float(r["bfs_min"])
            values.append(v)
        except (ValueError, TypeError):
            continue  # skip empty or invalid scores

    if values:
        values = np.array(values, dtype=float)

        # Plot histogram
        plt.hist(values, bins=30, edgecolor="black")
        plt.title("Distribution of bfs_min Scores")
        plt.xlabel("bfs_min")
        plt.ylabel("Frequency")

        hist_path = out_path.with_suffix(".png")
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"[combine] Histogram saved to {hist_path}")

        # Compute 5% quantile
        quan = np.quantile(values, 0.25)
        print(f"[combine] 25% quantile of bfs_min = {quan:.6f}")
    else:
        print("[combine] No valid bfs_min scores found — histogram skipped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
