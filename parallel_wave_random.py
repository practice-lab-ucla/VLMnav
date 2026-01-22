import subprocess
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import csv
import time
import math
import random

# ========================== CONFIG ==========================
WAVE_SIZE = 20
NUM_WAVES = 5
MAX_STEPS = 200

# Avoid some range
FORBIDDEN_RANGE = range(200, 221)

# Total environments in the whole pool (global count seen by main.py)
TOTAL_ENVIRONMENTS = 1000

# How many we are launching in THIS run
TOTAL_INSTANCES_LAUNCHED = max(0, int(NUM_WAVES) * int(WAVE_SIZE))

NUM_GPU = 1

PORT = 2000
CONFIG = "ObjectNav"
SCRIPT_PATH = "scripts/main.py"
PYTHON_BIN = "/home/qizhao/miniconda3/envs/vlm_nav/bin/python"
# ===========================================================

if TOTAL_INSTANCES_LAUNCHED <= 0:
    raise ValueError("Nothing to launch. Increase NUM_WAVES and/or WAVE_SIZE.")

if TOTAL_INSTANCES_LAUNCHED > TOTAL_ENVIRONMENTS:
    raise ValueError(
        f"Requested {TOTAL_INSTANCES_LAUNCHED} instances but only {TOTAL_ENVIRONMENTS} environments available."
    )

# Avoid sampling from a specific forbidden range (e.g., 200–220)
valid_ids = [i for i in range(TOTAL_ENVIRONMENTS) if i not in FORBIDDEN_RANGE]

# Choose random unique instance IDs from the valid pool
RANDOM_SEED = None  # e.g. 42 for deterministic sampling
if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

selected_instances = random.sample(valid_ids, TOTAL_INSTANCES_LAUNCHED)

# organize runs into waves preserving randomness
local_ids = list(range(TOTAL_INSTANCES_LAUNCHED))

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/parallel_run_{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)
WORKER_LOG_DIR = f"logs/worker_log_{timestamp}"
os.makedirs(WORKER_LOG_DIR, exist_ok=True)

print("🔧 Launch Configuration:")
print(f"- Waves: {NUM_WAVES}")
print(f"- Wave Size: {WAVE_SIZE}")
print(f"- Total Environments (global): {TOTAL_ENVIRONMENTS}")
print(f"- This run launches: {TOTAL_INSTANCES_LAUNCHED} (random unique instances, avoiding {FORBIDDEN_RANGE.start}-{FORBIDDEN_RANGE.stop - 1})")
print(f"- Max Steps per Episode: {MAX_STEPS}")
print(f"- Log Directory: {LOG_DIR}\n")


def run_instance(instance_id: int):
    gpu_id = instance_id % max(1, NUM_GPU)
    cmd = (
        f"RUN_ID={timestamp} "
        f"WORKER_LOG_DIR={WORKER_LOG_DIR} "
        f"EPISODE_LOG_DIR={LOG_DIR} "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{PYTHON_BIN} {SCRIPT_PATH} "
        f"--config {CONFIG} "
        f"--parallel "
        f"--instances {TOTAL_ENVIRONMENTS} "
        f"--instance {instance_id} "
        f"--max_steps {MAX_STEPS} "
        f"--port {PORT}"
    )
    log_file_path = os.path.join(LOG_DIR, f"instance_{instance_id}.log")
    with open(log_file_path, "w") as log_file:
        print(f"🚀 Launching instance {instance_id} on GPU {gpu_id}, logging to {log_file_path}")
        result = subprocess.run(cmd, shell=True, stdout=log_file, stderr=log_file)
    if result.returncode != 0:
        print(f"❌ Instance {instance_id} exited with code {result.returncode}. See {log_file_path}")
    else:
        print(f"✅ Instance {instance_id} finished, logs in {log_file_path}")


if __name__ == "__main__":
    start_time = datetime.now()

    computed_waves = math.ceil(TOTAL_INSTANCES_LAUNCHED / max(1, WAVE_SIZE))
    print(f"▶️ Planning to run {TOTAL_INSTANCES_LAUNCHED} instance(s) "
          f"in {computed_waves} wave(s) of up to {WAVE_SIZE} each")

    wave = 0
    wave_start = datetime.now()
    # chunk the selected_instances into waves of WAVE_SIZE
    for start in range(0, TOTAL_INSTANCES_LAUNCHED, WAVE_SIZE):
        wave += 1
        batch_instances = selected_instances[start:start + WAVE_SIZE]
        print(f"\n🌊 Wave {wave}/{computed_waves}: launching instances {batch_instances}")

        # use a process pool to run the batch in parallel
        with Pool(processes=len(batch_instances), maxtasksperchild=1) as pool:
            pool.map(run_instance, batch_instances)

        wave_end = datetime.now()
        wave_elapsed = wave_end - wave_start
        minutes, seconds = divmod(wave_elapsed.total_seconds(), 60)

        print(f"✅ Wave {wave} complete. "
              f"Used time: {int(minutes)} min {int(seconds)} sec")

    end_time = datetime.now()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)
    print(f"🏁 All planned instances finished. Total runtime: {int(minutes)} min {int(seconds)} sec")

    # # === Combine worker CSVs ===
    # try:
    #     combined_out = Path(WORKER_LOG_DIR) / "combined_workers.csv"
    #     combined_out.parent.mkdir(parents=True, exist_ok=True)
    #     worker_files = sorted(Path(WORKER_LOG_DIR).glob("worker_*.csv"))
    #     if not worker_files:
    #         print("[combine] No worker_*.csv files found. Skipping merge.")
    #     else:
    #         rows = []
    #         for f in worker_files:
    #             with f.open(newline="", encoding="utf-8") as fp:
    #                 reader = csv.DictReader(fp)
    #                 for r in reader:
    #                     rows.append({
    #                         "worker_id": f.stem.split("_")[-1],
    #                         "episode_ndx": r.get("episode_ndx", ""),
    #                         "scene_id": r.get("scene_id", ""),
    #                         "run_result": r.get("run_result", ""),
    #                         "steps_taken": r.get("steps_taken", ""),

    #                     })
    #         with combined_out.open("w", newline="", encoding="utf-8") as fp:
    #             writer = csv.DictWriter(fp, fieldnames=["worker_id", "episode_ndx", "scene_id", "run_result", "steps_taken"]) 

    #             writer.writeheader()
    #             writer.writerows(rows)
    #         print(f"[combine] Wrote {combined_out}")
    # except Exception as e:
    #     print(f"[combine] ERROR while combining worker CSVs: {e}")


    # === Combine worker CSVs ===
    try:
        combined_out = Path(WORKER_LOG_DIR) / "combined_workers.csv"
        combined_out.parent.mkdir(parents=True, exist_ok=True)

        worker_files = sorted(Path(WORKER_LOG_DIR).glob("worker_*.csv"))
        if not worker_files:
            print("[combine] No worker_*.csv files found. Skipping merge.")
        else:
            rows = []
            for f in worker_files:
                with f.open(newline="", encoding="utf-8") as fp:
                    reader = csv.DictReader(fp)
                    for r in reader:
                        rows.append({
                            # If you want to SKIP worker_id, remove this key and from fieldnames below
                            # "worker_id": f.stem.split("_")[-1],

                            "episode_ndx": r.get("episode_ndx", ""),
                            "scene_id": r.get("scene_id", ""),
                            "run_result": r.get("run_result", ""),
                            "steps_taken": r.get("steps_taken", ""),
                            "distance_to_g": r.get("distance_to_g", ""),
                            "dis_true": r.get("dis_true", ""),
                            "real_true": r.get("real_true", ""),
                            "dataset_geodesic": r.get("dataset_geodesic", ""),
                            "geo_start_to_goal": r.get("geo_start_to_goal", ""),
                            "path_length_m": r.get("path_length_m", ""),
                            "error": r.get("error", ""),
                        })

            fieldnames = [
                # "worker_id",
                "episode_ndx",
                "scene_id",
                "run_result",
                "steps_taken",
                "distance_to_g",
                "dis_true",
                "real_true",
                "dataset_geodesic",
                "geo_start_to_goal",
                "path_length_m",
                "error"
            ]

            with combined_out.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            print(f"[combine] Wrote {combined_out}")
    except Exception as e:
        print(f"[combine] ERROR while combining worker CSVs: {e}")
