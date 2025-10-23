import subprocess
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
import csv
import time
import math

# ========================== CONFIG ==========================
WAVE_SIZE = 10
NUM_WAVES = 30
MAX_STEPS = 200

# WAVE_SIZE = 1
# NUM_WAVES = 1
# MAX_STEPS = 5

# Total environments in the whole pool (global count seen by main.py)
TOTAL_ENVIRONMENTS = 1000

# Start offset into the 0..TOTAL_ENVIRONMENTS-1 space
# For the first 50, leave at 0. For the next 50 later, set to 50, etc.
START_INSTANCE = 0



NUM_GPU = 1

PORT = 2000
CONFIG = "ObjectNav"
SCRIPT_PATH = "scripts/main.py"
PYTHON_BIN = "/home/qizhao/miniconda3/envs/vlm_nav/bin/python"
# ===========================================================

# How many we are launching in THIS run
TOTAL_INSTANCES_LAUNCHED = max(0, int(NUM_WAVES) * int(WAVE_SIZE))

if START_INSTANCE + TOTAL_INSTANCES_LAUNCHED > TOTAL_ENVIRONMENTS:
    raise ValueError(
        f"Slice ({START_INSTANCE}..{START_INSTANCE + TOTAL_INSTANCES_LAUNCHED - 1}) "
        f"exceeds TOTAL_ENVIRONMENTS={TOTAL_ENVIRONMENTS}"
    )

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/parallel_run_{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)
WORKER_LOG_DIR = f"logs/worker_log_{timestamp}"
os.makedirs(WORKER_LOG_DIR, exist_ok=True)

print("🔧 Launch Configuration:")
print(f"- Waves: {NUM_WAVES}")
print(f"- Wave Size: {WAVE_SIZE}")
print(f"- Total Environments (global): {TOTAL_ENVIRONMENTS}")
print(f"- This run launches: {TOTAL_INSTANCES_LAUNCHED} (from {START_INSTANCE})")
print(f"- Max Steps per Episode: {MAX_STEPS}")
print(f"- Log Directory: {LOG_DIR}\n")

def run_instance(local_index: int):
    # Map local 0..TOTAL_INSTANCES_LAUNCHED-1 to global instance ids
    instance_id = START_INSTANCE + local_index

    gpu_id = instance_id % max(1, NUM_GPU)
    cmd = (
        f"RUN_ID={timestamp} "
        f"WORKER_LOG_DIR={WORKER_LOG_DIR} "
        f"EPISODE_LOG_DIR={LOG_DIR} "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{PYTHON_BIN} {SCRIPT_PATH} "
        f"--config {CONFIG} "
        f"--parallel "
        f"--instances {TOTAL_ENVIRONMENTS} "   # <-- pass the global total (1000)
        f"--instance {instance_id} "           # <-- the specific slice we're running now
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
    if TOTAL_INSTANCES_LAUNCHED <= 0:
        raise ValueError("Nothing to launch. Increase NUM_WAVES and/or WAVE_SIZE.")

    start_time = datetime.now()

    # Local indices we will launch this run
    local_ids = list(range(TOTAL_INSTANCES_LAUNCHED))
    computed_waves = math.ceil(TOTAL_INSTANCES_LAUNCHED / max(1, WAVE_SIZE))
    print(f"▶️ Planning to run {TOTAL_INSTANCES_LAUNCHED} instance(s) "
          f"in {computed_waves} wave(s) of up to {WAVE_SIZE} each")

    wave = 0
    wave_start = datetime.now()
    for start in range(0, TOTAL_INSTANCES_LAUNCHED, WAVE_SIZE):
        wave += 1
        batch_local = local_ids[start:start + WAVE_SIZE]
        batch_global = [START_INSTANCE + i for i in batch_local]
        print(f"\n🌊 Wave {wave}/{computed_waves}: launching instances {batch_global}")

        # wave_start = datetime.now()

        with Pool(processes=len(batch_local), maxtasksperchild=1) as pool:
            pool.map(run_instance, batch_local)

        wave_end = datetime.now()
        wave_elapsed = wave_end - wave_start
        minutes, seconds = divmod(wave_elapsed.total_seconds(), 60)

        print(f"✅ Wave {wave} complete. "
              f"Used time: {int(minutes)} min {int(seconds)} sec")
        
    end_time = datetime.now()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)
    print(f"🏁 All planned instances finished. Total runtime: {int(minutes)} min {int(seconds)} sec")

    # === Combine worker CSVs (unchanged) ===
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
                            "worker_id": f.stem.split("_")[-1],
                            "episode_ndx": r.get("episode_ndx", ""),
                            "scene_id": r.get("scene_id", ""),
                            "bfs_min": r.get("bfs_min", ""),
                        })
            with combined_out.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=["worker_id", "episode_ndx", "scene_id", "bfs_min"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"[combine] Wrote {combined_out}")
    except Exception as e:
        print(f"[combine] ERROR while combining worker CSVs: {e}")
