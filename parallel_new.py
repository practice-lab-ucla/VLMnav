import subprocess
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

# ========================== CONFIG ==========================
# NUM_INSTANCES = 1000             # How many partition the dataset split into 
# MAX_PARALLEL = 40               # How many to actually run
# EPISODES_PER_INSTANCE = 1     # Episodes each instance should run
# MAX_STEPS = 150                 # Max steps per episode


NUM_INSTANCES = 200             # How many partition the dataset split into 
MAX_PARALLEL = 40               # How many to actually run
EPISODES_PER_INSTANCE = 5    # Episodes each instance should run
MAX_STEPS = 250              # Max steps per episode

NUM_GPU = 1                    # Number of GPUs available (set to 1 if only one GPU)

PORT = 2000                   # Aggregator server port (optional)
CONFIG = "ObjectNav"          # Config file name (without .yaml)
SCRIPT_PATH = "scripts/main.py"  # Path to your main.py
PYTHON_BIN = "/home/qizhao/miniconda3/envs/vlm_nav/bin/python"  # Absolute path to Python in conda env
# ===========================================================

# Create a unique log folder for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/parallel_run_{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)
WORKER_LOG_DIR = f"logs/worker_log_{timestamp}"
os.makedirs(WORKER_LOG_DIR, exist_ok=True)

print("🔧 Launch Configuration:")
print(f"- Number of Total Instances: {NUM_INSTANCES}")
print(f"- Number of Instances to Run Now: {MAX_PARALLEL}")
print(f"- Episodes per Instance: {EPISODES_PER_INSTANCE}")
print(f"- Max Steps per Episode: {MAX_STEPS}")
print(f"- Log Directory: {LOG_DIR}\n")

def run_instance(instance_id):
    gpu_id = instance_id % NUM_GPU
    cmd = (
        f"RUN_ID={timestamp} "
        f"WORKER_LOG_DIR={WORKER_LOG_DIR} "
        f"EPISODE_LOG_DIR={LOG_DIR} "             
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"{PYTHON_BIN} {SCRIPT_PATH} "
        f"--config {CONFIG} "
        f"--parallel "
        f"--instances {NUM_INSTANCES} "
        f"--instance {instance_id} "
        f"--num_episodes {EPISODES_PER_INSTANCE} "
        f"--max_steps {MAX_STEPS} "
        f"--port {PORT}"
    )
    log_file_path = os.path.join(LOG_DIR, f"instance_{instance_id}.log")
    with open(log_file_path, "w") as log_file:
        print(f"🚀 Launching instance {instance_id} on GPU {gpu_id}, logging to {log_file_path}")
        subprocess.run(cmd, shell=True, stdout=log_file, stderr=log_file)
    print(f"✅ Instance {instance_id} finished, logs in {log_file_path}")



if __name__ == "__main__":
    start_time = datetime.now()
    instance_ids_to_run = list(range(min(MAX_PARALLEL, NUM_INSTANCES)))
    print(f"▶️ Running instances: {instance_ids_to_run}")

    with Pool(processes=len(instance_ids_to_run)) as pool:
        pool.map(run_instance, instance_ids_to_run)

    end_time = datetime.now()
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed.total_seconds(), 60)
    print(f"✅ Selected instances completed. Total runtime: {int(minutes)} min {int(seconds)} sec")


    # === Post-run: combine per-worker CSVs into one file ===
    try:
        from pathlib import Path
        import csv

        logs_dir = Path("logs")
        combined_out = Path(LOG_DIR) / "combined_workers.csv"
        combined_out.parent.mkdir(parents=True, exist_ok=True)

        # worker_files = sorted(logs_dir.glob("worker_*.csv"))
        # worker_files = sorted(Path(LOG_DIR).glob("worker_*.csv"))


        combined_out = Path(WORKER_LOG_DIR) / "combined_workers.csv"
        worker_files = sorted(Path(WORKER_LOG_DIR).glob("worker_*.csv"))



        if not worker_files:
            print("[combine] No worker_*.csv files found in 'logs/'. Skipping merge.")
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

            # Write combined file
            with combined_out.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=["worker_id", "episode_ndx", "scene_id", "bfs_min"])
                writer.writeheader()
                writer.writerows(rows)

            print(f"[combine] Wrote {combined_out}")
    except Exception as e:
        print(f"[combine] ERROR while combining worker CSVs: {e}")


