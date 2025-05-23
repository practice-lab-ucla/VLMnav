import subprocess
import os
from datetime import datetime
from multiprocessing import Pool

# ========================== CONFIG ==========================
NUM_INSTANCES = 20              # How many partition the dataset split into 
MAX_PARALLEL = 20               # How many to actually run
NUM_GPU = 1                    # Number of GPUs available (set to 1 if only one GPU)
EPISODES_PER_INSTANCE = 1     # Episodes each instance should run
MAX_STEPS = 2                 # Max steps per episode
PORT = 2000                   # Aggregator server port (optional)
CONFIG = "ObjectNav"          # Config file name (without .yaml)
SCRIPT_PATH = "scripts/main.py"  # Path to your main.py
PYTHON_BIN = "/home/qizhao/miniconda3/envs/vlm_nav/bin/python"  # Absolute path to Python in conda env
# ===========================================================

# Create a unique log folder for this run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs/parallel_run_{timestamp}"
os.makedirs(LOG_DIR, exist_ok=True)

print("🔧 Launch Configuration:")
print(f"- Number of Total Instances: {NUM_INSTANCES}")
print(f"- Number of Instances to Run Now: {MAX_PARALLEL}")
print(f"- Episodes per Instance: {EPISODES_PER_INSTANCE}")
print(f"- Max Steps per Episode: {MAX_STEPS}")
print(f"- Log Directory: {LOG_DIR}\n")

def run_instance(instance_id):
    gpu_id = instance_id % NUM_GPU
    cmd = (
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

