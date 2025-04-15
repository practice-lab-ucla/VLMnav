import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import os

# Scene path
scene_path = "data/scene_datasets/hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"

# Simulator Configuration
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path

# Agent Configuration
agent_cfg = habitat_sim.agent.AgentConfiguration()
agent_cfg.sensor_specifications = []

cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)

# ✅ Ensure navigation mesh is loaded
if not sim.pathfinder.is_loaded:
    print("⚠️ Navmesh missing! Generating...")
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.cell_size = 0.05
    navmesh_settings.cell_height = 0.0
    success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    if success:
        print("✅ Navmesh successfully generated!")
    else:
        print("❌ Navmesh generation failed!")
        exit(1)

# 🔹 Generate map at height 1.6m
height = 2.06
meters_per_pixel = 0.05
print(f"📏 Generating map at height: {height:.1f}m")

top_down_map = sim.pathfinder.get_topdown_view(
    meters_per_pixel= meters_per_pixel,
    height=height
)



print(f"Map shape: {top_down_map.shape}")


# the size of the map is 8.07 x 11.09 (meters)

occupancy = (~top_down_map.astype(bool)).astype(np.uint8)  # 0 = free, 1 = obstacle


# x = 2 
# y = 1  

# x = int(x/meters_per_pixel)
# y = int(y/meters_per_pixel)


# if occupancy[y, x] == 1:
#     print(f"Point ({x}, {y}) is OCCUPIED")
# else:
#     print(f"Point ({x}, {y}) is FREE")


# Save map
output_dir = "topdown_maps_single"
os.makedirs(output_dir, exist_ok=True)
filename = f"{output_dir}/topdown_map_h{height:.1f}.png"

plt.figure(figsize=(6, 6))
plt.imshow(top_down_map, cmap="gray", vmin=0, vmax=1)
plt.axis("off")
plt.title(f"Height = {height:.1f}m")
plt.savefig(filename, bbox_inches="tight", dpi=200)
plt.close()
print(f"✅ Saved: {filename}")


npy_filename = f"{output_dir}/occupancy_h{height:.1f}.npy"
np.save(npy_filename, occupancy)
print(f"✅ Saved occupancy grid: {npy_filename}")

sim.close()
