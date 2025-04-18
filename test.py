# from generate_map_fh import extract_and_save_topdown_map


# scene_path = "data/scene_datasets/hm3d/val/00877-4ok3usBNeis/4ok3usBNeis.basis.glb"
# agent_y = 2.06   # Automatically use the initial Y from agent
# occupancy, origin_xz = extract_and_save_topdown_map(scene_path, height=agent_y)




import numpy as np
from rrt_star_call import plan_rrt_star, plot_rrt_result

# Define start and goal in meters

start = (1.0274764  + 0.77397, 1.4455771  + 1.5698568)
goal = (2.0, 2.5)
map_path = "topdown_maps_single/occupancy_h2.06.npy"

# Run planner
path, nodes, occupancy, start_goal, reference_angle, reference_point = plan_rrt_star(start, goal, map_path)

# Plot result (only if path is valid)
if path:
    plot_rrt_result(path, nodes, occupancy, start_goal, map_path, reference_point=reference_point)

    if reference_angle is not None:
        print(f"✅ Reference angle (degrees): {np.degrees(reference_angle):.2f}°")
else:
    print("❌ No valid path found. Skipping plot.")
