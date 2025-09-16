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



# (5, 6)-(5, 5): 0.239
# (5, 5)-(4, 5): 0.355
# (5, 5)-(3, 6): 0.127
# (3, 6)-(2, 6): 0.208
# (5, 6)-(4, 4): 0.138
# (5, 6)-(3, 6): 0.076
# (3, 6)-(3, 5): 0.208
# (3, 5)-(3, 4): 0.196
# (3, 4)-(3, 3): 0.302
# (3, 4)-(4, 4): 0.042
# (3, 4)-(2, 3): 0.025
# (3, 5)-(3, 3): 0.139
# (3, 5)-(4, 4): 0.058
# (5, 6)-(5, 6): 0.046
# (4, 3)-(6, 4): 0.02
# (4, 3)-(6, 3): 0.018
# (4, 3)-(6, 4): 0.046
# (6, 4)-(6, 5): 0.375
# (6, 5)-(6, 6): 0.236
# (6, 6)-(7, 7): 0.369
# (7, 7)-(7, 7): 0.306
# (7, 7)-(7, 10): 0.096
# (7, 10)-(7, 10): 0.204
# (7, 10)-(6, 11): 0.27
# (6, 11)-(6, 11): 0.172
# (6, 11)-(6, 12): 0.331
# (6, 12)-(3, 11): 0.082