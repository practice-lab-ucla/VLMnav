from rrt_star_call import plan_rrt_star, plot_rrt_result

# Define start and goal in meters
# start = (7.03487, 1)
# goal = (2.0, 2.5)




start = (10, 1.5)  
goal = (2.0, 2.5)
map_path = "topdown_maps_single/occupancy_h2.1.npy"

# Run planner
path, nodes, occupancy, start_goal = plan_rrt_star(start, goal, map_path)

# Plot result
plot_rrt_result(path, nodes, occupancy, start_goal, map_path)
