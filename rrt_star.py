import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random
import math

# ----- RRT* Node Definition -----
class Node:
    def __init__(self, point):
        self.point = point  # (x, y)
        self.parent = None
        self.cost = 0.0

# ----- RRT* Parameters -----
# max_iterations = 3000
# step_size = 2
# search_radius = 20


meters_per_pixel = 0.05
max_iterations = 1000
step_size = 0.3
search_radius = 1
robot_radius_m = 0.05  # radius of the robot in meters
goal_tolerance = 1


# start = (8.075632, 1.7494259)  
# goal = (2.0, 2.5)


start = (8.5, 1.26951)  
goal = (1.0, 1.0)

step_size = step_size/meters_per_pixel
search_radius = search_radius/meters_per_pixel
goal_tolerance = goal_tolerance/meters_per_pixel

# Load occupancy map (0 = free, 1 = obstacle)
map_path = "topdown_maps_single/occupancy_h2.1.npy"
occupancy = np.load(map_path)





# Inflate obstacles using a disk kernel


inflate_px = int(robot_radius_m / meters_per_pixel)

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1)
)
occupancy = cv2.dilate(occupancy, kernel)






height, width = occupancy.shape

print( "height",height)

# Set start and goal (in pixel coordinates)

start = (start[0] / meters_per_pixel, start[1] / meters_per_pixel)
goal = (goal[0] / meters_per_pixel, goal[1] / meters_per_pixel)

# ----- Helper Functions -----
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_free(p):
    x, y = int(p[0]), int(p[1])
    return 0 <= x < width and 0 <= y < height and occupancy[y, x] == 0

def get_random_point():
    return (random.randint(0, width - 1), random.randint(0, height - 1))

def steer(from_point, to_point, step=step_size):
    vec = np.array(to_point) - np.array(from_point)
    dist = np.linalg.norm(vec)
    if dist < step:
        return to_point
    direction = vec / dist
    new_point = np.array(from_point) + direction * step
    return tuple(np.round(new_point).astype(int))

def is_collision_free(p1, p2):
    x0, y0 = p1
    x1, y1 = p2
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    while True:
        if not is_free((x0, y0)):
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy

    return True


def find_nearest(nodes, point):
    return min(nodes, key=lambda node: distance(node.point, point))

def get_neighbors(nodes, point, radius):
    return [node for node in nodes if distance(node.point, point) <= radius]

# ----- RRT* Planning -----
nodes = [Node(start)]
for i in range(max_iterations):
    rand_point = get_random_point()
    nearest_node = find_nearest(nodes, rand_point)
    new_point = steer(nearest_node.point, rand_point)

    if not is_free(new_point) or not is_collision_free(nearest_node.point, new_point):
        continue

    new_node = Node(new_point)
    new_node.parent = nearest_node
    new_node.cost = nearest_node.cost + distance(nearest_node.point, new_point)

    # Rewire
    neighbors = get_neighbors(nodes, new_point, search_radius)
    for neighbor in neighbors:
        new_cost = neighbor.cost + distance(neighbor.point, new_point)
        if new_cost < new_node.cost and is_collision_free(neighbor.point, new_point):
            new_node.parent = neighbor
            new_node.cost = new_cost

    nodes.append(new_node)

    # Check if goal is reached
    if distance(new_node.point, goal) < goal_tolerance:

        goal_node = Node(goal)
        goal_node.parent = new_node
        goal_node.cost = new_node.cost + distance(new_node.point, goal)
        nodes.append(goal_node)
        print(f"✅ Goal reached in {i} iterations.")
        break
else:
    print("❌ Failed to find path.")
    exit(1)

# ----- Extract Path -----
path = []
node = nodes[-1]
while node is not None:
    path.append(node.point)
    node = node.parent
path.reverse()

# ----- Visualization -----




## assign color for the inflated part

original_occupancy = np.load(map_path)

# Create RGB map: white = free, black = obstacle
vis_map = np.stack([255 * (1 - original_occupancy)] * 3, axis=-1).astype(np.uint8)

# Mark inflated-only regions in yellow (i.e., inflated - original)
inflated_only = (occupancy == 1) & (original_occupancy == 0)
vis_map[inflated_only] = [255, 255, 0] 




# use non inflated map
# vis_map = np.stack([255 * (1 - occupancy)] * 3, axis=-1).astype(np.uint8)




vis_map = np.ascontiguousarray(vis_map, dtype=np.uint8)  # 

# Draw tree (optional)
for node in nodes:
    if node.parent is not None:
        pt1 = tuple(map(int, node.point))
        pt2 = tuple(map(int, node.parent.point))
        cv2.line(vis_map, pt1, pt2, (200, 200, 200), 1)

# Draw path
for i in range(len(path) - 2):
    pt1 = tuple(map(int, path[i]))
    pt2 = tuple(map(int, path[i + 1]))
    cv2.line(vis_map, pt1, pt2, (0, 255, 0), 2)


# Draw start and goal
cv2.circle(vis_map, tuple(map(int, start)), 4, (0, 0, 255), -1)  # Red
cv2.circle(vis_map, tuple(map(int, goal)), 4, (255, 0, 0), -1)  # Blue


# Show result
plt.figure(figsize=(8, 8))
plt.imshow(vis_map)
plt.title("RRT* Path with Start/Goal")
plt.axis("off")
plt.savefig("rrt_star_result.png", bbox_inches="tight", dpi=200)
plt.close()  # This closes the figure immediately, avoids the pop-up
print("✅ Saved RRT* result to: rrt_star_result.png")

