import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import math
import os

class Node:
    def __init__(self, point):
        self.point = point
        self.parent = None
        self.cost = 0.0

def plan_rrt_star(start_m, goal_m, map_path, meters_per_pixel=0.01, step_size_m=0.1, search_radius_m=1, robot_radius_m=0.0, goal_tolerance_m=1, max_iterations=5000):
    # Convert parameters to pixel scale
    step_size = step_size_m / meters_per_pixel
    search_radius = search_radius_m / meters_per_pixel
    goal_tolerance = goal_tolerance_m / meters_per_pixel

    # Convert to pixel coordinates
    start = (start_m[0] / meters_per_pixel, start_m[1] / meters_per_pixel)
    goal = (goal_m[0] / meters_per_pixel, goal_m[1] / meters_per_pixel)

    # Load and inflate occupancy map
    occupancy = np.load(map_path)
    inflate_px = int(robot_radius_m / meters_per_pixel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1))
    occupancy = cv2.dilate(occupancy, kernel)
    height, width = occupancy.shape

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_free(p):
        x, y = int(p[0]), int(p[1])
        return 0 <= x < width and 0 <= y < height and occupancy[y, x] == 0

    def get_random_point():
        return (random.randint(0, width - 1), random.randint(0, height - 1))

    def steer(from_point, to_point):
        vec = np.array(to_point) - np.array(from_point)
        dist = np.linalg.norm(vec)
        if dist < step_size:
            return to_point
        direction = vec / dist
        new_point = np.array(from_point) + direction * step_size
        return tuple(np.round(new_point).astype(int))

    def is_collision_free(p1, p2):
        x0, y0 = map(int, p1)
        x1, y1 = map(int, p2)
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

    # RRT* Planning
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

        neighbors = get_neighbors(nodes, new_point, search_radius)
        for neighbor in neighbors:
            new_cost = neighbor.cost + distance(neighbor.point, new_point)
            if new_cost < new_node.cost and is_collision_free(neighbor.point, new_point):
                new_node.parent = neighbor
                new_node.cost = new_cost

        nodes.append(new_node)

        if distance(new_node.point, goal) < goal_tolerance:
            goal_node = Node(goal)
            goal_node.parent = new_node
            goal_node.cost = new_node.cost + distance(new_node.point, goal)
            nodes.append(goal_node)
            print(f"✅ Goal reached in {i} iterations.")
            break
    else:
        print("❌ Failed to find path. Skipping this planning attempt.")
        return [], [], occupancy, (start, goal), None, None

    # Reconstruct path
    path = []
    node = nodes[-1]
    while node:
        path.append(node.point)
        node = node.parent
    path.reverse()

###################################### get best angle ##############################################


    # --- Compute reference angle using Option 1: distance-based  ---
    reference_distance_px = int(1.5 / meters_per_pixel)
    reference_angle = None

    if not nodes or not path:
        print("❌ No path found, skipping reference angle.")
        return [], [], np.array([]), (start, goal), None


    pt1 = np.array(path[0])
    for pt in path[1:]:
        delta = np.array(pt) - pt1
        dist = np.linalg.norm(delta)
        if dist >= reference_distance_px:
            # reference_angle = math.atan2(delta[1], delta[0])  # angle in radians (Y over X)
            reference_angle = math.atan2(-delta[0], delta[1])  # -x over y 
            reference_point = pt
            break
    # fallback to using goal if threshold distance not exceeded
    if reference_angle is None:
        delta = np.array(goal) - pt1
        reference_angle = math.atan2(delta[1], delta[0])
        reference_point = pt

    return path, nodes, occupancy, (start, goal), reference_angle, reference_point


def plot_rrt_result(path, nodes, occupancy, start_goal, map_path, reference_point=None, output_dir="rrt_star_record"):

    # Helper to find next available filename
    def get_next_filename(prefix="rrt_star_result_", ext=".png"):
        os.makedirs(output_dir, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(output_dir, f"{prefix}{i}{ext}")):
            i += 1
        return os.path.join(output_dir, f"{prefix}{i}{ext}")

    save_path = get_next_filename()

    original_occupancy = np.load(map_path)
    vis_map = np.stack([255 * (1 - original_occupancy)] * 3, axis=-1).astype(np.uint8)
    inflated_only = (occupancy == 1) & (original_occupancy == 0)
    vis_map[inflated_only] = [255, 255, 0]
    vis_map = np.ascontiguousarray(vis_map, dtype=np.uint8)

    for node in nodes:
        if node.parent:
            pt1 = tuple(map(int, node.point))
            pt2 = tuple(map(int, node.parent.point))
            cv2.line(vis_map, pt1, pt2, (200, 200, 200), 1)

    for i in range(len(path) - 2):
        pt1 = tuple(map(int, path[i]))
        pt2 = tuple(map(int, path[i + 1]))
        cv2.line(vis_map, pt1, pt2, (0, 255, 0), 2)

    start, goal = start_goal
    cv2.circle(vis_map, tuple(map(int, start)), 4, (0, 0, 255), -1)
    cv2.circle(vis_map, tuple(map(int, goal)), 4, (255, 0, 0), -1)


    if reference_point is not None:
        cv2.circle(vis_map, tuple(map(int, reference_point)), 3, (0, 255, 255), -1)  # yellow dot


    plt.figure(figsize=(8, 8))
    plt.imshow(vis_map)
    plt.title("RRT* Path with Start/Goal")
    plt.axis("off")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Saved RRT* result to: {save_path}")
