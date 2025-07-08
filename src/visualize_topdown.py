import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def visualize_topdown_map_with_agent(
    map_path: str,
    agent_state,
    map_origin: tuple,
    step_idx: int = 0,
    meters_per_pixel: float = 0.005,
    grid_spacing_m: float = 0.5,
    save_path: str = None,
    show: bool = True,
    position_history: list = None  # List of (x, y) in world coordinates
):
    """
    Visualizes a top-down map with:
    - 0.5m × 0.5m grid
    - Agent's current location
    - Trajectory connecting visited grid centers
    - Highlights the last segment of the path

    Args:
        map_path (str): Path to .npy occupancy grid file.
        agent_state (habitat_sim.AgentState): Current agent state.
        map_origin (tuple): World coordinate origin used in top-down map.
        step_idx (int): Current step index.
        meters_per_pixel (float): Map resolution.
        grid_spacing_m (float): Grid spacing size in meters.
        save_path (str): Optional output path.
        show (bool): Whether to display via matplotlib.
        position_history (list): List of (x, y) world positions from the agent.
    """
    # Load map
    topdown_map = np.load(map_path)
    map_vis = (1 - topdown_map) * 255
    map_vis = cv2.cvtColor(map_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Agent position → pixel coordinates
    x = agent_state.position[0]
    y = agent_state.position[2]
    x_px = int((x - map_origin[0]) / meters_per_pixel)
    y_px = int((y - map_origin[1]) / meters_per_pixel)


    ###### get here they are in the grid #############



    spacing_px = int(grid_spacing_m / meters_per_pixel)
    # grid_x = x_px // spacing_px  # column index
    # grid_y = y_px // spacing_px  # row index


    grid_x = x_px // spacing_px + 1
    grid_y = y_px // spacing_px + 1













    ######################################### below is for visualization ################################


    # Draw agent current position (red dot)
    cv2.circle(map_vis, (x_px, y_px), radius=5, color=(0, 0, 255), thickness=-1)

    # Grid lines
    spacing_px = int(grid_spacing_m / meters_per_pixel)
    height, width = map_vis.shape[:2]
    for i in range(0, width, spacing_px):
        cv2.line(map_vis, (i, 0), (i, height), (200, 200, 200), 1)
    for j in range(0, height, spacing_px):
        cv2.line(map_vis, (0, j), (width, j), (200, 200, 200), 1)

    # Grid center dots
    for i in range(0, width, spacing_px):
        for j in range(0, height, spacing_px):
            center = (i + spacing_px // 2, j + spacing_px // 2)
            if center[0] < width and center[1] < height:
                cv2.circle(map_vis, center, radius=1, color=(0, 0, 0), thickness=-1)

    # Draw trajectory through grid centers
    if position_history and len(position_history) > 1:
        grid_points = []
        for x_w, y_w in position_history:
            x_p = int((x_w - map_origin[0]) / meters_per_pixel)
            y_p = int((y_w - map_origin[1]) / meters_per_pixel)
            x_center = (x_p // spacing_px) * spacing_px + spacing_px // 2
            y_center = (y_p // spacing_px) * spacing_px + spacing_px // 2
            grid_points.append((x_center, y_center))

        # Draw black dot at each center and path between centers
        for i, pt in enumerate(grid_points):

            cv2.circle(map_vis, pt, radius=3, color=(255, 0, 0), thickness=-1)  # Blue dot (BGR)


            if i > 0:
                pt_prev = grid_points[i - 1]
                # Use grees for last segment, blue otherwise
                color = (0, 255, 0) if i == len(grid_points) - 1 else (255, 0, 0)
                cv2.line(map_vis, pt_prev, pt, color=color, thickness=2)

    # Show or save
    if show:
        plt.figure(figsize=(8, 8))
        plt.imshow(map_vis)
        plt.title(f"Step {step_idx} — Agent @ ({x:.2f}, {y:.2f})")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, map_vis)


    return (grid_y, grid_x)

