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
    agent_grid_history: dict = None,  # {step_idx: (row, col)}
    teleport_step_flags: dict = None  # {step_idx: True/False}
):
    """
    Visualizes a top-down map with agent's trajectory.
    Prevents connecting steps with teleportation (rewind).
    """
    print("Teleport Step Flags:")
    for step, is_teleport in sorted(teleport_step_flags.items()):
        print(f"  Step {step}: {'TELEPORT' if is_teleport else 'normal'}")
    # Load map
    topdown_map = np.load(map_path)
    map_vis = (1 - topdown_map) * 255
    map_vis = cv2.cvtColor(map_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Agent position → pixel coordinates
    x = agent_state.position[0]
    y = agent_state.position[2]
    x_px = int((x - map_origin[0]) / meters_per_pixel)
    y_px = int((y - map_origin[1]) / meters_per_pixel)

    spacing_px = int(grid_spacing_m / meters_per_pixel)
    grid_x = x_px // spacing_px + 1
    grid_y = y_px // spacing_px + 1

    # # Draw agent position
    # cv2.circle(map_vis, (x_px, y_px), radius=5, color=(0, 0, 255), thickness=-1)

    # # Draw grid lines
    # height, width = map_vis.shape[:2]
    # for i in range(0, width, spacing_px):
    #     cv2.line(map_vis, (i, 0), (i, height), (200, 200, 200), 1)
    # for j in range(0, height, spacing_px):
    #     cv2.line(map_vis, (0, j), (width, j), (200, 200, 200), 1)

    # # Draw grid center dots
    # for i in range(0, width, spacing_px):
    #     for j in range(0, height, spacing_px):
    #         center = (i + spacing_px // 2, j + spacing_px // 2)
    #         if center[0] < width and center[1] < height:
    #             cv2.circle(map_vis, center, radius=1, color=(0, 0, 0), thickness=-1)

    # # Draw trajectory — skip teleport steps
    # if agent_grid_history and len(agent_grid_history) > 0:
    #     full_history = list(agent_grid_history.items()) + [(step_idx, (grid_y, grid_x))]

    #     for i in range(1, len(full_history)):
    #         step_prev, (row_prev, col_prev) = full_history[i - 1]
    #         step_curr, (row_curr, col_curr) = full_history[i]


            
    #         if teleport_step_flags and teleport_step_flags.get(step_curr, False):
    #             continue 


    #         center_prev = (
    #             col_prev * spacing_px - spacing_px // 2,
    #             row_prev * spacing_px - spacing_px // 2
    #         )
    #         center_curr = (
    #             col_curr * spacing_px - spacing_px // 2,
    #             row_curr * spacing_px - spacing_px // 2
    #         )

    #         color = (0, 255, 0) if i == len(full_history) - 1 else (255, 0, 0)
    #         cv2.line(map_vis, center_prev, center_curr, color=color, thickness=2)
    #         cv2.circle(map_vis, center_prev, radius=3, color=(255, 0, 0), thickness=-1)
    #         if i == len(full_history) - 1:
    #             cv2.circle(map_vis, center_curr, radius=3, color=(255, 0, 0), thickness=-1)

    # if show:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(map_vis)
    #     plt.title(f"Step {step_idx} — Agent @ ({x:.2f}, {y:.2f})")
    #     plt.axis("off")
    #     plt.tight_layout()
    #     plt.show()

    # if save_path:
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     cv2.imwrite(save_path, map_vis)

    return (grid_y, grid_x)
