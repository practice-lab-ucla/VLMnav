import numpy as np
import matplotlib.pyplot as plt
import cv2

# ---- Config ----
map_path = "topdown_maps_single/4ok3usBNeis.basis_h2.06.npy"
# [6.5, 2.06447, 3.25]

x_meters, y_meters = 0.9+0.77397, 1.4+1.5698568  # Replace with the point you want to visualize
meters_per_pixel = 0.005

# 📍 Agent Position: [1.243993  2.0644748 2.3801432]
# 🧭 Agent Rotation: quaternion(0.675563037395477, 0, 0.737302243709564, 0)


# ---- Load map ----
occupancy = np.load(map_path)
height, width = occupancy.shape

# ---- Convert to pixel coordinates ----
x_px = int(x_meters / meters_per_pixel)
y_px = int(y_meters / meters_per_pixel)


status = "occupied (obstacle)" if occupancy[y_px, x_px] == 1 else "free (navigable)"
print(f"🧱 Grid cell at ({x_px}, {y_px}) is: {status}")

# ---- Check bounds ----
if not (0 <= x_px < width and 0 <= y_px < height):
    raise ValueError(f"Point ({x_px}, {y_px}) is out of bounds for the map size {width}x{height}")

# ---- Visualization ----
# Convert occupancy to RGB
vis_map = np.stack([255 * (1 - occupancy)] * 3, axis=-1).astype(np.uint8)
vis_map = np.ascontiguousarray(vis_map)

# Mark the point
cv2.circle(vis_map, (x_px, y_px), 5, (0, 0, 255), -1)  # Red dot

# Show and save
plt.figure(figsize=(8, 8))
plt.imshow(vis_map)
plt.title(f"Point ({x_meters}m, {y_meters}m) on Map")
plt.axis("off")
plt.savefig("marked_map.png", bbox_inches="tight", dpi=200)
plt.show()

print(f"✅ Plotted point at pixel ({x_px}, {y_px})")
