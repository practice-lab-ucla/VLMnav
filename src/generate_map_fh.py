import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import os

def extract_and_save_topdown_map(scene_path, height, meters_per_pixel=0.005, output_dir="topdown_maps_single"):

    """
    Extracts a top-down occupancy map and saves it as .npy/.png files.
    
    Args:
        scene_path (str): Path to the .glb scene file.
        height (float): Y-coordinate (height) of the agent to use for slicing.
        output_dir (str): Folder to save the results.
        meters_per_pixel (float): Resolution of the top-down map.
    
    Returns:
        occupancy (np.ndarray): Occupancy grid (0 = free, 1 = obstacle)
        origin_xz (np.ndarray): World coordinate origin used for aligning with sim
    """

    # Simulator config
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)

    # Ensure navmesh is loaded
    if not sim.pathfinder.is_loaded:
        print("⚠️ Navmesh missing! Generating...")
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.cell_size = meters_per_pixel
        navmesh_settings.cell_height = 0.0
        success = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
        if not success:
            raise RuntimeError("❌ Navmesh generation failed!")
        print("✅ Navmesh successfully generated!")

    print(f"📏 Generating map at height: {height:.2f}m")
    top_down_map = sim.pathfinder.get_topdown_view(meters_per_pixel=meters_per_pixel, height=height)
    occupancy = (~top_down_map.astype(bool)).astype(np.uint8)  # 0 = free, 1 = obstacle

    bounds = sim.pathfinder.get_bounds()
    origin_xz = np.array([bounds[0][0], bounds[0][2]])  # [min_x, min_z]
    print(f"📍 Map origin (world coordinates): {origin_xz}")


    os.makedirs(output_dir, exist_ok=True)

    # Save occupancy grid
    # npy_filename = f"{output_dir}/occupancy_h{height:.2f}.npy"


    #### save the map with map name as well as height #####
    scene_name = os.path.splitext(os.path.basename(scene_path))[0]
    npy_filename = f"{output_dir}/{scene_name}_h{height:.2f}.npy"




    np.save(npy_filename, occupancy)
    print(f"✅ Saved occupancy grid: {npy_filename}")

    # # Save origin
    # origin_filename = f"{output_dir}/origin_h{height:.2f}.npy"
    # np.save(origin_filename, origin_xz)
    # # print(f"✅ Saved origin: {origin_filename}")

    # Save visual preview
    img_filename = npy_filename.replace('.npy', '.png')
    plt.figure(figsize=(6, 6))
    plt.imshow(top_down_map, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(f"Height = {height:.2f}m")
    plt.savefig(img_filename, bbox_inches="tight", dpi=200)
    plt.close()
    print(f"✅ Saved preview image: {img_filename}")

    sim.close()
    return occupancy, origin_xz