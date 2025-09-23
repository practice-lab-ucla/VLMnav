import logging
import math
import random
import habitat_sim
import numpy as np
import cv2
import ast
import concurrent.futures
import csv
import os

from simWrapper import PolarAction, SimWrapper
from utils import *
from vlm import *
from pivot import PIVOT
from scipy.spatial.transform import Rotation as R
from visualize_topdown import visualize_topdown_map_with_agent
from Modified_BFS import modified_bfs
from habitat_sim.utils.common import quat_from_coeffs




def get_agent_heading_angle(agent_quat):
    """
    Get the heading angle (in degrees) the agent is facing in global frame.
    The heading is measured from the X-axis in the XZ plane (e.g., 0° = +X, 90° = +Z).
    """
    quat_xyzw = [agent_quat.x, agent_quat.y, agent_quat.z, agent_quat.w]
    rot = R.from_quat(quat_xyzw)

    # Agent forward in local frame is Z
    forward_local = np.array([0, 0, 1])
    forward_global = rot.apply(forward_local)

    # Project to XZ plane
    x, z = forward_global[0], forward_global[2]

    # Compute heading angle (0 = +X, 90 = +Z)
    angle_rad = np.arctan2(x, -z)  # Flip Z because forward is Z
    angle_deg = np.degrees(angle_rad)

    return angle_deg % 360


class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass


class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        
        return agent_action, metadata


class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8




    @staticmethod
    def normalize_scores(confident_scores):
        """
        Normalizes a list of confidence scores to ensure they sum exactly to 1.
        """
        total = sum(confident_scores)
        
        if total == 0:
            return [1.0 / len(confident_scores)] * len(confident_scores)
        
        # Normalize scores
        normalized_scores = [s / total for s in confident_scores]
        normalized_scores = [round(n, 3) for n in normalized_scores]  # round to # ofdecimal places to avoid long numbers
        
        return normalized_scores


    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']



        self.simWrapper: SimWrapper = None
        self.resolution = (
            1080 // cfg['sensor_cfg']['res_factor'],
            1920 // cfg['sensor_cfg']['res_factor']
        )



        self.tree_action_queue = []
        self.tree_root_state = None
        self.tree_root_score_log = []


        self.current_episode_goal = None
        self.current_episode_idx = None 


        self.agent_grid_history = {}
        self.teleport_step_flags = {}
        self.step_action_log = []


        ## stored in _prompting
        ## store the history of action with respect to the score and position/orientation ++++ this is a dict
        self.step_action_log_history_dict = {}
        ## store the history of the score which is choosen by the agent action ++++ this is a dict 
        self.step_score_history_dict = {}


        ## store the history of the GSV score  ++++ this is a dict 
        ## stored in _stopping_module
        self.global_semantic_score = None

        ## stored in step
        self.gsv_per_step = {} 

        self.adjusted_score = {}


        self.turnaround_streak = 0


        ## currently same as first reach
        self.initiate_back_propagation = False
        self.first_reach = False

        # backtrack start index
        self.start_ndx = []



        self.goal_reached = False

        self.overall_stop = False


        self.goal_grid_location = None



        ### record the score


        self.step_action_ranking_dict = {}
        self.defer_rewind_to_root = False 
        
        self.best_bfs_path = set()
        self.best_bfs_min = None

        # which actions we have already tried at each step
        self.tried_actions_by_step = {}          # step_idx -> set of action indices
        # which step is the current rewind root (so we log tries against the right step)
        self.tree_root_step_ndx = None


        self.goal_steps = set() 



        self.rewind_origin_step = None              # where the current sibling-walk started
        self.immediate_turnaround_by_root = {}      # {root_step: set(action_indices that 1-step-turnaround)}
        self.last_root_action = None                # last sibling dispatched from current root


        self.parent_by_step = {}
        self.swipping_back = False






        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       
        self.pivot = PIVOT(self.actionVLM, self.fov, self.resolution, max_action_length=cfg['max_action_dist']) if cfg['pivot'] else None

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()







################################################################## here is where everything come together ####################################################################


    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        self.last_obs = obs.copy() 

        print(f"🟢 Executing step ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ {self.step_ndx}")

        # print("📍 Agent Position:", agent_state.position)
        # print("🧭 Agent Rotation (Quaternion):", agent_state.rotation)


        scene_filename = os.path.basename(self.simWrapper.scene_path)  # e.g., '4ok3usBNeis.basis.glb'
        scene_name = os.path.splitext(scene_filename)[0]  
        height = self.cfg.get('rrt_map_height')
        map_origin = self.cfg.get('map_origin')
        map_path = f"topdown_maps_single/{scene_name}_h{height:.2f}.npy"
        agent = self.simWrapper.sim.get_agent(0)




        if getattr(self, "defer_rewind_to_root", False):
            print("↩️ Rewinding to root before applying new action")


            # Step 1: Reset agent pose
            agent = self.simWrapper.sim.get_agent(0)
            new_state = habitat_sim.AgentState()
            new_state.position = self.tree_root_state.position
            new_state.rotation = self.tree_root_state.rotation

            print(f"📍 Restoring position: {new_state.position}")

            print(f"🧭 Restoring rotation (quat): {new_state.rotation}")


            # agent.set_state(new_state)
            # self.simWrapper.set_state(new_state)

            self.simWrapper.set_state(
                pos=new_state.position,
                quat=new_state.rotation
            )



            agent_state = self.simWrapper.sim.get_agent(0).get_state()
            # print("✅ Confirmed agent state after restore:")
            # print(f"  Pos: {agent_state.position}")
            # print(f"  Rot: {agent_state.rotation}")




            self.teleport_step_flags[self.step_ndx] = self.defer_rewind_to_root
            # print(f"step is ##########################################################################################{self.step_ndx}")

            # grid_row_col = visualize_topdown_map_with_agent(
            #     map_path=map_path,
            #     agent_state=agent_state,
            #     map_origin=map_origin,
            #     step_idx=self.step_ndx,
            #     meters_per_pixel=self.cfg.get('meters_per_pixel', 0.005),
            #     save_path=f"logs/topdown_step{self.step_ndx}.png",
            #     show=False,
            #     agent_grid_history=self.agent_grid_history,
            #     teleport_step_flags=self.teleport_step_flags  # this includes the current step
            # )

            grid_row_col = visualize_topdown_map_with_agent(
                map_path=map_path,
                agent_state=agent_state,
                map_origin=map_origin,
                step_idx=self.step_ndx,
                meters_per_pixel=self.cfg.get('meters_per_pixel', 0.005),
                grid_spacing_m=self.cfg.get('grid_spacing_m', 0.7),   
                save_path=f"logs/topdown_step{self.step_ndx}.png",
                show=False,
                agent_grid_history=self.agent_grid_history,
                teleport_step_flags=self.teleport_step_flags
            )


            print("grid_row_col:", grid_row_col)
            print("self.step_ndx:", self.step_ndx)


            self.defer_rewind_to_root = False  # consume flag

            # self.agent_grid_history.append(grid_row_col)
            self.agent_grid_history[self.step_ndx] = grid_row_col


            # print("📘 Agent Grid History:")
            # for step, (r, c) in sorted(self.agent_grid_history.items()):
            #     print(f"  Step {step}: Grid cell (row={r}, col={c})")












            # Step 2: Refresh observation
            obs = self.simWrapper.sim.get_sensor_observations(0)
            obs['agent_state'] = agent.get_state()

            # ✅ Restore 'goal' if it was present
            if hasattr(self, "last_obs") and "goal" in self.last_obs:
                obs["goal"] = self.last_obs["goal"]


            # Step 3: Override action proposal to be consistent with tree
            metadata = {}
            metadata['a_final'] = self.tree_root_a_final


######################################################################################## move this to previous step for stopping #############################################





            # 👉 If inside tree, continue taking queued actions
            if self.tree_action_queue:
                print("🌲 Continuing tree-style queue:", self.tree_action_queue)
                next_action = self.tree_action_queue.pop(0)


                self._link_parent_for_next_step(self.tree_root_step_ndx)
                print("trigger 1 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


                root_ndx = getattr(self, "tree_root_step_ndx", None)
                if root_ndx is not None:
                    if root_ndx not in self.tried_actions_by_step:
                        self.tried_actions_by_step[root_ndx] = set()
                    self.tried_actions_by_step[root_ndx].add(next_action)

                self.last_root_action = next_action



                # self.defer_rewind_to_root = len(self.tree_action_queue) > 0


                score = self.tree_root_score_log[next_action] if self.tree_root_score_log else "N/A"
                print(f"➡️ Taking queued action: {next_action} (score = {score})")
                print()

                
                ############### log the next score when rewind ###################
                self.adjusted_score[self.step_ndx] = score


                # print("🧮 Adjusted Score History:")
                # for step, adj_score in self.adjusted_score.items():
                #     print(f"  Step {step}: Adjusted Score = {adj_score}")








                # print(f"🎯 Next action to execute = {next_action}") 

                print("\n🌲 Current self.tree_root_a_final:")
                for i, (r, theta) in enumerate(self.tree_root_a_final):
                    print(f"  Action {i+1}: distance = {r:.3f} m, angle = {theta:.3f} rad ({np.degrees(theta):.2f}°)")

                print(f"📌 Next queued model index: {next_action}")







                agent_action = self._action_number_to_polar(next_action, list(self.tree_root_a_final))
                print(f"🛞 Converted to PolarAction: Distance = {agent_action.r}, Angle = {agent_action.theta:.5f}°")




                metadata['step_metadata'] = {
                    'action_number': next_action,
                    'success': 1,
                    'score': 1.0,
                    'confident_score': [],
                    'top_actions': self.tree_action_queue.copy()
                }
                metadata['logging_data'] = {}
                metadata['images'] = {
                    'color_sensor': obs['color_sensor']
                }


                a_final = self.tree_root_a_final




                # Project onto raw image
                raw_action_image = obs['color_sensor'].copy()
                self._project_onto_image(
                    a_final,
                    raw_action_image,
                    obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor']
                )
                metadata['images']['color_sensor'] = raw_action_image





                chosen_action_image = obs['color_sensor'].copy()
                self._project_onto_image(
                    a_final, chosen_action_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    chosen_action=next_action
                )
                metadata['images']['color_sensor_chosen'] = chosen_action_image


                self.step_ndx += 1



                # if (not self.initiate_back_propagation) and (next_action == 0):
                #     print("🔁 Queued action 0 while in rewind → rewinding one more step to parent of current root")
                #     self.step_rewind(self.tree_root_step_ndx, 0)


                if (not self.initiate_back_propagation) and (next_action == 0):

                    if self.swipping_back and getattr(self, "tree_root_step_ndx", None) == 0:
                        pass 

                    else:
                        if self.swipping_back:
                            if self.swipe_back_all_steps():
                                # Do NOT execute action 0. We just queued a new global swipe target.
                                # Return a no-op step; next tick will teleport & dispatch the queued action.
                                return None, {
                                    'step_metadata': {'action_number': -3, 'success': 1},
                                    'logging_data': {'note': 'GLOBAL_SWIPE_BACK_REQUEUE_ON_0'},
                                    'a_final': self.tree_root_a_final or [],
                                    'images': {
                                        # reuse the images we already prepared above
                                        'color_sensor': metadata['images']['color_sensor'],
                                        'color_sensor_chosen': metadata['images']['color_sensor_chosen'],
                                    }
                                }
                            else:
                                self.overall_stop = True
                                return PolarAction.stop, {
                                    'step_metadata': {'action_number': -1, 'success': 1},
                                    'logging_data': {'note': 'GLOBAL_SWIPE_BACK_EXHAUSTED'},
                                    'a_final': [],
                                    'images': {'color_sensor': obs['color_sensor']}
                                }
                        else:
                            print("🔁 Queued action 0 while in rewind → rewinding one more step to parent of current root")
                            self.step_rewind(self.tree_root_step_ndx, 0)



                return agent_action, metadata
            


# #######################################################################################################################################################




        agent_state = agent.get_state()



        self.teleport_step_flags[self.step_ndx] = self.defer_rewind_to_root
        # print(f"step is ##########################################################################################{self.step_ndx}")

        # grid_row_col = visualize_topdown_map_with_agent(
        #     map_path=map_path,
        #     agent_state=agent_state,
        #     map_origin=map_origin,
        #     step_idx=self.step_ndx,
        #     meters_per_pixel=self.cfg.get('meters_per_pixel', 0.005),
        #     save_path=f"logs/topdown_step{self.step_ndx}.png",
        #     show=False,
        #     agent_grid_history=self.agent_grid_history,
        #     teleport_step_flags=self.teleport_step_flags  # this includes the current step
        # )

        grid_row_col = visualize_topdown_map_with_agent(
            map_path=map_path,
            agent_state=agent_state,
            map_origin=map_origin,
            step_idx=self.step_ndx,
            meters_per_pixel=self.cfg.get('meters_per_pixel', 0.005),
            grid_spacing_m=self.cfg.get('grid_spacing_m', 0.7),   # 👈 add this line
            save_path=f"logs/topdown_step{self.step_ndx}.png",
            show=False,
            agent_grid_history=self.agent_grid_history,
            teleport_step_flags=self.teleport_step_flags
        )


        # self.agent_grid_history.append(grid_row_col)
        self.agent_grid_history[self.step_ndx] = grid_row_col
        


        # print("📘 Agent Grid History:")
        # for step, (r, c) in sorted(self.agent_grid_history.items()):
        #     print(f"  Step {step}: Grid cell (row={r}, col={c})")




        if self.step_ndx == 0:
            self.init_pos = agent_state.position

            self._initial_pose = habitat_sim.AgentState()
            self._initial_pose.position = np.array(agent_state.position, dtype=np.float32).copy()
            self._initial_pose.rotation = agent_state.rotation 


        agent_action, metadata = self._choose_action(obs)






        # if getattr(self, "terminate_after_local", False):
        #     # Clear rewind state
        #     self.tree_action_queue = []
        #     self.defer_rewind_to_root = False
        #     self.tree_root_state = None
        #     self.tree_root_step_ndx = None

        #     # Return STOP and mark failure
        #     return PolarAction.stop, {
        #         "step_metadata": {"action_number": -1, "success": 1},
        #         "logging_data": {"note": "LOCAL_REWIND_EXHAUSTED"},
        #         "a_final": [],
        #         "images": {"color_sensor": obs["color_sensor"]}
        #     }
        



        # only before goal / before back-propagation







        gsv = self.global_semantic_score
        self.gsv_per_step[self.step_ndx] = gsv
        log_history_action_state = self.step_action_log_history_dict.get(self.step_ndx)



        print(f"[Step {self.step_ndx}] Global Semantic Score (GSV)AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa: {gsv:.3f}")




        ### at the last step there is not step score because the goal is reached ###
        if self.step_ndx in self.step_score_history_dict:
            combined_score = self.step_score_history_dict[self.step_ndx] * gsv
            self.adjusted_score[self.step_ndx] = combined_score

            edges = self.generate_grid_edge_score_list_from_adjusted()
        else:
            print(f"model stopped skipping adjusted score.")

            edges = self.generate_grid_edge_score_list_from_adjusted()

            prev_step = self.step_ndx - 1
            prev_grid = self.agent_grid_history.get(prev_step)
            curr_grid = self.agent_grid_history.get(self.step_ndx)

            # If current grid is missing because the log didn't run, reconstruct using SAME grid logic
            if curr_grid is None:
                try:
                    agent = self.simWrapper.sim.get_agent(0)
                    s = agent.get_state()

                    # --- match visualize_topdown_map_with_agent ---
                    mpp = float(self.cfg.get('meters_per_pixel', 0.005))
                    grid_spacing_m = float(self.cfg.get('grid_spacing_m', 0.7))
                    origin_x, origin_y = self.cfg.get('map_origin')

                    x = float(s.position[0])
                    z = float(s.position[2])

                    x_px = int((x - origin_x) / mpp)
                    y_px = int((z - origin_y) / mpp)
                    spacing_px = int(grid_spacing_m / mpp)

                    grid_x = x_px // spacing_px + 1
                    grid_y = y_px // spacing_px + 1

                    curr_grid = (grid_y, grid_x)
                    # ---------------------------------------------

                    self.agent_grid_history[self.step_ndx] = curr_grid
                except Exception as e:
                    print(f"⚠️ Could not reconstruct current grid from pose: {e}")

            # (rest unchanged) synthesize minimal log if missing, then append (prev->curr) edge using prev_adj
            if self.step_ndx not in self.step_action_log_history_dict:
                grid_from = [prev_grid[0], prev_grid[1]] if prev_grid is not None else None
                grid_current = [curr_grid[0], curr_grid[1]] if curr_grid is not None else None
                try:
                    agent = self.simWrapper.sim.get_agent(0)
                    s = agent.get_state()
                    log_entry = {
                        "step": self.step_ndx,
                        "position": [round(float(p), 2) for p in s.position],
                        "rotation": [float(s.rotation.w), float(s.rotation.x),
                                    float(s.rotation.y), float(s.rotation.z)],
                        "grid_current": grid_current,
                        "grid_from": grid_from,
                        "actions": None,
                    }
                    self.step_action_log_history_dict[self.step_ndx] = log_entry
                    self.step_action_log.append(log_entry)
                except Exception as e:
                    print(f"⚠️ Could not synthesize minimal log for step {self.step_ndx}: {e}")

            if (
                prev_step >= 0
                and prev_grid is not None
                and curr_grid is not None
                and not self.teleport_step_flags.get(self.step_ndx, False)
            ):
                prev_adj = self.adjusted_score.get(prev_step)
                if prev_adj is None:
                    prev_gsv = self.gsv_per_step.get(prev_step)
                    prev_score = self.step_score_history_dict.get(prev_step)
                    if prev_gsv is not None and prev_score is not None:
                        prev_adj = prev_gsv * prev_score

                if prev_adj is not None:
                    candidate = (prev_grid[0], prev_grid[1], curr_grid[0], curr_grid[1], round(prev_adj, 3))
                    if candidate not in edges:
                        edges.append(candidate)


###################################################################################################################


        print("📊 Grid Transitions with Adjusted Scores:")

        for r1, c1, r2, c2, score in edges:
            print(f"({r1}, {c1})-({r2}, {c2}): {score}")


        # path, min_score = modified_bfs(edges)
        # print("testttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttttt")
        # print("Best Path:", path)
        # print("Maximized Minimum Score:", min_score)


        if self.initiate_back_propagation and self.goal_grid_location is None:
            if self.step_ndx in self.agent_grid_history:
                self.goal_grid_location = self.agent_grid_history[self.step_ndx]
                print(f"🎯 Goal grid location captured at step {self.step_ndx}: {self.goal_grid_location}")
            else:
                print("⚠️ Warning: Cannot capture goal grid location — not found in agent_grid_history.")


        # if self.first_reach and self.goal_grid_location is not None:
        if self.initiate_back_propagation:
            print("📊 Grid Transitions with Adjusted Scores:")
            # edges = self.generate_grid_edge_score_list_from_adjusted()
            for r1, c1, r2, c2, score in edges:
                print(f"({r1}, {c1})-({r2}, {c2}): {score}")

            path, min_score = modified_bfs(edges, self.goal_grid_location)

            print("✅ Running Modified BFS After Goal Reached")
            print("📍 Goal:", self.goal_grid_location)
            print("🏁 Best Path:", path)
            print("📉 Maximized Minimum Score:", min_score)


            self.best_bfs_path = set(path)
            self.best_bfs_min = min_score


            if self.best_bfs_min is not None:
                print(f"🔻 BFS bottleneck score before stopping: {self.best_bfs_min:.3f}")
            else:
                print("🔻 BFS bottleneck score before stopping: None")


        else:
            print("⏳ Awaiting goal reach... BFS not triggered yet.")
        


        ##### calculate number of turns for warning purpose #####
        selected_action = metadata['step_metadata']['action_number']







        if selected_action == 0:
            self.turnaround_streak += 1
        else:
            self.turnaround_streak = 0



        # if (selected_action == 0 and not self.initiate_back_propagation and self.tree_root_state is not None and self.tree_action_queue):
        #     # We’re in the middle of a sibling-walk; keep rewinding to this root
        #     should_rewind_to_sibling = True

        # ################# only do rewind to state when we ARE backtracking ##################

        # should_rewind_to_sibling = False



        curr_grid = self.agent_grid_history.get(self.step_ndx)

        should_rewind_to_sibling = False











        if self.initiate_back_propagation:
            # evaluate all rewind triggers first
            trigger_rewind = (
                selected_action == 0
                or self.goal_reached
                or (curr_grid is not None and curr_grid in self.best_bfs_path)
            )

            if trigger_rewind:
                if self.first_reach:
                    # First reach → do NOT rewind yet
                    print("✅ First reach detected — skipping rewind this time")
                    self.tree_action_queue = []
                    self.first_reach = False
                else:
                    # Any subsequent time → do rewind
                    if selected_action == 0:
                        print("🔁 Turn-around detected → will rewind to next sibling")
                    elif self.goal_reached:
                        print("🔁 Goal reached on this branch → will rewind to next sibling")
                    else:
                        print(f"🔁 On BFS best-path grid {curr_grid} → will rewind to next sibling")
                    should_rewind_to_sibling = True

        if should_rewind_to_sibling:
            self.defer_rewind_to_root = True








        






################################################################ for testing only ###############################################################3

        # # If the stopping head fires on THIS step, end the episode immediately (test mode)
        # if len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx:

        #     # produce images; add a "TERMINATING EPISODE" overlay via _project_onto_image()
        #     raw_action_image = obs['color_sensor'].copy()
        #     chosen_action_image = obs['color_sensor'].copy()
        #     self._project_onto_image(
        #         [],  # no arrows
        #         chosen_action_image,
        #         obs['agent_state'],
        #         obs['agent_state'].sensor_states['color_sensor']
        #     )

        #     # return STOP so env.py will mark done and end the loop
        #     return PolarAction.stop, {
        #         'step_metadata': {'action_number': -2, 'success': 1},
        #         'logging_data': {'note': 'TEST: goal seen → terminate (no backtracking)'},
        #         'a_final': [],
        #         'images': {
        #             'color_sensor': raw_action_image,
        #             'color_sensor_chosen': chosen_action_image
        #         }
        #     }















        # If we're already backtracking and standing on a node of the best BFS path,
        # skip re-traversing it—rewind to explore a different branch.
        if self.goal_reached:
            print("goal reached^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        else:
            print("goal not reached^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")










        if self.initiate_back_propagation and self.goal_reached and not self.tree_action_queue:
            # print(f"🔁 Initiating backtrack: min adjusted score = {self.min_score:.3f} at step {self.min_step}")
            # backtrack_success, a_final = self.rewind_and_explore_from_log_state(self.min_score, self.start_ndx)

            min_thresh = self.best_bfs_min

            

            if min_thresh is None:
                min_thresh = 1.0
                self.best_bfs_min = 1.0

            print(f"🔁 Initiating backtrack with BFS min = {min_thresh:.3f} ")
            print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            self.start_ndx = max(self.step_ndx - 1, 0)
            backtrack_success, a_final = self.rewind_and_explore_from_log_state(min_thresh, self.start_ndx)

            raw_action_image = obs['color_sensor'].copy()
            chosen_action_image = obs['color_sensor'].copy()

            self._project_onto_image(
                metadata['a_final'] if 'a_final' in metadata else [],
                raw_action_image,
                obs['agent_state'],
                obs['agent_state'].sensor_states['color_sensor']
            )
            self._project_onto_image(
                [], 
                chosen_action_image,
                obs['agent_state'],
                obs['agent_state'].sensor_states['color_sensor']
            )

            self.step_ndx += 1

            if backtrack_success:
                print("action foundddddddddddddddddddddddddddddddddddd")
                return None, {
                    'step_metadata': {'action_number': -3, 'success': 1},
                    'logging_data': {'note': 'backtrack from best BFS node'},
                    'a_final': a_final or [],
                    'images': {
                        'color_sensor': raw_action_image,
                        'color_sensor_chosen': chosen_action_image
                    }
                }
            else:
                print("🛑 No further rewind options from best-path node — stopping")
                self.overall_stop = True
                return PolarAction.stop, {
                    'step_metadata': {'action_number': -1, 'success': 1},
                    'logging_data': {'note': 'backtracking exhausted from best-path node'},
                    'a_final': [],
                    'images': {
                        'color_sensor': raw_action_image,
                        'color_sensor_chosen': chosen_action_image
                    }
                }






        # if self.initiate_back_propagation and len(self.tree_action_queue) == 1 and selected_action == 0:
        if self.initiate_back_propagation and not self.tree_action_queue and selected_action == 0:

            print("🔚 Finished all sub-actions at current rewind step")

            min_thresh = self.best_bfs_min 


            self.start_ndx = max(self.step_ndx - 1, 0)
            print(f"🔁 Trying deeper rewind to step {self.start_ndx} with min_score {min_thresh:.3f}")
            backtrack_success, a_final = self.rewind_and_explore_from_log_state(min_thresh, self.start_ndx)

            print("first trigger+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

            if backtrack_success:
                print(f"✅ Found new rewind target at step")
                # return None, {
                #     'step_metadata': {'action_number': -3, 'success': 1},
                #     'logging_data': {'note': 'deeper backtrack initiated'},
                #     'a_final': a_final,
                #     'images': {'color_sensor': obs['color_sensor']}
                # }
            raw_action_image = obs['color_sensor'].copy()
            chosen_action_image = obs['color_sensor'].copy()

            # Project all actions WITHOUT highlight → raw image
            self._project_onto_image(
                metadata['a_final'],
                raw_action_image,
                obs['agent_state'],
                obs['agent_state'].sensor_states['color_sensor']
            )

            # Project all actions WITH highlight → chosen image
            self._project_onto_image(
                metadata['a_final'],
                chosen_action_image,
                obs['agent_state'],
                obs['agent_state'].sensor_states['color_sensor'],
                chosen_action=0
            )
            
            
            self.step_ndx += 1


            if backtrack_success:
                return None, {
                    'step_metadata': {'action_number': -3, 'success': 1},
                    'logging_data': {'note': 'deeper backtrack initiated'},
                    'a_final': a_final,
                    'images': {
                        'color_sensor': raw_action_image,              # ✅ now has arrows
                        'color_sensor_chosen': chosen_action_image     # ✅ arrows + highlight
                    }
                }
            else:
                print("🛑 No further rewind options — stopping")
                self.overall_stop = True
                return PolarAction.stop, {
                    'step_metadata': {'action_number': -1, 'success': 1},
                    'logging_data': {'note': 'backtracking exhausted'},
                    'a_final': [],
                    'images': {'color_sensor': obs['color_sensor']}
                }






################################################################################################################################################

        # Current grid (row, col) for this step
        



        if (self.initiate_back_propagation and curr_grid in self.best_bfs_path and not self.goal_reached and not self.tree_action_queue):

            print(f"🟡 Backtracking and on best BFS path at step {self.step_ndx}, grid={curr_grid} → rewinding to explore a different branch")

            min_thresh = self.best_bfs_min 

            self.start_ndx = max(self.step_ndx - 1, 0)
            backtrack_success, a_final = self.rewind_and_explore_from_log_state(min_thresh, self.start_ndx)

            raw_action_image = obs['color_sensor'].copy()
            chosen_action_image = obs['color_sensor'].copy()

            self._project_onto_image(
                metadata['a_final'] if 'a_final' in metadata else [],
                raw_action_image,
                obs['agent_state'],
                obs['agent_state'].sensor_states['color_sensor']
            )
            self._project_onto_image(
                metadata['a_final'] if 'a_final' in metadata else [],
                chosen_action_image,
                obs['agent_state'],
                obs['agent_state'].sensor_states['color_sensor'],
                chosen_action=0
            )

            self.step_ndx += 1

            if backtrack_success:
                return None, {
                    'step_metadata': {'action_number': -3, 'success': 1},
                    'logging_data': {'note': 'backtrack from best BFS node'},
                    'a_final': a_final or [],
                    'images': {
                        'color_sensor': raw_action_image,
                        'color_sensor_chosen': chosen_action_image
                    }
                }
            else:
                print("🛑 No further rewind options from best-path node — stopping")
                self.overall_stop = True
                return PolarAction.stop, {
                    'step_metadata': {'action_number': -1, 'success': 1},
                    'logging_data': {'note': 'backtracking exhausted from best-path node'},
                    'a_final': [],
                    'images': {
                        'color_sensor': raw_action_image,
                        'color_sensor_chosen': chosen_action_image
                    }
                }


################################################################################################################################################




    


        ################# only do step rewind when we are NOT backtracking ##################

        # if (not self.initiate_back_propagation) and (selected_action == 0):
        #     self.step_rewind(self.step_ndx, selected_action)


        if (not self.initiate_back_propagation) and (selected_action == 0):
            if self.swipping_back:
                # We are back-swiping: pick another global best untried option
                if not self.swipe_back_all_steps():
                    self.overall_stop = True
                    return PolarAction.stop, {
                        'step_metadata': {'action_number': -1, 'success': 1},
                        'logging_data': {'note': 'GLOBAL_SWIPE_BACK_EXHAUSTED'},
                        'a_final': [],
                        'images': {'color_sensor': obs['color_sensor']}
                    }
            else:
                self.step_rewind(self.step_ndx, selected_action)












        selected_action = metadata['step_metadata']['action_number']
        # If we're in global back-swipe mode (pre-goal), keep swiping until the goal is reached
        if self.swipping_back and not self.initiate_back_propagation and not getattr(self, "defer_rewind_to_root", False) and selected_action == 0:
            if getattr(self, "tree_root_step_ndx", None) == 0:
                pass  # allow; do NOT re-sweep here
            
            else:
                if self.swipe_back_all_steps():
                    base = metadata['images'].get('color_sensor', obs['color_sensor'])
                    raw_action_image = base.copy()
                    chosen_action_image = base.copy()

                    self._project_onto_image(
                        metadata.get('a_final', []),
                        raw_action_image,
                        obs['agent_state'],
                        obs['agent_state'].sensor_states['color_sensor'],
                    )
                    self._project_onto_image(
                        metadata.get('a_final', []),
                        chosen_action_image,
                        obs['agent_state'],
                        obs['agent_state'].sensor_states['color_sensor'],
                        chosen_action=metadata['step_metadata'].get('action_number')
                    )

                    self.step_ndx += 1
                    return None, {
                        'step_metadata': {'action_number': -3, 'success': 1},
                        'logging_data': {'note': 'GLOBAL_SWIPE_BACK_INITIATED'},
                        'a_final': metadata.get('a_final', []),
                        'images': {
                            'color_sensor': raw_action_image,
                            'color_sensor_chosen': chosen_action_image
                        }
                    }

                else:
                    self.overall_stop = True
                    return [PolarAction.stop], {
                        'step_metadata': {'action_number': -1, 'success': 1},
                        'logging_data': {'note': 'GLOBAL_SWIPE_BACK_EXHAUSTED'},
                        'a_final': [],
                        'images': {'color_sensor': obs['color_sensor']}
                    }



        if (not self.initiate_back_propagation) and getattr(self, "terminate_after_local", False ) and not self.swipping_back:
            # Clear local rewind bookkeeping
            self.tree_action_queue = []
            self.defer_rewind_to_root = False
            self.tree_root_state = None
            self.tree_root_step_ndx = None

            # Enable global back-swipe mode
            self.swipping_back = True
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")
            print("TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT")

            # Try a global swipe immediately
            if self.swipe_back_all_steps():
                # Prepare a no-op return; next tick will teleport & run queued action
                base = metadata['images'].get('color_sensor', obs['color_sensor'])
                raw_action_image = base.copy()
                chosen_action_image = base.copy()

                # Draw ALL available actions on both images
                self._project_onto_image(
                    metadata.get('a_final', []),
                    raw_action_image,
                    obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                )
                self._project_onto_image(
                    metadata.get('a_final', []),
                    chosen_action_image,
                    obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    chosen_action=metadata['step_metadata'].get('action_number')  # may be None
                )

                self.step_ndx += 1
                return None, {
                    'step_metadata': {'action_number': -3, 'success': 1},
                    'logging_data': {'note': 'GLOBAL_SWIPE_BACK_CONTINUE'},
                    'a_final': metadata.get('a_final', []),
                    'images': {
                        'color_sensor': raw_action_image,
                        'color_sensor_chosen': chosen_action_image
                    }
                }

            else:
                # Nothing left anywhere → stop
                self.overall_stop = True
                return PolarAction.stop, {
                    "step_metadata": {"action_number": -1, "success": 1},
                    "logging_data": {"note": "GLOBAL_SWIPE_BACK_EXHAUSTED"},
                    "a_final": [],
                    "images": {"color_sensor": obs["color_sensor"]}
                }




































        step_metadata = metadata['step_metadata']


        



        if agent_action is None:
            print("✅ Reached goal or rewound — no action needed")
            print(f"🟢 self.goal_reached = {self.goal_reached}")

            if self.goal_reached:
                
                self.swipping_back = False

                chosen_action_image = obs['color_sensor'].copy()
                metadata['a_final'] = []  # 🛠️ Safely include empty a_final
                self._project_onto_image([], chosen_action_image, agent_state,
                                        agent_state.sensor_states['color_sensor'])
                metadata['images']['color_sensor_chosen'] = chosen_action_image

            self.step_ndx += 1


            # if self.goal_reached:
            #     self.goal_reached = False

            return agent_action, metadata



        ######################################################################################
        # confidence_score_for_distance = metadata['step_metadata'].get('score') 
        # # Adjust action distance based on confidence score
        # agent_action = self._adjust_action_distance(agent_action, confidence_score_for_distance)
        # # Print updated action details
        # print(f"Final Action Selected -> Distance: {agent_action.r}, Angle: {agent_action.theta}, Score: {confidence_score_for_distance}")
        # print("")
        ######################################################################################

        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image( 
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        self.step_ndx += 1



        # if self.goal_reached:
        #     self.goal_reached = False


        return agent_action, metadata
    




    def _link_parent_for_next_step(self, parent_step):
        """Record who the NEXT step's parent is (works for normal and rewind flows)."""
        child = self.step_ndx + 1
        self.parent_by_step[child] = parent_step
        print(f"👪 [parent-link] next step {child} ← parent {parent_step}")

    def _determine_parent_step(self, step_number):
        """Pure parent lookup: never uses grid/location."""
        if step_number in self.parent_by_step:
            return self.parent_by_step[step_number]
        prev = step_number - 1
        return prev if prev in self.step_action_log_history_dict else None




    # def _find_step_by_grid(self, grid_cell, upto_step):
    #     if grid_cell is None:
    #         return None
    #     for s in range(upto_step - 1, -1, -1):
    #         log_s = self.step_action_log_history_dict.get(s)
    #         if log_s and log_s.get("grid_current") == grid_cell:
    #             return s
    #     return None


    # def _at_state(self, target_state, pos_eps: float = 0.02, ang_eps_deg: float = 2.0) -> bool:
    #     """
    #     True iff current simulator agent pose matches target_state within tolerances.
    #     """
    #     import numpy as np, math
    #     from habitat_sim.utils.common import quat_to_angle_axis

    #     curr = self.simWrapper.sim.get_agent(0).get_state()

    #     # position check
    #     if np.linalg.norm(curr.position - target_state.position) > pos_eps:
    #         return False

    #     # orientation check by comparing minimal angles
    #     ang_curr, _ = quat_to_angle_axis(curr.rotation)
    #     ang_tgt, _ = quat_to_angle_axis(target_state.rotation)

    #     print(f"[DEBUG] curr: ang={ang_curr:.4f} ")
    #     print(f"[DEBUG] tgt : ang={ang_tgt:.4f} ")



    #     def _norm_angle(a):
    #         a = abs(a) % (2 * math.pi)
    #         return a if a <= math.pi else (2 * math.pi - a)

    #     return abs(_norm_angle(ang_curr) - _norm_angle(ang_tgt)) <= math.radians(ang_eps_deg)








    def swipe_back_all_steps(self):
        """
        Scan ALL previous steps for the highest-score UNTRIED non-zero action.
        Teleport to that step and queue exactly that action.
        Returns True if prepared; False if no candidates exist.
        """
        from habitat_sim import AgentState
        import numpy as np

        # newest → oldest
        for back_step in range(self.step_ndx, -1, -1):
            log = self.step_action_log_history_dict.get(back_step)
            if not log or not log.get("actions"):
                continue

            tried = self.tried_actions_by_step.get(back_step, set())

            # Collect untried, non-turnaround candidates with scores
            actions = sorted(log["actions"], key=lambda a: int(a.get("index", 0)))
            candidates = []
            max_idx = 0
            allow_zero_here = (back_step == 0)
            for a in actions:
                idx = int(a.get("index", 0))
                max_idx = max(max_idx, idx)
                # if idx == 0:
                #     continue
                # Allow 0 only at root (step 0)
                if idx == 0 and not allow_zero_here:
                    continue
                if idx in tried:
                    continue
                sc = a.get("adjusted")
                if sc is None:
                    sc = a.get("score", 0.0)
                candidates.append((idx, float(sc)))

            if not candidates:
                continue

            # Pick the single best untried option
            candidates.sort(key=lambda t: t[1], reverse=True)
            best_idx, _ = candidates[0]

            # Rebuild agent state at that step (same quat convention as rewind)
            restored = AgentState()
            restored.position = np.array(log["position"], dtype=np.float32)
            q = np.array(log["rotation"], dtype=np.float32)  # [w, x, y, z]
            q = q / (np.linalg.norm(q) or 1.0)
            q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
            restored.rotation = quat_from_coeffs(q_xyzw)

            # # Build a_final (exclude turnaround 0)
            # a_final = [(a["distance"], a["angle"]) for a in actions if int(a.get("index", 0)) != 0]

            # Build a_final for projection; keep excluding 0 so indices stay aligned
            # with your existing _action_number_to_polar contract (0 is special).
            a_final = [(a["distance"], a["angle"]) for a in actions if int(a.get("index", 0)) != 0]


            # Score log aligned by index (debug)
            scores_by_index = [None] * (max_idx + 1)
            for a in actions:
                idx = int(a.get("index", 0))
                scores_by_index[idx] = a.get("adjusted", a.get("score"))

            # Arm the rewind root so top-of-step() will teleport and execute best_idx
            self.tree_root_state = restored
            self.tree_root_a_final = a_final
            self.tree_action_queue = [best_idx]
            self.tree_root_score_log = scores_by_index
            self.tree_root_step_ndx = back_step
            self.rewind_origin_step = back_step
            self.immediate_turnaround_by_root.setdefault(back_step, set())
            self.defer_rewind_to_root = True

            # # Make sure the next step is parent-linked to this root
            # self._link_parent_for_next_step(back_step)


            return True

        return False





















    def step_rewind(self, current_step: int, selected_action: int):
        # Only trigger on action 0
        if selected_action != 0:
            return

        # # If we’re at initial pose or step 0, no-op
        # if self._at_state(self._initial_pose) or current_step <= 0:
        #     return


        parent_step = self.parent_by_step.get(current_step)


        if parent_step is None:

            print("the agent is at the initial state +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            # No parent to rewind to (e.g., step 0 root). Do nothing here so the
            # caller will execute the actual turn-around action (action 0) in-place.
            return


        print(f"🌳 [step_rewind] Rewinding from step {current_step} → parent step {parent_step}========================================")
        if parent_step is None:
            parent_step = current_step - 1  # conservative fallback

        # Build queue and restore pose from the *parent* step
        def build_queue_from_root(root_step: int):
            ranking = self.step_action_ranking_dict.get(root_step)
            prev_log = self.step_action_log_history_dict.get(root_step)
            if ranking is None or prev_log is None:
                return None, None, None

            tried = self.tried_actions_by_step.get(root_step, set())
            # remaining = [i for i, _ in ranking if i not in tried and i != 0]
            remaining = [i for i, _ in ranking if i not in tried]


            # remaining = [i for i, _ in ranking if i not in tried and i != 0]
            # if 0 not in tried:
            #     remaining.append(0)




            from habitat_sim import AgentState
            import numpy as np
            from habitat_sim.utils.common import quat_from_coeffs

            restored = AgentState()
            restored.position = np.array(prev_log['position'], dtype=np.float32)
            q = np.array(prev_log['rotation'], dtype=np.float32)     # [w, x, y, z]
            q = q / (np.linalg.norm(q) or 1.0)
            q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
            restored.rotation = quat_from_coeffs(q_xyzw)

            actions = prev_log.get('actions') or []
            sorted_actions = sorted(actions, key=lambda a: int(a.get('index', 0)))
            a_final = [(a['distance'], a['angle']) for a in sorted_actions if int(a.get('index', 0)) != 0]
            # a_final = [(a['distance'], a['angle']) for a in sorted_actions]


            max_idx = 0
            for a in actions:
                try:
                    max_idx = max(max_idx, int(a.get('index', 0)))
                except Exception:
                    pass
            scores_by_index = [None] * (max_idx + 1)
            for a in actions:
                idx = int(a.get('index', 0))
                scores_by_index[idx] = a.get('adjusted')

            return restored, a_final, (remaining, scores_by_index)

        restored, a_final, payload = build_queue_from_root(parent_step)
        if restored is None or not payload:
            # No data for parent; safely bail
            self.terminate_after_local = True
            return

        remaining, scores_by_index = payload
        if not remaining:
            # Nothing else to try at parent — still behave safely
            self.terminate_after_local = True
            return

        # Set rewind root to the parent step and teleport there on next tick
        self.tree_root_state = restored
        self.tree_root_a_final = a_final
        self.tree_action_queue = remaining
        self.tree_root_score_log = scores_by_index
        self.tree_root_step_ndx = parent_step
        self.rewind_origin_step = parent_step
        self.immediate_turnaround_by_root.setdefault(parent_step, set())
        self.defer_rewind_to_root = True











    def rewind_and_explore_from_log_state(self, min_adjusted: float, start_ndx: int = None):
        import magnum as mn
        from habitat_sim import AgentState
        import numpy as np

        stop_ndx = 0  # 🔁 Always go back to step 0

        print(f"🔁 Rewind range: from step {start_ndx} down to {stop_ndx}")

        for back_step in reversed(range(stop_ndx, start_ndx + 1)):
            print(f"🔎 Checking Step {back_step} for retryable actions...")


            if getattr(self, "goal_steps", None) and back_step in self.goal_steps:
                print(f"+++++++++++⏭️ Skipping goal step {back_step}")
                continue

            log = self.step_action_log_history_dict.get(back_step)
            if not log:
                print(f"+++++++++++ Step {back_step} has log in that step, it is a rewind state")
                continue

            actions = log.get("actions") or [] 


            try:
                pos_vals = log["position"]
                rot_vals = log["rotation"]
            except Exception as e:
                print(f"❌ Failed to parse state at step {back_step}: {e}")
                continue

            # Construct agent state
            restored_state = AgentState()
            restored_state.position = np.array(pos_vals, dtype=np.float32)

            q = np.array(rot_vals, dtype=np.float32)
            q /= np.linalg.norm(q)
            # Move w to the end for magnum quaternion
            q = np.concatenate((q[1:], q[:1]))  # [x, y, z, w] → [y, z, w, x]
            restored_state.rotation = quat_from_coeffs(q)

            # Extract adjusted scores and find retryable actions
            # adjusted_scores = {
            #     a["index"]: a["adjusted"]
            #     for a in log["actions"]
            #     if a["index"] != 0  # skip turn-around
            # }


            try:
                adjusted_scores = {
                    int(a["index"]): a.get("adjusted")
                    for a in actions
                    if int(a["index"]) != 0
                }
            except Exception as e:
                print(f"❌ Issue in actions at step {back_step}: {e}")
                continue







            if not adjusted_scores:
                print(f"⚠️ Step {back_step} has no actions to retry excluding turn arond")
                continue

            # Never retry anything we've already tried at this step (pre- or post-goal)
            tried = self.tried_actions_by_step.get(back_step, set())

            # Candidates: strictly above BFS bottleneck and not the turn-around (0)
            candidates = [idx for idx, sc in adjusted_scores.items()
                        if sc is not None and sc > min_adjusted and idx != 0]

            # Global rule: remove anything ever tried at this step
            actions_to_retry = [i for i in candidates if i not in tried]

            if not actions_to_retry:
                print(f"+++++++++++ Step {back_step} has no retryable actions above min_adjusted={min_adjusted:.3f} (all tried)")
                continue

            # Optional: execute higher-scoring alternatives first
            actions_to_retry.sort(key=lambda i: adjusted_scores[i], reverse=True)


            # Reconstruct a_final from log (excluding turn-around)
            a_final_dict = {
                a["index"]: (a["distance"], a["angle"])
                for a in log["actions"]
                if a["index"] != 0
            }
            a_final = [a_final_dict[k] for k in sorted(a_final_dict.keys())]

            print(f"🔁 Rewinding to Step {back_step}, retrying actions: {actions_to_retry}")
            print(f"🧿 Set tree root from Step {back_step}:")
            print(f"   📍 Position = {restored_state.position}")
            print(f"   🧭 Rotation (quat) = {restored_state.rotation}")

            print("\n📋 Reconstructed a_final list (indexed by action number):")
            for i, (r, theta) in enumerate(a_final):
                print(f"  a_final[{i}] = distance: {r:.2f}, angle: {theta:.2f}°")

            # Restore internal state
            self.tree_root_state = restored_state
            self.tree_action_queue = actions_to_retry
            self.tree_root_a_final = a_final
            self.tree_root_score_log = [a["adjusted"] for a in log["actions"]]  # ✅ log all adjusted scores
            self.tree_root_step_ndx = back_step 
            self.defer_rewind_to_root = True

            

            #### in the future we can start from the step -1 instead of rewinding to this step again ######
            self.start_ndx = back_step - 1




            return True, a_final

        print("⚠️ No valid rewind candidates found")
        return False, None












    def generate_grid_edge_score_list_from_adjusted(self):
        """
        Generate a list of transitions between grid cells with associated adjusted scores.
        Uses self.adjusted_score[step - 1] and skips steps without grid_from/grid_current.
        Returns a list of tuples: (from_row, from_col, to_row, to_col, adjusted_score)
        """
        edge_list = []

        sorted_steps = sorted(self.step_action_log_history_dict.keys())

        
        for step in sorted_steps:
            if step == 0:
                continue  # Skip step 0, no previous step

            log = self.step_action_log_history_dict.get(step)
            if not log:
                continue

            grid_from = log.get("grid_from")
            grid_to = log.get("grid_current")
            score = self.adjusted_score.get(step - 1)  # Score from previous step

            if grid_from is None or grid_to is None or score is None:
            # if grid_from is None or grid_to is None:
                continue  # Skip rewinds or missing info

            edge_list.append((grid_from[0], grid_from[1], grid_to[0], grid_to[1], round(score, 3)))

        return edge_list



















    
    def _adjust_action_distance(self, agent_action, confidence_score):
        """
        Adjusts the action distance based on the VLM confidence score.
        Ensures the adjusted distance does not exceed the max calibrated distance.
        """
        max_action_dist_calibration = self.cfg.get('max_action_dist_calibration')  # Default to 1.7 if not found


        # If the agent is determined to stop, do nothing
        if confidence_score is None:
            print("Target Found")
            return agent_action  
        
        calibrate_distance = self.cfg.get('discourage_ratio')

        # print(f"discourage ratio {calibrate_distance}LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
        
        # discourage_ratio = 0.75
        
    
        adjusted_distance = agent_action.r * confidence_score * calibrate_distance



        final_distance = min(adjusted_distance, max_action_dist_calibration)

        print(f"[VLMNavAgent] Original distance: {agent_action.r}, Adjusted distance: {final_distance}")

        agent_action.r = final_distance  # Update action distance
        return agent_action







    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):

        self.tree_action_queue = []
        self.tree_root_state = None
        self.tree_root_score_log = []

        

        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self._initial_pose = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()

        # this will be passed to env.py
        self.agent_grid_history = {}
        self.teleport_step_flags = {}
        self.step_action_log = []
        ## stored in _prompting
        ## store the history of action with respect to the score and position/orientation ++++ this is a dict
        self.step_action_log_history_dict = {}
        ## store the history of the score which is choosen by the agent action ++++ this is a dict 
        self.step_score_history_dict = {}
        ## store the history of the GSV score  ++++ this is a dict 
        ## stored in _stopping_module
        self.global_semantic_score = None
        ## stored in step
        self.gsv_per_step = {} 
        self.adjusted_score = {}
        self.turnaround_streak = 0
        

        self.initiate_back_propagation = False
        self.first_reach = False

        self.start_ndx = []


        self.goal_reached = False


        ## the FINAL call stop decision will kill the entire episode ##
        self.overall_stop = False

        self.goal_grid_location = None


        self.step_action_ranking_dict = {}
        self.defer_rewind_to_root = False 

        self.best_bfs_path = set()
        self.best_bfs_min = None

        # which actions we have already tried at each step
        self.tried_actions_by_step = {}          # step_idx -> set of action indices
        # which step is the current rewind root (so we log tries against the right step)
        self.tree_root_step_ndx = None

        self.goal_steps = set() 


        self.rewind_origin_step = None
        self.immediate_turnaround_by_root = {}
        self.last_root_action = None

        self.parent_by_step = {}
        self.swipping_back = False




        ####################################################### initialize a csv file that saves the RRT score ###########################3



    def _construct_prompt(self, **kwargs):
        raise NotImplementedError
    
    def _choose_action(self, obs):
        raise NotImplementedError

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()



        
        
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none' and self.cfg['project']:
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor']
                )
                images['color_sensor'] = new_image


    # #### ensure the final print will not print the default arrow #####
    #     if called_stop:
    #         logging.info('Model called stop')
    #         self.stopping_calls.append(self.step_ndx)

    #         if self.cfg['navigability_mode'] != 'none' and self.cfg['project']:
    #             new_image = obs['color_sensor'].copy()

    #             # 🚫 Don't draw default arrows if goal is reached
    #             if not self.goal_reached:
    #                 a_final = self._project_onto_image(
    #                     self._get_default_arrows(), new_image, obs['agent_state'],
    #                     obs['agent_state'].sensor_states['color_sensor']
    #                 )
    #             else:
    #                 a_final = []

    #             images['color_sensor'] = new_image



        




        step_metadata = {
            'action_number': -10,
            'success': 1,
            'pivot': 1 if self.pivot is not None else 0,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }





        return a_final, images, step_metadata, stopping_response
    






    

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        if not self.cfg['project']:
            # Actions for the w/o proj baseline
            a_final = {
                (self.cfg['max_action_dist'], -0.28 * np.pi): 1,
                (self.cfg['max_action_dist'], 0): 2,
                (self.cfg['max_action_dist'], 0.28 * np.pi): 3,
            }
            return a_final, images

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else: 
            a_initial = self._navigability(obs)

            # print(f"debug here ################################### 1",a_initial)


            a_final = self._action_proposer(a_initial, agent_state)


            # print(f"debug here ################################### 2",a_final)
        
        # print("After _action_proposer (a_final):")
        # for mag, theta in a_final:
        #     print(f"  θ = {np.rad2deg(theta):.2f}°, r = {mag:.2f}")


        a_final_projected = self._projection(a_final, images, agent_state)


        # print("After projection (a_final_projected):")
        # for mag, theta in a_final_projected:
        #     print(f"  θ = {np.rad2deg(theta):.2f}°, r = {mag:.2f}")





        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state)
        return a_final_projected, images

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop and prints confidence scores."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)

        if 'done' in dct and 'global_semantic_score' in dct:
            done = int(dct['done'])
            gsv = float(dct['global_semantic_score'])
            self.global_semantic_score = gsv



            print(f"Stopping Decision: {done}, Global Semantic Score: {gsv:.2f}")

            return done == 1, stopping_response

        return False, stopping_response
    




    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']








        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial
    


    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""

        # min_angle = self.fov/self.cfg['spacing_ratio']

        min_angle = self.cfg['hard_spacing']

        # print(f"min angle: {np.rad2deg(min_angle):.10f}°")



        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']
        observe_frac = self.cfg['observe_frac']

        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

            # print(f"2/3: {clip_frac}")



        # print("Voxel map shape:", self.voxel_map.shape)
        # print("Explored map shape:", self.explored_map.shape)



        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75  
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        ###################################################################################
        # print("Filtered actions (r > {:.2f}):".format(filter_thresh))
        # for r, theta, is_unexplored in filtered:
        #     print(f"  θ: {np.rad2deg(theta):.2f}°, r: {r:.2f}, unexplored: {is_unexplored}")
        ###################################################################################

        filtered.sort(key=lambda x: x[1])







        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            

                ### longes 0 is distance 1 is angle 2 is if explored

                out.append([min(longest[0] * observe_frac, clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])

                for i in range(longest_ndx + 1, len(f)):
                    if all(abs(f[i][1] - t) > (min_angle * 1.0) for t in thetas):
                        out.append([min(f[i][0] * observe_frac, clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])

                for i in range(longest_ndx - 1, -1, -1):
                    if all(abs(f[i][1] - t) > (min_angle * 1.0) for t in thetas):
                        out.append([min(f[i][0] * observe_frac, clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])


                # print("Thetas after forward/backward spacing:")
                # for t in sorted(thetas):
                #     print(f"  θ = {np.rad2deg(t):.2f}°")



                for r_i, theta_i, e_i in filtered:

                    # print(theta_i)

                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i * observe_frac, clip_mag), theta_i, e_i))
                        # thetas.add(theta)

                        thetas.add(theta_i)





    
        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]





        # longest = max(filtered, key=lambda x: x[0])
        # longest_theta = longest[1]
        # smallest_theta = longest[1]
        # longest_ndx = filtered.index(longest)
        # out.append([min(longest[0], clip_mag), longest[1], longest[2]])
        
        # for i in range(longest_ndx+1, len(filtered)):
        #     if filtered[i][1] - longest_theta > min_angle:
        #         out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
        #         longest_theta = filtered[i][1]
        # for i in range(longest_ndx-1, -1, -1):
        #     if smallest_theta - filtered[i][1] > min_angle:
        #         out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
        #         smallest_theta = filtered[i][1]









        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])



















        ############### here the function is only filtering out the action NOT changing it##################
        original_distance_dict = dict(a_initial)  
        # Restore original distances before returning



        min_dist = self.cfg.get('min_action_dist')
        out = [
            [mag, theta, e]
            for mag, theta, e in out
            if original_distance_dict.get(theta, mag) >= min_dist
        ]

        # If everything got filtered out, keep existing cooldown fallback
        if (not out) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()






        return [(original_distance_dict.get(theta, mag), theta) for mag, theta, _ in out]

    
        # return [(mag, theta) for mag, theta, _ in out]



    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """        


        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor']
        )


        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']):
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor']
                
            )


        return a_final_projected
        

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number and confidence scores.
        """

# ############################################# extract angle ################################################
#         print("🧭 Candidate action angles (relative to agent's heading):")
#         for idx, (_, theta_i) in enumerate(a_final):
#             angle_deg = np.degrees(theta_i)
#             print(f"  Action {idx + 1}: θ = {theta_i:.2f} rad / {angle_deg:.1f}°")


        prompt_type = 'action' if self.cfg['project'] else 'no_project'
        action_prompt = self._construct_prompt(goal, prompt_type, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.actionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])

            self._link_parent_for_next_step(self.step_ndx)

            print("trigger 222222222222222222222222222222222222222222222222222222222222")


            if self.step_ndx not in self.tried_actions_by_step:
                self.tried_actions_by_step[self.step_ndx] = set()
            self.tried_actions_by_step[self.step_ndx].add(step_metadata['action_number'])







            # the direct output of the score can be unnormalized, here we normalize the score

            # Get raw confidence scores from response
            conf_scores_raw = response_dict.get('confident_score', [])

            print(f"🔎 Raw confident scores (from VLM) RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRr: {conf_scores_raw}")

            # Normalize them
            conf_scores_norm = VLMNavAgent.normalize_scores(conf_scores_raw)

            print(f"✅ Normalized confident scores NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN: {conf_scores_norm}")

            # Update step metadata
            step_metadata['confident_score'] = conf_scores_norm

            # Set the score to the max normalized score
            step_metadata['score'] = max(conf_scores_norm) if conf_scores_norm else 0.0




            print(f"✅ Normalized confident scores: {conf_scores_norm}")
            # print(f"⭐ Max score (used as 'score'): {step_metadata['score']}")











            # Save sorted action indices by score (highest to lowest)  Save sorted action indices by score (highest to lowest)  Save sorted action indices by score (highest to lowest)
            action_ranking = sorted(
                [(i, score) for i, score in enumerate(conf_scores_norm)],
                key=lambda x: -x[1]
            )
            self.step_action_ranking_dict[self.step_ndx] = action_ranking






            turnaround_available = self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']
            conf_scores = step_metadata['confident_score']
            a_final = list(a_final)
            step_number = self.step_ndx

            self.step_score_history_dict[step_number] = step_metadata['score']

            # print("📊 Step Score History:")
            # for step, score in self.step_score_history_dict.items():
            #     print(f"  Step {step}: Score = {score}")


            # Save log entry for this step


            self._record_log_entry(step_number, a_final, conf_scores, turnaround_available)







            norm = VLMNavAgent.normalize_scores(step_metadata['confident_score'])  
            print(f"Highes Score Action: {step_metadata['action_number']}, normalized score for all actions: {norm}")

            
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            
            # score is the highest score
            logging_data['CONFIDENCE_SCORE'] = step_metadata.get('score')
            # confident_score is the score for each action
            logging_data['CONFIDENT_SCORE'] = step_metadata.get('confident_score')
            logging_data['PROMPT'] = action_prompt
            logging_data['RESPONSE'] = response




        return step_metadata, logging_data, response




    def _record_log_entry(self, step_number, a_final, conf_scores, turnaround_available):
        """
        Logs the agent's position, orientation, and scored actions at the given step.
        Stores the log in `self.step_action_log_history_dict` as a structured dictionary.
        """
        gsv = self.global_semantic_score

        actions = []

        if a_final is None or conf_scores is None:
            actions = None
        else:
            actions = []
            if turnaround_available:
                # Action 0: turn-around
                r_turn, theta_turn = a_final[-1]
                score_turn = conf_scores[0]
                adjusted_turn = score_turn * gsv
                actions.append({
                    "index": 0,
                    "angle": float(theta_turn),  # already in radians
                    "distance": float(r_turn),
                    "score": float(score_turn),
                    "adjusted": float(adjusted_turn),
                })
                aligned_actions = zip(range(1, len(a_final)), a_final[:-1], conf_scores[1:])
            else:
                aligned_actions = zip(range(len(a_final)), a_final, conf_scores)

            for i, r_theta, score in aligned_actions:
                r, theta = r_theta
                adjusted = score * gsv
                actions.append({
                    "index": i,
                    "angle": float(theta),
                    "distance": float(r),
                    "score": float(score),
                    "adjusted": float(adjusted),
                })






        # Get agent pose

        agent = self.simWrapper.sim.get_agent(0)
        agent_state = agent.get_state()
        state = agent_state
        pos = state.position
        rot = state.rotation

        if step_number in self.agent_grid_history:
            curr_grid = self.agent_grid_history[step_number]
            grid_current = [curr_grid[0], curr_grid[1]]
        else:
            curr_grid = None


        if step_number > 0 and (step_number - 1) in self.agent_grid_history:
            prev_grid = self.agent_grid_history[step_number - 1]
            grid_from = [prev_grid[0], prev_grid[1]]
        else:
            grid_from = None




        # parent = self._find_step_by_grid(grid_from, step_number)
        # if parent is None:
        #     # fallback: linear parent if present
        #     parent = step_number - 1 if step_number - 1 in self.step_action_log_history_dict else None
        # self.parent_by_step[step_number] = parent


        parent = self._determine_parent_step(step_number)
        self.parent_by_step[step_number] = parent





        log_entry = {
            "step": step_number,
            # "position": [round(float(p), 2) for p in pos],
            "position": [round(float(p), 6) for p in pos],
            "rotation": [float(rot.w), float(rot.x), float(rot.y), float(rot.z)],
            "grid_current": grid_current,
            "grid_from": grid_from,
            "actions": actions
        }

        # Save to log
        self.step_action_log_history_dict[step_number] = log_entry
        self.step_action_log.append(log_entry)

        # Print human-readable format
        print(f"Step {log_entry['step']}")
        print(f"📍 Agent Location: {log_entry['position']}")
        print(f"🧭 Agent Rotation (quat): {log_entry['rotation']}")

        print(f"📦 Grid From: {log_entry['grid_from']}")
        print(f"📦 Grid Currently at  : {log_entry['grid_current']}")

        if actions is not None:
            for a in log_entry['actions']:
                print(f"  Action {a['index']}: angle = {a['angle']}, distance = {a['distance']}m, score = {a['score']}, adjusted = {a['adjusted']}")
        else:
            print("⚠️ No actions recorded (a_final or conf_scores was None)")

        # for a in log_entry['actions']:
        #     print(f"  Action {a['index']}: angle = {a['angle']}, distance = {a['distance']}m, score = {a['score']}, adjusted = {a['adjusted']}")























    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask

    def _get_default_arrows(self):
        """
        Get the action options for when the agent calls stop the first time, or when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7



        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle),
        ]

        
        default_actions.sort(key=lambda x: x[1])


        return default_actions

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i

        for i in range(num_points - 4):
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        # if chosen_action == -1:
        
        if self.goal_reached:

            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
            self.goal_reached = False

            return projected

        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if (self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        return projected


    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        
        # print(f"2/3: {clip_frac},max: {clip_dist}")

        # clip_frac = 10000.0
        # clip_dist = 10000.0

        clipped = min(clip_frac * r, clip_dist)
        
        # clipped = 0

        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15

            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)

        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]
        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        try:
            eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            logging.error(f'Error parsing response {response}')
            return {}



class GOATAgent(VLMNavAgent):
 
    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        if goal['mode'] == 'image':
            stopping_images = [obs['color_sensor'], goal['goal_image']]
        else:
            stopping_images = [obs['color_sensor']]

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, stopping_images, goal)
        if goal['mode'] == 'image':
            images['goal_image'] = goal['goal_image']

        step_metadata.update({
            'goal': goal['name'],
            'goal_mode': goal['mode']
        })

        # If model calls stop two times in a row, we return the stop action and terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor'],
                    goal_image=goal['goal_image'] if goal['mode'] == 'image' else None
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata
    
    def _construct_prompt(self, goal: dict, prompt_type: str, num_actions=0):
        """Constructs the prompt, depending on the goal modality. """
        if goal['mode'] == 'object':
            task = f'Navigate to the nearest {goal["name"]}'
            first_instruction = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
        if goal['mode'] == 'description':
            first_instruction = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it. "
            task = first_instruction
        if goal['mode'] == 'image':
            task = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            first_instruction = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate as close as you can to it. "

        if prompt_type == 'stopping':        
            stopping_prompt = (f"The agent has the following navigation task: \n{task}\n. The agent has sent you an image taken from its current location{' as well as the goal image. ' if goal['mode'] == 'image' else '. '} "
                                f'Your job is to determine whether the agent is close to the specified {goal["name"].upper()}'
                                f"First, tell me what you see in the image, and tell me if there is a {goal['name']} that matches the description. Then, return 1 if the agent is close to the {goal['name']}, and 0 if it isn't. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt

        if prompt_type == 'pivot':
            return f'{first_instruction} Use your prior knowledge about where items are typically located within a home. '
        
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see, and if you have any leads on finding the {goal['name']}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        
        if prompt_type == 'action':
            action_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
            f"There are {num_actions-1} red arrow(s) superimposed onto your observation, which represent potential actions. " 
            f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. {'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
            f"First, tell me what you see, and if you have any leads on finding the {goal['name']}. Second, tell me which general direction you should go in. "
            f"Lastly, explain which action is the best and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def reset_goal(self):
        """Called after every subtask of GOAT. Notably does not reset the voxel map, only resets all areas to be unexplored"""
        self.stopping_calls = [self.step_ndx-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']


class ObjectNavAgent(VLMNavAgent):



    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']


        ########################### print agent location ###############################
        print("📍 Agent Position:", agent_state.position)
        print("🧭 Agent Rotation:", agent_state.rotation)

        yaw_deg = get_agent_heading_angle(agent_state.rotation)
        print("🧭 Agent Rotation in euler degree:", yaw_deg)

        ########################### RRT star here ###############################
        map_origin = self.cfg.get('map_origin')
        # print(f"printing map_origin {map_origin}")







        x_start = agent_state.position[0] - map_origin[0] # X position in meters
        y_start = agent_state.position[2] - map_origin[1]  # 
        start = (x_start, y_start) ## this is only for the location inside of the image 






        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal



        if isinstance(a_final, dict):
            a_final = list(a_final.keys())

        # check if turn around is added into an option
        turnaround_available = (self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown']
        turn_around_action = (0.75, np.pi)
        if turnaround_available and turn_around_action not in a_final:
            
            a_final.append(turn_around_action)


############################################################################################# NAV agent####################################################3



        # if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1 and not self.initiate_back_propagation:
        if len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx and not self.initiate_back_propagation:

            if not self.first_reach:
                self.start_ndx=self.step_ndx - 1


                self.first_reach = True




            print(11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)




        # if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1 and not self.initiate_back_propagation:

        if len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx:
        # if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:


            self.initiate_back_propagation = True

            self.goal_reached = True

            self.goal_steps.add(self.step_ndx)


            ################## we can still record the log with conf_score to be none
            step_number = self.step_ndx
            a_final = None
            conf_scores = None
            self._record_log_entry(step_number, a_final, conf_scores, turnaround_available)
            ##################



            # ############ for testing only #############
            # agent_action = PolarAction.stop
            # return agent_action, {
            #     'step_metadata': {'action_number': -2, 'success': 1},
            #     'logging_data': {'note': 'backtrack initiated'},
            #     'a_final': a_final,
            #     'images': {'color_sensor': obs['color_sensor']}
            # }
            # ############ for testing only #############


            return None, {
                'step_metadata': {'action_number': -2, 'success': 1},
                'logging_data': {'note': 'backtrack initiated'},
                'a_final': a_final,
                'images': {'color_sensor': obs['color_sensor']}
            }
  








            # ⛔ Only stop if backtrack failed
            print("🛑 No backtrack options — stopping agent.")
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop


            logging_data = {}


            print("stooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooop")

            logging_data['STOPPING RESPONSE'] = stopping_response
            metadata = {
                'step_metadata': step_metadata,
                'logging_data': logging_data,
                'a_final': a_final,
                'images': images

            }






            return agent_action, metadata




        global_angles = []

        for idx, (_, theta_i) in enumerate(a_final):
            angle_deg_relative = np.degrees(theta_i)
            angle_deg_global = (angle_deg_relative + yaw_deg) % 360
            global_angles.append(angle_deg_global)
            # print(f"  Action {idx + 1}: θ = {angle_deg_relative:.1f}° (relative), {angle_deg_global:.1f}° (global)")














        else:
            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor']
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }




        #####################################################################################################
        gsv = self.global_semantic_score




























            #####################################################
            # please note that the maximum error is saved in _post_episode function under env.py
















 

        return agent_action, metadata


    def _construct_prompt(self, goal: str, prompt_type: str, num_actions: int=0):
        if prompt_type == 'stopping':


            # stopping_prompt = (
            #                 f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            #                 f"Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note that a chair is NOT a sofa, which is NOT a bed. "
            #                 f"First, describe what you see in the image and whether a {goal} is present. "
            #                 f"Second, you have two actions to choose from. First action: return 1 if the agent is VERY CLOSE to the {goal}. Second action: return 0 if it is far away, does not exist, or you are not sure. "
            #                 f"Third, based on what is visible in the image, provide a score between 0.0 and 1.0 representing how much this scene is worth exploring further. "
            #                 f"This is called the global semantic score. A score close to 1.0 means the scene appears promising and informative, suggesting that moving forward or scanning the area may help locate the {goal}. "
            #                 f"A score close to 0.0 means the scene appears uninformative, irrelevant, or unlikely to contain useful paths or cues. "
            #                 f"Format your response in JSON format:\n"
            #                 f"{{'done': <1 or 0>, 'global_semantic_score': <float between 0.0 and 1.0>}}"
            #             )
            
            stopping_prompt = (
                            f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                            f"Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note that a chair is NOT a sofa, which is NOT a bed. "
                            f"First, describe what you see in the image and whether a {goal} is present. "
                            f"Second, you have two actions to choose from. First action: return 1 if the agent is VERY CLOSE to the {goal}. Second action: return 0 if it is far away, does not exist, or you are not sure. "
                            f"Third, Independently, rate the SCENE'S EXPLORATION POTENTIAL as a float in [0.0, 1.0], "
                            f"named global_semantic_score. This score MUST depend only on the current environment, "
                            f"NOT on whether the goal is present or visible. High scores mean the scene has open, "
                            f"traversable, informative paths (e.g., clear corridors, multiple branches, large visible free space). "
                            f"Low scores mean likely dead-ends, cluttered/tight spaces, blocked passages, or no promising directions."
                            f"Important rules for global_semantic_score:"
                            f"Do NOT increase the score just because the {goal} is visible."
                            f"Base it on openness, navigability cues, line of sight, and apparent paths."
                            f"Examples: \n"
                            f"0.0 to 0.1 → the view is completely blocked, directly facing a wall, with CLEARLY NO navigable path\n"
                            f"0.1 to 0.3 → the view has no clear outlet, close to a wall, or almost blocked\n"
                            f"0.3 to 0.7 → the view has a clear outlet or large navigable space "
                            f"(the higher the score, the clearer and more navigable it looks)\n"
                            f"0.7 to 1.0 → the view has multiple outlets, corridors, or very large navigable space to navigate\n"
                            f"Respond in JSON:\n"
                            f"{{'done': <1 or 0>, 'global_semantic_score': <float 0.0 to 1.0>}}"
                        )










            return stopping_prompt
        


        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action achieves that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal.upperstopping_prompt()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            
            turnaround_available = self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']

            # action_prompt = (
            #     f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
            #     f"Use your prior knowledge about where items are typically located within a home. "
            #     f"There are {num_actions} actions that you can choose from. "
            #     f"Actions are shown with red arrows superimposed onto your observation, labeled with numbers in white circles. "
            #     f"{'NOTE: If you see a white circle with number 0, it means there is an action for turn around. Choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. '}"
            #     f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
            #     f"Second, tell me which general direction you should go in. "
            #     f"Lastly, explain which action achieves that best and return it as JSON in the format: "
            #     f"{{'action': <action_key>, 'score': <confidence_score>, 'confident_score': [<score_0>, <score_1>, ..., <score_n>]}}. "
            #     f"'action' must be an integer not a string. "
            #     f"You must generate exactly {num_actions} confidence scores, one for each action shown. "
            #     f"The 'confident_score' list represents probabilities for each action and MUST sum exactly to 1.0. "
            #     f"{'If Action 0 (turn around) is available, its confidence score must appear first in the list, followed by Action 1, Action 2, etc.' if turnaround_available else 'The scores should be listed in order: Action 1, Action 2, Action 3, and so on.'}"
            # )

            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions} actions that you can choose from. "
                f"Actions are shown with red arrows superimposed onto your observation, labeled with numbers in white circles. "
                f"{'NOTE: If you see a white circle with number 0, it means there is an action for turn around. Choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. '}"
                f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
                f"Second, tell me which general direction you should go in. "
                f"Lastly, explain which action achieves that best and return it as JSON in the format: "
                f"{{'action': <action_key>, 'score': <confidence_score>, 'confident_score': [<score_0>, <score_1>, ..., <score_n>]}}. "
                f"The 'confident_score' list represents probabilities for each action "
                f"'action' must be an integer not a string and an independent confidence value in [0, 1]  "
                f"Do NOT normalize or force the scores to sum to 1. "
                f"You must generate exactly {num_actions} confidence scores, one for each action shown. "
                f"{'If Action 0 (turn around) is available, its confidence score must appear first in the list, followed by Action 1, Action 2, etc.' if turnaround_available else 'The scores should be listed in order: Action 1, Action 2, Action 3, and so on.'}"
            )



























            # action_prompt = (
            #     f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
            #     f"Use your prior knowledge about where items are typically located within a home. "
            #     f"There are {num_actions} actions that you can choose from. "
            #     f"Actions are shown with red arrows superimposed onto your observation, labeled with numbers in white circles. "
            #     f"{'NOTE: If you see a white circle with number 0, it means there is an action for turn around. Choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. '}"
            #     f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
            #     f"Second, tell me which general direction you should go in. "
            #     f"Lastly, explain which action achieves that best and return it as JSON in the format: "
            #     f"{{'action': <action_key>, 'score': <confidence_score>, 'confident_score': [<score_0>, <score_1>, ..., <score_n>]}}. "
            #     f"'action' must be an integer not a string. "
            #     f"Generate exactly {num_actions} scores, one for each action shown. "
            #     f"Each s_i is an independent confidence value in [0, 1] for action i. "
            #     f"Higher scores mean the action is more likely to bring you closer to the {goal.upper()} or otherwise more promising. "
            #     f"Lower scores mean the action is less likely to help reach the goal, blocked, or less useful. "
            #     f"Do NOT normalize or force the scores to sum to 1. "
            #     f"{'If Action 0 (turn around) is available, its confidence score must appear first in the list, followed by Action 1, Action 2, etc.' if turnaround_available else 'The scores should be listed in order: Action 1, Action 2, Action 3, and so on.'}"
            #     f"If two actions are visually/geometrically similar (e.g., small angle difference or targeting the same opening/corridor), "
            #     f"their scores should be close (e.g., difference ≤ 0.10)."
            # )

            
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')




######################### score cannot be zero ###############################



