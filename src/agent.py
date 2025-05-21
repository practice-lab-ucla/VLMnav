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
from rrt_star_call import plan_rrt_star, plot_rrt_result




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
        self.tree_root_state = []






        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       
        self.pivot = PIVOT(self.actionVLM, self.fov, self.resolution, max_action_length=cfg['max_action_dist']) if cfg['pivot'] else None

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        self.last_obs = obs.copy() 


        if getattr(self, "defer_rewind_to_root", False):
            print("↩️ Rewinding to root before applying new action")
            self.defer_rewind_to_root = False  # consume flag

            # Step 1: Reset agent pose
            agent = self.simWrapper.sim.get_agent(0)
            new_state = habitat_sim.AgentState()
            new_state.position = self.tree_root_state.position
            new_state.rotation = self.tree_root_state.rotation
            agent.set_state(new_state)

            # Step 2: Refresh observation
            obs = self.simWrapper.sim.get_sensor_observations(0)
            obs['agent_state'] = agent.get_state()

            # ✅ Restore 'goal' if it was present
            if hasattr(self, "last_obs") and "goal" in self.last_obs:
                obs["goal"] = self.last_obs["goal"]


            # Step 3: Override action proposal to be consistent with tree
            metadata = {}
            metadata['a_final'] = self.tree_root_a_final








            # 👉 If inside tree, continue taking queued actions
            if self.tree_action_queue:
                print("🌲 Continuing tree-style queue:", self.tree_action_queue)
                print()
                next_action = self.tree_action_queue.pop(0)
                agent_action = self._action_number_to_polar(next_action, list(self.tree_root_a_final))

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

                chosen_action_image = obs['color_sensor'].copy()
                self._project_onto_image(
                    a_final, chosen_action_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor'],
                    chosen_action=next_action
                )
                metadata['images']['color_sensor_chosen'] = chosen_action_image

                print(f"➡️ Taking queued action: {next_action}")
                self.step_ndx += 1
                return agent_action, metadata
            








# ########################### print agent location ###############################
#         print("📍 Agent Position:", agent_state.position)
#         print("🧭 Agent Rotation:", agent_state.rotation)

#         quat = agent_state.rotation  # This is a habitat_sim.geo.Quaternionf
#         quat_xyzw = [quat.x, quat.y, quat.z, quat.w]
#         r = R.from_quat(quat_xyzw)
#         euler_deg = r.as_euler('xyz', degrees=True)
#         print("🧭 Agent Rotation in euler degree:", euler_deg)


#         x_start = agent_state.position[0] + 1 # X position in meters
#         y_start = agent_state.position[2] + 1.5 # 
#         start = (x_start, y_start)
#         goal = (2.0, 2.5)
#         map_path = "topdown_maps_single/occupancy_h2.1.npy"


#         path, nodes, occupancy, start_goal = plan_rrt_star(start, goal, map_path)

#         if not path:
#             print("⚠️ No valid path found. Skipping plot.")
#         else:
#             plot_rrt_result(path, nodes, occupancy, start_goal, map_path)



        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        agent_action, metadata = self._choose_action(obs)



        # print(f"get issue@@@@@@@@@@@@@@@@@@@", self.tree_action_queue)

        # if it is empty create a tree



        step_metadata = metadata['step_metadata']




        # === STEP 2: Tree-style top-actions selection ===
        if step_metadata.get('action_number') != -1:  # ✅ Only proceed if not terminating
            threshold = self.cfg.get('vlm_score_threshold')
            turnaround_available = self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']
            action_offset = 0 if turnaround_available else 1

            scored_actions = [
                (i + action_offset, score)
                for i, score in enumerate(step_metadata['confident_score'])
                if score >= threshold
            ]

            scored_actions.sort(key=lambda x: x[1], reverse=True)
            top_actions = [idx for idx, _ in scored_actions]
            step_metadata['top_actions'] = top_actions

            print(f'✅ Tree-style top actions selected: {top_actions}')
        else:
            print("⛔ Skipping tree-style selection — agent has chosen to stop.")














        top_actions = metadata['step_metadata'].get('top_actions', [])
        if not self.tree_action_queue and len(top_actions) > 1:

            print(f"top action number size", len(top_actions))



            self.tree_root_state = obs['agent_state']
            self.tree_action_queue = top_actions.copy()


            

            self.tree_root_a_final = metadata['a_final']  # ✅ Save original a_final

            print(f"🌲 Tree-style queue initialized: {self.tree_action_queue}")




        # ##################### override the action with the tree action ################
        # if self.tree_action_queue:
        #     print("🌲 Tree-style queue active:", self.tree_action_queue)

            ########################## Remove the first item in the list and return it
            next_action = self.tree_action_queue.pop(0)  # Get next high-scoring action

            print(f"override previous action, action to take now:",next_action)

            # apply the action in the original recorded a_final
            agent_action = self._action_number_to_polar(next_action, list(self.tree_root_a_final))


            # Update metadata to reflect overridden action
            metadata['step_metadata']['action_number'] = next_action




        ######################################################################  rewind##########################
        selected_action = metadata['step_metadata']['action_number']



        print(f"degbut the queue remaining action",self.tree_action_queue)

        if selected_action == 0 and self.tree_action_queue:
            print("🔁 Turned around — will rewind to root next step")
            self.defer_rewind_to_root = True

            # agent = self.simWrapper.sim.get_agent(0)
            # new_state = habitat_sim.AgentState()
            # new_state.position = self.tree_root_state.position
            # new_state.rotation = self.tree_root_state.rotation
            # agent.set_state(new_state)

            # obs = self.simWrapper.sim.get_sensor_observations(0)
            # obs['agent_state'] = agent.get_state()

            # metadata['a_final'] = self.tree_root_a_final


                
        # new_state.position = np.array([9.5, 2.06447, 1])
        # new_state.rotation = np.array([0.0, -0.76604444, 0.0, -0.64278761])


        confidence_score = metadata['step_metadata'].get('score') 



        # Adjust action distance based on confidence score
        agent_action = self._adjust_action_distance(agent_action, confidence_score)
        # Print updated action details
        print(f"Final Action Selected -> Distance: {agent_action.r}, Angle: {agent_action.theta}, Score: {confidence_score}")
        print("")

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
        return agent_action, metadata
    

    
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
        
        adjusted_distance = agent_action.r * confidence_score
        final_distance = min(adjusted_distance, max_action_dist_calibration)

        print(f"[VLMNavAgent] Original distance: {agent_action.r}, Adjusted distance: {final_distance}")

        agent_action.r = final_distance  # Update action distance
        return agent_action







    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):

        self.tree_action_queue = []
        self.tree_root_state = None



        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()

        # this will be passed to env.py
        self.max_rrt_score_error = 0.0  # Reset at start of episode



        ####################################################### initialize a csv file that saves the RRT score ###########################3
        RRT_SCORE_LOG_PATH = "score_data/rrt_score_log.csv"

        if not os.path.exists(RRT_SCORE_LOG_PATH):
            os.makedirs(os.path.dirname(RRT_SCORE_LOG_PATH), exist_ok=True)
            with open(RRT_SCORE_LOG_PATH, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Step", "RRT_Score_error"])




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
        

        a_final_projected = self._projection(a_final, images, agent_state)


        # print(f"debug here ################################### 3",a_final_projected)

        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state)
        return a_final_projected, images

    # def _stopping_module(self, stopping_images: list[np.array], goal):
    #     """Determines if the agent should stop."""
    #     stopping_prompt = self._construct_prompt(goal, 'stopping')
    #     stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
    #     dct = self._eval_response(stopping_response)
    #     if 'done' in dct and int(dct['done']) == 1:
    #         return True, stopping_response
        
    #     return False, stopping_response

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop and prints confidence scores."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)

        if 'done' in dct and 'confident_score' in dct:
            done = int(dct['done'])
            confident_scores = dct['confident_score']
            normalized_scores = VLMNavAgent.normalize_scores(confident_scores)  # Ensure scores sum to 1
            
            print(f"Stopping Decision: {done}, Confidence Scores for Stop and Not stop: {normalized_scores}")

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
        min_angle = self.fov/self.cfg['spacing_ratio']


        # min_angle = 1
        # print(f"debug here hhhhhhhhhhhhhhhhhhhhhhhhhhhh",min_angle )


        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']

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

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

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
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                # for i in range(longest_ndx+1, len(f)):
                #     if f[i][1] - longest_theta > (min_angle*0.9):
                #         out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                #         thetas.add(f[i][1])
                #         longest_theta = f[i][1]
                # for i in range(longest_ndx-1, -1, -1):
                #     if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                #         out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                #         thetas.add(f[i][1])
                #         smallest_theta = f[i][1]

                for i in range(longest_ndx + 1, len(f)):
                    if all(abs(f[i][1] - t) > (min_angle * 0.9) for t in thetas):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])

                for i in range(longest_ndx - 1, -1, -1):
                    if all(abs(f[i][1] - t) > (min_angle * 0.9) for t in thetas):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])




                # print(thetas)
                for r_i, theta_i, e_i in filtered:

                    # print(theta_i)

                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta)

                # print("#######################################")
                # print(thetas)
    
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


        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])


        ############### here the function is only filtering out the action NOT changing it##################
        original_distance_dict = dict(a_initial)  
        # Restore original distances before returning
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
            step_metadata['score'] = float(response_dict.get('score', 0))  # Default to 0 if not provided

            # print(f"the score i, {step_metadata['score']}")

            step_metadata['confident_score'] = response_dict.get('confident_score', [])

            # print(f"Chosen Action: {step_metadata['action_number']}, Confidence Score: {step_metadata['score']}, Confident Score: {step_metadata['confident_score']}")




            # ===  STEP 2: Tree-style top-actions selection ===
            threshold = self.cfg.get('vlm_score_threshold')


            turnaround_available = self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']
            action_offset = 0 if turnaround_available else 1

            scored_actions = [
                (i + action_offset, score)
                for i, score in enumerate(step_metadata['confident_score'])
                if score >= threshold
            ]

            # Sort by confidence in ascending order
            scored_actions.sort(key=lambda x: x[1])

            # Extract only the action indices, now ordered from lowest to highest confidence
            top_actions = [idx for idx, _ in scored_actions]
            step_metadata['top_actions'] = top_actions




            print(f'getting actions ################################################',top_actions)
            # ===================================================







            norm = VLMNavAgent.normalize_scores(step_metadata['confident_score'])  
            print(f"Highes Score Action: {step_metadata['action_number']}, normalized score for all actions: {norm}")

            
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['CONFIDENCE_SCORE'] = step_metadata.get('score')
            logging_data['CONFIDENT_SCORE'] = step_metadata.get('confident_score')
            logging_data['PROMPT'] = action_prompt
            logging_data['RESPONSE'] = response

        return step_metadata, logging_data, response


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
            (self.cfg['stopping_action_dist'], angle)
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
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
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
        

        print(f"printing map_origin {map_origin}")

        x_start = agent_state.position[0] - map_origin[0] # X position in meters
        y_start = agent_state.position[2] - map_origin[1]  # 

        # x_start = agent_state.position[0] + 0.77397 # X position in meters
        # y_start = agent_state.position[2] + 1.5698568 # 



        ######################################### initiate RRT
        start = (x_start, y_start)
        goal = (2.0, 2.5)
        # goal = (99.0, 99.0)
        height = self.cfg.get('rrt_map_height')
        map_path = f"topdown_maps_single/occupancy_h{height:.2f}.npy"
        path, nodes, occupancy, start_goal, reference_angle, reference_point = plan_rrt_star(start, goal, map_path)


        if reference_angle is not None:
            reference_angle_deg = np.degrees(reference_angle)
            print('Reference angle from RRT* (degrees):', reference_angle_deg)
        else:
            print('⚠️ No reference angle from RRT* (path not found)')



        if not path:
            print("⚠️ No valid path found. Skipping plot.")
        else:
            plot_rrt_result(path, nodes, occupancy, start_goal, map_path, reference_point)
        ####################################################





        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        ########################### Extract action angles ###############################
        # print("🧭 Candidate action angles (relative to agent's heading):")
        # for idx, (_, theta_i) in enumerate(a_final):
        #     angle_deg = np.degrees(theta_i)
        #     print(f"  Action {idx + 1}: θ = {theta_i:.2f} rad / {angle_deg:.1f}°")


        ########################### calculate the best option and print###############################

        # yaw_deg = euler_deg[1]
        # for idx, (_, theta_i) in enumerate(a_final):
        #     angle_deg_relative = np.degrees(theta_i)
        #     angle_deg_global = (angle_deg_relative + yaw_deg) 
        #     print(f"  Action {idx + 1}: θ = {angle_deg_relative:.1f}° (relative), {angle_deg_global:.1f}° (global)")
        ################################################################################





        # If the model calls stop two times in a row, terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
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









        if reference_angle is not None and 'confident_score' in step_metadata:
            reference_angle_deg = np.degrees(reference_angle)
            print(f"✅ RRT* Reference Global Angle: {reference_angle_deg:.1f}°")

            # yaw_deg = euler_deg[1]
            yaw_deg = get_agent_heading_angle(agent_state.rotation)

            global_angles = []
            for idx, (_, theta_i) in enumerate(a_final):
                angle_deg_relative = np.degrees(theta_i)
                angle_deg_global = (angle_deg_relative + yaw_deg) % 360
                global_angles.append(angle_deg_global)
                print(f"  Action {idx + 1}: θ = {angle_deg_relative:.1f}° (relative), {angle_deg_global:.1f}° (global)")

            # Find the best matching angle
            diffs = [abs((angle - reference_angle_deg + 180) % 360 - 180) for angle in global_angles]
            best_action_idx = int(np.argmin(diffs))
            closest_angle = global_angles[best_action_idx]

            # Extract correct score
            confident_scores = step_metadata.get('confident_score', [])
            turnaround_available = self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']

            # score_idx = best_action_idx + 1 if turnaround_available else best_action_idx


            num_actions = len(a_final)

            if turnaround_available:
                # Map visual action index → score index
                if best_action_idx == num_actions - 1:  # turnaround action (last in a_final)
                    score_idx = 0
                else:
                    score_idx = best_action_idx + 1
            else:
                score_idx = best_action_idx


            # score_idx = best_action_idx

            # print('debug ############################',score_idx)

            # print('debug ############################',len(confident_scores))




            if 0 <= score_idx < len(confident_scores):
                confidence = confident_scores[score_idx]

                step_metadata['rrt_score'] = confident_scores[score_idx]
                print(f"🎯 Best VLM Option Matching RRT*: Action {best_action_idx + 1} (θ ≈ {closest_angle}°), Confidence: {confidence}")


            else:
                print("⚠️ Best matching index out of range of confidence scores.")



            rrt_score = step_metadata.get('rrt_score', None)
            rrt_score_error = 1 - rrt_score
            self.max_rrt_score_error = max(self.max_rrt_score_error, rrt_score_error)



            print(f"im pritingggggggggggggggggggggggggggggggggggggggggggggggggg",self.max_rrt_score_error)

            # Log episode, step, and score 
            try:
                with open("score_data/rrt_score_log.csv", mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        self.episode_ndx,
                        self.step_ndx,
                        rrt_score_error
                    ])
            except Exception as e:
                print(f"⚠️ Failed to log RRT score: {e}")


























        return agent_action, metadata

    def _construct_prompt(self, goal: str, prompt_type: str, num_actions: int=0):
        if prompt_type == 'stopping':
            stopping_prompt = (f"The agent has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
                            f"Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note that a chair is NOT a sofa, which is NOT a bed. "
                            f"First, describe what you see in the image and whether a {goal} is present. "
                            f"Second, you have two actions to choose from. First action: return 1 if the agent is VERY CLOSE to the {goal}. Second action: return 0 if it is far away, does not exist, or you are not sure. "
                            f"Along with your decision, provide confidence scores for each of actions. The first score should be the score for the first action, and the second score should be the score for the second action."
                            f"Format your response in JSON format: "
                            f"{{'done': <1 or 0>, 'confident_score': [<confidence_for_stopping>, <confidence_for_not_stopping>]}}. "
                            f"The 'confident_score' list represents probabilities for each action and MUST sum exactly to 1.0. "
                            f"Normalize the values if necessary.")
            return stopping_prompt

    

        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action achieves that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            
            turnaround_available = self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']

            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                f"Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions} actions that you can choose from. "
                f"Actions are shown with red arrows superimposed onto your observation, labeled with numbers in white circles. "
                f"{'NOTE: If you see a white circle with number 0, it means there is an action for turn around. Choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if turnaround_available else ''}"
                f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
                f"Second, tell me which general direction you should go in. "
                f"Lastly, explain which action achieves that best and return it as JSON in the format: "
                f"{{'action': <action_key>, 'score': <confidence_score>, 'confident_score': [<score_0>, <score_1>, ..., <score_n>]}}. "
                f"'action' must be an integer not a string. "
                f"You must generate exactly {num_actions} confidence scores, one for each action shown. "
                f"The 'confident_score' list represents probabilities for each action and MUST sum exactly to 1.0. "
                f"{'If Action 0 (turn around) is available, its confidence score must appear first in the list, followed by Action 1, Action 2, etc.' if turnaround_available else 'The scores should be listed in order: Action 1, Action 2, Action 3, and so on.'}"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

                # f"{'If Action 0 (turn around) is available, its confidence score must appear first in the list.' if turnaround_available else ''}"

                # f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                # f"Use your prior knowledge about where items are typically located within a home. "
                # f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                # f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                # f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                # f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
                # f"Second, tell me which general direction you should go in. "
                # f"Lastly, explain which action achieves that best and return it as JSON in the format: "
                # f"{{'action': <action_key>, 'score': <confidence_score>, 'confident_score': [<score_1>, <score_2>, ..., <score_n>]}}. "
                # f"The 'score' must be exactly equal to the confidence value of the chosen action in 'confident_score'. "
                # f"The 'confident_score' list represents probabilities for each action and MUST sum exactly to 1.0. "
                # f"Normalize the values if necessary."


                # f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible. "
                # f"Use your prior knowledge about where items are typically located within a home. "
                # f"There are {num_actions} of actions that you can choose from"
                # f"Actions with red arrows superimposed onto your observation, which represent potential actions. " 
                # f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                # f"{'NOTE: If you see a white circle with number 0, it means there is an action for turn around. Choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                # f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. "
                # f"Second, tell me which general direction you should go in. "
                # f"Lastly, explain which action achieves that best and return it as JSON in the format: "
                # f"{{'action': <action_key>, 'score': <confidence_score>, 'confident_score': [<score_1>, <score_2>, ..., <score_n>]}}. "
                # f"You must generate exactly {num_actions} confidence scores, one for each action shown. "
                # f"The 'confident_score' list represents probabilities for each action and MUST sum exactly to 1.0. "
                # f"Normalize the values if necessary."