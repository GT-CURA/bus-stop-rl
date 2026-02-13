from src.stop_detector import StopDetector
import numpy as np 
from settings import S
from src.rl_env.graph import Graph
from src.utils.tools import haversine
from src.utils.context import RoadContext
import cv2 

class Episode():
    def __init__(self, stop_detector: StopDetector, context: RoadContext, stop, pic):
        self.reward = 0.0
        self.steps = 0
        self.found = False
        self.space_presses = 0
        self.stop = stop
        self.steps_since_found = 0
        self.stop_detector = stop_detector
        self.zoom_amt = 0
        self.current_node = None
        self.zoom_presses = 0 
        self.consec_acts = 0
        self.consec_pan = 0

        # Determine geo info
        self.initial_lat, self.initial_lng, self.initial_heading = pic.lat, pic.lng, pic.heading

        # Setup context for this stop
        self.context = context
        self.context.set_context(stop)

        # Build graph class
        self.graph = Graph(context)

        # Build current node
        self.current_node = self.graph.add_node(pic, False)

        # Can't think of a better way to do this 
        self.opposite = {
            "Backwards": "Forwards",
            "Forwards":"Backwards",
            "Counterclockwise":"Clockwise",
            "Clockwise":"Counterclockwise",
            "Next":"Next",
            "Return":"Return",
            "Zoom":None
        }
        self.prev_action = None

        # Announce new stop to console
        self.announce_reset()

    def get_features(self, img, output, pic):
        # Get features, bb info from stop detector
        yolo_vec = self.stop_detector.extract_features(img, output)

        # Get spatial info from SV URL
        lat, lng, heading = pic.lat, pic.lng, pic.heading

        # Calculate distance vector. Grows smaller after 50 meters
        dist = haversine(self.initial_lat, self.initial_lng, lat, lng)
        dist_scaled = np.tanh(dist / 50)

        # Normalize heading difference
        delta_heading = np.radians(heading - self.initial_heading)
        delta_heading = np.arctan2(np.sin(delta_heading), np.cos(delta_heading))

        # Use sin/cos for smoothness
        heading_sin = np.sin(delta_heading)
        heading_cos = np.cos(delta_heading)

        # Handle zoming
        zoom_amt = min(self.zoom_amt / 2, 1)

        # Provide steps after found
        remaining_steps = min(self.steps_since_found / S.free_steps_after_found, 1)
        
        # Provide spacebar presses 
        spacebar_presses = min(self.space_presses / S.free_spacebar_presses, 1)

        # Put all into a vec 
        spatial_vec = np.array([
            dist_scaled,
            heading_sin,
            heading_cos,
            zoom_amt,
            spacebar_presses,
            remaining_steps
        ], dtype=np.float32)

        # Adjust images 
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        small = cv2.resize(gray, (S.out_size, S.out_size), interpolation=cv2.INTER_AREA)
        normalized = small.astype(np.float32) / 255.0
        img_vec = normalized.flatten()

        # Get graph features
        graph_vec = self.graph.get_features(self.current_node)

        # Concat features
        return np.concatenate([yolo_vec, spatial_vec, graph_vec, img_vec]).astype(np.float32)

    def update(self, action, img, pic):
        # Update steps if key != enter
        if action != "Next":
            self.steps += 1 

        # Log space bar presses and zooms
        if action == "Return":
            self.space_presses += 1

        # Handle zooming 
        if action == "Zoom":
            self.zoom_presses += 1 

            # Update zoom level 
            if self.zoom_amt < 2:
                self.zoom_amt += 1
        else:
            self.zoom_amt = 0
            self.zoom_presses = 0

        # Run stop detector model to get conf for assessment
        output = self.stop_detector.run(img)

        # Build/update current node object 
        add_neighbors = True if action != "Return" else False
        self.current_node = self.graph.add_node(pic, add_neighbors)

        # Use output to derive initial score
        conf, found = self.stop_detector.score_output(
            output, 
            self.current_node, 
            pic, 
            self.steps,
            self.found
        )

        # Update guesses
        self.graph.update_hypotheses(self.current_node, self.steps, action)

        # See if this episode is finished
        done = False
        if action == "Next":
            reward, done = self.check_done(found)

        # Determine score if not
        else:
            reward, done = self.score(conf, action, found, pic)

         # Update "found" status
        if found and not self.found:
            self.found = True
            self.steps_since_found = 1
        elif self.found: 
            self.steps_since_found += 1

        # Extract features from observation
        features = self.get_features(img, output, pic)

        # Add to total reward for this episode (for logging)
        self.reward += reward 

        # Announce results to console 
        self.announce_step(action, reward)
        return features, reward, done
    
    def check_done(self, found):
        # Don't allow before bus stop has been found or attempts exhausted
        if not found and not self.found:
            if self.steps <= S.min_steps:
                return S.premature_end, False
            
        # Base move on reward
        reward = S.move_on_reward

        # Reward moving on before using all free steps
        if self.found and self.steps_since_found <= S.free_steps_after_found:
            reward += S.efficiency_bonus

        # Write "best" image
        # if S.save_best_img and self.best_vp["img"] is not None:
        #     stop_name = self.stop.place_name.replace("/", "-")

        #     # Run model again :( to get annotations on a copy of the best image
        #     save_img = self.best_vp["img"].copy()
        #     filename=f"{S.log_dir}/images/{stop_name}_best.jpg"
        #     if S.annotate_best_img:
        #         results = self.stop_detector.run(save_img)
        #         results.save()
        #     else:
        #         (filename, save_img)
        # Tell model to finish this episode
        return reward, True

    def score(self, conf, action, found, pic):
        """ Determines penalties and rewards based on episode data. """
        done = False
        reward = 0.0

        # Forcibly move on at max steps
        if self.steps >= S.max_steps:
            return -.5, True
        
        # Forcibly move on at max steps since found 
        if self.steps_since_found >= S.max_steps_after_found:
            return -.5, True
        
        # Dampen reward
        raw_reward = conf * S.dampen_scalor

        # Punish going over a certain number of moves since finding the stop
        move_cap_penalty = 0.0
        if self.steps_since_found > S.free_steps_after_found:
            move_cap_penalty = (self.steps_since_found - S.free_steps_after_found) * S.after_found_punishment

        # Prevent spacebar spamming
        rtrn_penalty = 0.0
        if action == "Return":
            if self.space_presses > S.free_spacebar_presses:
                rtrn_penalty = min(S.spacebar_penalty * self.space_presses, .3)

        # Add exponential cost for zooming 
        zoom_cost = 0.0
        if action == "Zoom":
            zoom_cost = S.zoom_cost
            if self.zoom_presses > 2:
                raw_reward = 0
        zoom_cost = min(zoom_cost, .5)

        # Add bonus for finding stop (once)
        found_bonus = 0.0
        if found and not self.found: 
            found_bonus = S.found_boost

        # Add bonus if visiting a new node
        new_node_bonus = 0.0
        if self.current_node.visits == 1:
            new_node_bonus = S.new_node_bonus 

        # Prevent getting stuck in action loop
        undo_penalty = 0.0
        if self.prev_action is not None and self.opposite[action] == self.prev_action:
            self.consec_acts += 1 
            undo_penalty = S.undo_penalty * (self.consec_acts ** 2)
            undo_penalty = min(undo_penalty, .3)
        else:
            self.consec_acts = 0

        self.prev_action = action
        
        # Calulate spatial rewards
        graph_rwd = self.graph.calc_graph_rwd(pic.pano_id)
        direction_rwd = self.graph.calc_direction_rwd(self.current_node, pic)
        coord_rwd = self.graph.calc_coord_rwd(pic)
        
        # Only allow spatial penalties if zooming 
        if action == "Zoom":
            graph_rwd = min(0, graph_rwd)
            direction_rwd = min(0, direction_rwd)
            coord_rwd = min(0, coord_rwd)

        # Apply weights to spatial rewards 
        graph_rwd *= S.graph_weight
        direction_rwd *= S.heading_weight
        coord_rwd *= S.coord_weight
        
        # Scale rewards
        reward = (raw_reward + graph_rwd + direction_rwd + coord_rwd 
                  + new_node_bonus +  found_bonus
                  - rtrn_penalty - move_cap_penalty - undo_penalty
                  - zoom_cost)
        
        # Clip reward to ensure stability 
        final_reward = np.clip(reward, -.5, .5)
        
        # Announce 
        if S.msg_score_breakdown:
            print(f"Graph reward: {graph_rwd} | Direction reward: {direction_rwd} | Coord Reward: {coord_rwd}")
        return final_reward, done
    
    def announce_reset(self):
        print("\n\n\n\n\n\n", "="*12, f"[STOP LOADED]","="*12)
        print(f"Spawn point ({self.stop.og_lat}, {self.stop.og_lng})")
        print(f"Stop ID {self.stop.place_name}")

    def announce_step(self, key, reward):
        """ Print stop info and action to console at each step. """
        print("\n\n", "="*15, f"[STEP {self.steps}]","="*15)
        print(f"Action: {key} \nReward: {reward:.3f} \nSteps Since Found: {self.steps_since_found}")
        print(f"Stop: {self.stop.place_name} ({self.stop.og_lat}, {self.stop.og_lng})")