import numpy as np
from settings import S
from src.stop_detector import StopDetector
from src.utils.misc import Misc
from src.rl_env.graph import Graph

class Episode():
    def __init__(self, stop, stop_detector: StopDetector, pic):
        self.log = []
        self.reward = 0.0
        self.steps = 0
        self.found = False
        self.space_presses = 0
        self.stop = stop
        self.steps_since_found = 0
        self.prev_move = None
        self.stop_detector = stop_detector
        self.zoom_amt = 0
        self.current_node = None

        # Determine geo info
        self.initial_lat, self.initial_lng, self.initial_heading = pic.lat, pic.lng, pic.heading

        # Build graph class
        self.graph = Graph()

    def get_features(self, img, output, pic, add_neighbors=True):
        # Build node
        self.current_node = self.graph.add_node(pic, add_neighbors)

        # Get features, bb info from stop detector
        yolo_feats = self.stop_detector.extract_features(img, output)

        # Get spatial info from SV URL
        lat, lng, heading = pic.lat, pic.lng, pic.heading
        
        # Calculate diff between initial and new lats
        delta_lat = lat - self.initial_lat
        delta_lon = lng - self.initial_lng

        # Calculate distance vector. Grows smaller after 50 meters
        dist = Misc.haversine(self.initial_lat, self.initial_lng, lat, lng)
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
            delta_lat,
            delta_lon,
            dist_scaled,
            heading_sin,
            heading_cos,
            zoom_amt,
            spacebar_presses,
            remaining_steps
        ], dtype=np.float32)

        # Concat features
        return np.concat([yolo_feats, spatial_vec])

    def update(self, key, img, pic):
        # Update steps if key != enter
        if key != "Next":
            self.steps += 1 

        # Log space bar presses and zooms
        if key == "Return":
            self.space_presses += 1
            self.zoom_amt = 0

        # Update zoom level
        if key == "Zoom":
            self.zoom_amt += 1
        elif key in ["Forwards", "Backwards"]:
            self.zoom_amt = 0

        # Run stop detector model to get conf for assessment
        output = self.stop_detector.run(img)

        # Extract features from observation
        add_neighbors = True
        if key == "Return":
            add_neighbors = False
        features = self.get_features(img, output, pic, add_neighbors)

        # Use output to derive initial score
        conf, found = self.stop_detector.score_output(
            output, 
            self.current_node, 
            pic, 
            self.steps,
            self.initial_lat,
            self.initial_lng)

        # Update guesses
        self.graph.update_hypotheses(self.current_node, self.steps)

        # See if this episode is finished
        done = False
        if key == "Next":
            reward, done = self.check_done(found)

        # Determine score if not
        else:
            reward, done = self.score(conf, key, found, pic)

         # Update "found" status
        if found and not self.found:
            self.found = True
            self.steps_since_found = 1
        elif self.found: 
            self.steps_since_found += 1

        self.log.append(key)

        # Add to total reward for this episode (for logging)
        self.reward += reward 

        # Write log if done
        # if done:
        #     self.log_manager.add(self)
    
        # Announce results to console 
        Misc.announce(self, key, reward)
        self.prev_move = key
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

    def score(self, conf, key, found, pic):
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
        if key == "Return":
            if self.space_presses > S.free_spacebar_presses:
                rtrn_penalty = min(S.spacebar_penalty * self.space_presses, .3)

        # Add bonus for finding stop (once)
        found_bonus = 0.0
        if found and not self.found: 
            found_bonus = S.found_boost

        # Add bonus if visiting a new node
        new_node_bonus = 0.0
        if self.current_node.visits == 1:
            new_node_bonus = S.new_node_bonus 

        # Calulate spatial rewards 
        graph_rwd = self.graph.calc_graph_rwd(pic.pano_id)
        direction_rwd = self.graph.calc_direction_rwd(self.current_node, pic.heading)
        coord_rwd = self.graph.calc_coord_rwd(pic.lat, pic.lng)
        
        # Apply weights to spatial rewards 
        graph_rwd *= S.graph_weight
        direction_rwd *= S.heading_weight
        coord_rwd *= S.coord_weight
        
        # Scale rewards
        reward = (raw_reward + graph_rwd + direction_rwd + coord_rwd 
                  + new_node_bonus +  found_bonus
                  - rtrn_penalty - move_cap_penalty)
        
        # Clip reward to ensure stability 
        final_reward = np.clip(reward, -.5, .5)
        print(f"Graph reward: {graph_rwd} \nDirection reward: {direction_rwd} \nCoord Reward: {coord_rwd}")
        return final_reward, done