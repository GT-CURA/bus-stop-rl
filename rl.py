from resources.streetview import StreetView
from resources.stop_detector import StopDetector
from resources.logging import LogManager
from resources.misc import Misc
from resources.loader import StopLoader
from settings import S
import numpy as np
import gymnasium as gym
from atexit import register

class StreetViewEnv(gym.Env):
    def __init__(self, streetview: StreetView, stop_loader: StopLoader):
        # Set stuff up!!
        super().__init__()
        self.sv = streetview
        self.stop_detector = StopDetector()
        self.stop_loader = stop_loader

        # PPO model design
        self.action_space = gym.spaces.Discrete(len(S.action_map))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(S.frame_dim,),
            dtype=np.float32
        )

        # Episode specific
        self.reset_next = True
        self.episode = None

        # Setup logging, register for exit
        self.log_manager = LogManager(flush_every=2, flush_interval=10)
        register(self.log_manager.shutdown)

    def reset(self, seed=None, options=None):
        # Get the next stop, load it 
        stop = self.stop_loader.load_stop()

        # Create new episode
        self.episode = Episode(stop, self.sv.page.url, self.stop_detector, self.log_manager, self.sv.current_pic)
        
        # Set up screenshot stack 
        img = self.sv.get_img()
        yolo_output = self.stop_detector.run(img)
        features = self.episode.get_features(img, yolo_output, self.sv.current_pic)

        # Reset episode-specific vars
        self.reset_next = False

        # Give model the observation
        features = np.array(features, dtype=self.observation_space.dtype)
        return features, {}

    def step(self, action):
        # Get key, take screenshot
        done = False
        key = S.action_map[action]

        # Handle spacebar
        if key == "Key.space":
            self.sv.goto_start()

        # Handle other keys, skipping enter
        elif key != "Key.enter":  
            self.sv.do_action(key)
        
        # Run stop detector on changed env
        img = self.sv.get_img() 

        # Udate episode, let it score etc.
        obs, reward, done = self.episode.update(key, img, self.sv.current_pic)
        
        return obs, reward, done, False, {"raw_reward": reward}

# Class for storing episode data
class Episode():
    def __init__(self, stop, url: str, stop_detector: StopDetector, log_manager: LogManager, pic):
        self.log = []
        self.reward = 0.0
        self.steps = 0
        self.found = False
        self.best_img = (float(-999), None)
        self.space_presses = 0
        self.amenity_scores = {}
        self.stop = stop
        self.steps_since_found = 0
        self.found_viewpoints = []
        self.new_viewpoint = False
        self.stop_detector = stop_detector
        self.log_manager = log_manager
        self.zoom_amt = 0
        
        # Determine geo info
        self.initial_lat, self.initial_lon, self.initial_heading = pic.lat, pic.lng, pic.heading

    def get_features(self, img, output, pic):
        # Get features, bb info from stop detector
        yolo_feats, found = self.stop_detector.extract_features(img, output)

        # Get spatial info from SV URL
        lat, lon, heading = pic.lat, pic.lng, pic.heading
        
        # Calculate diff between initial and new lats
        delta_lat = lat - self.initial_lat
        delta_lon = lon - self.initial_lon

        # Calculate distance vector. Grows smaller after 50 meters
        dist = Misc.haversine(self.initial_lat, self.initial_lon, lat, lon)
        dist_scaled = np.tanh(dist / 50)

        # Normalize heading difference
        delta_heading = np.radians(heading - self.initial_heading)
        delta_heading = np.arctan2(np.sin(delta_heading), np.cos(delta_heading))

        # Use sin/cos for smoothness
        heading_sin = np.sin(delta_heading)
        heading_cos = np.cos(delta_heading)

        # # Zoom count 
        zoom_amt = min(self.zoom_amt / 2, 1)
        zoom_scaled = zoom_amt * 2 -1

        # Tell model if these coords and heading have been used before
        self.new_viewpoint = False
        if found:
            view = (round(lat, 6), round(lon, 6), round(heading, 1))
            if view not in self.found_viewpoints:
                self.found_viewpoints.append(view)
                self.new_viewpoint = True
        viewpoint_count = min(len(self.found_viewpoints)/3, 1)

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
            viewpoint_count,
            remaining_steps,
            spacebar_presses, 
            zoom_scaled
        ], dtype=np.float32)

        # Concat features
        return np.concat([yolo_feats, spatial_vec])

    def update(self, key, img, pic):
        # Update steps
        self.steps += 1 

        # Punish spacebar spamming
        if key == "Key.space":
            self.space_presses += 1

        # # Handle zooming 
        # if key == "=":
        #     self.zoom_amt += 1
        # elif key in ["w","s","Key.space"]:
        #     self.zoom_amt = 0

        # Run stop detector model to get conf for assessment
        output = self.stop_detector.run(img)
        conf, found, boxes, box_sz = self.stop_detector.score_output(output)
        
        # Extract features from observation
        features = self.get_features(img, output, pic)

        # See if this episode is finished
        done = False
        if key == "Key.enter":
            reward, done = self.check_done(found)

        # Determine score if not
        else:
            reward, done = self.score(conf, key, found, box_sz)

         # Update "found" status
        if found and not self.found:
            self.found = True
            self.steps_since_found = 1
        elif self.found: 
            self.steps_since_found += 1

        # Unpack model results for log
        if boxes:
            for amenity, score in boxes.items():
                if amenity not in self.amenity_scores or score > self.amenity_scores[amenity]:
                    self.amenity_scores[amenity] = score
        self.log.append(key)

        # Check if this is the best image (to save it later), add to reward
        if reward > self.best_img[0]:
            self.best_img = (reward, img)
        self.reward += reward 

        # Update box size
        self.prev_box_sz = box_sz if box_sz else 0

        # Write log if done 
        if done:
            self.log_manager.add(self)

        # Announce results to console 
        Misc.announce(self, key, reward)
        return features, reward, done
    
    def check_done(self, found):
        # Don't allow before bus stop has been found or attempts exhausted
        if not found and not self.found:
            if self.steps <= S.min_steps:
                return S.premature_end, False
            
        # If found, don't allow without multiple perspectives (unless surpassed free steps)
        elif len(self.found_viewpoints) < 2:
            if self.steps_since_found > S.free_steps_after_found:
                return -.3, True
            else:
                return -.4, False
        
        # Base move on reward
        reward = S.move_on_reward

        # Reward moving on before using all free steps
        if self.found and self.steps_since_found <= S.free_steps_after_found:
            reward += S.efficiency_bonus

        # Write "best" image
        if self.best_img and S.save_screenshots:
            stop_name = self.stop.place_name.replace("/", "-")

            # Run model again :( to get annotations on a copy of the best image
            save_img = self.best_img[1].copy()
            results = self.stop_detector.run(save_img)
            results.save(filename=f"{S.log_dir}/{stop_name}_labeled.jpg")
        
        # Tell model to finish this episode
        return reward, True

    def score(self, conf, key, found, box_sz):
        """ Determines penalties and rewards based on episode data. """
        done = False
        reward = 0.0

        # Forcibly move on at max steps
        if self.steps >= S.max_steps:
            return -.85, True
        
        # Forcibly move on at max steps since found 
        if self.steps_since_found >= S.max_steps_after_found:
            return -.85, True
        
        # Dampen reward
        reward = conf * S.dampen_scalor

        # Punish going over a certain number of moves since finding the stop
        if self.steps_since_found > S.free_steps_after_found:
            reward -= (self.steps_since_found - S.free_steps_after_found) * S.after_found_punishment

        # Prevent spacebar spamming
        if key == "Key.space":
            if self.space_presses > S.free_spacebar_presses:
                reward -= S.spacebar_penalty * self.space_presses 

        # Add bonus if already found
        if found and self.found:
            reward += S.consecutive_boost

        # Incentivize larger boxes
        sz_reward = box_sz * S.size_scalar
        sz_reward = min(sz_reward, S.max_sz_pts)
        reward += sz_reward

        # Scale rewards
        reward = np.clip(reward, -1.0, 1.0)
        return reward, done