from src.streetview.sv import StreetView
from src.stop_detector import StopDetector
from src.utils.logging import LogManager
from src.utils.loader import StopLoader
from src.rl_env.episode import Episode
from settings import S
import numpy as np
import gymnasium as gym
from atexit import register
from time import sleep

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
        self.episode = Episode(stop, self.stop_detector, self.sv.current_pic)
        
        # Set up frame stack 
        img = self.sv.get_img()
        yolo_output = self.stop_detector.run(img)
        features = self.episode.get_features(img, yolo_output, self.sv.current_pic)

        # Must run scoring function to build first node
        self.stop_detector.score_output(
            yolo_output, 
            self.episode.current_node,
            self.sv.current_pic,
            0,
            self.episode.initial_lat,
            self.episode.initial_lng
        )

        # Reset episode-specific vars
        self.reset_next = False

        # Give agent the observation
        return features, {}

    def step(self, action):
        # Wait time between steps
        sleep(S.wait_time)

        # Get key, get img
        done = False
        key = S.action_map[action]

        # Handle spacebar
        if key == "Return":
            self.sv.goto_start()

        # Handle other keys, skipping enter
        elif key != "Next":  
            self.sv.do_action(key)
        
        # Run stop detector on changed env
        img = self.sv.get_img() 

        # Udate episode, let it score etc.
        obs, reward, done = self.episode.update(key, img, self.sv.current_pic)
        
        return obs, reward, done, False, {"raw_reward": reward}