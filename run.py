# SB3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Project modules
from settings import S
from src.feature_extractor import StopMLPPolicy
from src.rl_env.env import StreetViewEnv
from src.streetview.sv import StreetView
from src.utils.loader import StopLoader
from src.utils.server.server import start_server

def make_env(path: str, ignore_path: str = None):
    # Create streetview and loader
    sv = StreetView()
    stop_loader = StopLoader(sv)

    # Load stops, launch SV
    stop_loader.load_stops(path, ignore_path)

    # Pass YOLO to loader :(
    env = StreetViewEnv(sv, stop_loader)
    stop_loader.stop_detector = env.stop_detector
    return env, len(stop_loader.stops)

def train(save_path: str, stops_path: str, weights_path = None):
    """
    Train the agent, either a fresh version or from a saved path.

    :param save_path: Path to save the weights to, including checkpoints.
    :param stops_path: Path of the csv or json of stops to train on. 
    :param weights_path: If resuming training, specify path to pretrained weights.
    """
    env, _ = make_env(stops_path)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=S.stack_sz)

    # Resume training 
    if weights_path:
        agent = PPO.load(weights_path, env=vec_env)
    
    else:
        # Create PPO agent
        agent = PPO(
            policy=StopMLPPolicy,
            env=vec_env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=64,
            n_steps=2048,
            policy_kwargs=dict(normalize_images=False),
            tensorboard_log=S.log_dir,
            device="cuda:0"
        )

    # Creates checkpoint files while training and tensorboard log
    checkpoint_callback = CheckpointCallback(
        save_freq=4096,
        save_path='./weights/',
        name_prefix='PPO'
    )

    # Setup custom log
    logger = configure(S.log_dir, ["csv", "stdout"])
    agent.set_logger(logger)

    # Begin learning
    agent.learn(total_timesteps=409600, callback=checkpoint_callback)
    
    # Save weights, close gym
    agent.save(save_path)

def infer(stops_path: str, weights_path: str, ignore_path: str = None):
    """
    Primary inference loop. Used to actually 'run' the agent on a collection of bus stops.

    :param stops_path: Path of the csv or json of stops to find. 
    :param weights_path: Path to the agent's weights, trained previously. Exclude .zip
    """
    # Wrap environment
    env, num_stops = make_env(stops_path, ignore_path)
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=S.stack_sz)

    # Load the agent
    agent = PPO.load(weights_path, env=vec_env)
    
    # Must reset environment initially
    obs = vec_env.reset()

    # Iterate through each episode, allowing agent to run until done for each
    for ep in range(num_stops):
        while True:

            # Get action, do step
            action, _ = agent.predict(obs, deterministic=False)
            obs, reward, done, info = vec_env.step(action)

            # Break if episode finished
            if done[0]:
                break

if __name__ == "__main__":
    if S.run_server: 
        start_server(port=5000)
        
    # Run training/inference loop here!
    train("weights/PPO", "assets/easy.csv")
    # infer("assets/study_area.csv", "573440")