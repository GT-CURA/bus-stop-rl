# SB3 & misc
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3 import PPO
from resources.custom_policies import StopMLPPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Project moduless
from rl import StreetView, StreetViewEnv
from settings import S 
from resources.loader import StopLoader

def make_env():
    sv = StreetView()
    stop_loader = StopLoader(sv)
    stop_loader.load_stops("assets/all_scores.json", shuffle_stops=False, num_positives=600)
    sv.launch()
    return StreetViewEnv(sv, stop_loader)

def train(save_path: str, load_path = None):
    """
    Train the model, either a fresh version or from a saved path.

    :param env: The training environment from rl module.
    :param save_path: Path to save the model to, including checkpoints.
    :param load_path: If resuming training, specify existsing model path.
    """
    vec_env = DummyVecEnv([make_env])
    vec_env = VecFrameStack(vec_env, n_stack=S.stack_sz)

    # Resume training 
    if load_path:
        model = PPO.load(load_path, env=vec_env)
    
    else:
        # Create PPO model
        model = PPO(
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
        save_freq=750,
        save_path='./models/',
        name_prefix='PPO'
    )

    # Setup custom log
    logger = configure(S.log_dir, ["csv", "stdout"])
    model.set_logger(logger)

    # Begin learning
    model.learn(total_timesteps=25600, callback=checkpoint_callback)
    
    # Save model, close gym
    model.save(save_path)

def infer(env: StreetViewEnv, model_path: str, num_stops = 20):
    # Wrap environment just like in 2training
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # Load the model
    model = PPO.load(model_path, env=vec_env)

    # Run inference
    for episode in range(num_stops):
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = vec_env.step(action)

if __name__ == "__main__":
    train("models/PPO", "PPO_11250_steps")
    # infer(vec_env, "models/PPO")