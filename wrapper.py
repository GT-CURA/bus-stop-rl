import gymnasium as gym
import numpy as np

class ActionOverrideWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        forced_action: int,
        force_prob: float = 0.05,
        enabled: bool = True
    ):
        super().__init__(env)
        self.forced_action = forced_action
        self.force_prob = force_prob
        self.enabled = enabled
        self.steps = 0

    def step(self, action):
        if self.steps > 2 and self.enabled:
            if np.random.rand() < self.force_prob:
                action = self.forced_action
                info = {"forced_action": True}
            else:
                info = {"forced_action": False}

        self.steps += 1
        obs, reward, terminated, truncated, env_info = self.env.step(action)
        return obs, reward, terminated, truncated, env_info

    def reset(self, *, seed = None, options = None):
        self.steps = 0
        return super().reset(seed=seed, options=options)