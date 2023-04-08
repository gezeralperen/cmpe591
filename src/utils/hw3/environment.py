import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from homework3 import Hw3Env

class Hw3GymEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):
        self.env = Hw3Env(render_mode="offscreen")
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 128, 128), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs.numpy(), reward, done, {}

    def reset(self):
        self.env.reset()
        return self.env.state().numpy()

    def render(self, mode='human'):
        if mode == 'human':
            self.env.render()
        else:
            return self.env.state().numpy()

    def seed(self, seed=None):
        self.env.seed(seed)
        return [seed]
