import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import multiprocessing as mp
import torch
import time

from utils.hw3.environment import Hw3GymEnv
from homework3 import Memory



def startTraining():
    load = False
    env = DummyVecEnv([lambda: Hw3GymEnv()])  # create the Gym environment
    model = PPO('MlpPolicy', env, verbose=1)  # create the model using PPO2 and the Gym environment
    if(load):
        model.load('models/PPO.pt')
    model.learn(total_timesteps=100_000, progress_bar=True)
    model.save('models/PPO.pt')

    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render()
    env.close()