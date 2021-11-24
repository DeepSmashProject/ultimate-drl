import random
import gym
#from ultimate_env import UltimateEnv
from ray.rllib.env.env_context import EnvContext
import numpy as np
import cv2
import torch
from torchvision import transforms as transforms
from gym.spaces import Discrete, Box
from PIL import Image
########################################
# Trading environment that takes only 
# one position and ends the episode 
# when the position reaches 0
#######################################
class TestEnv(gym.Env):
    def __init__(self, config: EnvContext):
        self.env = gym.make("Breakout-v0")
        self.action_space = Discrete(4)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(84, 84, 1),
            dtype='uint8'
        )

    def reset(self):
        obs = self.env.reset()
        obs = self._preprocess(obs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        #print(obs.shape)
        obs = self._preprocess(obs)
        return obs, reward, done, info

    def _preprocess(self, obs):
        # obs: (84, 84, 1)
        obs = np.array(obs)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84))
        obs = obs[:, :, np.newaxis]
        return obs


Env = TestEnv

env = TestEnv({})
obs = env.reset()
print(obs.shape, env.observation_space, env.action_space)
obs, _, _, _ = env.step(0)
print(obs.shape, obs)