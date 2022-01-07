import random
import gym
from ultimate_gym import UltimateEnv
from yuzulib.game.ssbu import Action
from gym.spaces import Discrete, Box
import cv2
import numpy as np
# 29 action
action_list = [
    Action.ACTION_JAB,
    Action.ACTION_RIGHT_TILT,
    Action.ACTION_LEFT_TILT,
    Action.ACTION_UP_TILT,
    Action.ACTION_DOWN_TILT,
    Action.ACTION_RIGHT_SMASH,
    Action.ACTION_LEFT_SMASH,
    Action.ACTION_UP_SMASH,
    Action.ACTION_DOWN_SMASH,
    Action.ACTION_NEUTRAL_SPECIAL,
    Action.ACTION_RIGHT_SPECIAL,
    Action.ACTION_LEFT_SPECIAL,
    Action.ACTION_UP_SPECIAL,
    Action.ACTION_UP_RIGHT_SPECIAL,
    Action.ACTION_UP_LEFT_SPECIAL,
    Action.ACTION_DOWN_SPECIAL,
    Action.ACTION_GRAB,
    Action.ACTION_SHIELD,
    Action.ACTION_JUMP,
    Action.ACTION_RIGHT_JUMP,
    Action.ACTION_LEFT_JUMP,
    Action.ACTION_SHORT_HOP,
    Action.ACTION_RIGHT_SHORT_HOP,
    Action.ACTION_LEFT_SHORT_HOP,
    #Action.ACTION_UP_TAUNT,
    #Action.ACTION_DOWN_TAUNT,
    #Action.ACTION_LEFT_TAUNT,
    #Action.ACTION_RIGHT_TAUNT,
    Action.ACTION_SPOT_DODGE,
    Action.ACTION_RIGHT_ROLL,
    Action.ACTION_LEFT_ROLL,
    #Action.ACTION_RIGHT_DASH,
    #Action.ACTION_LEFT_DASH,
    #Action.ACTION_RIGHT_WALK,
    #Action.ACTION_LEFT_WALK,
    #Action.ACTION_CROUCH,
    #Action.ACTION_RIGHT_CRAWL,
    #Action.ACTION_LEFT_CRAWL,
    Action.ACTION_RIGHT_STICK,
    Action.ACTION_LEFT_STICK,
    Action.ACTION_UP_STICK,
    Action.ACTION_DOWN_STICK,
    #Action.ACTION_NO_OPERATION
]

class BaseEnv(gym.Env):
    def __init__(self):
        self.env = UltimateEnv(fps=6)
        self.action_space = Discrete(len(action_list))
        self.prev_damaged_player = None
        self.p1_damage = 0
        self.p2_damage = 0
        self.observation_space = Box(
                low=0,
                high=255,
                shape=(84, 84, 3),
                dtype='uint8'
            )

    def step(self, action_num):
        action = action_list[action_num]
        obs, reward, done, info = self.env.step(action)
        # reward fix to 0-1
        reward = reward / 50 # 10% damage is 0.2 reward
        # combo bonus
        if info["diff_damage"][0] != 0: 
            if self.prev_damaged_player == 0:
                reward = reward * 0.8
            self.prev_damaged_player = 0
            self.p1_damage = info["damage"][0]
        elif info["diff_damage"][1] != 0:
            if self.prev_damaged_player == 1:
                reward = reward * 1.2
            self.prev_damaged_player = 1
            self.p2_damage = info["damage"][1]
        
        # early killed bonus
        if info["kill"][0]:
            reward = -1
            reward += ((self.p1_damage-100)/100)*0.5
        if info["kill"][1]:
            reward = 1
            reward += ((100-self.p2_damage)/100)*0.5
        obs = self._preprocess(obs)

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset(without_reset=False)
        obs = self._preprocess(obs)
        self.prev_damaged_player = None
        self.p1_damage = 0
        self.p2_damage = 0
        return obs

    def _preprocess(self, obs):
        # to (256, 256) Gray
        obs = cv2.resize(obs, (256, 256))
        #obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        #obs = obs[:, :, np.newaxis]
        obs = obs.astype(np.uint8)
        return obs

if __name__ == '__main__':
    env = BaseEnv()
    obs = env.reset()
    done = False
    while not done:
        action = random.randrange(len(action_list))
        obs, reward, done, info = env.step(action)
        print("obs: {}, reward: {}, done: {}, info: {}".format(obs.shape, reward, done, info))