import random
import gym
from ultimate_gym import UltimateEnv, Screen
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
from ray.rllib.env.env_context import EnvContext
from gym.spaces import Discrete, Box, Dict
import cv2
import numpy as np
from collections import deque
########################################
# Update
# 1. add reward -0.1 when the players' action missed
# 2. add reward -0.1 when the player do smash action over 3 times
# 3. add reward +0.01 every time step.
### 4 None. add reward +0.1 when the player action jump or upB at outside of the stage until every 1 times
#######################################

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
    Action.ACTION_DOWN_SPECIAL,
    Action.ACTION_GRAB,
    Action.ACTION_SHIELD,
    Action.ACTION_JUMP,
    Action.ACTION_SHORT_HOP,
    #Action.ACTION_UP_TAUNT,
    #Action.ACTION_DOWN_TAUNT,
    #Action.ACTION_LEFT_TAUNT,
    #Action.ACTION_RIGHT_TAUNT,
    Action.ACTION_SPOT_DODGE,
    Action.ACTION_RIGHT_ROLL,
    Action.ACTION_LEFT_ROLL,
    Action.ACTION_RIGHT_DASH,
    Action.ACTION_LEFT_DASH,
    Action.ACTION_RIGHT_WALK,
    Action.ACTION_LEFT_WALK,
    Action.ACTION_CROUCH,
    #Action.ACTION_RIGHT_CRAWL,
    #Action.ACTION_LEFT_CRAWL,
    Action.ACTION_RIGHT_STICK,
    Action.ACTION_LEFT_STICK,
    Action.ACTION_UP_STICK,
    Action.ACTION_DOWN_STICK,
    Action.ACTION_NO_OPERATION
]

class BaseEnv(gym.Env):
    def __init__(self, config: EnvContext):
        screen = Screen(fps=15)
        controller = Controller()
        training_mode = TrainingMode(
            controller=controller,
            stage=Stage.STAGE_HANENBOW, 
            player=Fighter.FIGHTER_MARIO,
            cpu=Fighter.FIGHTER_MARIO,
            cpu_level=9,
        )
        game_path = "/workspace/games/SSBU/Super Smash Bros Ultimate [v0].nsp"
        dlc_dir = "/workspace/games/SSBU/DLC/"
        self.env = UltimateEnv(game_path, dlc_dir, screen, controller, training_mode, without_setup=True)
        self.action_space = Discrete(len(action_list))
        self.observation_space = Dict({
            "observation": Box(
                low=0,
                high=255,
                shape=(84, 84, 1),
                dtype='uint8'
            ),
            "damage": Box(
                low=0,
                high=1,
                shape=(2,),
                dtype='float'
            )
        })
        self.smash_buffer = deque([], 3)

    def step(self, action_num):
        action = action_list[action_num]
        obs, reward, done, info = self.env.step(action)
        reward = self._add_reward(reward, action, info)
        obs = self._preprocess(obs)
        observation = {
            "observation": obs,
            "damage": np.array([info["damage"][0]/150 if info["damage"][0]/150 < 1 else 1, info["damage"][1]/150 if info["damage"][1]/150 < 1 else 1])
        }
        # print("reward: {}, info: {}, done: {}".format(reward, info, done))
        return observation, reward, done, info

    def _add_reward(self, reward, action, info):
        # reward fix to 0-1
        reward = reward / 100 # 10% damage is 0.1 reward
        # 3. add reward +0.01 every time step.
        reward += 0.01
        # 2. add reward -0.1 when the player do smash action over 3 times
        if action in [Action.ACTION_RIGHT_SMASH, Action.ACTION_LEFT_SMASH, Action.ACTION_UP_SMASH, Action.ACTION_DOWN_SMASH]:
            self.smash_buffer.append(True)
        else:
            self.smash_buffer.append(False)
        if self.smash_buffer.count(True) == len(self.smash_buffer):
            reward -= 0.1
        # 1. add reward -0.1 when the players' action missed
        if action in [Action.ACTION_RIGHT_SMASH, Action.ACTION_LEFT_SMASH, Action.ACTION_UP_SMASH, Action.ACTION_DOWN_SMASH]:
            if info["diff_damage"][1] == 0:
                reward -= 0.1
        
        # if killed add +-1 reward
        reward = -1 if info["kill"][0] == True else reward
        reward = 1 if info["kill"][1] == True else reward
        return reward

    def reset(self):
        obs = self.env.reset(without_reset=True)
        obs = self._preprocess(obs)
        observation = {
            "observation": obs,
            "damage": np.array([0,0])
        }
        return observation

    def _preprocess(self, obs):
        obs = cv2.resize(obs, (84, 84))
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
        obs = obs[:, :, np.newaxis]
        obs = obs.astype(np.uint8)
        return obs

Env = BaseEnv

if __name__ == '__main__':
    env = BaseEnv({})
    obs = env.reset()
    print(obs.shape, env.observation_space, env.action_space)
    obs, _, _, _ = env.step(action_list[0])
    print(obs.shape, obs)