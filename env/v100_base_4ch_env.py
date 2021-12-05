import random
import gym
from ultimate_gym import UltimateEnv, Screen
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
from ray.rllib.env.env_context import EnvContext
from gym.spaces import Discrete, Box, Dict
import cv2
import numpy as np
from collections import deque
from utils.yolov5.detector import Detector
########################################
# Trading environment that takes only 
# one position and ends the episode 
# when the position reaches 0
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
    #Action.ACTION_NO_OPERATION
]

class BaseEnv(gym.Env):
    def __init__(self, config: EnvContext):
        screen = Screen(fps=15)
        controller = Controller()
        self.detector = Detector()
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
                shape=(84, 84, 4),
                dtype='uint8'
            ),
            "damage": Box(
                low=0,
                high=1,
                shape=(2,),
                dtype='float'
            )
        })
        self.p1_p_buffer = deque([], self.buffer_size) # position buffer
        self.p2_p_buffer = deque([], self.buffer_size)

    def step(self, action_num):
        obs, reward, done, info = self.env.step(action_list[action_num])
        # reward fix to 0-1
        reward = reward / 50 # 10% damage is 0.2 reward
        # if killed add +-1 reward
        reward = -1 if info["kill"][0] == True else reward
        reward = 1 if info["kill"][1] == True else reward
        obs = self._preprocess(obs)
        observation = {
            "observation": obs,
            "damage": np.array([info["damage"][0]/150 if info["damage"][0]/150 < 1 else 1, info["damage"][1]/150 if info["damage"][1]/150 < 1 else 1])
        }
        print("reward: {}, done: {}, info: {}".format(reward, done, info))
        return observation, reward, done, info

    def reset(self):
        obs = self.env.reset(without_reset=True)
        obs = self._preprocess(obs)
        observation = {
            "observation": obs,
            "damage": np.array([0,0])
        }
        return observation

    def _preprocess(self, obs):
        obs_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        results = self.detector.detect(obs)
        stage_obs = np.zeros_like(obs_gray)
        player1_obs = np.zeros_like(obs_gray) # red mario
        player2_obs = np.zeros_like(obs_gray) # purple mario
        for r in results:
            xyxy = r["xyxy"]
            if r["cls"] == 0: # purple mario
                player2_obs[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 255
            elif r["cls"] == 1: # red mario
                player1_obs[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 255
            elif r["cls"] == 2: # stage
                stage_obs[xyxy[0]:xyxy[2], xyxy[1]:xyxy[3]] = 255
            
        obs = np.stack([obs_gray, stage_obs, player1_obs, player2_obs], axis=2)
        obs = cv2.resize(obs, (84, 84)) # (84, 84 , 4)
        obs = obs.astype(np.uint8)
        return obs

Env = BaseEnv

if __name__ == '__main__':
    env = BaseEnv({})
    obs = env.reset()
    print(obs.shape, env.observation_space, env.action_space)
    obs, _, _, _ = env.step(action_list[0])
    print(obs.shape, obs)