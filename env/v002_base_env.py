import random
import sys
import gym
from ultimate_gym import UltimateEnv, Screen, Controller
from libultimate.enums import Action
from yuzulib.client import register_to_server
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

# TODO: killが2回おきてepisodeがスキップしてしまう。
# TODO: serverに接続してるものがいるかどうか
# TODO: 研究室サーバに複数設置できるか
# TODO: rllibから接続できるか、実行できるか
class BaseEnv(gym.Env):
    def __init__(self, config: EnvContext):
        width, height = 128, 128
        # 0,3,4,6,7
        server_address = self.get_server_address()
        screen = Screen(fps=6, address=server_address, width=width, height=height, grayscale=False)
        controller = Controller(address=server_address)
        self.env = UltimateEnv(screen, controller)
        self.action_space = Discrete(len(action_list))
        '''self.observation_space = Dict({
            "observation": Box(
                low=0,
                high=255,
                shape=(width, height, 1),
                dtype='uint8'
            ),
            "damage": Box(
                low=0,
                high=1,
                shape=(2,),
                dtype='float'
            )
        })'''
        self.observation_space = Box(
                low=0,
                high=255,
                shape=(width, height, 1),
                dtype='uint8'
            )

    def get_server_address(self):
        server_list = ["http://192.168.207.230:6000", "http://192.168.207.233:6000", "http://192.168.207.234:6000", "http://192.168.207.236:6000", "http://192.168.207.237:6000"]
        for address in server_list:
            print("try connection to {}".format(address))
            ok = register_to_server(address)
            if ok:
                print("register address: {}".format(address))
                return address
            else:
                print("cannot register address: {}".format(address))
        print("Error: all server address is registerd by other env")
        sys.exit(1)


    def step(self, action_num):
        obs, reward, done, info = self.env.step(action_list[action_num])
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[:, :, np.newaxis]
        obs = obs.astype(np.float32) / 255
        # reward fix to 0-1
        reward = reward / 50 # 10% damage is 0.2 reward
        # if killed add +-1 reward
        reward = -1 if info["kill"][0] == True else reward
        reward = 1 if info["kill"][1] == True else reward
        #p1_damage = info["damage"][0]/150 if info["damage"][0]/150 < 1 else 1
        #p2_damage = info["damage"][1]/150 if info["damage"][1]/150 < 1 else 1

        #observation = {
        #    "observation": np.array(obs),
        #    "damage": np.array([p1_damage, p2_damage])
        #}
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset(without_reset=True)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = obs[:, :, np.newaxis]
        obs = obs.astype(np.float32) / 255
        #observation = {
        #    "observation": np.array(obs),
        #    "damage": np.array([0,0])
        #}
        return obs


Env = BaseEnv

if __name__ == '__main__':
    env = BaseEnv({})
    obs = env.reset()
    print(obs["damage"], env.observation_space, env.action_space)
    obs, _, _, _ = env.step(0)