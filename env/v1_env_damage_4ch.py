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
        reward = reward / 100 # 10% damage is 0.1 reward
        # if killed add +-1 reward
        reward = -1 if info["kill"][0] == True else reward
        reward = 1 if info["kill"][1] == True else reward
        obs = self._preprocess(obs)
        observation = {
            "observation": obs,
            "damage": np.array([info["damage"][0]/150 if info["damage"][0]/150 < 1 else 1, info["damage"][1]/150 if info["damage"][1]/150 < 1 else 1])
        }
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
        # create 4ch images
        ## stage obs
        stage_obs = self.extract_color(obs, [10, 50, 120] , [50, 150, 200])
        stage_obs = self.extract_max_white_area(stage_obs)
        ## no background obs
        background_obs = self.extract_color(obs, [30, 110, 40] , [80, 230, 140])
        background_obs = self.extract_max_white_area(background_obs)
        background_obs = cv2.bitwise_not(background_obs)
        no_background_obs = cv2.bitwise_and(background_obs, obs_gray)
        # 
        mario1_obs = self.extract_color(obs, [10, 10, 10] , [100, 50, 80]) # purple mario
        mario1_position = self.get_position(mario1_obs)
        mario2_obs = self.extract_color(obs, [80, 80, 0] , [130, 130, 70]) # white mario
        mario2_position = self.get_position(mario2_obs)
        no_background_stage_obs = cv2.bitwise_xor(background_obs, stage_obs)
        mario1_position_obs = self.extract_position_white_area(no_background_stage_obs, mario1_position[0], mario1_position[1])
        mario2_position_obs = self.extract_position_white_area(no_background_stage_obs, mario2_position[0], mario2_position[1])

        obs = np.stack([no_background_obs, stage_obs, mario1_position_obs, mario2_position_obs], axis=2)
        obs = cv2.resize(obs, (84, 84)) # (84, 84 , 4)
        obs = obs.astype(np.uint8)
        return obs

    def extract_color(self, img, lower, upper):
        img = cv2.inRange(img, np.array(lower), np.array(upper))
        return img

    def extract_max_white_area(self, img):
        contours = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # 一番面積が大きい輪郭を選択する。
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))

        # 黒い画像に一番大きい輪郭だけ塗りつぶして描画する。
        out = np.zeros_like(img)
        mask = cv2.drawContours(out, [max_cnt], -1, color=255, thickness=-1)

        # 背景画像と前景画像を合成
        img = np.where(mask==255, img, out)
        return img

    def extract_position_white_area(self, img, x, y):
        contours = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        # 一番面積が大きい輪郭を選択する。
        i = 0
        for cnt in contours:
            if len(cnt) > 300:
                i += 1
                cnt = np.squeeze(cnt)
                minX, maxX, minY, maxY = np.min(cnt[:, 0]), np.max(cnt[:, 0]), np.min(cnt[:, 1]), np.max(cnt[:, 1])
                if minX < x and x < maxX and minY < y and y < maxY:
                    out = np.zeros_like(img)
                    mask = cv2.drawContours(out, [cnt], -1, color=255, thickness=-1)

                    # 背景画像と前景画像を合成
                    img = np.where(mask==255, img, out)
                    return img
        none_img = np.zeros_like(img)
        return none_img

    def get_position(self, img):
        contours = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        # 一番面積が大きい輪郭を選択する。
        if len(contours) == 0:
            return 0, 0
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
        x = (min(max_cnt[:, :, 0]) + (max(max_cnt[:, :, 0])-min(max_cnt[:, :, 0]))/2)[0]
        y = (min(max_cnt[:, :, 1]) + (max(max_cnt[:, :, 1])-min(max_cnt[:, :, 1]))/2)[0]
        return x, y

Env = BaseEnv

if __name__ == '__main__':
    env = BaseEnv({})
    obs = env.reset()
    print(obs.shape, env.observation_space, env.action_space)
    obs, _, _, _ = env.step(action_list[0])
    print(obs.shape, obs)