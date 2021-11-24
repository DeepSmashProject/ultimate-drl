import random
import gym
from ultimate_gym import UltimateEnv, Screen
from libultimate import Controller, Action, Fighter, Stage, TrainingMode
from ray.rllib.env.env_context import EnvContext
########################################
# Trading environment that takes only 
# one position and ends the episode 
# when the position reaches 0
#######################################
class CustomEnv(gym.Env):
    def __init__(self, config: EnvContext):
        screen = Screen(fps=30)
        controller = Controller()
        training_mode = TrainingMode(
            controller=controller,
            stage=Stage.STAGE_HANENBOW, 
            player=Fighter.FIGHTER_MARIO,
            cpu=Fighter.FIGHTER_MARIO,
            cpu_level=7,
        )
        game_path = "/workspace/games/SSBU/Super Smash Bros Ultimate [v0].nsp"
        dlc_dir = "/workspace/games/SSBU/DLC/"
        self.env = UltimateEnv(game_path, dlc_dir, screen, controller, training_mode, without_setup=False)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        return obs

Env = CustomEnv