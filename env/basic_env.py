import random
import gym
#from ultimate_env import UltimateEnv

########################################
# Trading environment that takes only 
# one position and ends the episode 
# when the position reaches 0
#######################################
class CustomEnv(gym.Wrapper):
    def __init__(self):
        #env = UltimateEnv()
        env = None
        super().__init__(env)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        return obs, reward, done, info

Env = CustomEnv