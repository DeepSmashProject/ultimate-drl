import argparse
import numpy as np
import os
import random
import ray
from ray.tune import Experiment
from ray import tune
from importlib import import_module
import yaml
from pathlib import Path
import sys
import torch
import cloudpickle

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-f', '--file', default="",
                        help='Config file path')
    parser.add_argument('-e', '--experiment_dir', default="",
                        help='Result directory path')
    parser.add_argument('-w', '--weight', default="",
                        help='Evaluate weight file name')
    args = parser.parse_args()
    if not os.path.isfile(args.file) or args.weight == "" or args.experiment_dir == "":
        print("Error: argument '--file' is not found")
        sys.exit(1)
    results_path = Path(os.path.dirname(__file__)).joinpath("results").resolve()

    with open(args.file, 'r') as yml:
        config = yaml.load(yml)

    #path = "{}/{}/DQN_BreakoutNoFrameskip-v4_6c7bd_00000_0_2021-11-20_23-01-57/checkpoint_000100".format(results_path, config["name"])
    #print(path)
    with open(args.experiment_dir, 'rb') as f:
        model = cloudpickle.load(f)
    print(model([]))
    '''print(config)
    env = import_module("env.{}".format(config['env'])).Env
    model = import_module("model.{}".format(config['model'])).Model
    weight_path = "{}/{}".format(args.experiment_dir, args.weight) #TODO
    model.load_state_dict(torch.load(weight_path))
    print(env, model)
    #env = gym.make()
    total_reward = 0
    for i in range(args.iter):
        obs = env.reset()
        done = False
        while not done:
            action = model(obs)
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            obs = next_obs
    env.close()
    print("Mean Reward: {}".format(total_reward/args.iter))'''
    print("finished.")