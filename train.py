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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-f', '--file', default="",
                        help='Config file path')
    args = parser.parse_args()
    if not os.path.isfile(args.file):
        print("Error: argument '--file' is not found")
        sys.exit(1)
    results_path = Path(os.path.dirname(__file__)).joinpath("results").resolve()

    with open(args.file, 'r') as yml:
        config = yaml.load(yml)

    print(config)
    env = import_module("env.{}".format(config['env'])).Env
    model = import_module("model.{}".format(config['model'])).Model
    print(env, model)

    train_config = config["config"]
    train_config["env"] = env
    train_config["model"] = {
        "custom_model": model,
    }
    experiment_spec = Experiment(
        config["name"],
        run=config["run"],
        stop=config["stop"],
        config=train_config,
        local_dir=str(results_path),
        checkpoint_freq=10,
        max_failures=2
    )
    print("Training automatically with Ray Tune")
    analysis = tune.run_experiments(experiment_spec)

    ray.shutdown()
    print("Finished", analysis)