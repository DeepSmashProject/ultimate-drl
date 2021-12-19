import argparse
import numpy as np
import os
import random
import ray
from ray.tune import Experiment
from ray import tune
from importlib import import_module
from yuzulib.client import unregister_to_server
import yaml
from pathlib import Path
import sys
path = str(Path(os.path.dirname(__file__)).resolve())
sys.path.append(path)
path = str(Path(os.path.dirname(__file__)).joinpath("utils/yolov5").resolve())
sys.path.append(path)
print(sys.path, path)

def unregister_server():
    server_list = ["http://192.168.207.230:6000", "http://192.168.207.233:6000", "http://192.168.207.234:6000", "http://192.168.207.236:6000", "http://192.168.207.237:6000"]
    for address in server_list:
        print("try unregister from {}".format(address))
        ok = unregister_to_server(address)
        if ok:
            print("unregister address: {}".format(address))
        else:
            print("cannot unregister address: {}".format(address))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('-f', '--file', default="",
                        help='Config file path')
    parser.add_argument('-r', '--restore', default=None,
                        help='checkpoint file path')
    args = parser.parse_args()
    if not os.path.isfile(args.file):
        print("Error: argument '--file' is not found")
        sys.exit(1)
    results_path = Path(os.path.dirname(__file__)).joinpath("results").resolve()

    print(args.file)
    with open(args.file, 'r') as yml:
        config = yaml.safe_load(yml)

    print(config)
    env = import_module("env.{}".format(config['env'])).Env
    callbacks = import_module("callbacks.{}".format(config['callbacks'])).Callbacks
    print(env)

    train_config = config["config"]
    if "model_config" in config and "model" in config:
        print("apply custom model")
        model = import_module("model.{}".format(config['model'])).Model
        model_config = config["model_config"]
        model_config["custom_model"] = model
        train_config["model"] = model_config
    train_config["env"] = env
    train_config["callbacks"] = callbacks
    experiment_spec = Experiment(
        config["name"],
        run=config["run"],
        stop=config["stop"],
        config=train_config,
        local_dir=str(results_path),
        checkpoint_freq=10,
        restore=args.restore
        #max_failures=1
    )

    unregister_server()
    print("Training automatically with Ray Tune")
    ray.init()
    analysis = tune.run_experiments(experiment_spec)
    #analysis = tune.run_experiments(experiment_spec, callbacks=[])
    ray.shutdown()
    print("Finished", analysis)