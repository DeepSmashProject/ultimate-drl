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
import time
import ray
from ray.tune import Experiment
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.env.env_context import EnvContext
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
import gym
from gym.spaces import Discrete, Box
import numpy as np
from ray.rllib.utils.framework import try_import_tf, try_import_torch
path = str(Path(os.path.dirname(__file__)).resolve())
sys.path.append(path)
path = str(Path(os.path.dirname(__file__)).joinpath("utils/yolov5").resolve())
sys.path.append(path)
print(sys.path, path)
class RPGEnv(gym.Env):
    metadata = {'render.modes': ['human', 'ansi']}
    FIELD_TYPES = [
        'S',  # 0: スタート
        'G',  # 1: ゴール
        '~',  # 2: 芝生(敵の現れる確率1/10)
        'w',  # 3: 森(敵の現れる確率1/2)
        '=',  # 4: 毒沼(1step毎に1のダメージ, 敵の現れる確率1/2)
        'A',  # 5: 山(歩けない)
        'Y',  # 6: 勇者
    ]
    MAP = np.array([
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # "AAAAAAAAAAAA"
        [5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],  # "AA~~~~~~~~~~"
        [5, 5, 2, 0, 2, 2, 5, 2, 2, 4, 2, 2],  # "AA~S~~A~~=~~"
        [5, 2, 2, 2, 2, 2, 5, 5, 4, 4, 2, 2],  # "A~~~~~AA==~~"
        [2, 2, 3, 3, 3, 3, 5, 5, 2, 2, 3, 3],  # "~~wwwwAA~~ww"
        [2, 3, 3, 3, 3, 5, 2, 2, 1, 2, 2, 3],  # "~wwwwA~~G~~w"
        [2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2],  # "~~~~~~==~~~~"
    ])
    MAX_STEPS = 100

    def __init__(self, config: EnvContext):
        print("waiting setup...")
        self.id = random.random()
        time.sleep(2)
        # action_space, observation_space, reward_range を設定する
        self.action_space = gym.spaces.Discrete(4)  # 東西南北
        self.observation_space = gym.spaces.Box(
            low=0,
            high=len(self.FIELD_TYPES),
            shape=self.MAP.shape,
            dtype=np.float32
        )
        print(self.MAP.shape)
        self.reward_range = [-1., 100.]
        self.reset()

    def reset(self):
        # 諸々の変数を初期化する
        self.pos = self._find_pos('S')[0]
        self.goal = self._find_pos('G')[0]
        self.done = False
        self.damage = 0
        self.steps = 0
        return self._observe()

    def step(self, action):
        # 1ステップ進める処理を記述。戻り値は observation, reward, done(ゲーム終了したか), info(追加の情報の辞書)
        if action == 0:
            next_pos = self.pos + [0, 1]
        elif action == 1:
            next_pos = self.pos + [0, -1]
        elif action == 2:
            next_pos = self.pos + [1, 0]
        elif action == 3:
            next_pos = self.pos + [-1, 0]

        if self._is_movable(next_pos):
            self.pos = next_pos
            moved = True
        else:
            moved = False

        observation = self._observe()
        reward = self._get_reward(self.pos, moved)
        self.damage += self._get_damage(self.pos)
        self.done = self._is_done()
        self.steps += 1
        return observation, reward, self.done, {}

    def render(self, mode='human', close=False):
        # human の場合はコンソールに出力。ansiの場合は StringIO を返す
        #outfile = StringIO() if mode == 'ansi' else sys.stdout
        print('\n'.join(' '.join(
                self.FIELD_TYPES[elem] for elem in row
                ) for row in self._observe()
            ) + '\n'
        )

    def _get_reward(self, pos, moved):
        # 報酬を返す。報酬の与え方が難しいが、ここでは
        # - ゴールにたどり着くと 100 ポイント
        # - ダメージはゴール時にまとめて計算
        # - 1ステップごとに-1ポイント(できるだけ短いステップでゴールにたどり着きたい)
        # とした
        # 報酬は0~1
        if moved and (self.goal == pos).all():
            return max(100 - self.damage, 0)
        else:
            return -1

    def _get_damage(self, pos):
        # ダメージの計算
        field_type = self.FIELD_TYPES[self.MAP[tuple(pos)]]
        if field_type == 'S':
            return 0
        elif field_type == 'G':
            return 0
        elif field_type == '~':
            return 10 if np.random.random() < 1/10. else 0
        elif field_type == 'w':
            return 10 if np.random.random() < 1/2. else 0
        elif field_type == '=':
            return 11 if np.random.random() < 1/2. else 1

    def _is_movable(self, pos):
        # マップの中にいるか、歩けない場所にいないか
        return (
            0 <= pos[0] < self.MAP.shape[0]
            and 0 <= pos[1] < self.MAP.shape[1]
            and self.FIELD_TYPES[self.MAP[tuple(pos)]] != 'A'
        )

    def _observe(self):
        # マップに勇者の位置を重ねて返す
        observation = np.array(self.MAP.copy(), dtype=np.float32)
        observation[tuple(self.pos)] = self.FIELD_TYPES.index('Y')
        return observation

    def _is_done(self):
        # 今回は最大で self.MAX_STEPS までとした
        if (self.pos == self.goal).all():
            return True
        elif self.steps > self.MAX_STEPS:
            return True
        else:
            return False

    def _find_pos(self, field_type):
        return np.array(list(zip(*np.where(
        self.MAP == self.FIELD_TYPES.index(field_type)
    ))))
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
    env = RPGEnv
    #callbacks = import_module("callbacks.{}".format(config['callbacks'])).Callbacks
    print(env)

    train_config = config["config"]
    if "model_config" in config and "model" in config:
        print("apply custom model")
        model = import_module("model.{}".format(config['model'])).Model
        model_config = config["model_config"]
        model_config["custom_model"] = model
        train_config["model"] = model_config
    train_config["env"] = env
    #train_config["callbacks"] = callbacks
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

    #unregister_server()
    print("Training automatically with Ray Tune")
    ray.init()
    analysis = tune.run_experiments(experiment_spec)
    #analysis = tune.run_experiments(experiment_spec, callbacks=[])
    ray.shutdown()
    print("Finished", analysis)