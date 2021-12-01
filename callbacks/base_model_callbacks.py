from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from typing import Dict
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
import numpy as np
from collections import deque

class BaseModelCallbacks(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.buffer_num = 100
        self.win_flag_buffer = deque([], self.buffer_num)

    def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, **kwargs):
        pass

    def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       **kwargs):
        info = episode.last_info_for()
        kill = info["kill"]
        self.win_flag_buffer.append(kill[1]) # player2 dead or not
        win_rate = self.win_flag_buffer.count(True) / len(self.win_flag_buffer) if len(self.win_flag_buffer) >= self.buffer_num else 0
        episode.custom_metrics["win_rate"] = win_rate

    def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_train_result(self, trainer, result: dict, **kwargs):
        pass

    def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
        pass

Callbacks = BaseModelCallbacks