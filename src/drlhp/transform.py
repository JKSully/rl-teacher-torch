import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchrl
from torchrl.envs import EnvBase
from torchrl.envs.transforms import Transform, RewardScaling

from tensordict import TensorDict, NestedKey

from collections.abc import Sequence

from drlhp.reward_predictor import ComparisonRewardPredictor


class HumanPreferenceTransform(Transform):
    def __init__(self,
                 label_schedule, # TODO
                 reward_predictor: nn.Module = None,
                 in_keys: Sequence[NestedKey] | None = None,
                 out_keys: Sequence[NestedKey] | None = None,):
        if in_keys is None:
            in_keys = ['obs']
        if out_keys is None:
            out_keys = ['obs']
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.reward_predictor = reward_predictor if reward_predictor is not None else ComparisonRewardPredictor()


    def forward(self, tensordict: TensorDict):
        pass

    def to(self, *args, **kwargs):
        pass

    def transform_done_spec(self, done_spec):
        return super().transform_done_spec(done_spec)

    def transform_env_batch_size(self, batch_size):
        return super().transform_env_batch_size(batch_size)

    def transform_env_device(self, device):
        return super().transform_env_device(device)

    def transform_input_spec(self, input_spec):
        return super().transform_input_spec(input_spec)

    def transform_observation_spec(self, observation_spec):
        return super().transform_observation_spec(observation_spec)

    def transform_output_spec(self, output_spec):
        return super().transform_output_spec(output_spec)

    def transform_reward_spec(self, reward_spec):
        return super().transform_reward_spec(reward_spec)
