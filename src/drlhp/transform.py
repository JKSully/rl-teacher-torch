import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchrl
from torchrl.envs import EnvBase
from torchrl.envs.transforms import Transform, CatTensors
from torchrl.modules import MLP

from tensordict import NestedKey, TensorDictBase

from collections.abc import Sequence
from copy import copy
from math import prod

from drlhp.reward_predictor import ComparisonRewardPredictor
from drlhp.label_schedules import ConstantLabelSchedule, LabelSchedule, LabelAnnealer


class HumanPreferenceTransform(Transform):
    def __init__(self,
                 reward_predictor: nn.Module = None,
                 in_keys: Sequence[NestedKey] = None,
                 out_keys: Sequence[NestedKey] = None,
                 label_schedule: LabelSchedule = None,):

        if in_keys is None:
            in_keys = ['obs_segments', 'act_segments',
                       'alt_obs_segments', 'alt_act_segments']
        if out_keys is None:
            out_keys = ['logits']  # Or reward?
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        input_dim = prod(self.parent.observation_spec.shape) + \
            prod(self.parent.action_spec.shape)

        self.reward_predictor = reward_predictor if reward_predictor is not None else ComparisonRewardPredictor(
            input_dim)
        self.label_schedule = label_schedule if label_schedule is not None else ConstantLabelSchedule()
        self.label_annealer = LabelAnnealer()  # TODO: input args

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Steps:
        # 1. Get a set (2) of observations and actions from the top of the buffer
        # 2. The preference is the q_value and the non-preferred segment is the other
        # 3. Get the reward prediction for each segment
        # 4. Wait for the human to provide a preference
        # 5. Run forward pass on the reward predictor

        raise NotImplementedError("forward method not implemented")

    # Can _apply_transform take in multiple tensors (ie obs and act)?
    def _apply_transform(self, obs: torch.Tensor):
        pass

    def to(self, *args, **kwargs):
        self.reward_predictor.to(*args, **kwargs)
        return self

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
