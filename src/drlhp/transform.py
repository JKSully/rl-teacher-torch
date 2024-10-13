import torch
from torch import Tensor, Size, device
import torch.nn as nn
import torch.nn.functional as F
from torchrl.data import TensorSpec

import torchrl
from torchrl.envs import EnvBase
from torchrl.envs.transforms import Transform, RewardScaling

from tensordict import TensorDict, NestedKey

from collections.abc import Sequence

from drlhp.reward_predictor import ComparisonRewardPredictor
from drlhp.summaries import AgentLogger
from drlhp.label_schedules import LabelAnnealer, LabelSchedule, ConstantLabelSchedule

class HumanPreferenceTransform(Transform):
    def __init__(self,
                 reward_predictor: nn.Module = None,
                 in_keys: Sequence[NestedKey] | None = None,
                 out_keys: Sequence[NestedKey] | None = None,
                 label_scheduler: LabelSchedule = None) -> None:
        if in_keys is None:
            in_keys = ['obs']
        if out_keys is None:
            out_keys = ['obs']
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        self.reward_predictor = reward_predictor if reward_predictor is not None else ComparisonRewardPredictor()
        self.label_scheduler = label_scheduler if label_scheduler is not None else ConstantLabelSchedule()
        self.network  = nn.ModuleList([self.reward_predictor])

    def forward(self, tensordict: TensorDict):
        raise NotImplementedError

    def to(self, *args, **kwargs):
        self.network.to(*args, **kwargs)
        return self

    def transform_done_spec(self, done_spec) -> TensorSpec:
        return super().transform_done_spec(done_spec)

    def transform_env_batch_size(self, batch_size) -> Size:
        return super().transform_env_batch_size(batch_size)

    def transform_env_device(self, device) -> device:
        return super().transform_env_device(device)

    def transform_input_spec(self, input_spec) -> TensorSpec:
        return super().transform_input_spec(input_spec)

    def transform_observation_spec(self, observation_spec) -> TensorSpec:
        return super().transform_observation_spec(observation_spec)

    def transform_output_spec(self, output_spec) -> TensorSpec:
        return super().transform_output_spec(output_spec)

    def transform_reward_spec(self, reward_spec) -> TensorSpec:
        return super().transform_reward_spec(reward_spec)
