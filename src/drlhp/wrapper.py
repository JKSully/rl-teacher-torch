import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import torchrl
from torchrl.envs import EnvBase, IsaacGymWrapper
from torchrl.envs.common import _EnvWrapper
from torchrl.data.utils import DEVICE_TYPING

from tensordict import NestedKey, TensorDictBase

from collections.abc import Sequence

from drlhp.reward_predictor import ComparisonRewardPredictor
from drlhp.label_schedules import ConstantLabelSchedule, LabelSchedule, LabelAnnealer

# Not in use for now


class HumanPreferenceWrapper(_EnvWrapper):
    def __init__(self,
                 env: EnvBase,
                 reward_predictor: nn.Module = None,
                 label_schedule: LabelSchedule = None,
                 **kwargs):
        if env is not None:
            kwargs['env'] = env

        super().__init__(**kwargs, allow_done_after_reset=True)

        self.reward_predictor = reward_predictor if reward_predictor is not None else ComparisonRewardPredictor()
        self.label_schedule = label_schedule if label_schedule is not None else ConstantLabelSchedule()
        self.label_annealer = LabelAnnealer()
        self.train_reward_predictor = True
        self.collect_preferences = False

    def _check_kwargs(self, kwargs):
        return super()._check_kwargs(kwargs)  # TODO

    def _init_env(self):
        return super()._init_env()  # TODO

    def _build_env(self, **kwargs):
        return super()._build_env(**kwargs)  # TODO

    def _make_specs(self, env):
        return super()._make_specs(env)  # TODO
