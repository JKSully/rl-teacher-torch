import torch
from torch import Tensor
import torch.nn as nn

import torchrl
from torchrl.envs import EnvBase

from drlhp.nn import FullyConnectedMLP


class ComparisonRewardPredictor(nn.Module):
    """
    Reward predictor that takes in two segments, the preferred segment and the alternative segment, and predicts the reward of the preferred segment
    """

    def __init__(self, env: EnvBase, network: nn.Module = None) -> None:
        """
        Args:
            env (EnvBase): The environment object
            network (nn.Module): The neural network model. Must take in the observation and action tensors
        """
        super().__init__()
        self.env = env
        self.network = network if network is not None else FullyConnectedMLP(
            env.observation_spec.shape, env.action_spec.shape)

    def forward(self, obs: Tensor, act: Tensor, alt_obs: Tensor, alt_act: Tensor) -> Tensor:
        """
        Args:
            obs (Tensor): The observation tensor
            act (Tensor): The action tensor
            alt_obs (Tensor): The alternative observation tensor
            alt_act (Tensor): The alternative action tensor

        Returns:
            Tensor: The reward tensor
        """
        q_value = self.network(obs, act)
        alt_q_value = self.network(alt_obs, alt_act)

        segment_reward_pred = q_value.sum(dim=1)
        segment_reward_pred_alt = alt_q_value.sum(dim=1)
        reward_logits = torch.stack(
            # (B, 2)
            [segment_reward_pred, segment_reward_pred_alt], dim=1)

        return reward_logits
