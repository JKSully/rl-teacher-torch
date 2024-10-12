import torch
from torch import Tensor
import torch.nn as nn

from torchrl.modules import MLP

import numpy as np


class FullyConnectedMLP(nn.Module):
    """
    A fully connected multi-layer perceptron for q-value estimation
    """

    def __init__(self, obs_shape, act_shape, h_size=64) -> None:
        super(nn.Module, self).__init__()
        input_dim = np.prod(obs_shape) + np.prod(act_shape)
        self.network = MLP(in_features=input_dim, out_features=1, depth=2,
                           dropout=0.5, num_cells=h_size, activation_class=nn.LeakyReLU)

    def forward(self, obs: Tensor, act: Tensor):
        flat_obs = obs.view(obs.size(0), -1)
        x = torch.cat([flat_obs, act], dim=1)
        return self.network(x)
