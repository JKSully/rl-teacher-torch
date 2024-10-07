import numpy as np
import torch
from torch import tensor
from torch.nn import Module, LeakyReLU
from torchrl.modules import MLP


class FullyConnectedMLP(Module):
    def __init__(self, obs_shape, act_shape, h_size=64)->None:
        super(Module, self).__init__()
        input_dim = np.prod(obs_shape) + np.prod(act_shape)
        self.model = MLP(in_features=input_dim, out_features=1, depth=2, dropout=0.5, num_cells=h_size, activation_class=LeakyReLU)
    
    def forward(self, obs, act):
        flat_obs = obs.view(obs.size(0), -1)
        x = torch.cat([flat_obs, act], dim=1)
        return self.model(x)
