from torchrl.envs import EnvBase
from torchrl.objectives import ClipPPOLoss
from torchrl.modules import ProbabilisticActor, ValueOperator, MLP
from torchrl.data import TensorDictReplayBuffer

from tensordict import TensorDict
from tensordict.nn import TensorDictModule

import torch

from DRLHP.segment_sampling import get_timesteps_per_episode
from DRLHP.nn import FullyConnectedMLP


def train_ppo(
        env: EnvBase,
        predictor,
        summary_writer=None,
        runtime: int = 1800,
        max_timesteps_per_episode: int = None,
        # Replace below as necessary for PPO
        max_kl: float = 0.001,
        seed: int = 0,
        discount_factor: float = 0.995,
        cg_damping: float = 0.1,
        device: str = None,
):
    device = device if device is not None else 'cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu'
    run_indefinitely = (runtime <= 0)
    if max_timesteps_per_episode is None:
        max_timesteps_per_episode = get_timesteps_per_episode(env)

    # Actor
    mlp_actor = MLP(num_cells=64, depth=3,
                    in_features=env.observation_spec.shape[0], out_features=env.action_spec.shape[0]).to(device)
    actor = TensorDictModule(mlp_actor, in_keys=['observation'], out_keys=[
                             'action', 'log_prob'])

    # Critic
    mlp_value = FullyConnectedMLP(
        obs_shape=env.observation_spec.shape, act_shape=env.action_spec.shape).to(device)
    critic = TensorDictModule(mlp_value, in_keys=['observation', 'action'], out_keys=[
        'advantage'])  # TODO: check out_keys
