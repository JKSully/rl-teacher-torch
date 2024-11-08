import torch

from torchrl.envs import GymEnv
from torchrl.modules import MLP
from torchrl.collectors import SyncDataCollector
from torchrl.modules.tensordict_module import AdditiveGaussianWrapper, AdditiveGaussianModule
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.envs.transforms import TransformedEnv, Compose

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule


from tqdm import tqdm
import matplotlib.pyplot as plt

from drlhp.video import PreferenceLogger


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    env = GymEnv('Pendulum-v1', device=DEVICE)
    env = TransformedEnv(env, Compose(lambda x: x)) 

    print(env.observation_spec.shape)
    print(env.action_spec.shape)

    obs_dim = 3
    action_dim = 1

    mlp_actor = MLP(num_cells=64, depth=3, in_features=obs_dim,
                    out_features=action_dim).to(DEVICE)
    actor = TensorDictModule(
        mlp_actor, in_keys=['observation'], out_keys=['action'])
    
    policy = TensorDictSequential(actor, AdditiveGaussianModule(env.action_spec))

    mlp_critic = MLP(num_cells=64, depth=3, in_features=obs_dim, out_features=action_dim).to(DEVICE)  # Fix input features

    critic = TensorDictSequential(TensorDictModule(
        mlp_critic, in_keys=['observation'], out_keys=['state_value']))
    
    collector = SyncDataCollector(
        env, policy, frames_per_batch=1_000, total_frames=1_000_000
    )

    buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(100_000, device=DEVICE)
    )

    loss_module = ClipPPOLoss(actor, critic).to(DEVICE)

    optim = torch.optim.Adam(loss_module.parameters(), lr=2e-4)

    for data in tqdm(collector):
        buffer.extend(data)
        sample = buffer.sample(50)
        loss = loss_module(sample)
        loss = loss['loss_actor'] + loss['loss_value']
        loss.backward()
        optim.step()
        optim.zero_grad()




if __name__ == '__main__':
    main()