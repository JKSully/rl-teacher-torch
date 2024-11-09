import torch
from torch import nn

from torchrl.envs import GymEnv
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.envs.transforms import TransformedEnv, Compose, ObservationNorm, DoubleToFloat, StepCounter
from torchrl.modules import TanhNormal
from torchrl.collectors import SyncDataCollector
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import NormalParamExtractor
from torchrl.data import TensorSpec, ReplayBuffer
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import CSVLogger

from tensordict.nn import TensorDictModule

from collections import defaultdict

from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

from drlhp.video import PreferenceLogger


class ActorNet(nn.Module):
    def __init__(self, num_cells: int, input_dim: int, action_spec: TensorSpec):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, 2 * action_spec.shape[-1]),
            NormalParamExtractor()
        )

    def forward(self, x):
        return self.net(x)

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class ValueNet(nn.Module):
    def __init__(self, num_cells: int, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, num_cells),
            nn.Tanh(),
            nn.Linear(num_cells, 1),
        )

    def forward(self, x):
        return self.net(x)

    def to(self, *args, **kwargs):
        self.net.to(*args, **kwargs)
        return super().to(*args, **kwargs)


@dataclass
class hyperparameters:
    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0
    frames_per_batch = 1000
    # For a complete training, bring the number of frames up to 1M
    total_frames = 10_000
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimization steps per batch of data collected
    clip_epsilon = (
        0.2  # clip value for PPO loss: see the equation in the intro for more context.
    )
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

def main():
    #DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    DEVICE = 'cpu'

    logger = CSVLogger(exp_name='IDP', log_dir='IDP_videos', video_format='mp4')

    env = GymEnv('InvertedDoublePendulum-v4', from_pixels=True, pixels_only=False, device=DEVICE)
    env = TransformedEnv(env, Compose(
        ObservationNorm(loc=0.0, scale=1.0, in_keys=['observation']),
        DoubleToFloat(),
        StepCounter(),
        VideoRecorder(logger, tag='run_video')
    )) 

    input_dim = env.observation_spec['observation'].shape[-1]

    actor_net = ActorNet(num_cells=hyperparameters.num_cells, input_dim=input_dim, action_spec=env.action_spec).to(DEVICE)
    policy_module = TensorDictModule(actor_net, in_keys=['observation'], out_keys=['loc', 'scale'])
    policy_module = ProbabilisticActor(
        module=policy_module, 
        spec=env.action_spec, 
        in_keys=['loc', 'scale'], 
        distribution_class=TanhNormal, 
        distribution_kwargs={
            'low': env.action_spec.space.low,
            'high': env.action_spec.space.high
        },
        return_log_prob=True
    )

    value_net = ValueNet(num_cells=hyperparameters.num_cells, input_dim=input_dim).to(DEVICE)
    value_module = ValueOperator(module=value_net, in_keys=['observation'])

    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=hyperparameters.frames_per_batch,
        total_frames=hyperparameters.total_frames,
        split_trajs=False,
        device=DEVICE,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=hyperparameters.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(gamma=hyperparameters.gamma, lmbda=hyperparameters.lmbda, value_network=value_module, average_gae=True).to(DEVICE)
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=hyperparameters.clip_epsilon,
        entropy_bonus=bool(hyperparameters.entropy_eps),
        entropy_coef=hyperparameters.entropy_eps,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=hyperparameters.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=hyperparameters.total_frames // hyperparameters.frames_per_batch, eta_min=0.0)

    logs = defaultdict(list)
    pbar = tqdm(total=hyperparameters.total_frames)
    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        for _ in range(hyperparameters.num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(hyperparameters.frames_per_batch // hyperparameters.sub_batch_size):
                subdata = replay_buffer.sample(hyperparameters.sub_batch_size)
                loss_vals = loss_module(subdata.to(DEVICE))
                loss_value = loss_vals['loss_objective'] + loss_vals['loss_critic'] + loss_vals['loss_entropy']
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), hyperparameters.max_grad_norm)
                optim.step()
                optim.zero_grad()


        logs['reward'].append(tensordict_data['next', 'reward'].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = f'average reward={logs["reward"][-1]: 4.4f} (init={logs["reward"][0]: 4.4f})'
        logs['step_count'].append(tensordict_data['step_count'].max().item())
        stepcount_str = f'step count (max): {logs["step_count"][-1]}'
        logs['lr'].append(optim.param_groups[0]['lr'])
        lr_str = f'lr policy: {logs["lr"][-1]: 4.4f}'
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(max_steps=1000, policy=policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(
                    eval_rollout["next", "reward"].sum().item()
                )
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                del eval_rollout

        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()


if __name__ == '__main__':
    main()