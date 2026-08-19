from collections import defaultdict
from dataclasses import dataclass

import hydra
import matplotlib.pyplot as plt
import torch
from gymnasium.spaces import Dict
from omegaconf import DictConfig
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.collectors import Collector
from torchrl.data import ReplayBuffer, TensorSpec
from torchrl.data.replay_buffers import LazyTensorStorage, SamplerWithoutReplacement
from torchrl.envs import GymEnv
from torchrl.envs.transforms import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import TanhNormal
from torchrl.modules.distributions import NormalParamExtractor
from torchrl.modules.tensordict_module import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.record import VideoRecorder
from torchrl.record.loggers import CSVLogger
from tqdm import tqdm

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
            NormalParamExtractor(),
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


@hydra.main(version_base="1.2", config_path="", config_name="test_logger")
def main(cfg: DictConfig):
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    DEVICE = "cpu"

    logger = CSVLogger(exp_name="IDP", log_dir="IDP_videos", video_format="mp4")

    env = GymEnv(
        "InvertedDoublePendulum-v4", from_pixels=True, pixels_only=False, device=DEVICE
    )
    env = TransformedEnv(
        env,
        Compose(
            [
                VideoRecorder(logger=logger, tag="run_video"),
                ObservationNorm(loc=0.0, scale=1.0, in_keys=["observation"]),
                DoubleToFloat(),
                StepCounter(),
            ]
        ),
    )

    input_dim = env.observation_spec["observation"].shape[-1]

    actor_net = ActorNet(
        num_cells=cfg.hyperparameters.num_cells,
        input_dim=input_dim,
        action_spec=env.action_spec,
    ).to(DEVICE)
    policy_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
    )
    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )

    value_net = ValueNet(
        num_cells=cfg.hyperparameters.num_cells, input_dim=input_dim
    ).to(DEVICE)
    value_module = ValueOperator(module=value_net, in_keys=["observation"])

    collector = Collector(
        env,
        policy_module,
        frames_per_batch=cfg.hyperparameters.frames_per_batch,
        total_frames=cfg.hyperparameters.total_frames,
        split_trajs=False,
        device=DEVICE,
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.hyperparameters.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    advantage_module = GAE(
        gamma=cfg.hyperparameters.gamma,
        lmbda=cfg.hyperparameters.lmbda,
        value_network=value_module,
        average_gae=True,
    ).to(DEVICE)
    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=cfg.hyperparameters.clip_epsilon,
        entropy_bonus=bool(cfg.hyperparameters.entropy_eps),
        entropy_coeff=cfg.hyperparameters.entropy_eps,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.hyperparameters.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=cfg.hyperparameters.total_frames // cfg.hyperparameters.frames_per_batch,
        eta_min=0.0,
    )

    logs = defaultdict(list)
    pbar = tqdm(total=cfg.hyperparameters.total_frames)
    eval_str = ""

    for i, tensordict_data in enumerate(collector):
        for _ in range(cfg.hyperparameters.num_epochs):
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(
                cfg.hyperparameters.frames_per_batch
                // cfg.hyperparameters.sub_batch_size
            ):
                subdata = replay_buffer.sample(cfg.hyperparameters.sub_batch_size)
                loss_vals = loss_module(subdata.to(DEVICE))
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.hyperparameters.max_grad_norm
                )
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean())
        pbar.update(tensordict_data.numel())
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(tensordict_data["step_count"].max())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_rollout = env.rollout(max_steps=1000, policy=policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean())
                logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum())
                logs["eval step_count"].append(eval_rollout["step_count"].max())
                eval_str = (
                    f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                    f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                    f"eval step-count: {logs['eval step_count'][-1]}"
                )
                env.transform.dump()
                del eval_rollout

        pbar.set_description(
            ", ".join([eval_str, cum_reward_str, stepcount_str, lr_str])
        )

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


if __name__ == "__main__":
    main()
