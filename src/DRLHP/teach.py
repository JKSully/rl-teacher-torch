import os
import os.path as osp
import random
from collections import deque
from time import time, sleep
from typing import Union
import argparse

import numpy as np

import torch
from torch import Tensor
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Adam

from tensordict import TensorDict
from tensordict.nn import TensorDictSequential

import torchrl
from torchrl.envs import EnvBase
from torchrl.envs import GymEnv

from DRLHP.comparison_collectors import SyntheticComparisonCollector, HumanComparisonCollector
from DRLHP.segment_sampling import get_timesteps_per_episode
from DRLHP.label_schedules import LabelAnnealer, ConstantLabelSchedule
from DRLHP.nn import FullyConnectedMLP
from DRLHP.segment_sampling import sample_segment_from_path, segments_from_rand_rollout, get_timesteps_per_episode
from DRLHP.summaries import AgentLogger, make_summary_writer
from DRLHP.utils import slugify, corrcoef
from DRLHP.video import SegmentVideoRecorder
from DRLHP.train import train_ppo


CLIP_LENGTH = 1.5


class TraditionalRLRewardPredictor:
    def __init__(self, summary_writer):
        self.agent_logger = AgentLogger(summary_writer)

    def predict_reward(self, path):
        self.agent_logger.log_episode(path)
        return path["original_rewards"]

    def path_callback(self, path):
        pass


class ComparisonRewardPredictor(Module):
    def __init__(self, env: EnvBase, summary_writer, comparison_collector, agent_logger: AgentLogger, label_schedule, DEVICE: torch.device):
        super(ComparisonRewardPredictor, self).__init__()

        self.summary_writer = summary_writer
        self.agent_logger = agent_logger
        self.comparison_collector = comparison_collector
        self.label_schedule = label_schedule

        self.recent_segments = deque(maxlen=200)
        self._frames_per_segment = CLIP_LENGTH * \
            get_timesteps_per_episode(env=env)
        self._steps_since_last_training = 0
        self._n_timestamps_per_predictor_training = 1e2
        self._elapsed_predictor_training_iters = 0

        self.obs_shape = env.observation_spec.shape
        self.discrete_action_space = not hasattr(env.action_spec, "shape")
        self.act_shape = (
            env.action_spec.ndim,) if self.discrete_action_space else env.action_spec.shape

        # NN setup
        self.device = DEVICE
        self.network = FullyConnectedMLP(
            self.obs_shape, self.act_shape, 1).to(self.device)
        self.optim = Adam(self.network.parameters(), lr=1e-3)
        self.criterion = CrossEntropyLoss()

    def _predict_rewards(self, obs_segments: Tensor, act_segments: Tensor, network: Module):
        batchsize = obs_segments.shape[0]
        segment_length = obs_segments.shape[1]

        obs = obs_segments.view(-1, *self.obs_shape)
        act = act_segments.view(-1, *self.act_shape)

        rewards: Tensor = network(obs, act)

        return rewards.view(batchsize, segment_length)

    def forward(self, obs_segments: Tensor, act_segments: Tensor, alt_obs_segments: Tensor, alt_act_segments: Tensor):
        q_value = self._predict_rewards(
            obs_segments, act_segments, self.network)
        alt_q_value = self._predict_rewards(
            alt_obs_segments, alt_act_segments, self.network)

        segment_reward_pred_left = q_value.sum(dim=1)
        segment_reward_pred_right = alt_q_value.sum(dim=1)
        reward_logits = torch.stack(
            # (batchsize, 2)
            [segment_reward_pred_left, segment_reward_pred_right], dim=1)

        return reward_logits

    def predict_reward(self, path: dict | TensorDict):
        """Predict the reward for each step in a path."""
        self.network.eval()

        if isinstance(path, dict):
            with torch.no_grad():
                q_value = self.network(
                    torch.tensor(path["obs"], dtype=torch.float32).to(
                        self.device).unsqueeze(0),
                    torch.tensor(path["actions"], dtype=torch.float32).to(
                        self.device).unsqueeze(0)
                )
        elif isinstance(path, TensorDict):
            with torch.no_grad():
                q_value = self.network(
                    path["obs"],
                    path["actions"]
                )

        return q_value[0]

    def path_callback(self, path: TensorDict):
        path_length = len(path["obs"])
        self._steps_since_last_training += path_length

        self.agent_logger.log_episode(path)

        segment = sample_segment_from_path(path, int(self._frames_per_segment))
        if segment is not None:
            self.recent_segments.append(segment)

        if len(self.comparison_collector) < int(self.label_schedule.n_desired_labels):
            self.comparison_collector.add_segment_pair(
                random.choice(self.recent_segments),
                random.choice(self.recent_segments)
            )

        if self._steps_since_last_training >= int(self._n_timestamps_per_predictor_training):
            self.train_predictor()
            self._steps_since_last_training = 0

    def train_predictor(self):
        self.comparison_collector.label_unlabeled_comparisons()

        minibatch_size = min(
            64, len(self.comparison_collector.labeled_decisive_comparisons))
        labeled_comparisons = random.sample(
            self.comparison_collector.labeled_decisive_comparisons, minibatch_size)

        left_obs = torch.tensor(
            [comp["left"]["obs"] for comp in labeled_comparisons], dtype=torch.float32).to(self.device)
        left_act = torch.tensor([comp["left"]["actions"]
                                for comp in labeled_comparisons], dtype=torch.float32).to(self.device)
        right_obs = torch.tensor(
            [comp["right"]["obs"] for comp in labeled_comparisons], dtype=torch.float32).to(self.device)
        right_act = torch.tensor([comp["right"]["actions"]
                                 for comp in labeled_comparisons], dtype=torch.float32).to(self.device)
        labels = torch.tensor(
            [comp["label"] for comp in labeled_comparisons], dtype=torch.long).to(self.device)

        self.network.train()
        self.optim.zero_grad()

        reward_logits = self.forward(left_obs, left_act, right_obs, right_act)
        loss = self.criterion(reward_logits, labels)

        loss.backward()
        self.optim.step()

        self._elapsed_predictor_training_iters += 1
        self._write_training_summaries(loss.item())

    def _write_training_summaries(self, loss):
        self.agent_logger.log_scalar("predictor/loss", loss)

        recent_paths = self.agent_logger.get_recent_paths_with_padding()
        if len(recent_paths) > 1 and self.agent_logger.summary_step % 10 == 0:
            validation_obs = torch.tensor(
                [path["obs"] for path in recent_paths], dtype=torch.float32).to(self.device)
            validation_act = torch.tensor(
                [path["actions"] for path in recent_paths], dtype=torch.float32).to(self.device)

            self.network.eval()
            with torch.no_grad():
                q_value = self.network(validation_obs, validation_act)

            ep_reward_pred = q_value.sum(dim=1).cpu().numpy()
            reward_true = np.array([path['original_rewards']
                                   for path in recent_paths])
            ep_reward_true = reward_true.sum(axis=1)
            self.agent_logger.log_simple(
                "predictor/correlations", corrcoef(ep_reward_true, ep_reward_pred))

        self.agent_logger.log_simple(
            "predictor/num_training_iters", self._elapsed_predictor_training_iters)
        self.agent_logger.log_simple(
            "labels/desired_labels", self.label_schedule.n_desired_labels)
        self.agent_logger.log_simple(
            "labels/total_comparisons", len(self.comparison_collector))
        self.agent_logger.log_simple(
            "labels/labeled_comparisons", len(
                self.comparison_collector.labeled_decisive_comparisons)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_id', type=str, required=True)
    parser.add_argument('--predictor', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--n_labels', type=int, default=None)
    parser.add_argument('--pretrain_labels', type=int, default=None)
    parser.add_argument('--num_timesteps', default=5e6, type=int)
    parser.add_argument('--agent', default='ppo', type=str)
    parser.add_argument('--pretrain_iters', default=100_000, type=int)
    parser.add_argument('--no_videos', action='store_true')
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    print("Setting things up...")

    env_id = args.env_id
    device = torch.device(args.device)
    run_name = f'{env_id}/{args.name}-{int(time())}'
    summary_writer = make_summary_writer(run_name)

    env = GymEnv(env_id, device=device)

    num_timesteps = args.num_timesteps
    experiment_name = slugify(args.name)

    if args.predictor == 'rl':
        predictor = TraditionalRLRewardPredictor(summary_writer)
    else:
        agent_logger = AgentLogger(summary_writer)

        pretrain_labels = args.pretrain_labels if args.pretrain_labels is not None else args.n_labels // 4

        if args.n_labels is not None:
            label_schedule = LabelAnnealer(
                agent_logger, final_timesteps=num_timesteps, final_labels=args.n_labels, pretrain_labels=pretrain_labels)
        else:
            print("No label limit given. We will request one label every few seconds.")
            label_schedule = ConstantLabelSchedule(
                pretrain_labels=pretrain_labels)

        if args.predictor == 'synthetic':
            comparison_collector = SyntheticComparisonCollector()
        elif args.predictor == 'human':
            bucket = os.environ.get("RL_TEACHER_GCS_BUCKET")
            assert bucket is not None and bucket.startswith(
                "gs://"), "env variable RL_TEACHER_GCS_BUCKET must start with gs://"
            comparison_collector = HumanComparisonCollector(
                env=env, experiment_name=experiment_name)
        else:
            raise ValueError(
                "Bad value for --predictor: {}".format(args.predictor))

        predictor = ComparisonRewardPredictor(
            env,
            summary_writer,
            comparison_collector=comparison_collector,
            agent_logger=agent_logger,
            label_schedule=label_schedule,
        )

        print("Starting random rollouts to generate pretraining segments. No learning will take place...")
        pretrain_segments = segments_from_rand_rollout(
            env, n_desired_segments=pretrain_labels*2, clip_length_in_seconds=CLIP_LENGTH, workers=args.workers
        )
        for idx, segment in enumerate(pretrain_segments):
            comparison_collector.add_segment_pair(
                segment, pretrain_segments[idx + pretrain_labels]
            )

        while len(comparison_collector.labeled_comparisons) < int(pretrain_labels * 0.75):
            comparison_collector.label_unlabeled_comparisons()
            if args.predictor == "synthetic":
                print("{} synthetic labels generated.".format(
                    len(comparison_collector.labeled_comparisons)))
            elif args.predictor == "human":
                print("{}/{} comparisons labeled. Please add more labels w/ the human-feedback-api. Sleeping...".format(
                    len(comparison_collector.labeled_comparisons), pretrain_labels))
                sleep(5)

        for i in range(args.pretrain_iters):
            predictor.train_predictor()
            if i % 100 == 0:
                print("{}/{} predictor pretraining iters...".format(i,
                      args.pretrain_iters))

    if not args.no_videos:
        predictor = SegmentVideoRecorder(
            predictor, env, save_dir=osp.join("/tmp/rl_teacher_vids", run_name))

    print("Starting joint training of predictor and agent")
    # TODO: Implement agent training
    if args.agent == 'ppo':
        # Train ppo:

        raise NotImplementedError("PPO training not implemented yet")
    else:
        raise ValueError("{} is not a valid agent".format(args.agent))


if __name__ == '__main__':
    main()
