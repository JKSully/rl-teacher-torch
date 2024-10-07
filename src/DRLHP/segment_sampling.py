import math
from multiprocessing import Pool
import numpy as np
import torch

from torchrl.envs import EnvBase
from tensordict import TensorDict
from typing import Dict, Union, Callable


def get_timesteps_per_episode(env: EnvBase):
    """ Get the maximum number of timesteps per episode for an environment. """
    # Use hasattr to avoid errors when the environment does not have the attribute
    if hasattr(env, 'max_episode_steps'):
        return env.max_episode_steps
    if hasattr(env, 'spec.max_episode_steps'):
        return env.spec.max_episode_steps
    if hasattr(env, '_env'):
        return get_timesteps_per_episode(env._env)
    if hasattr(env, 'env'):
        return get_timesteps_per_episode(env.env)

    return None  # FINISH THIS


def _slice_path(path: Dict | TensorDict, segment_length: int, start_pos: int = 0):
    return {
        # k: torch.as_tensor(v[start_pos:(start_pos + segment_length)])
        k: np.asarray(v[start_pos:(start_pos + segment_length)])
        for k, v in path.items()
        if k in ['obs', 'actions', 'original_rewards', 'human_obs']}


def create_segment_q_states(segment: Union[TensorDict, Dict]):
    obs_Ds = segment["obs"]
    act_Ds = segment["actions"]
    # return np.concatenate([obs_Ds, act_Ds], axis=1)
    return torch.cat([obs_Ds, act_Ds], dim=1)


def sample_segment_from_path(path: Dict | TensorDict, segment_length: int):
    """Returns a segment sampled from a random place in a path. Returns None if the path is too short"""
    path_length = len(path["obs"])
    if path_length < segment_length:
        return None

    start_pos = np.random.randint(0, path_length - segment_length + 1)

    # Build segment
    segment = _slice_path(path, segment_length, start_pos)

    # Add q_states
    segment["q_states"] = create_segment_q_states(segment)
    return segment


def random_action(env: EnvBase, ob):
    """ Pick an action by uniformly sampling the environment's action space. """
    return env.action_spec.sample()


def do_rollout(env: EnvBase, action_function):
    """ Builds a path by running through an environment using a provided function to select actions. """
    obs, rewards, actions, human_obs = [], [], [], []
    max_timesteps_per_episode = get_timesteps_per_episode(env)
    ob = env.reset()
    # Primary environment loop
    # TODO: Migrate to TorchRL from Gymnasium
    for _ in range(max_timesteps_per_episode):
        action = action_function(env, ob)
        obs.append(ob)
        actions.append(action)
        ob, rew, terminated, truncated, info = env.step(action)
        rewards.append(rew)
        human_obs.append(info.get("human_obs"))
        if terminated or truncated:
            break
    # Build path dictionary
    path = {
        "obs": np.array(obs),
        "original_rewards": np.array(rewards),
        "actions": np.array(actions),
        "human_obs": np.array(human_obs)}
    return path


def basic_segments_from_rand_rollout(
        env: EnvBase, n_desired_segments, clip_length_in_seconds,
        # Multiprocessing parameters
        seed=0, _verbose=True, _multiplier=1
):
    segments = []
    np.random.seed(seed)  # This is a subsitute for space_prng.seed(seed)
    segment_length = int(clip_length_in_seconds * env.fps)
    while len(segments) < n_desired_segments:
        path = do_rollout(env, random_action)
        segments_for_this_path = max(
            1, int(0.25 * len(path["obs"]) / segment_length))
        for _ in range(segments_for_this_path):
            segment = sample_segment_from_path(path, segment_length)
            if segment is not None:
                segments.append(segment)

            if _verbose and len(segments) % 10 == 0 and len(segments) > 0:
                print('Collected {}/{} segments'.format(len(segments)
                      * _multiplier, n_desired_segments * _multiplier))

    if _verbose:
        print('Successfully collected {} segments'.format(
            len(segments) * _multiplier))
    return segments


def segments_from_rand_rollout(env: EnvBase, n_desired_segments, clip_length_in_seconds, workers: int):
    if workers < 2:
        return basic_segments_from_rand_rollout(env, n_desired_segments, clip_length_in_seconds)

    pool = Pool(processes=workers)
    segments_per_worker = int(math.ceil(n_desired_segments / workers))

    jobs = [
        (env, segments_per_worker,
         clip_length_in_seconds, i, i == 0, workers)
        for i in range(workers)
    ]

    results = pool.starmap(basic_segments_from_rand_rollout, jobs)
    pool.close()
    return [segment for sublist in results for segment in sublist]
