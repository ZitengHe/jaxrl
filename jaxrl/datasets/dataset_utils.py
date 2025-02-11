from typing import Tuple

import gym
from tqdm import tqdm

from jaxrl.datasets.awac_dataset import AWACDataset
from jaxrl.datasets.d4rl_dataset import D4RLDataset
from jaxrl.datasets.dataset import Dataset
from jaxrl.datasets.rl_unplugged_dataset import RLUnpluggedDataset
from jaxrl.utils import make_env


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]
    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs

def make_env_and_dataset(env_name: str, seed: int, dataset_name: str,
                         video_save_folder: str) -> Tuple[gym.Env, Dataset]:
    env = make_env(env_name, seed, video_save_folder)

    if 'd4rl' in dataset_name:
        dataset = D4RLDataset(env)
        if 'antmaze' in env_name:
            dataset.rewards -= 1.0
            # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
            # but I found no difference between (x - 0.5) * 4 and x - 1.0
        elif ('halfcheetah' in env_name or 'walker2d' in env_name
            or 'hopper' in env_name):
            normalize(dataset)
    elif 'awac' in dataset_name:
        dataset = AWACDataset(env_name)
    elif 'rl_unplugged' in dataset_name:
        dataset = RLUnpluggedDataset(env_name.replace('-', '_'))
    else:
        raise NotImplementedError(f'{dataset_name} is not available!')

    return env, dataset
