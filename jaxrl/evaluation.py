from typing import Dict

import flax.linen as nn
import gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    stats = {'return': [], 'length': []}
    successes = None
    for _ in range(num_episodes):
        observation, done = env.reset(), False
        total_reward = 0.0
        while not done:
            action = agent.sample_actions(observation, temperature=0.0)
            if len(action.shape) == 2 and action.shape[0]==1:
                action = action.squeeze(0)
            observation, reward, done, info = env.step(action)
            total_reward = total_reward + reward

        stats['return'].append(total_reward)
        stats['length'].append(info['episode']['length'])

        # while not done:
        #     action = agent.sample_actions(observation, temperature=0.0)
        #     observation, _, done, info = env.step(action)
        # for k in stats.keys():
        #     stats[k].append(info['episode'][k])

        if 'is_success' in info:
            if successes is None:
                successes = 0.0
            successes += info['is_success']

    for k, v in stats.items():
        stats[k] = np.mean(v)

    if successes is not None:
        stats['success'] = successes / num_episodes
    return stats
