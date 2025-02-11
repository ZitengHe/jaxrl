import os
import random
import time
import sys
sys.path.insert(0, '/home/dodo/hzt/jaxrl/')

import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import (AWACLearner, DDPGLearner, REDQLearner, SACLearner,
                          SACV1Learner)
from jaxrl.datasets import ReplayBuffer
from jaxrl.evaluation import evaluate
from jaxrl.utils import make_env

import wandb

os.environ["WANDB_API_KEY"] = "7412a0b0ed5f0d1eb4548fbcf4ece8e6b85d8028"

FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'HalfCheetah-v3', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('updates_per_step', 1, 'Gradient updates per step.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_integer('start_training', int(1e4),
                     'Number of training steps to start training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')
flags.DEFINE_boolean('track', False, 'Track experiments with Weights and Biases.')
flags.DEFINE_string('wandb_project_name', "jaxrl", "The wandb's project name.")
flags.DEFINE_string('wandb_entity', None, "the entity (team) of wandb's project")
flags.DEFINE_string('wandb_mode', 'disabled', 'Some Spetial Label')



config_flags.DEFINE_config_file(
    'config',
    # 'configs/sac_v1_two_head.py',
    # 'configs/sac_v1_default.py',
    '/home/dodo/hzt/jaxrl/examples/configs/sac_v1_two_head.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)


def main(_):
    project = 'SAC-JAX-RL'
    group = 'sacv1_two_head'

    kwargs = dict(FLAGS.config)
    algo = kwargs['algo']
    run_name = f"{FLAGS.env_name}__{algo}__state_pred_loss-3__{FLAGS.seed}__{int(time.time())}"

    wandb.init(project=project, 
            group=group,
            name=run_name,
            # name=FLAGS.env_name+'_seed-'+str(FLAGS.seed)+'_'+str(np.random.randint(0, 1000)),
            config={
                'algo': algo,
                'env_name': FLAGS.env_name,
                'seed': FLAGS.seed,
                'actor_lr': FLAGS.config.actor_lr,
                'vf_lr': FLAGS.config.value_lr,
                'qf_lr': FLAGS.config.critic_lr,
                'hidden_dims': FLAGS.config.hidden_dims,
                'share_hidden_dims': FLAGS.config.share_hidden_dims,
                'state_hidden_dims': FLAGS.config.state_hidden_dims,
                'action_hidden_dims': FLAGS.config.action_hidden_dims,
                'inv_hidden_dims': FLAGS.config.inv_hidden_dims,
                'actor_lr': FLAGS.config.actor_lr,
                'vf_lr': FLAGS.config.value_lr,
                'qf_lr': FLAGS.config.critic_lr,
                'temp_lr': FLAGS.config.temp_lr,
                'inv_lr': FLAGS.config.inv_lr,
            },
            
            monitor_gym=True,
            save_code=True,
            settings=wandb.Settings(init_timeout=10),
            mode=FLAGS.wandb_mode,
        #    mode='disabled',
            )

    # summary_writer = SummaryWriter(
    #     os.path.join(FLAGS.save_dir, run_name))

    if FLAGS.save_video:
        video_train_folder = os.path.join(FLAGS.save_dir, 'video', 'train')
        video_eval_folder = os.path.join(FLAGS.save_dir, 'video', 'eval')
    else:
        video_train_folder = None
        video_eval_folder = None

    env = make_env(FLAGS.env_name, FLAGS.seed, video_train_folder)
    eval_env = make_env(FLAGS.env_name, FLAGS.seed + 42, video_eval_folder)

    np.random.seed(FLAGS.seed)
    random.seed(FLAGS.seed)


    replay_buffer_size = kwargs.pop('replay_buffer_size')
    if algo == 'sac':
        agent = SACLearner(FLAGS.seed,
                           env.observation_space.sample()[np.newaxis],
                           env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'redq':
        agent = REDQLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis],
                            policy_update_delay=FLAGS.updates_per_step,
                            **kwargs)
    elif algo == 'sacv1' or algo == 'sacv1_two_head':
        agent = SACV1Learner(FLAGS.seed,
                             env.observation_space.sample()[np.newaxis],
                             env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'awac':
        agent = AWACLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    elif algo == 'ddpg':
        agent = DDPGLearner(FLAGS.seed,
                            env.observation_space.sample()[np.newaxis],
                            env.action_space.sample()[np.newaxis], **kwargs)
    else:
        raise NotImplementedError()

    replay_buffer = ReplayBuffer(env.observation_space, env.action_space,
                                 replay_buffer_size or FLAGS.max_steps)

    eval_returns = []
    observation, done = env.reset(), False
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        if i < FLAGS.start_training:
            action = env.action_space.sample()
        else:
            action = agent.sample_actions(observation)
        next_observation, reward, done, info = env.step(action)

        if not done or 'TimeLimit.truncated' in info:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(observation, action, reward, mask, float(done),
                             next_observation)
        observation = next_observation

        if done:
            observation, done = env.reset(), False

            log_data = {f'training/{k}': v for k, v in info['episode'].items()}
            log_data["timesteps"] = info['total']['timesteps']

            if 'is_success' in info:
                log_data['training/success'] = info['is_success']

            wandb.log(log_data, step=i)

        if i >= FLAGS.start_training:
            for _ in range(FLAGS.updates_per_step):
                batch = replay_buffer.sample(FLAGS.batch_size)
                update_info = agent.update(batch)

            if i % FLAGS.log_interval == 0:
                log_data = {f'training/{k}': v for k, v in update_info.items()}
                wandb.log(log_data, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, eval_env, FLAGS.eval_episodes)

            eval_log_data = {f'evaluation/average_{k}s': v for k, v in eval_stats.items()}
            wandb.log(eval_log_data, step=i)

            # # 记录返回值
            # eval_returns.append((info['total']['timesteps'], eval_stats['return']))
            # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
            #         eval_returns,
            #         fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
