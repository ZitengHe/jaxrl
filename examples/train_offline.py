import os
import sys
sys.path.insert(0, '/home/air/hzt/jaxrl/')
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter

from jaxrl.agents import BCLearner, SACV1Learner
from jaxrl.datasets import make_env_and_dataset, split_into_trajectories
from jaxrl.evaluation import evaluate
import wandb
import d4rl

os.environ["WANDB_API_KEY"] = "7412a0b0ed5f0d1eb4548fbcf4ece8e6b85d8028"

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'sacv1', 'ori or base or plus or pro')
flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_enum('dataset_name', 'd4rl', ['d4rl', 'awac', 'rl_unplugged'],
                  'Dataset name.')
flags.DEFINE_string('wandb_mode', 'disabled', 'Some Spetial Label')
flags.DEFINE_string('inv_dyna_path', None, 'only pretrain or not')

flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 1000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 10000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 256, 'Mini batch size.')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_float(
    'percentile', 100.0,
    'Dataset percentile (see https://arxiv.org/abs/2106.01345).')
flags.DEFINE_float('percentage', 100.0,
                   'Pencentage of the dataset to use for training.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_boolean('save_video', False, 'Save videos during evaluation.')

flags.DEFINE_float('actor_lr', 3e-4, 'Learning rate for the actor network.')
flags.DEFINE_float('vf_lr', 3e-4, 'Learning rate for the value function network.')
flags.DEFINE_float('qf_lr', 3e-4, 'Learning rate for the Q-function network.')
flags.DEFINE_float('temp_lr', 3e-4, 'Temperature parameter for the inverse model.')
flags.DEFINE_float('inv_lr', 1e-4, 'Learning rate for the inverse model.')
flags.DEFINE_float('discount', 0.99, 'Discount factor for reinforcement learning.')

flags.DEFINE_float('expectile', 0.7, 'The actual tau for expectiles.')
flags.DEFINE_float('temperature_a', 3.0, 'Temperature parameter for action.')
flags.DEFINE_float('temperature_s', 3.0, 'Temperature parameter for state.')
flags.DEFINE_float('dropout_rate', None, 'Dropout rate for hidden layers.')
flags.DEFINE_boolean('value_layer_norm', False, 'Whether to use layer normalization for value.')
flags.DEFINE_float('value_dropout_rate', -1, 'Dropout rate for value network.')

flags.DEFINE_string('policy_type', 'gaussian', 'gaussian or deterministic for policy')
flags.DEFINE_boolean('policy_layer_norm', False, 'Whether to use layer normalization for policy.')
flags.DEFINE_float('policy_dropout_rate', -1, 'Dropout rate for policy network.')
flags.DEFINE_string('policy_activations', 'relu', 'Activation function for policy network.')
flags.DEFINE_string('opt_decay_schedule', 'cosine', 'Opt decay schedule.')

flags.DEFINE_float('tau', 0.005, 'Soft update factor for target networks.')
flags.DEFINE_float('inv_penalty', 1.0, 'Penalty for the inverse model.')
flags.DEFINE_float('bc_a', 1.0, 'bc weight')
flags.DEFINE_float('bc_s', 1.0, 'bc weight')
flags.DEFINE_string('sample_strategy', 'por', 'iql or maxQ or por')
flags.DEFINE_boolean('delta_policy', False, 'Whether to use delta policy.')
flags.DEFINE_float('inv_dropout_rate', -1, 'Dropout rate for inverse dynamics model network.')


config_flags.DEFINE_config_file(
    'config',
    '/home/air/hzt/jaxrl/examples/configs/sac_v1_default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)




def main(_):
    project = 'SAC-JAX-RL'
    group = 'sacv1'
    wandb.init(project=project, 
            group=group,
            name=FLAGS.env_name+'_seed-'+str(FLAGS.seed)+'_'+str(np.random.randint(0, 1000)),
            config={
                'mode': FLAGS.mode,
                'env_name': FLAGS.env_name,
                'seed': FLAGS.seed,
                'actor_lr': FLAGS.actor_lr,
                'vf_lr': FLAGS.vf_lr,
                'qf_lr': FLAGS.qf_lr,
                'hidden_dims': FLAGS.config.hidden_dims,
                'share_hidden_dims': FLAGS.config.share_hidden_dims,
                'state_hidden_dims': FLAGS.config.state_hidden_dims,
                'action_hidden_dims': FLAGS.config.action_hidden_dims,
                'inv_hidden_dims': FLAGS.config.inv_hidden_dims,
                'discount': FLAGS.discount,
                'expectile': FLAGS.expectile,
                'temperature_a': FLAGS.temperature_a,
                'temperature_s': FLAGS.temperature_s,
                'dropout_rate': FLAGS.dropout_rate,
                'layer_norm': FLAGS.value_layer_norm,
                'value_dropout_rate': FLAGS.value_dropout_rate,
                'policy_layer_norm': FLAGS.policy_layer_norm,
                'policy_dropout_rate': FLAGS.policy_dropout_rate,
                'tau': FLAGS.tau,
                'inv_penalty': FLAGS.inv_penalty,
                'bc_a': FLAGS.bc_a,
                'bc_s': FLAGS.bc_s,
                'sample_strategy': FLAGS.sample_strategy,
                'policy_type': FLAGS.policy_type,
                'delta_policy': FLAGS.delta_policy,
                'inv_dropout_rate': FLAGS.inv_dropout_rate,
                'inv_lr': FLAGS.inv_lr,
            },
            settings=wandb.Settings(init_timeout=10),
            mode=FLAGS.wandb_mode,
        #    mode='disabled',
            )
    # summary_writer = SummaryWriter(
    #     os.path.join(FLAGS.save_dir, 'tb', str(FLAGS.seed)))

    video_save_folder = None if not FLAGS.save_video else os.path.join(
        FLAGS.save_dir, 'video', 'eval')

    env, dataset = make_env_and_dataset(FLAGS.env_name, FLAGS.seed,
                                        FLAGS.dataset_name, video_save_folder)

    if FLAGS.percentage < 100.0:
        dataset.take_random(FLAGS.percentage)

    if FLAGS.percentile < 100.0:
        dataset.take_top(FLAGS.percentile)

    # kwargs = dict(FLAGS.config)
    # kwargs['num_steps'] = FLAGS.max_steps
    if FLAGS.mode == 'sacv1':
        agent = SACV1Learner(FLAGS.seed,
                        env.observation_space.sample()[np.newaxis],
                        env.action_space.sample()[np.newaxis], 
                        actor_lr=FLAGS.actor_lr,
                        value_lr=FLAGS.vf_lr,
                        critic_lr=FLAGS.qf_lr,
                        temp_lr=FLAGS.temp_lr,
                        hidden_dims=FLAGS.config.hidden_dims,
                        discount=FLAGS.discount,
                        tau=FLAGS.tau,)

    # eval_returns = []
    for i in tqdm.tqdm(range(1, FLAGS.max_steps + 1),
                       smoothing=0.1,
                       disable=not FLAGS.tqdm):
        batch = dataset.sample(FLAGS.batch_size)

        update_info = agent.update(batch)

        # if i % FLAGS.log_interval == 0:
        #     for k, v in update_info.items():
        #         wandb.log(update_info, step=i)
        if (i+1) % FLAGS.log_interval == 0:
            wandb.log(update_info, step=i)

        if i % FLAGS.eval_interval == 0:
            eval_stats = evaluate(agent, env, FLAGS.eval_episodes)
            normalized_returns = d4rl.get_normalized_score(FLAGS.env_name,
                                                           eval_stats['return']) * 100.0
            # for k, v in eval_stats.items():
            #     summary_writer.add_scalar(f'evaluation/average_{k}s', v, i)
            # summary_writer.flush()

            wandb.log({
                'evaluation/return mean': eval_stats['return'].mean(),
                'evaluation/normalized return mean': normalized_returns.mean(),
                },
                step=i
                )

            # eval_returns.append((i, eval_stats['return']))
            # np.savetxt(os.path.join(FLAGS.save_dir, f'{FLAGS.seed}.txt'),
            #            eval_returns,
            #            fmt=['%d', '%.1f'])


if __name__ == '__main__':
    app.run(main)
