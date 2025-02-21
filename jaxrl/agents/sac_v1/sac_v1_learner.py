"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

from gym import Env
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxrl.agents.sac import temperature
from jaxrl.agents.sac.actor import update as update_actor
from jaxrl.agents.sac.actor import update_inv_dyna, update_two_head
from jaxrl.agents.sac.critic import target_update
from jaxrl.agents.sac_v1.critic import update_q, update_v
from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, PRNGKey
# from jaxrl.networks.policies
import flax.linen as nn

@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_value: Model, temp: Model, batch: Batch, discount: float,
    tau: float, target_entropy: float, update_target: bool
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:

    new_critic, critic_info = update_q(critic, target_value, batch, discount)

    rng, key = jax.random.split(rng)
    new_actor, actor_info = update_actor(key, actor, new_critic, temp, batch)

    rng, key = jax.random.split(rng)
    new_value, value_info = update_v(key, new_actor, new_critic, value, temp,
                                     batch, True)

    if update_target:
        new_target_value = target_update(new_value, target_value, tau)
    else:
        new_target_value = target_value

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy_a'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_value, new_target_value, new_temp, {
        **{f"training/{k}": v for k, v in critic_info.items()},
        **{f"training/{k}": v for k, v in value_info.items()},
        **{f"training/{k}": v for k, v in actor_info.items()},
        **{f"training/{k}": v for k, v in alpha_info.items()},
    }

@functools.partial(jax.jit, static_argnames=('update_target', 'env', 'beta'))
def _update_sac_two_head_jit(rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_value: Model, inv_dyna: Model, temp: Model, temp_state:Model, batch: Batch, discount: float,
    tau: float, target_entropy: float, update_target: bool, env: Env, beta: float
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    new_critic, critic_info = update_q(critic, target_value, batch, discount)

    rng, key = jax.random.split(rng)
    # new_inv_dyna, inv_dyna_info = update_inv_dyna(key, inv_dyna, batch)
    new_inv_dyna, inv_dyna_info = inv_dyna, {}
    new_actor, actor_info = update_two_head(key, actor, value, new_critic, new_inv_dyna, temp, temp_state, batch, env=env, beta=beta)

    rng, key = jax.random.split(rng)
    new_value, value_info = update_v(key, new_actor, new_critic, value, temp,
                                     batch, True)

    if update_target:
        new_target_value = target_update(new_value, target_value, tau)
    else:
        new_target_value = target_value

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy_a'],
                                              target_entropy)
    if temp_state is not None:
        new_temp_state, alpha_state_info = temperature.update(temp_state, actor_info['entropy_s'],
                                              target_entropy)
        alpha_state_info = {**{f"temp_state/{k}": v for k, v in alpha_state_info.items()}}

    else:
        new_temp_state = None
        alpha_state_info = {}
    return rng, new_actor, new_critic, new_value, new_target_value, new_inv_dyna, new_temp, new_temp_state, {
        **{f"training/{k}": v for k, v in critic_info.items()},
        **{f"training/{k}": v for k, v in value_info.items()},
        **{f"training/{k}": v for k, v in actor_info.items()},
        **{f"training/{k}": v for k, v in alpha_info.items()},
        **{f"training/{k}": v for k, v in alpha_state_info.items()},
        **{f"training/{k}": v for k, v in inv_dyna_info.items()},
    }

class SACV1Learner(object):

    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 algo: str = 'sacv1',
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 inv_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 share_hidden_dims: Sequence[int] = (1024,),
                 state_hidden_dims: Sequence[int] = (1024,1024),
                 action_hidden_dims: Sequence[int] = (512,),
                 inv_hidden_dims: Sequence[int] = (1024, 1024),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 init_temperature: float = 1.0,
                 individual_temp: bool = False,
                 reduction_ratio: int = 16,
                 beta: float = 10,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        state_dim = observations.shape[-1]
        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim / 2
        else:
            self.target_entropy = target_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount
        self.algo = algo
        self.beta = beta

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key, temp_key_state = jax.random.split(rng, 5)
        if algo == 'sacv1':
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim,)
        elif algo == 'sacv1_attention':
            actor_def = policies.NormalTanhAttentionPolicy(hidden_dims, action_dim, reduction_ratio=reduction_ratio)
        elif algo == 'sacv1_two_head':
            actor_def = policies.TwoHeadsGaussianPolicy(share_hidden_dims=share_hidden_dims,
                                                        state_hidden_dims=state_hidden_dims,
                                                        action_hidden_dims=action_hidden_dims,
                                                        state_dim=state_dim,
                                                        action_dim=action_dim,)

            inv_dynamics_def = policies.InverseDerministicPolicy(hidden_dims=inv_hidden_dims,
                                                                 action_dim=action_dim,)
            input = jnp.concatenate([observations, observations], -1)
            inv_dynamics = Model.create(inv_dynamics_def,
                             inputs=[actor_key, input],
                             tx=optax.adam(learning_rate=inv_lr))
            self.inv_dyna = inv_dynamics
        elif algo == 'sacv1_two_head_attention':
            actor_def = policies.TwoHeadsAttentionGaussianPolicy(share_hidden_dims=share_hidden_dims,
                                                        state_hidden_dims=state_hidden_dims,
                                                        action_hidden_dims=action_hidden_dims,
                                                        state_dim=state_dim,
                                                        action_dim=action_dim,
                                                        reduction_ratio=reduction_ratio,
                                                        )
            inv_dynamics_def = policies.InverseDerministicPolicy(hidden_dims=inv_hidden_dims,
                                                                 action_dim=action_dim,)
            input = jnp.concatenate([observations, observations], -1)
            inv_dynamics = Model.create(inv_dynamics_def,
                             inputs=[actor_key, input],
                             tx=optax.adam(learning_rate=inv_lr))
            self.inv_dyna = inv_dynamics
        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optax.adam(learning_rate=actor_lr))
        # # 打印  actor   
        # tabulate_actor = nn.tabulate(actor_def, jax.random.PRNGKey(0), compute_flops=True, compute_vjp_flops=True)
        # print(tabulate_actor(observations))
        # exit()

        # # 打印  inverse model   
        # tabulate_inv = nn.tabulate(inv_dyna_def, jax.random.PRNGKey(0), compute_flops=True, compute_vjp_flops=True)

        # print(tabulate_inv(jnp.concatenate([observations, observations], -1)))
        # exit()

        critic_def = critic_net.DoubleCritic(hidden_dims)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = critic_net.ValueCritic(hidden_dims)
        value = Model.create(value_def,
                             inputs=[critic_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_value = Model.create(value_def,
                                    inputs=[critic_key, observations])

        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=optax.adam(learning_rate=temp_lr))
        self.temp_state = None
        if individual_temp:
            temp_state = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key_state],
                            tx=optax.adam(learning_rate=temp_lr))
            self.temp_state = temp_state
            

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_value = target_value
        self.temp = temp
        self.rng = rng
        self.step = 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        if np.isnan(actions).any():
            print(self.actor.params)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)
    
    def sample_next_states(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, next_states = policies.sample_next_states(self.rng, self.actor.apply_fn,
                                               self.actor.params, observations,
                                               temperature)
        if np.isnan(next_states).any():
            print(self.actor.params)
        self.rng = rng

        next_states = np.asarray(next_states)
        return np.clip(next_states, -1, 1)

    def update(self, batch: Batch, env: Env=None) -> InfoDict:
        self.step += 1
        
        if self.algo == 'sacv1' or self.algo == 'sacv1_attention':

            new_rng, new_actor, new_critic, new_value, new_target_value, new_temp, info = _update_jit(
                self.rng, self.actor, self.critic, self.value, self.target_value,
                self.temp, batch, self.discount, self.tau, self.target_entropy,
                self.step % self.target_update_period == 0)

            self.rng = new_rng
            self.actor = new_actor
            self.critic = new_critic
            self.value = new_value
            self.target_value = new_target_value
            self.temp = new_temp
        elif self.algo == 'sacv1_two_head' or self.algo == 'sacv1_two_head_attention':

            # new_rng, new_actor, new_critic, new_value, new_target_value, new_temp, info = _update_jit(
            #     self.rng, self.actor, self.critic, self.value, self.target_value,
            #     self.temp, batch, self.discount, self.tau, self.target_entropy,
            #     self.step % self.target_update_period == 0)

            # self.rng = new_rng
            # self.actor = new_actor
            # self.critic = new_critic
            # self.value = new_value
            # self.target_value = new_target_value
            # self.temp = new_temp

            new_rng, new_actor, new_critic, new_value, new_target_value, new_inv_dyna, new_temp, new_temp_state, info = _update_sac_two_head_jit(
            self.rng, self.actor, self.critic, self.value, self.target_value, self.inv_dyna, 
            self.temp, self.temp_state, batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0, env, self.beta)

            self.rng = new_rng
            self.actor = new_actor
            self.critic = new_critic
            self.value = new_value
            self.target_value = new_target_value
            self.temp = new_temp
            self.temp_state = new_temp_state
            self.inv_dyna = new_inv_dyna

        return info


class CBAM(nn.Module):
    reduction_ratio: int = 16   # 如果任务复杂度较高，可以尝试更小的 reduction_ratio（如8）

    @nn.compact
    def __call__(self, x):
        b, *dims = x.shape
        x = x.reshape(b, -1, dims[-1])  # 保持通道维度最后

        # 通道注意力（共享MLP）
        def shared_mlp():
            return nn.Sequential([
                nn.Dense(dims[-1] // self.reduction_ratio),
                nn.relu,
                nn.Dense(dims[-1])
            ])
        
        channel_avg = jnp.mean(x, axis=1, keepdims=True)
        channel_max = jnp.max(x, axis=1, keepdims=True)
        mlp = shared_mlp()
        channel_attention = nn.sigmoid(mlp(channel_avg) + mlp(channel_max))
        x = x * channel_attention

        # 空间注意力
        spatial_avg = jnp.mean(x, axis=-1, keepdims=True)  # (B, S, 1)
        spatial_max = jnp.max(x, axis=-1, keepdims=True)   # (B, S, 1)
        spatial_concat = jnp.concatenate([spatial_avg, spatial_max], axis=-1)
        spatial_attention = nn.sigmoid(
            nn.Conv(1, kernel_size=(3,), padding='SAME')(spatial_concat)  # 更小的kernel
        )
        return (x * spatial_attention).reshape(b, *dims)
    