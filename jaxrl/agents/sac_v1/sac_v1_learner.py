"""Implementations of algorithms for continuous control."""

import functools
from typing import Optional, Sequence, Tuple

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

@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_sac_two_head_jit(rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_value: Model, inv_dyna: Model, temp: Model, batch: Batch, discount: float,
    tau: float, target_entropy: float, update_target: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    new_critic, critic_info = update_q(critic, target_value, batch, discount)

    rng, key = jax.random.split(rng)
    new_inv_dyna, inv_dyna_info = update_inv_dyna(key, inv_dyna, batch)
    new_actor, actor_info = update_two_head(key, actor, value, new_critic, new_inv_dyna, temp, batch)

    rng, key = jax.random.split(rng)
    new_value, value_info = update_v(key, new_actor, new_critic, value, temp,
                                     batch, True)

    if update_target:
        new_target_value = target_update(new_value, target_value, tau)
    else:
        new_target_value = target_value

    new_temp, alpha_info = temperature.update(temp, actor_info['entropy_a'],
                                              target_entropy)

    return rng, new_actor, new_critic, new_value, new_target_value, new_inv_dyna, new_temp, {
        **{f"training/{k}": v for k, v in critic_info.items()},
        **{f"training/{k}": v for k, v in value_info.items()},
        **{f"training/{k}": v for k, v in actor_info.items()},
        **{f"training/{k}": v for k, v in alpha_info.items()},
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

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        if algo == 'sacv1':
            actor_def = policies.NormalTanhPolicy(hidden_dims, action_dim,)
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

    def update(self, batch: Batch) -> InfoDict:
        self.step += 1
        
        if self.algo == 'sacv1':

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
        elif self.algo == 'sacv1_two_head':

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

            new_rng, new_actor, new_critic, new_value, new_target_value, new_inv_dyna, new_temp, info = _update_sac_two_head_jit(
            self.rng, self.actor, self.critic, self.value, self.target_value, self.inv_dyna, 
            self.temp, batch, self.discount, self.tau, self.target_entropy,
            self.step % self.target_update_period == 0)

            self.rng = new_rng
            self.actor = new_actor
            self.critic = new_critic
            self.value = new_value
            self.target_value = new_target_value
            self.temp = new_temp
            self.inv_dyna = new_inv_dyna

        return info
