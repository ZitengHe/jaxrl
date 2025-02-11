import functools
from typing import Any, Callable, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.module import init
from tensorflow_probability.substrates import jax as tfp
import distrax

tfd = tfp.distributions
tfb = tfp.bijectors

from jaxrl.networks.common import MLP, Params, PRNGKey, default_init

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


class MSEPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        actions = nn.Dense(self.action_dim,
                           kernel_init=default_init())(outputs)
        return nn.tanh(actions)


class NormalTanhPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    final_fc_init_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True
    init_mean: Optional[jnp.ndarray] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim,
                         kernel_init=default_init(
                             self.final_fc_init_scale))(outputs)
        if self.init_mean is not None:
            means += self.init_mean

        if self.state_dependent_std:
            log_stds = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                    self.final_fc_init_scale))(outputs)
        else:
            log_stds = self.param('log_stds', nn.initializers.zeros,
                                  (self.action_dim, ))

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds = jnp.clip(log_stds, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means = nn.tanh(means)

        base_dist = tfd.MultivariateNormalDiag(loc=means,
                                               scale_diag=jnp.exp(log_stds) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist,
                                               bijector=tfb.Tanh())
        else:
            return base_dist


class NormalTanhMixturePolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    num_components: int = 5
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> tfd.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        logits = nn.Dense(self.action_dim * self.num_components,
                          kernel_init=default_init())(outputs)
        means = nn.Dense(self.action_dim * self.num_components,
                         kernel_init=default_init(),
                         bias_init=nn.initializers.normal(stddev=1.0))(outputs)
        log_stds = nn.Dense(self.action_dim * self.num_components,
                            kernel_init=default_init())(outputs)

        shape = list(observations.shape[:-1]) + [-1, self.num_components]
        logits = jnp.reshape(logits, shape)
        mu = jnp.reshape(means, shape)
        log_stds = jnp.reshape(log_stds, shape)

        log_stds = jnp.clip(log_stds, LOG_STD_MIN, LOG_STD_MAX)

        components_distribution = tfd.Normal(loc=mu,
                                             scale=jnp.exp(log_stds) *
                                             temperature)

        base_dist = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(logits=logits),
            components_distribution=components_distribution)

        dist = tfd.TransformedDistribution(distribution=base_dist,
                                           bijector=tfb.Tanh())

        return tfd.Independent(dist, 1)

class InverseDerministicPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    tanh_squash_output: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)
        outputs = nn.Dense(self.action_dim, kernel_init=default_init())(outputs)

        if not self.tanh_squash_output:
            outputs = nn.tanh(outputs)
        else:
            return outputs


class TwoHeadsGaussianPolicy(nn.Module):
    share_hidden_dims: Sequence[int]
    action_hidden_dims: Sequence[int]
    state_hidden_dims: Sequence[int]
    state_dim: int
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    layer_norm: Optional[bool] = False
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> Tuple[tfd.Distribution, tfd.Distribution]:
        z = MLP(hidden_dims=self.share_hidden_dims,
                activations=self.activations,
                activate_final=True,
                dropout_rate=self.dropout_rate,
                layer_norm=self.layer_norm)(observations,
                                                training=training)
        outpus_action = MLP(self.action_hidden_dims, 
                           activations=self.activations,
                           dropout_rate=self.dropout_rate, 
                           layer_norm=self.layer_norm)(z)
        means_action = nn.Dense(self.action_dim,
                                kernel_init=default_init(
                                self.log_std_scale
                                ))(outpus_action)
        outputs_state = MLP(self.state_hidden_dims, 
                          activations=self.activations, 
                          dropout_rate=self.dropout_rate, 
                          layer_norm=self.layer_norm)(z)
        means_state = nn.Dense(self.state_dim,
                               kernel_init=default_init(
                               self.log_std_scale
                                ))(outputs_state)
        if self.state_dependent_std:
            log_stds_action = nn.Dense(self.action_dim,
                                       kernel_init=default_init(
                                       self.log_std_scale))(means_action)
            log_stds_state = nn.Dense(self.state_dim,
                                      kernel_init=default_init(
                                      self.log_std_scale))(outputs_state)
        else:
            log_stds_action = self.param('log_stds_action', nn.initializers.zeros,
                                  (self.action_dim, ))
            log_stds_state = self.param('log_stds_state', nn.initializers.zeros,
                                  (self.state_dim, ))
            

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds_action = jnp.clip(log_stds_action, log_std_min, log_std_max)
        log_stds_state = jnp.clip(log_stds_state, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means_action = nn.tanh(means_action)

        # # 加上下边这个多元高斯分布，无法打印网络结构
        # base_dist_action = distrax.MultivariateNormalDiag(loc=means_action,
        #                                        scale_diag=jnp.exp(log_stds_action) *
        #                                        temperature)
        # base_dist_state = distrax.MultivariateNormalDiag(loc=means_state,
        #                                        scale_diag=jnp.exp(log_stds_state) *
        #                                        temperature)

        # 加上下边这个多元高斯分布，无法打印网络结构
        base_dist_action = tfd.MultivariateNormalDiag(loc=means_action,
                                               scale_diag=jnp.exp(log_stds_action) *
                                               temperature)
        base_dist_state = tfd.MultivariateNormalDiag(loc=means_state,
                                               scale_diag=jnp.exp(log_stds_state) *
                                               temperature)

        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist_action, bijector=tfb.Tanh()), tfd.TransformedDistribution(distribution=base_dist_state, bijector=tfb.Tanh())
        else:
            # return base_dist_action, base_dist_state
            return base_dist_action

class TwoHeadsDeterministicPolicy(nn.Module):
    share_hidden_dims: Sequence[int]
    action_hidden_dims: Sequence[int]
    state_hidden_dims: Sequence[int]
    state_dim: int
    action_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    state_dependent_std: bool = True
    dropout_rate: Optional[float] = None
    layer_norm: Optional[bool] = False
    log_std_scale: float = 1.0
    log_std_min: Optional[float] = None
    log_std_max: Optional[float] = None
    tanh_squash_distribution: bool = True

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 temperature: float = 1.0,
                 training: bool = False) -> Tuple[tfd.Distribution, jnp.ndarray]:
        z = MLP(hidden_dims=self.share_hidden_dims,
                activations=self.activations,
                activate_final=True,
                dropout_rate=self.dropout_rate,
                layer_norm=self.layer_norm)(observations,
                                                training=training)
        means_action = MLP(self.action_hidden_dims+(self.action_dim, ), 
                           activations=self.activations,
                           dropout_rate=self.dropout_rate, 
                           layer_norm=self.layer_norm)(z)

        pre_state = MLP(self.state_hidden_dims + (self.state_dim, ), 
                          activations=self.activations, 
                          dropout_rate=self.dropout_rate, 
                          layer_norm=self.layer_norm)(z)

        if self.state_dependent_std:
            log_stds_action = nn.Dense(self.action_dim,
                                       kernel_init=default_init(
                                       self.log_std_scale))(z)
        else:
            log_stds_action = self.param('log_stds_action', nn.initializers.zeros,
                                  (self.action_dim, ))
            

        log_std_min = self.log_std_min or LOG_STD_MIN
        log_std_max = self.log_std_max or LOG_STD_MAX
        log_stds_action = jnp.clip(log_stds_action, log_std_min, log_std_max)

        if not self.tanh_squash_distribution:
            means_action = nn.tanh(means_action)

        # 加上下边这个多元高斯分布，无法打印网络结构
        base_dist_action = distrax.MultivariateNormalDiag(loc=means_action,
                                               scale_diag=jnp.exp(log_stds_action) *
                                               temperature)
        if self.tanh_squash_distribution:
            return tfd.TransformedDistribution(distribution=base_dist_action, bijector=tfb.Tanh()), pre_state
        else:
            return base_dist_action, pre_state

@functools.partial(jax.jit, static_argnames=('actor_apply_fn', 'distribution'))
def _sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    if distribution == 'det':
        return rng, actor_apply_fn({'params': actor_params}, observations,
                                   temperature)
    else:
        try:
            dist = actor_apply_fn({'params': actor_params}, observations,
                                temperature)
            rng, key = jax.random.split(rng)
            return rng, dist.sample(seed=key)
        except:
            dist_a, dist_s = actor_apply_fn({'params': actor_params}, observations,
                                temperature)
            rng, key = jax.random.split(rng)
            return rng, dist_a.sample(seed=key)


def sample_actions(
        rng: PRNGKey,
        actor_apply_fn: Callable[..., Any],
        actor_params: Params,
        observations: np.ndarray,
        temperature: float = 1.0,
        distribution: str = 'log_prob') -> Tuple[PRNGKey, jnp.ndarray]:
    return _sample_actions(rng, actor_apply_fn, actor_params, observations,
                           temperature, distribution)
