from typing import Tuple

from gym import Env
import jax
import jax.numpy as jnp
import jax.lax as lax

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
import distrax

def update(key: PRNGKey, actor: Model, critic: Model, temp: Model,
           batch: Batch) -> Tuple[Model, InfoDict]:

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        try:
            dist = actor.apply_fn({'params': actor_params}, batch.observations)

            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
        except:
            dist_a, _ = actor.apply_fn({'params': actor_params}, batch.observations)
            actions = dist_a.sample(seed=key)
            log_probs = dist_a.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy_a': -log_probs.mean()
        }

    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def update_two_head(key: PRNGKey, actor: Model, value: Model, critic: Model, inv_dyna: Model, temp: Model, temp_state: Model,
           batch: Batch, beta: float=5, env: Env=None) -> Tuple[Model, InfoDict]:
    

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        info = {}
        dist_a, dist_s = actor.apply_fn({'params': actor_params},
                       batch.observations,
                       training=True,   
                       rngs={'dropout': key})
        
        actions = dist_a.sample(seed=key)
        log_probs_a = dist_a.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)



        # pre_state = dist_s.sample(seed=key)
        # input_inv = jnp.concatenate([batch.observations, pre_state], -1)
        # frozen_params_inv = jax.lax.stop_gradient(inv_dyna.params)
        # dist_a_inv = inv_dyna.apply_fn({'params': frozen_params_inv}, input_inv)

        ####### AWR   或许也是有道理的？
        ### 
        # v = value(batch.observations)
        # target_v = batch.rewards + batch.masks * next_v
        # exp_s = jnp.exp(beta * (target_v - v))
        # exp_s = jnp.minimum(exp_s, 20.0)
        # state_loss = -(exp_s * log_probs_s).mean()
        ### 
        next_state = dist_s.sample(seed=key)
        log_probs_s = dist_s.log_prob(next_state)
        next_v = value(next_state)
        if beta > 0:
            v = value(batch.observations)
            next_v = value(batch.next_observations)
            target_v = batch.rewards + batch.masks * next_v
            exp_s = jnp.exp(beta * (target_v - v))
            exp_s = jnp.minimum(exp_s, 20.0)
            log_probs_s = dist_s.log_prob(batch.next_observations)

            state_loss = -(exp_s * log_probs_s).mean()
            info.update({
                'AWR_adv_v': (target_v-v).mean(),
                'AWR_exp_s': exp_s.mean(),
            })
        else:
            if temp_state is not None:
                state_loss = (log_probs_s * temp_state() - next_v).mean()
            else:
                state_loss = (log_probs_s * temp() - next_v).mean()
        action_loss = (log_probs_a * temp() - q).mean()

        # for action in actions:
        #     print(action.shape)
        #     next_observation, reward, done, info = env.step(action)
        # exit()
        # inv_loss = ((dist_a.mean() - dist_a_inv)**2).mean()
        state_pred_loss = jnp.mean((batch.next_observations - next_state) ** 2)

        actor_loss = action_loss + state_loss

        # actor_loss = action_loss
        info.update({
            'actor_loss': actor_loss,
            'state_loss': state_loss,
            # 'inv_loss': inv_loss,  # 如果需要的话可以取消注释
            'pre_state_MSE': state_pred_loss,
            'entropy_a': -log_probs_a.mean(),
            'entropy_s': -log_probs_s.mean(),
        })
        return actor_loss, info
    
    new_actor, info = actor.apply_gradient(actor_loss_fn)

    return new_actor, info

def update_inv_dyna(key: PRNGKey, inv_dyna:Model,  # type: ignore
                      batch: Batch) -> Tuple[Model, InfoDict]:
    
    def inv_loss_fn(inv_dyna_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        input = jnp.concatenate([batch.observations, batch.next_observations], -1)
        dist_a = inv_dyna.apply_fn({'params': inv_dyna_params},
                                input,
                                rngs={'dropout': key})
        if type(dist_a) == distrax._src.distributions.mvn_diag.MultivariateNormalDiag:
            log_probs_a = dist_a.log_prob(batch.actions)
            inv_dyna_loss = -log_probs_a.mean()
        elif type(dist_a) == jax._src.interpreters.ad.JVPTracer:
            inv_dyna_loss = ((dist_a - batch.actions) ** 2).mean()
 

        return inv_dyna_loss, {'inv_dyna_loss': inv_dyna_loss.mean(),}

    new_inv_dyna, info = inv_dyna.apply_gradient(inv_loss_fn)
    return new_inv_dyna, info