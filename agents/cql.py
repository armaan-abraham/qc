import copy
from typing import Any, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from einops import repeat, einsum, rearrange
from flax import linen as nn

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.encoders import encoder_modules
from utils.networks import Actor, ActorVectorField
from agents.cql_util import construct_attn_mask_seq, get_action_predict_pos, coherent_q_loss

class Value(nn.Module):
    d_model_observation: int
    d_model_action: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    encoder: nn.Module = None # observation encoder

    def setup(self):
        self.act_embed = nn.Dense(self.d_model_action, name="act_embed")
        self.act_pos_embed = nn.Embed(
            num_embeddings=self.max_seq_len,
            features=self.d_model_action,
            name="act_pos_embed"
        )
        self.obs_embed = nn.Dense(self.d_model_observation, name="obs_embed")
        self.d_attn_head = self.d_model_action // self.num_heads
        blocks = []
        for layer_idx in range(self.num_layers):
            blocks.append(
                {
                    "obs_ln1": nn.LayerNorm(name=f"obs_ln1_{layer_idx}"),
                    "act_ln1": nn.LayerNorm(name=f"act_ln1_{layer_idx}"),
                    "obs_ln2": nn.LayerNorm(name=f"obs_ln2_{layer_idx}"),
                    "act_ln2": nn.LayerNorm(name=f"act_ln2_{layer_idx}"),
                    "act_query": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"act_query_{layer_idx}"
                    ),
                    "obs_key": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"obs_key_{layer_idx}"
                    ),
                    "act_key": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"act_key_{layer_idx}"
                    ),
                    "obs_value": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"obs_value_{layer_idx}"
                    ),
                    "act_value": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"act_value_{layer_idx}"
                    ),
                    "obs_mlp_1": nn.Dense(self.d_model_observation * 4, name=f"obs_mlp_1_{layer_idx}"),
                    "obs_mlp_2": nn.Dense(self.d_model_observation, name=f"obs_mlp_2_{layer_idx}"),
                    "act_mlp_1": nn.Dense(self.d_model_action * 4, name=f"act_mlp_1_{layer_idx}"),
                    "act_mlp_2": nn.Dense(self.d_model_action, name=f"act_mlp_2_{layer_idx}"),
                }
            )
        self.blocks = blocks
        self.final_act_ln = nn.LayerNorm(name="final_act_ln")
        self.q_value_head = nn.Dense(1, name="q_value_head")


    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, multi_action: bool = True) -> jnp.ndarray:
        assert observations.ndim == 3
        assert actions.ndim == 3
        assert actions.shape[0:2] == observations.shape[0:2]
        assert self.num_heads < self.d_model_action
        assert self.d_model_action % self.num_heads == 0

        batch_size, seq_len = observations.shape[0:2]
        assert seq_len <= self.max_seq_len

        if multi_action:
            # Construct attention mask
            attn_mask_seq = construct_attn_mask_seq(seq_len)
            attn_mask = repeat(
                attn_mask_seq,
                "seq_obs seq_q seq_k -> batch seq_obs seq_q seq_k",
                batch=batch_size,
            )

        # If observation encoder is provided, encode observations
        if self.encoder is not None:
            observations = self.encoder(observations)

        # Embed each action
        act_embed = self.act_embed(actions)
        if multi_action:
            act_resid = repeat(act_embed, "batch seq_obs d_model -> batch seq_obs seq_act d_model", seq_act=seq_len)
            # Define positional embeddings for each action position relative to
            # each observation position. This will start at 0 for the action at
            # the observation time step and then increment by 1. This will be
            # invalid for actions preceding the observation, but these will be
            # masked out.
            act_resid += repeat(
                self.act_pos_embed(get_action_predict_pos(seq_len)),
                "seq_obs seq_act d_model -> batch seq_obs seq_act d_model",
                batch=batch_size,
            )
        else:
            # Each q value is for the action at the observation time step
            act_resid = act_embed + self.act_pos_embed(jnp.array([0]))

        obs_resid = self.obs_embed(observations)

        for layer_idx in range(self.num_layers):
            if multi_action:
                assert act_resid.shape == (batch_size, seq_len, seq_len, self.d_model_action)
            else:
                assert act_resid.shape == (batch_size, seq_len, self.d_model_action)
            assert obs_resid.shape == (batch_size, seq_len, self.d_model_observation)

            # === Pre-attn layer norm ===

            obs_resid_ln1 = self.blocks[layer_idx]["obs_ln1"](obs_resid)
            act_resid_ln1 = self.blocks[layer_idx]["act_ln1"](act_resid)

            # === Attention ===

            if multi_action:
                # Each action for a given observation attends to all previous
                # actions for that observation and the observation. Observations do
                # not attend to anything. Compute queries for actions, and keys and
                # values for both actions and observations.

                attn_q = self.blocks[layer_idx]["act_query"](act_resid_ln1)
                assert attn_q.shape == (batch_size, seq_len, seq_len, self.num_heads, self.d_attn_head)

                obs_resid_attn = repeat(obs_resid_ln1, "batch seq_obs d_model -> batch seq_obs 1 d_model")
                assert obs_resid_attn.shape == (batch_size, seq_len, 1, self.d_model_observation)
                # [batch seq_obs 1 num_heads d_head]
                attn_k_obs = self.blocks[layer_idx]["obs_key"](obs_resid_attn)
                assert attn_k_obs.shape == (batch_size, seq_len, 1, self.num_heads, self.d_attn_head)
                # [batch seq_obs seq_act num_heads d_head]
                attn_k_act = self.blocks[layer_idx]["act_key"](act_resid_ln1)
                assert attn_k_act.shape == (batch_size, seq_len, seq_len, self.num_heads, self.d_attn_head)
                # [batch seq_obs seq_act+1 num_heads d_head]
                attn_k = jnp.concatenate([
                    attn_k_obs,
                    attn_k_act,
                ], axis=2)
                assert attn_k.shape == (batch_size, seq_len, seq_len + 1, self.num_heads, self.d_attn_head)

                attn_v_obs = self.blocks[layer_idx]["obs_value"](obs_resid_attn)
                attn_v_act = self.blocks[layer_idx]["act_value"](act_resid_ln1)
                attn_v = jnp.concatenate([
                    attn_v_obs,
                    attn_v_act,
                ], axis=2)
                assert attn_v.shape == (batch_size, seq_len, seq_len + 1, self.num_heads, self.d_attn_head)

                attn_weights = einsum(
                    attn_q,
                    attn_k,
                    "batch seq_obs seq_q num_heads d_head, batch seq_obs seq_k num_heads d_head -> batch seq_obs seq_q seq_k num_heads",
                ) / jnp.sqrt(self.d_attn_head)

                # Attn mask is equivalently applied to all heads
                attn_weights = jnp.where(attn_mask[..., None] > 0, attn_weights, -1e6)

                attn_probs = nn.softmax(attn_weights, axis=3)

                attn_output = rearrange(einsum(
                    attn_probs,
                    attn_v,
                    "batch seq_obs seq_q seq_k num_heads, batch seq_obs seq_k num_heads d_head -> batch seq_obs seq_q num_heads d_head",
                ), "batch seq_obs seq_q num_heads d_head -> batch seq_obs seq_q (num_heads d_head)")
            else:
                # Single action case: action attends to observation only
                attn_v = self.blocks[layer_idx]["obs_value"](obs_resid_ln1)
                assert attn_v.shape == (batch_size, seq_len, self.num_heads, self.d_attn_head)

                attn_output = rearrange(
                    attn_v,
                    "batch seq_obs num_heads d_head -> batch seq_obs (num_heads d_head)",
                )

            act_resid += attn_output

            # === Pre-MLP layer norm ===

            obs_resid_ln2 = self.blocks[layer_idx]["obs_ln2"](obs_resid)
            act_resid_ln2 = self.blocks[layer_idx]["act_ln2"](act_resid)

            # === MLP ===

            obs_mlp = self.blocks[layer_idx]["obs_mlp_1"](obs_resid_ln2)
            obs_mlp = nn.gelu(obs_mlp)
            obs_mlp = self.blocks[layer_idx]["obs_mlp_2"](obs_mlp)
            obs_resid += obs_mlp

            act_mlp = self.blocks[layer_idx]["act_mlp_1"](act_resid_ln2)
            act_mlp = nn.gelu(act_mlp)
            act_mlp = self.blocks[layer_idx]["act_mlp_2"](act_mlp)
            act_resid += act_mlp

        # Compute q values from action representations
        act_resid = self.final_act_ln(act_resid)
        q_values = self.q_value_head(act_resid).squeeze(-1)

        if multi_action:
            # Result contains (multi-action) Q value for each state and each
            # contiguous future action sequence starting from that state.
            assert q_values.shape == (batch_size, seq_len, seq_len)
        else:
            assert q_values.shape == (batch_size, seq_len)
        return q_values


class PastAwareActorVectorField(nn.Module):
    """Actor vector field network for flow matching"""

    d_model_observation: int
    d_model_action: int
    num_layers: int
    num_heads: int
    max_seq_len: int
    action_dim: int
    encoder: nn.Module = None # observation encoder

    def setup(self) -> None:
        self.obs_embed = nn.Dense(self.d_model_observation, name="obs_embed")
        self.obs_pos_embed = nn.Embed(num_embeddings=self.max_seq_len, features=self.d_model_observation, name="obs_pos_embed")

        self.action_embed = nn.Dense(self.d_model_action, name="action_embed")

        self.d_attn_head = self.d_model_observation // self.num_heads

        blocks = []
        for layer_idx in range(self.num_layers):
            blocks.append(
                {
                    "obs_ln1": nn.LayerNorm(name=f"obs_ln1_{layer_idx}"),
                    "obs_ln2": nn.LayerNorm(name=f"obs_ln2_{layer_idx}"),
                    "act_ln2": nn.LayerNorm(name=f"act_ln2_{layer_idx}"),
                    "obs_query": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"act_query_{layer_idx}"
                    ),
                    "obs_key": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"obs_key_{layer_idx}"
                    ),
                    "obs_value": nn.DenseGeneral(
                        features=(self.num_heads, self.d_attn_head),
                        axis=-1,
                        name=f"obs_value_{layer_idx}"
                    ),
                    "obs_to_act": nn.Dense(self.d_model_action, name=f"obs_to_act_{layer_idx}"),
                    "obs_mlp_1": nn.Dense(self.d_model_observation * 4, name=f"obs_mlp_1_{layer_idx}"),
                    "obs_mlp_2": nn.Dense(self.d_model_observation, name=f"obs_mlp_2_{layer_idx}"),
                    "act_mlp_1": nn.Dense(self.d_model_action * 4, name=f"act_mlp_1_{layer_idx}"),
                    "act_mlp_2": nn.Dense(self.d_model_action, name=f"act_mlp_2_{layer_idx}"),
                }
            )
        self.blocks = blocks
        self.final_act_ln = nn.LayerNorm(name="final_act_ln")
        self.action_head = nn.Dense(self.action_dim, name="action_head")

    @nn.compact
    def __call__(self, observations, actions, times, is_encoded=False):
        """Return the vectors at the given states, actions, and times

        Args:
            observations: Observations.
            actions: Actions.
            times: Times.
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        
        assert observations.ndim == 3
        assert actions.ndim == 3
        assert times.ndim == 3
        assert observations.shape[0:2] == actions.shape[0:2] == times.shape[0:2]
        batch_size, seq_len = observations.shape[0:2]

        # Jointly embed actions and times
        act_resid = self.action_embed(
            jnp.concatenate([actions, times], axis=-1)
        )

        # Embed observations
        obs_resid = self.obs_embed(observations) + repeat(
            self.obs_pos_embed(jnp.arange(seq_len)),
            "seq d_model -> batch seq d_model",
            batch=batch_size,
        )

        attn_mask = jnp.tril(jnp.ones((seq_len, seq_len)))

        # Transformer layers
        for layer_idx in range(self.num_layers):
            # === Pre-attn layer norm ===

            obs_resid_ln1 = self.blocks[layer_idx]["obs_ln1"](obs_resid)

            # === Attention ===

            # Each observation attends to all previous observations
            attn_q = self.blocks[layer_idx]["obs_query"](obs_resid_ln1)
            assert attn_q.shape == (batch_size, seq_len, self.num_heads, self.d_attn_head)

            attn_k = self.blocks[layer_idx]["obs_key"](obs_resid_ln1)
            assert attn_k.shape == (batch_size, seq_len, self.num_heads, self.d_attn_head)

            attn_v = self.blocks[layer_idx]["obs_value"](obs_resid_ln1)
            assert attn_v.shape == (batch_size, seq_len, self.num_heads, self.d_attn_head)

            attn_weights = einsum(
                attn_q,
                attn_k,
                "batch seq_q num_heads d_head, batch seq_k num_heads d_head -> batch seq_q seq_k num_heads",
            ) / jnp.sqrt(self.d_attn_head)

            # Attn mask is equivalently applied to all heads
            # Broadcast mask over batch and head dims
            attn_weights = jnp.where(attn_mask[None, :, :, None] > 0, attn_weights, -1e6)

            attn_probs = nn.softmax(attn_weights, axis=2)

            attn_output = rearrange(einsum(
                attn_probs,
                attn_v,
                "batch seq_q seq_k num_heads, batch seq_k num_heads d_head -> batch seq_q num_heads d_head",
            ), "batch seq_q num_heads d_head -> batch seq_q (num_heads d_head)")

            obs_resid += attn_output
            act_resid += self.blocks[layer_idx]["obs_to_act"](obs_resid)

            # === Pre-MLP layer norm ===

            obs_resid_ln2 = self.blocks[layer_idx]["obs_ln2"](obs_resid)
            act_resid_ln2 = self.blocks[layer_idx]["act_ln2"](act_resid)

            # === MLP ===

            obs_mlp = self.blocks[layer_idx]["obs_mlp_1"](obs_resid_ln2)
            obs_mlp = nn.gelu(obs_mlp)
            obs_mlp = self.blocks[layer_idx]["obs_mlp_2"](obs_mlp)
            obs_resid += obs_mlp

            act_mlp = self.blocks[layer_idx]["act_mlp_1"](act_resid_ln2)
            act_mlp = nn.gelu(act_mlp)
            act_mlp = self.blocks[layer_idx]["act_mlp_2"](act_mlp)
            act_resid += act_mlp
        
        act_resid = self.final_act_ln(act_resid)
        acts_out = self.action_head(act_resid)
        assert acts_out.shape == (batch_size, seq_len, self.action_dim)

        return acts_out


class CQLAgent(flax.struct.PyTreeNode):
    """Coherent Q learning (CQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        # Batch data must have proper sequence structure
        assert batch['observations'].ndim == 3 # [batch, seq_len, obs_dim]
        assert batch['actions'].ndim == 3 # [batch, seq_len, act_dim]
        assert batch['rewards'].ndim == 2 # [batch, seq_len]
        assert batch['masks'].ndim == 2 # [batch, seq_len]
        assert batch['terminals'].ndim == 2 # [batch, seq_len]
        assert batch['observations'].shape[0:2] == batch['actions'].shape[0:2] == batch['rewards'].shape[0:2] == batch['masks'].shape[0:2] == batch['terminals'].shape[0:2]

        batch_size, seq_len = batch['observations'].shape[0:2]
        rng, sample_rng = jax.random.split(rng)
        a_star_next = self.sample_actions(batch['next_observations'], rng=sample_rng)
        assert a_star_next.shape == (batch_size, seq_len, self.config['action_dim'])
        q_a_star_next = jax.lax.stop_gradient(self.network.select('target_critic')(batch['next_observations'], a_star_next, multi_action=False))
        assert q_a_star_next.shape == (batch_size, seq_len)

        q = self.network.select('critic')(batch['observations'], batch['actions'], multi_action=True, params=grad_params)
        assert q.shape == (batch_size, seq_len, seq_len)

        q_loss = coherent_q_loss(
            q=q,
            q_a_star_next=q_a_star_next,
            rewards=batch['rewards'],
            completion_mask=~batch['masks'].astype(bool),
            continuation_mask=~batch['terminals'].astype(bool),
            discount=self.config['discount'],
        )
        return q_loss, {
            'critic_loss': q_loss,
            'q_mean': q.mean(),
            'q_std': q.std(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q_a_star_next_mean': q_a_star_next.mean(),
            'q_a_star_next_std': q_a_star_next.std(),
            'q_a_star_next_max': q_a_star_next.max(),
            'q_a_star_next_min': q_a_star_next.min(),
        }
        

    def actor_loss(self, batch, grad_params, rng):
        # Batch data must have proper sequence structure
        assert batch['observations'].ndim == 3 # [batch, seq_len, obs_dim]
        assert batch['actions'].ndim == 3 # [batch, seq_len, act_dim]
        assert batch['observations'].shape[0:2] == batch['actions'].shape[0:2]
        batch_size, seq_len, action_dim = batch['actions'].shape

        if self.config['actor_type'] == 'flow':
            rng, x_rng, t_rng = jax.random.split(rng, 3)

            # BC flow loss.
            x_0 = jax.random.normal(x_rng, (batch_size, seq_len, action_dim))
            x_1 = batch['actions']
            t = jax.random.uniform(t_rng, (batch_size, seq_len, 1))
            x_t = (1 - t) * x_0 + t * x_1
            vel = x_1 - x_0

            pred = self.network.select('actor')(batch['observations'], x_t, t, params=grad_params)
        
            bc_flow_loss = jnp.mean((pred - vel) ** 2)

            return bc_flow_loss, {
                'bc_flow_loss': bc_flow_loss,
            }

        else: # gaussian
            actor_dists = self.network.select('actor')(batch['observations'], params=grad_params)
            actor_actions_unclipped = actor_dists.mode()
            actor_actions = jnp.clip(actor_actions_unclipped, -1, 1)

            # Behavorial cloning loss
            log_probs_mean = actor_dists.log_prob(jnp.clip(batch['actions'], -1 + 1e-5, 1 - 1e-5)).mean()
            bc_loss = -log_probs_mean

            # Q loss
            q_loss = -self.network.select('critic')(batch['observations'], actor_actions, multi_action=False).mean()

            return bc_loss * self.config['bc_alpha'] + q_loss, {
                'bc_loss': bc_loss,
                'q_loss': q_loss,
                'actions_unclipped_mean': actor_actions_unclipped.mean(),
                'actions_unclipped_std': actor_actions_unclipped.std(),
                'actions_unclipped_max': actor_actions_unclipped.max(),
                'actions_unclipped_min': actor_actions_unclipped.min(),
                'log_probs_mean': log_probs_mean,
            }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        rng = rng if rng is not None else self.rng

        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        info['mean_completions'] = (1 - batch['masks']).sum(axis=1).mean()
        info['mean_terminations'] = batch['terminals'].sum(axis=1).mean()

        loss = critic_loss + actor_loss
        return loss, info
        

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @staticmethod
    def _update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')

        return self.replace(network=new_network, rng=new_rng), info
    
    @jax.jit
    def update(self, batch):
        return self._update(self, batch)
    
    @jax.jit
    def batch_update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    @jax.jit
    def sample_actions(
        self,
        observations,
        rng=None,
    ):
        assert observations.ndim == 3
        assert observations.shape[1] <= self.config['horizon_length']
        assert observations.shape[2] == self.config['obs_dim']
        batch_size, seq_len = observations.shape[0:2]

        if self.config['actor_type'] == 'flow':
            noises = jax.random.normal(
                rng,
                (
                    (batch_size * self.config["actor_num_samples"], seq_len, self.config['action_dim'])
                ),
            )
            observations = repeat(
                observations,
                "batch seq obs_dim -> (batch sample) seq obs_dim",
                sample=self.config["actor_num_samples"],
            )
            assert observations.shape[:-1] == noises.shape[:-1]

            actions = self.compute_flow_actions(observations, noises)
            
            q = rearrange(
                self.network.select("critic")(observations, actions, multi_action=False),
                "(batch sample) seq -> (batch seq) sample",
                batch=batch_size,
                sample=self.config["actor_num_samples"],
            )
            actions = rearrange(
                actions,
                "(batch sample) seq act_dim -> (batch seq) sample act_dim",
                batch=batch_size,
                sample=self.config["actor_num_samples"],
            )
            actions_opt = actions[jnp.arange(batch_size * seq_len), jnp.argmax(q, axis=-1)]

            return rearrange(
                actions_opt,
                "(batch seq) act_dim -> batch seq act_dim",
                batch=batch_size,
                seq=seq_len,
            )
        else:
            dist = self.network.select('actor')(observations)
            actions = dist.sample(seed=rng)
            actions = jnp.clip(actions, -1, 1)
            return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        assert observations.ndim == 3
        assert noises.ndim == 3
        assert observations.shape[0:2] == noises.shape[0:2]
        assert observations.shape[2] == self.config['obs_dim']
        assert noises.shape[2] == self.config['action_dim']
        batch_size, seq_len = observations.shape[0:2]

        actions = noises

        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full((batch_size, seq_len, 1), i / self.config['flow_steps'])
            vels = self.network.select('actor')(observations, actions, t, is_encoded=True)
            actions = actions + vels / self.config['flow_steps']
        actions = jnp.clip(actions, -1, 1)

        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        assert ex_observations.ndim == 3, ex_observations.shape
        assert ex_actions.ndim == 3, ex_actions.shape

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        config['action_dim'] = ex_actions.shape[-1]
        config['obs_dim'] = ex_observations.shape[-1]
        batch_size, seq_len = ex_observations.shape[0:2]

        # Define encoders.
        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        # Define networks.
        critic_def = Value(
            d_model_observation=config['d_model_observation'],
            d_model_action=config['d_model_action'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            max_seq_len=config['horizon_length'],
            encoder=encoders.get('critic'),
        )
        if config['actor_type'] == 'gaussian':
            actor_def = Actor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['action_dim'],
                layer_norm=True,
                encoder=encoders.get('actor'),
                const_std=True,
                tanh_squash=False,
            )
            actor_params = (ex_observations,)
        else:
            actor_def = PastAwareActorVectorField(
                d_model_observation=config['actor_d_model_observation'],
                d_model_action=config['actor_d_model_action'],
                num_layers=config['actor_num_layers'],
                num_heads=config['actor_num_heads'],
                max_seq_len=config['horizon_length'],
                encoder=encoders.get('actor'),
                action_dim=config['action_dim'],
            )
            actor_params = (ex_observations, ex_actions, jnp.zeros((batch_size, seq_len, 1)))

        network_info = dict(
            actor=(actor_def, actor_params),
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        if config["weight_decay"] > 0.:
            network_tx = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params

        params[f'modules_target_critic'] = params[f'modules_critic']


        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


    
def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cql',
            
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).

            horizon_length=ml_collections.config_dict.placeholder(int), # Will be set

            # Critic
            d_model_observation=256,
            d_model_action=64,
            num_layers=4,
            num_heads=4,

            # Actor
            actor_type='gaussian', # gaussian or flow

            # Gaussian actor
            actor_hidden_dims=(512, 512, 512),

            # Flow actor
            flow_steps=10,  # Number of flow steps.
            actor_num_samples=32, # Number of action samples for actor flow
            actor_d_model_observation=256,
            actor_d_model_action=256,
            actor_num_layers=4,
            actor_num_heads=4,

            tau=0.005,  # Target network update rate.
            weight_decay=1e-3,
            discount=0.99,  # Discount factor.
            lr=3e-4,  # Learning rate.
            batch_size=256,

            bc_alpha=0.0, # Behavioral cloning loss weight.

        )
    )
    return config