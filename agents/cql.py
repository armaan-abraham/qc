import copy
from typing import Any, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from einops import repeat, einsum, reduce
from flax import linen as nn

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from rlpd_networks import MLP, Actor
from rlpd_distributions import TanhNormal
from .cql_util import construct_attn_mask_seq

from functools import partial


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_temp = self.param(
            "log_temp",
            init_fn=lambda key: jnp.full((), jnp.log(self.initial_temperature)),
        )
        return jnp.exp(log_temp)

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
        self.blocks = []
        for layer_idx in range(self.num_layers):
            self.blocks.append(
                {
                    "obs_ln1": nn.LayerNorm(name=f"obs_ln1_{layer_idx}"),
                    "act_ln1": nn.LayerNorm(name=f"act_ln1_{layer_idx}"),
                    "obs_ln2": nn.LayerNorm(name=f"obs_ln2_{layer_idx}"),
                    "act_ln2": nn.LayerNorm(name=f"act_ln2_{layer_idx}"),
                    "act_query": nn.DenseGeneral(
                        features=(self.num_heads, self.d_model_action // self.num_heads),
                        axis=-1,
                        name=f"act_query_{layer_idx}"
                    ),
                    "obs_key": nn.DenseGeneral(
                        features=(self.num_heads, self.d_model_observation // self.num_heads),
                        axis=-1,
                        name=f"obs_key_{layer_idx}"
                    ),
                    "act_key": nn.DenseGeneral(
                        features=(self.num_heads, self.d_model_action // self.num_heads),
                        axis=-1,
                        name=f"act_key_{layer_idx}"
                    ),
                    "obs_value": nn.DenseGeneral(
                        features=(self.num_heads, self.d_model_observation // self.num_heads),
                        axis=-1,
                        name=f"obs_value_{layer_idx}"
                    ),
                    "act_value": nn.DenseGeneral(
                        features=(self.num_heads, self.d_model_action // self.num_heads),
                        axis=-1,
                        name=f"act_value_{layer_idx}"
                    ),
                    "obs_mlp_1": nn.Dense(self.d_model_observation * 4, name=f"obs_mlp_1_{layer_idx}"),
                    "obs_mlp_2": nn.Dense(self.d_model_observation, name=f"obs_mlp_2_{layer_idx}"),
                    "act_mlp_1": nn.Dense(self.d_model_action * 4, name=f"act_mlp_1_{layer_idx}"),
                    "act_mlp_2": nn.Dense(self.d_model_action, name=f"act_mlp_2_{layer_idx}"),
                }
            )
        self.final_act_ln = nn.LayerNorm(name="final_act_ln")
        self.q_value_head = nn.Dense(1, name="q_value_head")


    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions: jnp.ndarray, mask: jnp.ndarray, multi_action: bool = True) -> jnp.ndarray:
        assert observations.ndim == 3
        assert actions.ndim == 3
        assert mask.ndim == 2
        assert actions.shape[0:2] == observations.shape[0:2] == mask.shape
        assert self.num_heads < self.d_model_action
        assert self.d_model_action % self.num_heads == 0
        d_head = self.d_model_action // self.num_heads

        batch_size, seq_len = observations.shape[0:2]

        if multi_action:
            # Construct attention mask
            attn_mask_seq = construct_attn_mask_seq(seq_len)
            attn_mask = repeat(
                attn_mask_seq,
                "seq_obs seq_q seq_k -> batch seq_obs seq_q seq_k",
                batch=batch_size,
            )
            # Each query should have at least 1 key
            assert jnp.all(reduce(attn_mask, "batch seq_obs seq_q seq_k -> batch seq_obs seq_q", "min") >= 0)

        # If observation encoder is provided, encode observations
        if self.encoder is not None:
            observations = self.encoder(observations)

        # Embed each action
        act_embed = self.act_embed(actions)
        if multi_action:
            act_resid = repeat(act_embed, "batch seq_obs d_model -> batch seq_obs seq_act d_model", seq_act=seq_len)
            # Define positional embeddings for each action position relative to
            # each observation position. This will start at 0 for the action at
            # the observation time step and then increment by 1.
            act_predict_idx = jnp.maximum(repeat(jnp.arange(seq_len), "seq_act -> seq_obs seq_act", seq_obs=seq_len) - jnp.arange(seq_len)[:, None], 0)
            act_resid += repeat(
                self.act_pos_embed(jnp.arange(act_predict_idx)),
                "seq_obs seq_act d_model -> batch seq_obs seq_act d_model",
                batch=batch_size,
            )
        else:
            # Each q value is for the action at the observation time step
            act_resid = act_embed + self.act_pos_embed(0)

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
                # actions for that observation and the observation. Observations
                # do not attend to anything (markovian). Compute queries for
                # actions, and keys and values for both actions and observations.

                attn_q = self.blocks[layer_idx]["act_query"](act_resid_ln1)
                assert attn_q.shape == (batch_size, seq_len, seq_len, self.num_heads, d_head)

                obs_resid_attn = repeat(obs_resid_ln1, "batch seq_obs d_model -> batch seq_obs 1 d_model")
                # [batch seq_obs 1 num_heads d_head]
                attn_k_obs = self.blocks[layer_idx]["obs_key"](obs_resid_attn)
                assert attn_k_obs.shape == (batch_size, seq_len, 1, self.num_heads, d_head)
                # [batch seq_obs seq_act num_heads d_head]
                attn_k_act = self.blocks[layer_idx]["act_key"](act_resid_ln1)
                assert attn_k_act.shape == (batch_size, seq_len, seq_len, self.num_heads, d_head)
                # [batch seq_obs seq_act+1 num_heads d_head]
                attn_k = jnp.concatenate([
                    attn_k_obs,
                    attn_k_act,
                ], axis=2)
                assert attn_k.shape == (batch_size, seq_len, seq_len + 1, self.num_heads, d_head)

                attn_v_obs = self.blocks[layer_idx]["obs_value"](obs_resid_attn)
                attn_v_act = self.blocks[layer_idx]["act_value"](act_resid_ln1)
                attn_v = jnp.concatenate([
                    attn_v_obs,
                    attn_v_act,
                ], axis=2)
                assert attn_v.shape == (batch_size, seq_len, seq_len + 1, self.num_heads, d_head)

                attn_weights = einsum(
                    attn_q,
                    attn_k,
                    "batch seq_obs seq_q num_heads d_head, batch seq_obs seq_k num_heads d_head -> batch seq_obs seq_q seq_k num_heads",
                ) / jnp.sqrt(d_head)

                # Attn mask is equivalently applied to all heads
                attn_weights = jnp.where(attn_mask[..., None] > 0, attn_weights, -1e6)

                attn_probs = nn.softmax(attn_weights, axis=3)

                attn_output = einsum(
                    attn_probs,
                    attn_v,
                    "batch seq_obs seq_q seq_k num_heads, batch seq_obs seq_k num_heads d_head -> batch seq_obs seq_q (num_heads d_head)",
                )
            else:
                # Single action case: action attends to observation only
                attn_v = self.blocks[layer_idx]["obs_value"](obs_resid_ln1)
                assert attn_v.shape == (batch_size, seq_len, self.num_heads, d_head)

                attn_output = reduce(
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



class CQLAgent(flax.struct.PyTreeNode):
    """Coherent Q learning (CQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        pass 

    def actor_loss(self, batch, grad_params, rng):
        pass

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        pass

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
        """Sample actions from the actor."""
        dist = self.network.select('actor')(observations)
        actions = dist.sample(seed=rng)
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
        pass
    
def get_config():
    raise NotImplementedError
    config = ml_collections.ConfigDict(
        dict(
            agent_name='cql',
        )
    )
    return config