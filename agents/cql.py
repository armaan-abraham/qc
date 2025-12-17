import copy
from functools import partial
from typing import Any, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from einops import repeat, einsum, rearrange, reduce
from flax import linen as nn

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.encoders import encoder_modules
from utils.networks import Actor, ActorVectorField, Value
from agents.cql_util import coherent_q_loss

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
        assert batch['utils_to_terminals'].ndim == 2 # [batch, seq_len]
        assert batch['times_to_terminals'].ndim == 2 # [batch, seq_len]
        assert batch['terminals_are_completions'].ndim == 2 # [batch, seq_len]
        assert batch['masks'].ndim == 2 # [batch, seq_len]
        assert batch['terminals'].ndim == 2 # [batch, seq_len]
        assert batch['observations'].shape[0:2] == batch['actions'].shape[0:2] == batch['rewards'].shape[0:2] == batch['masks'].shape[0:2] == batch['terminals'].shape[0:2] == batch['utils_to_terminals'].shape[0:2] == batch['times_to_terminals'].shape[0:2]

        batch_size, seq_len = batch['observations'].shape[0:2]
        assert seq_len == self.config['horizon_length']
        assert batch_size == self.config['batch_size']
        rng, sample_rng = jax.random.split(rng)
        a_star_next = self.sample_actions(batch['next_observations'], rng=sample_rng, greedy=True)
        assert a_star_next.shape == (batch_size, seq_len, self.config['action_dim'])
        q_a_star_next_ens = jax.lax.stop_gradient(self.network.select('target_critic')(batch['next_observations'], actions=a_star_next))
        assert q_a_star_next_ens.shape == (self.config['num_critics'], batch_size, seq_len)
        q_a_star_next = reduce(q_a_star_next_ens, 'ensemble batch seq -> batch seq', 'mean')

        q_ens = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        assert q_ens.shape == (self.config['num_critics'], batch_size, seq_len)

        q_loss_ens = jax.vmap(
            coherent_q_loss,
            in_axes=(0, None, None, None, None, None, None, None, None, None)
        )(
            q_ens,
            q_a_star_next,
            batch['rewards'],
            batch['times_to_terminals'],
            batch['utils_to_terminals'],
            batch['terminals_are_completions'],
            ~batch['masks'].astype(bool),
            self.config['discount'],
            self.config['distant_coherence_weight'],
            self.config['completion_coherence_weight'],
        )
        assert q_loss_ens.shape == (self.config['num_critics'],)

        q_loss = jnp.mean(q_loss_ens)

        return q_loss, {
            'critic_loss': q_loss,
            'q_mean': q_ens.mean(),
            'q_std': q_ens.std(),
            'q_max': q_ens.max(),
            'q_min': q_ens.min(),
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
            q_loss = -self.network.select('critic')(batch['observations'], actions=actor_actions).mean()

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

    @partial(jax.jit, static_argnames=('greedy',))
    def sample_actions(
        self,
        observations,
        rng=None,
        greedy=False,
    ):
        assert observations.shape[-1] == self.config['obs_dim']

        # Observations can be any shape, as long as the last dimension is the
        # observation dimension.
        if self.config['actor_type'] == 'flow':
            batch_dims = observations.shape[:-1]
            observations = observations.reshape((-1, observations.shape[-1]))
            num_observations = observations.shape[0]
            noises = jax.random.normal(
                rng,
                (
                    (num_observations, self.config['actor_num_samples'], self.config['action_dim'])
                ),
            )
            observations = repeat(
                observations,
                "obs obs_dim -> obs samples_per_obs obs_dim",
                samples_per_obs=self.config["actor_num_samples"],
            )
            assert observations.shape[:-1] == noises.shape[:-1]

            actions = self.compute_flow_actions(observations, noises)
            actions = jnp.clip(actions, -1, 1)
            assert actions.shape == (num_observations, self.config['actor_num_samples'], self.config['action_dim'])

            actions_seq = rearrange(
                actions,
                "batch sample act_dim -> (batch sample) 1 act_dim",
            )
            observations_seq = rearrange(
                observations,
                "batch sample obs_dim -> (batch sample) 1 obs_dim",
            )

            q_ens = self.network.select("critic")(observations_seq, actions=actions_seq)
            assert q_ens.shape == ((self.config['num_critics'], num_observations * self.config['actor_num_samples'], 1))
            q_ens = rearrange(
                q_ens,
                "ensemble (batch sample) 1 -> ensemble batch sample",
                batch=num_observations,
                sample=self.config["actor_num_samples"],
            )
            q = reduce(q_ens, 'ensemble batch sample -> batch sample', 'mean')
            assert q.shape == (num_observations, self.config['actor_num_samples'])

            return actions[jnp.arange(num_observations), jnp.argmax(q, axis=-1)].reshape(batch_dims + (self.config['action_dim'],))
        else:
            dist = self.network.select('actor')(observations)

            if greedy:
                # For evaluation: sample from policy without epsilon-greedy exploration
                actions = dist.sample(seed=rng)
            else:
                # For training: epsilon-greedy action selection
                rng, eps_rng, sample_rng, random_rng = jax.random.split(rng, 4)

                # Sample from the policy
                policy_actions = dist.sample(seed=sample_rng)

                # Generate random actions uniformly in [-1, 1]
                random_actions = jax.random.uniform(
                    random_rng,
                    shape=policy_actions.shape,
                    minval=-1.0,
                    maxval=1.0,
                )

                # Epsilon-greedy: with probability epsilon, use random action
                use_random = jax.random.uniform(eps_rng, shape=policy_actions.shape[:-1]) < self.config['epsilon']
                actions = jnp.where(use_random, random_actions, policy_actions)

            actions = jnp.clip(actions, -1, 1)
            return actions
    
    @jax.jit
    def compute_flow_actions(
        self,
        observations,
        noises,
    ):
        """Compute actions from the BC flow model using the Euler method."""
        batch_dims = noises.shape[:-1]
        actions = noises
        # Euler method.
        for i in range(self.config['flow_steps']):
            t = jnp.full(batch_dims + (1,), i / self.config['flow_steps'])
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
            hidden_dims=config['critic_hidden_dims'],
            layer_norm=True,
            num_ensembles=config['num_critics'],
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
            actor_def = ActorVectorField(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['action_dim'],
                layer_norm=True,
                encoder=encoders.get('actor'),
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
            optimizer = optax.adamw(learning_rate=config['lr'], weight_decay=config["weight_decay"])
        else:
            optimizer = optax.adam(learning_rate=config['lr'])

        if config["max_grad_norm"] != -1.:
            network_tx = optax.chain(
                optax.clip_by_global_norm(config['max_grad_norm']),
                optimizer,
            )
        else:
            network_tx = optimizer

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
            critic_hidden_dims=(512, 512, 512, 512),
            num_critics=2,
            distant_coherence_weight=1.0,
            completion_coherence_weight=1.0,

            # Actor
            actor_type='flow', # gaussian or flow
            actor_num_samples=16, # Number of action samples for actor flow
            flow_steps=10,  # Number of flow steps.
            actor_hidden_dims=(512, 512, 512, 512),

            tau=0.005,  # Target network update rate.
            weight_decay=0,
            max_grad_norm=-1.,  # Maximum gradient norm for clipping (-1 to disable).
            discount=0.99,  # Discount factor.
            lr=3e-4,  # Learning rate.
            batch_size=128,

            bc_alpha=0.0, # Behavioral cloning loss weight.

            epsilon=0.0, # Epsilon for epsilon-greedy action selection (Gaussian actor only).

        )
    )
    return config