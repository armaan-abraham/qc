import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum

from utils.datasets import get_pair_rel_utils

def one_step_bellman_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):
    targets = rewards + discount * q_a_star_next * (1.0 - completion_mask.astype(jnp.float32))
    return jnp.mean((q - targets) ** 2)


def distant_coherence_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    times_to_terminals: jnp.ndarray,
    utils_to_terminals: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):
    batch_size, seq_len = q.shape

    pair_rel_utils, pair_rel_times, valid_pair_rel_utils = get_pair_rel_utils(
        utils_to_terminals,
        times_to_terminals,
        discount,
    )  # [batch, seq_len, seq_len]

    # === Lower bound loss ===

    # Compare earlier q values to relative util + later q(a*)

    # These are the utils at the later observations computed from the
    # immediately subsequent reward and q(a*). These are the same values that
    # are used in the one-step Bellman update, repeated for comparison to the
    # earlier observations.
    obs_post_util_to_go = repeat(
        rewards + discount * q_a_star_next * (1.0 - completion_mask.astype(jnp.float32)),
        "batch obs_post -> batch obs_pre obs_post",
        obs_pre=seq_len,
    )
    obs_post_util_to_go_discount = discount ** jnp.abs(pair_rel_times)
    # print("obs post util to go discount:")
    # print(obs_post_util_to_go_discount)

    # Element i j for a sequence in the batch is the observed utility between
    # observation i and j plus the expected utility to go from taking the
    # optimal policy starting at observation j in terms of Q(s_j, a*), which is
    # a lower bound on the expected utility to go for taking action a_i and then
    # following the optimal policy, estimated by Q(s_i, a_i).
    mixed_util_from_obs_pre = pair_rel_utils + obs_post_util_to_go_discount * obs_post_util_to_go
    q_obs_pre = repeat(
        q,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    )
    diffs = mixed_util_from_obs_pre - q_obs_pre
    lower_bound_loss = jnp.where(
        valid_pair_rel_utils,
        jnp.maximum(0.0, diffs),
        0.0,
    ) ** 2 / jnp.maximum(1.0, valid_pair_rel_utils.sum())


    # === Upper bound loss ===

    # Compare earlier q(a*) to observed util + later q values
    
    # We are comparing to the earlier q(a*), which are actually one observation
    # after the one we use to compute the relative utils, so we need to subtract
    # the reward immediately before the q(a*) observations from the relative
    # utils and divide by the discount.
    rel_util = (pair_rel_utils - repeat(
        rewards,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    )) / discount

    obs_post_q = repeat(
        q,
        "batch obs_post -> batch obs_pre obs_post",
        obs_pre=seq_len,
    )

    obs_post_q_discount = discount ** (jnp.abs(pair_rel_times) - 1)

    # Element i j for a sequence in the batch is the observed utility between
    # observations i+1 and j plus the expected utility to go from taking action
    # a_j and then following the optimal policy afterward in terms of Q*(s_j,
    # a_j). Q*(s_{i+1}, a*) is an upper bound on this value.
    mixed_util_from_obs_pre = rel_util + obs_post_q * obs_post_q_discount

    # We don't need to worry about the completion mask here because we will zero
    # out any difference elements where the time of i is not less than the time
    # of j, which means any rows in this repeated result corresponding to
    # terminals will be invalidated in the mask.
    q_a_star_obs_pre = repeat(
        q_a_star_next,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    )
    diffs = mixed_util_from_obs_pre - q_a_star_obs_pre
    upper_bound_loss = jnp.where(
        valid_pair_rel_utils,
        jnp.maximum(0.0, diffs),
        0.0,
    ) ** 2 / jnp.maximum(1.0, valid_pair_rel_utils.sum())

    return lower_bound_loss + upper_bound_loss

def coherent_q_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    times_to_terminals: jnp.ndarray,
    utils_to_terminals: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):
    return (
        one_step_bellman_loss(
            q,
            q_a_star_next,
            rewards,
            completion_mask,
            discount,
        )
        + distant_coherence_loss(
            q,
            q_a_star_next,
            rewards,
            times_to_terminals,
            utils_to_terminals,
            completion_mask,
            discount,
        )
    )



if __name__ == "__main__":
    from utils.datasets import Dataset
    import numpy as np

    jax.random.PRNGKey(2)
    np.random.seed(5)

    discount = 0.9
    data = {
        'observations': np.arange(6),
        'next_observations': np.arange(6) + 1,
        'actions': np.arange(6) * 2,
        'rewards': np.arange(6) * 0.5 + 1,
        'masks': np.array(
            [0, 1, 1, 0, 1, 1]
        ),
        'terminals': np.array(
            [1, 0, 0, 1, 0, 1]
        ),
    }
    print("Initial data:")
    for k, v in data.items():
        print(f"{k}:")
        print(v)
    print()

    dataset = Dataset.create(discount=discount, **data)
    batch = dataset.sample_in_trajectories(batch_size=2, sequence_length=4)
    # Convert to jax
    batch = {k: jnp.array(v) for k, v in batch.items()}
    print("Sampled batch:")
    for k, v in batch.items():
        print(f"{k}:")
        print(v)
    print()

    q = jnp.array([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 8.0, 10.0, 12.0],
    ])
    q_a_star_next = jnp.array([
        [2.0, 3.0, 4.0, 5.0],
        [5.0, 9.0, 11.0, 13.0],
    ])

    loss = coherent_q_loss(
        q,
        q_a_star_next,
        batch['rewards'],
        batch['times_to_terminals'],
        batch['utils_to_terminals'],
        1 - batch['masks'],
        discount,
    )
    print("coherent q loss")
    print(loss)