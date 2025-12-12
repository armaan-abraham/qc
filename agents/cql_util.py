import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum, reduce

from utils.datasets import get_pair_rel_utils

def one_step_bellman_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):
    targets = rewards + discount * q_a_star_next * (1.0 - completion_mask.astype(jnp.float32))
    loss = jnp.mean((q - targets) ** 2)
    return loss


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
    assert jnp.issubdtype(q.dtype, jnp.floating)
    assert jnp.issubdtype(q_a_star_next.dtype, jnp.floating)
    assert jnp.issubdtype(rewards.dtype, jnp.floating)
    assert jnp.issubdtype(times_to_terminals.dtype, jnp.integer)
    assert completion_mask.dtype == jnp.bool_
    assert discount >= 0.0 and discount <= 1.0

    pair_rel_utils, pair_rel_times, valid_pair_rel_utils = get_pair_rel_utils(
        utils_to_terminals,
        times_to_terminals,
        discount,
    )  # [batch, seq_len, seq_len]
    valid_pair_rel_utils_float = valid_pair_rel_utils.astype(jnp.float32)

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

    # Element i j for a sequence in the batch is the observed utility between
    # observation i and j plus the expected utility to go from taking the
    # optimal policy starting at observation j in terms of Q(s_j, a*), which is
    # a lower bound on the expected utility to go for taking action a_i and then
    # following the optimal policy, estimated by Q(s_i, a_i).
    lower_bounds = pair_rel_utils + obs_post_util_to_go_discount * obs_post_util_to_go
    lower_bounds_mean_denom = reduce(
        valid_pair_rel_utils_float,
        "batch obs_pre obs_post -> batch obs_pre",
        "sum",
    )

    lower_bounds_mean = reduce(
        lower_bounds * valid_pair_rel_utils_float,
        "batch obs_pre obs_post -> batch obs_pre",
        "sum",
    ) / jnp.maximum(1.0, lower_bounds_mean_denom)
    upper_bounds_min = jnp.where(
        lower_bounds_mean_denom > 0.0,
        lower_bounds_mean,
        -jnp.inf,
    )

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

    obs_post_q_discount = discount ** (jnp.abs(pair_rel_times) - 1)
    
    # We don't need to worry about the completion mask here because we will zero
    # out any difference elements where the time of i is not less than the time
    # of j, which means any rows in this repeated result corresponding to
    # terminals will be invalidated in the mask.
    q_a_star_obs_pre = repeat(
        q_a_star_next,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    )

    upper_bounds = (q_a_star_obs_pre - rel_util) / obs_post_q_discount
    upper_bounds_mean_denom = reduce(
        valid_pair_rel_utils_float,
        "batch obs_pre obs_post -> batch obs_post",
        "sum",
    )
    upper_bounds_mean = reduce(
        upper_bounds * valid_pair_rel_utils_float,
        "batch obs_pre obs_post -> batch obs_post",
        "sum",
    ) / jnp.maximum(1.0, upper_bounds_mean_denom)
    lower_bounds_max = jnp.where(
        upper_bounds_mean_denom > 0.0,
        upper_bounds_mean,
        jnp.inf,
    )

    # Compute losses

    # Clip bounds based on means

    upper_bounds = jnp.maximum(upper_bounds, repeat(
        upper_bounds_min,
        "batch obs_post -> batch obs_pre obs_post",
        obs_pre=seq_len,
    ))

    lower_bounds = jnp.minimum(lower_bounds, repeat(
        lower_bounds_max,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    ))

    q_obs_post = repeat(
        q,
        "batch obs_post -> batch obs_pre obs_post",
        obs_pre=seq_len,
    )

    diffs = q_obs_post - upper_bounds

    upper_bound_loss = jnp.sum(
        jnp.where(
            valid_pair_rel_utils,
            jnp.maximum(0.0, diffs),
            0.0,
        ) ** 2 
    ) / jnp.maximum(1.0, valid_pair_rel_utils.sum())

    q_obs_pre = repeat(
        q,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    )

    diffs = lower_bounds - q_obs_pre

    lower_bound_loss = jnp.sum(
        jnp.where(
            valid_pair_rel_utils,
            jnp.maximum(0.0, diffs),
            0.0,
        ) ** 2 
    ) / jnp.maximum(1.0, valid_pair_rel_utils.sum())

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
    assert q.ndim == 2
    assert q.shape == q_a_star_next.shape == rewards.shape == times_to_terminals.shape == utils_to_terminals.shape == completion_mask.shape
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
    from utils.datasets import Dataset, get_utils_and_times_to_terminals
    import numpy as np

    discount = 0.9

    rewards = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 7, 9, 11],
        ], dtype=jnp.float32
    )
    q_a_star_next = jnp.array(
        [
            [10, 11, 12, 13],
            [40, 42, 44, 46],
        ], dtype=jnp.float32
    )
    completion_mask = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    ).astype(bool)
    continuation_mask = jnp.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
        ]
    ).astype(bool)
    q = jnp.array(
        [
            [11, 13, 15, 17],
            [42, 44, 46, 48],
        ], dtype=jnp.float32
    )

    
    utils_to_terminals = []
    times_to_terminals = []
    for batch_idx in range(rewards.shape[0]):
        utils_to_terminals_seq, times_to_terminals_seq = get_utils_and_times_to_terminals(
            rewards[batch_idx],
            ~continuation_mask[batch_idx],
            discount,
        )
        utils_to_terminals.append(utils_to_terminals_seq)
        times_to_terminals.append(times_to_terminals_seq)
    utils_to_terminals = jnp.stack(utils_to_terminals, axis=0)
    times_to_terminals = jnp.stack(times_to_terminals, axis=0)

    print("utils to terminals")
    print(utils_to_terminals)
    print("times to terminals")
    print(times_to_terminals)


    loss = coherent_q_loss(
        q,
        q_a_star_next,
        rewards,
        times_to_terminals,
        utils_to_terminals,
        completion_mask,
        discount,
    )
    print("coherent q loss")
    print(loss)