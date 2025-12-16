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
    loss = jnp.mean((q - targets) ** 2)
    return loss


def distant_coherence_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    times_to_terminals: jnp.ndarray,
    utils_to_terminals: jnp.ndarray,
    terminals_are_completions: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):

    batch_size, seq_len = q.shape
    assert jnp.issubdtype(q.dtype, jnp.floating)
    assert jnp.issubdtype(q_a_star_next.dtype, jnp.floating)
    assert jnp.issubdtype(rewards.dtype, jnp.floating)
    assert jnp.issubdtype(times_to_terminals.dtype, jnp.integer)
    assert jnp.issubdtype(utils_to_terminals.dtype, jnp.floating)
    assert completion_mask.dtype == jnp.bool_

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
    diffs_sum = jnp.sum(
        jnp.where(
            valid_pair_rel_utils,
            jnp.maximum(0.0, diffs),
            0.0,
        ) ** 2 
    )
    diffs_denom = valid_pair_rel_utils.sum()

    # Add lower bound loss based on utils to terminals if terminals are
    # completions
    diffs = utils_to_terminals - q
    diffs_sum += jnp.sum(
        jnp.where(
            terminals_are_completions,
            jnp.maximum(0.0, diffs),
            0.0,
        ) ** 2
    )
    diffs_denom += jnp.sum(terminals_are_completions)

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
    diffs_sum += jnp.sum(
        jnp.where(
            valid_pair_rel_utils,
            jnp.maximum(0.0, diffs),
            0.0,
        ) ** 2 
    )
    diffs_denom += valid_pair_rel_utils.sum()

    return diffs_sum / jnp.maximum(diffs_denom, 1.0)

def coherent_q_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    times_to_terminals: jnp.ndarray,
    utils_to_terminals: jnp.ndarray,
    terminals_are_completions: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
    distant_coherence_weight: float,
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
            terminals_are_completions,
            completion_mask,
            discount,
        ) * distant_coherence_weight
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
            [0, 0, 0, 1],
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
            [5, 7, 9, 11],
            [42, 44, 46, 48],
        ], dtype=jnp.float32
    )

    
    utils_to_terminals = []
    times_to_terminals = []
    terminals_are_completions = []
    for batch_idx in range(rewards.shape[0]):
        utils_to_terminals_seq, times_to_terminals_seq, terminals_are_completions_seq = get_utils_and_times_to_terminals(
            rewards[batch_idx],
            ~continuation_mask[batch_idx],
            ~completion_mask[batch_idx],
            discount,
        )
        utils_to_terminals.append(utils_to_terminals_seq)
        times_to_terminals.append(times_to_terminals_seq)
        terminals_are_completions.append(terminals_are_completions_seq)

    utils_to_terminals = jnp.stack(utils_to_terminals, axis=0)
    times_to_terminals = jnp.stack(times_to_terminals, axis=0)
    terminals_are_completions = jnp.stack(terminals_are_completions, axis=0)

    print("utils to terminals")
    print(utils_to_terminals)
    print("times to terminals")
    print(times_to_terminals)
    print("terminals are completions")
    print(terminals_are_completions)


    loss = coherent_q_loss(
        q,
        q_a_star_next,
        rewards,
        times_to_terminals,
        utils_to_terminals,
        terminals_are_completions,
        completion_mask,
        discount,
        distant_coherence_weight=1.0,
    )
    print("coherent q loss")
    print(loss)