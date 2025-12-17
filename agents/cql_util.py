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
    loss = jnp.sum((q - targets) ** 2)
    denom = q.size
    return loss, denom


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

    # === Mutual Q loss ===
    mixed_util_from_obs_pre = pair_rel_utils + repeat(
        q,
        "batch obs_post -> batch obs_pre obs_post",
        obs_pre=seq_len,
    ) * discount ** (jnp.abs(pair_rel_times))

    opt_util_from_obs_pre = repeat(
        q,
        "batch obs_pre -> batch obs_pre obs_post",
        obs_post=seq_len,
    )

    diffs = mixed_util_from_obs_pre - opt_util_from_obs_pre
    diffs_sum = jnp.sum(
        jnp.where(
            valid_pair_rel_utils,
            jnp.maximum(0.0, diffs),
            0.0,
        ) ** 2
    )
    diffs_denom = valid_pair_rel_utils.size / 2

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
    diffs_denom += q.size

    return diffs_sum, diffs_denom

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
    one_step_loss, one_step_denom = one_step_bellman_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        discount,
    )
    distant_loss, distant_denom = distant_coherence_loss(
        q,
        q_a_star_next,
        rewards,
        times_to_terminals,
        utils_to_terminals,
        terminals_are_completions,
        completion_mask,
        discount,
    )
    total_loss = one_step_loss + distant_coherence_weight * distant_loss
    total_denom = one_step_denom + distant_coherence_weight * distant_denom
    return total_loss / total_denom

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