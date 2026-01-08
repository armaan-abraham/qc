import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum

def all_between(A):
    """
    Create array B where B[i,j] is True if all elements A[min(i,j):max(i,j)] are True.
    
    Args:
        A: Boolean array of shape (n,)
    
    Returns:
        Array of shape (n, n)
    """
    n = A.shape[0]
    
    cumsum = jnp.cumsum(A)
    
    # Create indices
    i = jnp.arange(n)[:, None]  # (n, 1)
    j = jnp.arange(n)[None, :]  # (1, n)
    
    # Prepend 0 to cumsum for easier indexing
    cumsum_padded = jnp.concatenate([jnp.array([0]), cumsum])
    print("cumsum_padded:", cumsum_padded)
    
    lo = jnp.minimum(i, j)
    hi = jnp.maximum(i, j)
    
    range_sum = cumsum_padded[hi] - cumsum_padded[lo]
    range_len = hi - lo
    
    B = range_sum == range_len
    
    return B

def get_utils_to_seq_end(
    rewards: jnp.ndarray,
    discount: float,
):
    batch_size, seq_len = rewards.shape

    utils_to_seq_end = jnp.zeros((batch_size, seq_len), dtype=rewards.dtype)

    # These utils are only meaningful when used to compute relative utils
    # between transitions in the same trajectory
    for t in reversed(range(seq_len)):
        utils_to_seq_end = utils_to_seq_end.at[:, t].set(
            rewards[:, t] + discount * jnp.where(
                t + 1 < seq_len,
                utils_to_seq_end[:, t + 1],
                0.0,
            )
        )
    
    return utils_to_seq_end

def get_chunk_utils(
    rewards: jnp.ndarray,
    utils_to_seq_end: jnp.ndarray,
    completion_mask: jnp.ndarray,
    continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
):
    batch_size, seq_len = rewards.shape
    num_chunks = seq_len // action_chunk_size

    q_idx = jnp.arange(0, seq_len, action_chunk_size)
    v_next_idx = q_idx + action_chunk_size - 1
    chunk_utils = utils_to_seq_end[:, q_idx] - (
        discount ** (action_chunk_size - 1) * (
            rewards[:, v_next_idx] + discount * v_next_idx
        )
    )
    assert chunk_utils.shape == (batch_size, num_chunks)
    continuation_mask_by_chunk = rearrange(
        continuation_mask,
        "batch (num_chunks chunk_size) -> batch num_chunks chunk_size",
        chunk_size=action_chunk_size,
    )
    # A chunk is valid if all non-final transitions are continuations
    chunk_valids = jnp.all(
        continuation_mask_by_chunk[:, :, :-1],
        axis=-1,
    )
    assert chunk_valids.shape == (batch_size, num_chunks)
    chunk_completions = completion_mask[:, v_next_idx]
    assert chunk_completions.shape == (batch_size, num_chunks)
    return chunk_utils, chunk_valids, chunk_completions

def get_rectified_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    utils_to_seq_end: jnp.ndarray,
    chunk_utils: jnp.ndarray,
    chunk_valids: jnp.ndarray,
    chunk_completions: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
):
    seq_len = utils_to_seq_end.shape[1]
    batch_size, num_chunks = q.shape

    chunk_terminals = (~chunk_valids) | chunk_completions

    q_idx = jnp.arange(0, seq_len, action_chunk_size)
    v_idx = q_idx + action_chunk_size - 1
    chunk_start_utils_to_seq_end = utils_to_seq_end[:, q_idx]
    pairwise_time_diffs = (
        repeat(
            jnp.arange(num_chunks),
            "chunk_pre -> chunk_pre chunk_post",
            chunk_post=num_chunks,
        ) - repeat(
            jnp.arange(num_chunks),
            "chunk_post -> chunk_pre chunk_post",
            chunk_pre=num_chunks,
        )
    ) * action_chunk_size
    chunk_start_pairwise_utils = repeat(
        chunk_start_utils_to_seq_end,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_chunks,
    ) - repeat(
        chunk_start_utils_to_seq_end,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_chunks,
    ) * discount ** pairwise_time_diffs
    
    # Differences are valid if time diff is positive and all intermediate chunks
    # are valid
    occurs_after = pairwise_time_diffs > 0
    intermediates_valid = jax.vmap(all_between, in_axes=0)(
        chunk_valids.astype(bool)
    )
    assert intermediates_valid.shape == (batch_size, num_chunks, num_chunks)
    pairwise_utils_valid = occurs_after & intermediates_valid

    # Compute lower bound loss by comparing earlier q values to later v_next values
    chunk_post_utils_to_seq_end = 
    lower_bound_violations = jnp.maximum((
        chunk_start_pairwise_utils + repeat(
            chunk_utils,
            "batch chunk_post -> batch chunk_pre chunk_post",
            chunk_pre=num_chunks,
        )
    ) - repeat(
        q,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_chunks,
    ), 0.0)
    lower_bound_loss = jnp.sum(

    )





def get_lql_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    rewards: jnp.ndarray,
    completion_mask: jnp.ndarray,
    continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int = 1,
):
    batch_size, seq_len = rewards.shape
    assert seq_len % action_chunk_size == 0
    num_chunks = seq_len // action_chunk_size
    assert q.shape == v_next.shape == (batch_size, num_chunks)
    assert completion_mask.shape == continuation_mask.shape == (batch_size, seq_len)
    assert completion_mask.dtype == bool
    assert continuation_mask.dtype == bool

    utils_to_seq_end = get_utils_to_seq_end(
        rewards,
        discount,
    )
    assert utils_to_seq_end.dtype == jax.float32

    chunk_utils, chunk_valids, chunk_completions = get_chunk_utils(
        rewards,
        utils_to_seq_end,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size,
    )
    assert chunk_utils.dtype == jnp.float32
    assert chunk_valids.dtype == jnp.bool
    assert chunk_completions.dtype == jnp.bool

    bellman_loss = jnp.sum(
        (
            q - (chunk_utils + v_next * (discount ** action_chunk_size) * (1 - chunk_completions.astype(q.dtype)))
        ) ** 2 * chunk_valids
    ) / jnp.maximum(jnp.sum(chunk_valids.astype(jnp.int32)), 1)

    rectified_loss = get_rectified_loss(
        q,
        v_next,
        utils_to_seq_end,
        chunk_utils,
        chunk_valids,
        chunk_completions,
        discount,
        action_chunk_size,
    )

    return bellman_loss + rectified_loss


if __name__ == "__main__":

    a = jnp.array([1, 0, 1, 0]).astype(bool)
    print("all between")
    print(all_between(a))
     

    discount = 0.9

    rewards = jnp.array(
        [
            [1, 2, 3, 4],
            [5, 7, 9, 11],
        ]
    )
    q_a_star_next = jnp.array(
        [
            [10, 11, 12, 13],
            [40, 42, 44, 46],
        ]
    )
    completion_mask = jnp.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    ).astype(bool)
    continuation_mask = jnp.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]
    ).astype(bool)
    q = jnp.array(
        [
            [11, 13, 15, 17],
            [42, 44, 46, 48],
        ]
    )
