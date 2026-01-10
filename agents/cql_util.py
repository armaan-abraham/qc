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
    
    # Pad with an extra zero at the end for easier indexing
    utils_to_seq_end = jnp.concatenate(
        [utils_to_seq_end, jnp.zeros((batch_size, 1), dtype=rewards.dtype)],
        axis=1,
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
    """
    Compute the utilities in each action chunk.
    """
    batch_size, seq_len = rewards.shape
    num_chunks = seq_len // action_chunk_size
    assert utils_to_seq_end.shape == (batch_size, seq_len + 1)

    chunk_start_idx = jnp.arange(0, seq_len, action_chunk_size)
    # This will use the util to seq end from the next chunk, which may not be
    # part of the same trajectory even if the current chunk is valid, but this
    # is okay because the util is to *sequence end*, not episode or trajectory
    # end.
    chunk_utils = utils_to_seq_end[:, chunk_start_idx] - (
        discount ** action_chunk_size * utils_to_seq_end[:, chunk_start_idx + action_chunk_size]
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
    chunk_completion_mask = completion_mask[:, chunk_start_idx + action_chunk_size - 1]
    assert chunk_completion_mask.shape == (batch_size, num_chunks)
    chunk_continuation_mask = continuation_mask[:, chunk_start_idx + action_chunk_size - 1]
    return chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask

def get_rectified_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    utils_to_seq_end: jnp.ndarray,
    chunk_utils: jnp.ndarray,
    chunk_valids: jnp.ndarray,
    chunk_completion_mask: jnp.ndarray,
    chunk_continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
):
    """
    q and v_next are expected to be provided for each chunk (q at chunk start
    and v_next at chunk end).
    """
    seq_len = utils_to_seq_end.shape[1] - 1 # -1 for padding
    batch_size, num_chunks = q.shape

    chunk_start_idx = jnp.arange(0, seq_len, action_chunk_size)
    chunk_start_utils_to_seq_end = utils_to_seq_end[:, chunk_start_idx]
    pairwise_time_diffs = (
         repeat(
            jnp.arange(num_chunks),
            "chunk_post -> chunk_pre chunk_post",
            chunk_pre=num_chunks,
        ) - repeat(
            jnp.arange(num_chunks),
            "chunk_pre -> chunk_pre chunk_post",
            chunk_post=num_chunks,
        )
    ) * action_chunk_size
    print("pairwise time diffs")
    print(pairwise_time_diffs)
    # Compute pairwise utilities between each chunk starting point. Element
    # (i,j) for i < j is the utility from chunk i to j.
    chunk_start_pairwise_utils = repeat(
        chunk_start_utils_to_seq_end,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_chunks,
    ) - repeat(
        chunk_start_utils_to_seq_end,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_chunks,
    ) * discount ** pairwise_time_diffs
    print("chunk start pairwise utils")
    print(chunk_start_pairwise_utils)
    
    # Differences are valid if time diff is positive and all chunks in [i,j) are
    # valid nonterminals
    intermediates_valid = jax.vmap(all_between, in_axes=0)(
        chunk_valids & chunk_continuation_mask
    )
    print("intermediates valid")
    print(intermediates_valid.astype(jnp.int32))
    occurs_after = pairwise_time_diffs > 0
    assert intermediates_valid.shape == (batch_size, num_chunks, num_chunks)
    pairwise_utils_valid = occurs_after & intermediates_valid
    print("pairwise utils valid")
    print(pairwise_utils_valid.astype(jnp.int32))

    # Compute lower bound loss by comparing earlier q values to later v_next
    # values.

    # Compute the estimated utility-to-go from the observed utils between chunks
    # plus the v_next value function estimate at the later chunk.
    chunk_post_util_to_go = repeat(
        chunk_utils + v_next * (discount ** action_chunk_size) * (1 - chunk_completion_mask.astype(q.dtype)),
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_chunks,
    )
    chunk_post_util_to_go_discount = discount ** pairwise_time_diffs
    mixed_util_from_chunk_pre = chunk_start_pairwise_utils + chunk_post_util_to_go * chunk_post_util_to_go_discount
    # Mixed utils are valid if pairs are valid and the later chunk is valid (as
    # v_next is used there)
    mixed_util_from_chunk_pre_valid = pairwise_utils_valid & repeat(
        chunk_valids,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_chunks,
    )
    util_from_chunk_pre = repeat(
        q,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_chunks,
    )
    lower_bound_loss = jnp.sum(
        jnp.maximum(
            mixed_util_from_chunk_pre - util_from_chunk_pre,
            0.0,
        ) ** 2 * mixed_util_from_chunk_pre_valid
    ) / jnp.maximum(jnp.sum(mixed_util_from_chunk_pre_valid.astype(jnp.int32)), 1)
    print("lower bound loss", lower_bound_loss)


    # Compute upper bound loss by comparing later q values to earlier v_next values.

    # Compute estimated utility-to-go from the end of the earlier chunk
    util_from_chunk_end_pre = repeat(
        v_next,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_chunks,
    )

    # Compute the estimated utility-to-go from the end of the earlier chunk
    # using the observed utils between chunks plus the q value function estimate
    # at the later chunk.
    # We need to subtract the util of the earlier chunk from the relative util
    # due to using v_next at the earlier chunk end
    mixed_util_from_chunk_end_pre = (chunk_start_pairwise_utils - repeat(
        chunk_utils,
        "batch chunk_pre -> batch chunk_pre chunk_post",
        chunk_post=num_chunks,
    )) / discount ** action_chunk_size + repeat(
        q,
        "batch chunk_post -> batch chunk_pre chunk_post",
        chunk_pre=num_chunks,
    ) * discount ** (pairwise_time_diffs - action_chunk_size)

    upper_bound_loss = jnp.sum(
        jnp.maximum(
            mixed_util_from_chunk_end_pre - util_from_chunk_end_pre,
            0.0,
        ) ** 2 * pairwise_utils_valid
    ) / jnp.maximum(jnp.sum(pairwise_utils_valid.astype(jnp.int32)), 1)

    print("upper bound loss", upper_bound_loss)

    return lower_bound_loss + upper_bound_loss

def get_bellman_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    chunk_utils: jnp.ndarray,
    chunk_valids: jnp.ndarray,
    chunk_completion_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int,
):
    print("chunk utils")
    print(chunk_utils)
    targets = (chunk_utils + v_next * (discount ** action_chunk_size) * (1 - chunk_completion_mask.astype(q.dtype)))
    print("targets")
    print(targets)
    bellman_loss = jnp.sum(
        (
            q - targets
        ) ** 2 * chunk_valids
    ) / jnp.maximum(jnp.sum(chunk_valids.astype(jnp.int32)), 1)
    return bellman_loss

def get_lql_loss(
    q: jnp.ndarray,
    v_next: jnp.ndarray,
    rewards: jnp.ndarray,
    completion_mask: jnp.ndarray,
    continuation_mask: jnp.ndarray,
    discount: float,
    action_chunk_size: int = 1,
):
    assert jnp.issubdtype(q.dtype, jnp.floating)
    assert jnp.issubdtype(v_next.dtype, jnp.floating)
    assert jnp.issubdtype(rewards.dtype, jnp.floating)
    batch_size, seq_len = rewards.shape
    assert seq_len % action_chunk_size == 0
    num_chunks = seq_len // action_chunk_size
    assert q.shape == v_next.shape == (batch_size, num_chunks)
    assert completion_mask.shape == continuation_mask.shape == (batch_size, seq_len)
    assert completion_mask.dtype == jnp.bool_
    assert continuation_mask.dtype == jnp.bool_

    utils_to_seq_end = get_utils_to_seq_end(
        rewards,
        discount,
    )
    assert utils_to_seq_end.dtype == jnp.float32

    chunk_utils, chunk_valids, chunk_completion_mask, chunk_continuation_mask = get_chunk_utils(
        rewards,
        utils_to_seq_end,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size,
    )
    assert chunk_utils.dtype == jnp.float32
    assert chunk_valids.dtype == jnp.bool
    assert chunk_completion_mask.dtype == jnp.bool
    print("chunk valids")
    print(chunk_valids.astype(jnp.int32))
    print("chunk completions")
    print(chunk_completion_mask.astype(jnp.int32))

    bellman_loss = get_bellman_loss(
        q,
        v_next,
        chunk_utils,
        chunk_valids,
        chunk_completion_mask,
        discount,
        action_chunk_size,
    )
    print("bellman loss", bellman_loss)

    rectified_loss = get_rectified_loss(
        q,
        v_next,
        utils_to_seq_end,
        chunk_utils,
        chunk_valids,
        chunk_completion_mask,
        chunk_continuation_mask,
        discount,
        action_chunk_size,
    )
    print("rectified loss", rectified_loss)

    return bellman_loss + rectified_loss


if __name__ == "__main__":
    print("*************************")

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
            [1, 1, 1, 1],
            [1, 1, 1, 0],
        ]
    ).astype(bool)
    q = jnp.array(
        [
            [11, 13, 15, 17],
            [42, 44, 46, 48],
        ], dtype=jnp.float32
    )

    action_chunk_size = 1
    loss = get_lql_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        continuation_mask,
        discount,
        action_chunk_size=action_chunk_size,
    )
    print("LQL loss", loss)