import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum

def get_target_q(
    rewards: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):
    """Termination mask is 1 where episode terminates (not including episode
    boundaries that are not terminations)."""
    # Compute reward for every multi action q function
    batch_size, seq_len = rewards.shape

    rewards_by_obs = jnp.concatenate(
        [
            rewards,
            jnp.zeros((batch_size, seq_len)),
        ],
        axis=1,
    )
    range_idxs = jnp.arange(seq_len)[:, None] + jnp.arange(seq_len)[None, :]
    # Index into rewards by obs to get sequences of rewards to sum over for each observation
    rewards_by_obs = rewards_by_obs[:, range_idxs]
    assert rewards_by_obs.shape == (batch_size, seq_len, seq_len)

    # Multiply each observation-wise reward sequence by discounting coefficients
    discounts = discount ** jnp.arange(seq_len)

    rewards_by_obs = rewards_by_obs * discounts

    # Discounted reward targets are just cumulative sums
    target_rewards = jnp.cumsum(rewards_by_obs, axis=-1)

    # Do zero padding trick and discounting for target q values
    q_a_star_next_pad = jnp.concatenate(
        [
            # Q is 0 
            jnp.where(
                completion_mask,
                0.0,
                q_a_star_next,
            ),
            jnp.zeros((batch_size, seq_len)),
        ],
        axis=-1
    )
    range_idxs = jnp.arange(seq_len)[:, None] + jnp.arange(seq_len)[None, :]
    target_q_a_star = q_a_star_next_pad[:, range_idxs]
    discounts = discount ** (jnp.arange(seq_len) + 1)
    target_q_a_star = target_q_a_star * discounts

    target_q = target_rewards + target_q_a_star

    # Shift over predictions so that the last dimension corresponds to the last
    # action being predicted
    src_i, src_j = jnp.triu_indices(seq_len)
    src_j = seq_len - 1 - src_j
    target_q_shift = jnp.zeros_like(target_q)
    dst_i = src_i
    dst_j = src_j + dst_i
    target_q = target_q_shift.at[:, dst_i, dst_j].set(target_q[:, src_i, src_j])
    
    return target_q


def construct_valid_q_mask(seq_len: int) -> jnp.ndarray:
    # This is for a single sequence
    # Construct valid mask for multi-action q values
    return jnp.triu(jnp.ones((seq_len, seq_len))).astype(bool)


def construct_valid_q_mask_batch(continuation_mask: jnp.ndarray) -> jnp.ndarray:
    batch_size, seq_len = continuation_mask.shape
    valid_q_mask_base = construct_valid_q_mask(seq_len)
    valid_q_mask = repeat(
        valid_q_mask_base,
        "seq_obs seq_act -> batch seq_obs seq_act",
        batch=batch_size,
    )
    continuation_mask_by_obs = repeat(
        continuation_mask,
        "batch seq_act -> batch seq_obs seq_act",
        seq_obs=seq_len,
    )
    intersections = valid_q_mask & ~continuation_mask_by_obs
    # Find index of first True in each row
    first_true_idx = jnp.argmax(intersections, axis=-1)
    
    has_true = jnp.any(intersections, axis=-1)
    
    col_indices = jnp.arange(seq_len)
    
    # Mask positions after first True (and only if row has a True)
    mask = (col_indices > first_true_idx[:, :, None]) & has_true[:, :, None]
    result = jnp.where(mask, 0, valid_q_mask).astype(bool)

    return result


def coherent_q_loss(
    q: jnp.ndarray,
    q_a_star_next: jnp.ndarray,
    rewards: jnp.ndarray,
    completion_mask: jnp.ndarray,
    continuation_mask: jnp.ndarray,
    discount: float,
):
    assert completion_mask.dtype == bool
    assert continuation_mask.dtype == bool
    assert q.shape == rewards.shape == q_a_star_next.shape
    # TODO: check boundaries

    batch_size, seq_len = rewards.shape

    # These are the bellman-derived values for each multi action q for observed
    # actions in terms of observed rewards and q values from the next
    # observation and policy-generated action. These have no gradient flow.
    target_q = get_target_q(
        rewards,
        q_a_star_next,
        completion_mask,
        discount,
    )

    diag_i, diag_j = jnp.diag_indices(seq_len)
    target_q_one_step = target_q[:, diag_i, diag_j]
    one_step_loss = jnp.mean((q - target_q_one_step) ** 2)
    
    valid_q_mask_batch = construct_valid_q_mask_batch(continuation_mask)
    assert valid_q_mask_batch.dtype == bool

    # For any given observation, Q functions with fewer parametrized actions
    # should be greater. This is because taking the optimal policy earlier will
    # result in a higher (or equal) utility than taking any actual sequence of
    # actions.
    q_expand = repeat(
        q,
        "batch seq_obs -> batch seq_obs seq_act",
        seq_act=seq_len,
    )
    diffs = q_expand - target_q
    diffs_i, diffs_j = jnp.triu_indices(seq_len, k=1)
    diffs = diffs[:, diffs_i, diffs_j]
    valid_diffs = valid_q_mask_batch[:, diffs_i, diffs_j]
    inequality_loss = jnp.sum(
        jnp.where(
            valid_diffs,
            jnp.maximum(diffs, 0.0) ** 2,
            0.0,
        )
    ) / jnp.maximum(jnp.sum(valid_diffs), 1)
    return one_step_loss + inequality_loss
    

if __name__ == "__main__":
    # Simple test
    # seq_len = 4
    # attn_mask_seq = construct_attn_mask_seq(seq_len)
    # print(attn_mask_seq)

    rewards = jnp.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    )
    q_a_star_next = jnp.array(
        [
            [10, 20, 30],
            [40, 45, 50],
        ]
    )
    completion_mask = jnp.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ]
    ).astype(bool)
    continuation_mask = jnp.array(
        [
            [1, 1, 0],
            [1, 0, 1],
        ]
    ).astype(bool)
    discount = 0.9
    target_q = get_target_q(
        rewards,
        q_a_star_next,
        completion_mask,
        discount,
    )
    print("target q")
    print(target_q)
    # print("valid q mask")
    # print(construct_valid_q_mask(4).astype(int))

    # continuation_mask = jnp.array(
    #     [
    #         [1, 0, 1, 1],
    #         [1, 1, 1, 1],
    #         [1, 1, 1, 0],
    #         [0, 1, 1, 1],
    #         [0, 1, 0, 1],
    #     ]
    # )

    # print("valid q mask batch")
    # print(construct_valid_q_mask_batch(continuation_mask).astype(int))

    q = jnp.array(
        [
            [11, 12, 13],
            [42, 48, 54],
        ]
    )
    q_loss = coherent_q_loss(
        q,
        q_a_star_next,
        rewards,
        completion_mask,
        continuation_mask,
        discount,
    )
    print("coherent q loss")
    print(q_loss)