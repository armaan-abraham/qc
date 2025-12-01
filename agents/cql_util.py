import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum, reduce

def get_utils(
    rewards: jnp.ndarray,
    q_next: jnp.ndarray,
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
                q_next,
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
    ineq_weight: float = 1.0,
):
    assert completion_mask.dtype == bool
    assert continuation_mask.dtype == bool
    assert q.shape == rewards.shape == q_a_star_next.shape
    # TODO: check boundaries

    batch_size, seq_len = rewards.shape

    # These are the bellman-derived values for each multi action q for observed
    # actions in terms of observed rewards and q values from the next
    # observation and policy-generated (optimal) action. These have no gradient
    # flow.
    rollouts_q_a_star = get_utils(
        rewards,
        q_a_star_next,
        completion_mask,
        discount,
    )
    print("rollouts_q_a_star")
    print(rollouts_q_a_star)

    diag_i, diag_j = jnp.diag_indices(seq_len)
    target_q_one_step = rollouts_q_a_star[:, diag_i, diag_j]
    loss = jnp.mean((q - target_q_one_step) ** 2)

    if seq_len > 1:
        valid_q_mask_batch = construct_valid_q_mask_batch(continuation_mask)
        print("valid_q_mask_batch")
        print(valid_q_mask_batch)
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
        print("q_expand")
        print(q_expand)
        diffs = rollouts_q_a_star - q_expand
        print("diffs")
        print(diffs)
        diffs = diffs.at[~valid_q_mask_batch].set(-jnp.inf)
        diffs_i, diffs_j = jnp.tril_indices(seq_len)
        print("diffs indices")
        print(diffs_i)
        print(diffs_j)
        diffs = diffs.at[:, diffs_i, diffs_j].set(-jnp.inf)
        print("masked diffs")
        print(diffs)
        diffs_max = reduce(
            diffs,
            "batch seq_obs seq_act -> batch seq_obs",
            "max",
        )
        print("diffs max")
        print(diffs_max)
        valid_diffs_max = diffs_max != -jnp.inf
        print("valid_diffs_max")
        print(valid_diffs_max)
        upward_ineq_loss = jnp.sum(
            jnp.where(
                valid_diffs_max,
                jnp.maximum(diffs_max, 0.0) ** 2,
                0.0,
            )
        ) / jnp.maximum(jnp.sum(valid_diffs_max), 1)

        downward_ineq_loss = 0.0

        if seq_len > 2:

            # Compare every q value to all previous q star values by chaining two
            # inequalities for any given observation: 
            # 1: q(s,a*) > q(s,a)
            # 2: q(s,a1) > q(s,a1,a2)
            # Because q(s1,a1,a2) = r12 + gamma * q(s2,a2), then we also know that
            # q(s1, a1*) > r12 + gamma * q(s2, a2).

            # Compute the utility of starting at observation s and then taking actions
            # a1...an, [batch seq_obs seq_act-1], where we start at seq_obs and then take
            # actions up to the action at the corresponding index plus 1.

            # This is the utility for starting at the indexed observation, taking the
            # actions up to the same index in the action dim, then taking the action
            # after that in the actual rollout, then taking the optimal policy.
            rollouts_q = get_utils(
                rewards[:, 1:-1],
                q[:, 2:],
                completion_mask[:, 1:-1],
                discount,
            )
            print("rollouts_q")
            print(rollouts_q)
            # This is the utility for starting at the indexed observation and following
            # the optimal policy
            q_a_star_next_expand = repeat(
                q_a_star_next[:, :-2],
                "batch seq_obs -> batch seq_obs seq_act",
                seq_act=seq_len-2
            )
            print("q_a_star_next_expand")
            print(q_a_star_next_expand)
            diffs = rollouts_q - q_a_star_next_expand
            print("diffs 2")
            print(diffs)
            # The valid mask invalidates after discontinuations because for computing q
            # a star values, we only need the next observation which is provided at the
            # same index for each observation. However, here we are actually using the
            # next observation by using subsequent indexes in the same observation array
            # because we're using the corresponding observed actions. This necessitates
            # the 2: shift. The continuation mask with :-2 indexing comes from the fact
            # that we have only computed q a star given the next observation array which
            # is standard for the one-step backup. We could in principle instead compute
            # q a star values for the observation array, which would allow us to omit
            # this extra shift, but then we wouldn't be able to do the standard one-step
            # backup for every transition.
            valid_diffs = valid_q_mask_batch[:, 1:-1, 1:-1] & valid_q_mask_batch[:, 1:-1, 2:] & continuation_mask[:, :-2, None]
            print("valid_diffs 2")
            print(valid_diffs)
            diffs = diffs.at[~valid_diffs].set(-jnp.inf)
            print("masked diffs 2")
            print(diffs)
            diffs_max = reduce(
                diffs,
                "batch seq_obs seq_act -> batch seq_act",
                "max",
            )
            print("diffs max 2")
            print(diffs_max)
            valid_diffs_max = diffs_max != -jnp.inf
            print("valid_diffs_max 2")
            print(valid_diffs_max)
            downward_ineq_loss = jnp.sum(
                jnp.where(
                    valid_diffs_max,
                    jnp.maximum(diffs_max, 0.0) ** 2,
                    0,
                )
            ) / jnp.maximum(jnp.sum(valid_diffs_max), 1)

        # Upward inequality loss pushes q values up, downward pushes them down, and
        # one step is the standard bellman loss.
        loss += (upward_ineq_loss + downward_ineq_loss) * ineq_weight

    return loss
    

if __name__ == "__main__":
    # Simple test
    # seq_len = 4
    # attn_mask_seq = construct_attn_mask_seq(seq_len)
    # print(attn_mask_seq)

    rewards = jnp.array(
        [
            [1, 4, 7],
            [5, 6, 8],
        ]
    )
    q_a_star_next = jnp.array(
        [
            [10, 30, 40],
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
            [1, 1, 1],
            [1, 1, 1],
        ]
    ).astype(bool)
    discount = 0.9

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
            [11, 13, 15],
            [42, 45, 48],
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