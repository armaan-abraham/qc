import jax
import jax.numpy as jnp
from einops import repeat, rearrange, einsum

def get_utils(
    rewards: jnp.ndarray,
    q_next_expand: jnp.ndarray,
    completion_mask: jnp.ndarray,
    discount: float,
):

    batch_size, seq_len_trans = rewards.shape
    seq_len_net = q_next_expand.shape[1]
    assert (seq_len_trans - 1) % (seq_len_net - 1) == 0
    network_transition_interval = (seq_len_trans - 1) // (seq_len_net - 1)

    # Expand q_next to have same dimension as rewards, then index into final
    # result to only include utils that end with a valid q_next
    q_next_expand = jnp.zeros((batch_size, seq_len_net, network_transition_interval), dtype=q_next_expand.dtype)
    q_next_expand = q_next_expand.at[:, :, 0].set(q_next_expand)
    q_next_expand = rearrange(
        q_next_expand,
        "batch seq_net interval -> batch (seq_net interval)",
    )[:, :seq_len_trans]
    assert q_next_expand.shape == (batch_size, seq_len_trans)

    rewards_by_obs = jnp.concatenate(
        [
            rewards,
            jnp.zeros((batch_size, seq_len_trans)),
        ],
        axis=1,
    )
    range_idxs = jnp.arange(seq_len_trans)[:, None] + jnp.arange(seq_len_trans)[None, :]
    # Index into rewards by obs to get sequences of rewards to sum over for each observation
    rewards_by_obs = rewards_by_obs[:, range_idxs]
    assert rewards_by_obs.shape == (batch_size, seq_len_trans, seq_len_trans)

    # Multiply each observation-wise reward sequence by discounting coefficients
    discounts = discount ** jnp.arange(seq_len_trans)

    rewards_by_obs = rewards_by_obs * discounts

    # Discounted reward targets are just cumulative sums
    target_rewards = jnp.cumsum(rewards_by_obs, axis=-1)

    # Do zero padding trick and discounting for target q values
    q_next_expand_pad = jnp.concatenate(
        [
            # Q is 0 
            jnp.where(
                completion_mask,
                0.0,
                q_next_expand,
            ),
            jnp.zeros((batch_size, seq_len_trans)),
        ],
        axis=-1
    )
    range_idxs = jnp.arange(seq_len_trans)[:, None] + jnp.arange(seq_len_trans)[None, :]
    target_q_next = q_next_expand_pad[:, range_idxs]
    discounts = discount ** (jnp.arange(seq_len_trans) + 1)
    target_q_next = target_q_next * discounts

    target_q = target_rewards + target_q_next

    # Shift over predictions so that the last dimension corresponds to the last
    # action being predicted
    src_i, src_j = jnp.triu_indices(seq_len_trans)
    src_j = seq_len_trans - 1 - src_j
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
    assert q.shape == q_a_star_next.shape
    assert rewards.shape == completion_mask.shape == continuation_mask.shape
    assert q.shape[0] == rewards.shape[0]

    batch_size, seq_len_trans = rewards.shape
    seq_len_net = q.shape[1]
    assert (seq_len_trans - 1) % (seq_len_net - 1) == 0
    network_transition_interval = (seq_len_trans - 1) // (seq_len_net - 1)
    

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

    rollouts_q_a_star_net = rollouts_q_a_star[:, ::network_transition_interval, ::network_transition_interval]

    diag_i, diag_j = jnp.diag_indices(seq_len_net)
    target_q_one_step = rollouts_q_a_star_net[:, diag_i, diag_j]
    loss = jnp.mean((q - target_q_one_step) ** 2)
    

    if seq_len_trans > 1:
        valid_q_mask_batch = construct_valid_q_mask_batch(continuation_mask)
        valid_q_mask_batch_net = valid_q_mask_batch[:, ::network_transition_interval, ::network_transition_interval]
        assert valid_q_mask_batch.dtype == bool

        # For any given observation, Q functions with fewer parametrized actions
        # should be greater. This is because taking the optimal policy earlier will
        # result in a higher (or equal) utility than taking any actual sequence of
        # actions.
        q_expand = repeat(
            q,
            "batch seq_obs -> batch seq_obs seq_act",
            seq_act=seq_len_net,
        )
        diffs = rollouts_q_a_star_net - q_expand
        diffs_i, diffs_j = jnp.triu_indices(seq_len_net, k=1)
        diffs = diffs[:, diffs_i, diffs_j]
        valid_diffs = valid_q_mask_batch_net[:, diffs_i, diffs_j]
        upward_ineq_loss = jnp.sum(
            jnp.where(
                valid_diffs,
                jnp.maximum(diffs, 0.0) ** 2,
                0.0,
            )
        ) / jnp.maximum(jnp.sum(valid_diffs), 1)

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
        # actions up to the same index in the action dim, and then taking the action
        # after that in the actual rollout.
        rollouts_q = get_utils(
            rewards[:, 1:-1],
            q[:, 2:],
            completion_mask[:, 1:-1],
            discount,
        )
        # Insert q along the diagonal and push everything else to the right
        rollouts_q = jnp.pad(rollouts_q, ((0,0),(0,1),(1,0)))
        diag_i, diag_j = jnp.diag_indices(seq_len-1)
        rollouts_q = rollouts_q.at[:, diag_i, diag_j].set(q[:, 1:])
        # Element i, j for i >= j in rollouts_q corresponds to the observed
        # utility taking actions i to j-1 and then following the optimal policy
        # afterwards (estimated by q).

        # This is the utility for starting at the indexed observation and following
        # the optimal policy
        q_a_star_next_expand = repeat(
            q_a_star_next[:, :-1],
            "batch seq_obs -> batch seq_obs seq_act",
            seq_act=seq_len-1
        )
        diffs = rollouts_q - q_a_star_next_expand
        valid_diffs = valid_q_mask_batch[:, 1:, 1:] & continuation_mask[:, :-1, None]
        downward_ineq_loss = jnp.sum(
            jnp.where(
                valid_diffs,
                jnp.maximum(diffs, 0.0) ** 2,
                0,
            )
        ) / jnp.maximum(jnp.sum(valid_diffs), 1)

        # Upward inequality loss pushes q values up, downward pushes them down, and
        # one step is the standard bellman loss.
        loss += upward_ineq_loss + downward_ineq_loss

    return loss
    

if __name__ == "__main__":

    discount = 0.9

    # rewards = jnp.array(
    #     [
    #         [1, 2, 3, 4],
    #         [5, 7, 9, 11],
    #     ]
    # )
    # q_a_star_next = jnp.array(
    #     [
    #         [10, 11, 12, 13],
    #         [40, 42, 44, 46],
    #     ]
    # )
    # completion_mask = jnp.array(
    #     [
    #         [0, 0, 0, 0],
    #         [0, 1, 0, 1],
    #     ]
    # ).astype(bool)
    # continuation_mask = jnp.array(
    #     [
    #         [0, 1, 0, 1],
    #         [1, 1, 1, 0],
    #     ]
    # ).astype(bool)
    # q = jnp.array(
    #     [
    #         [11, 13, 15, 17],
    #         [42, 44, 46, 48],
    #     ]
    # )

    batch_size = 8
    seq_len = 8
    key = jax.random.PRNGKey(0)
    rewards = jax.random.uniform(key, (batch_size, seq_len)) * 10
    key, subkey = jax.random.split(key)
    q_a_star_next = jax.random.uniform(subkey, (batch_size, seq_len)) * 50
    key, subkey = jax.random.split(key)
    q = jax.random.uniform(subkey, (batch_size, seq_len)) * 50
    key, subkey = jax.random.split(key)
    completion_mask = jax.random.bernoulli(subkey, p=0.2, shape=(batch_size, seq_len))
    key, subkey = jax.random.split(key)
    continuation_mask = jax.random.bernoulli(subkey, p=0.7, shape=(batch_size, seq_len))

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