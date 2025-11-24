import jax.numpy as jnp
from einops import repeat, rearrange

def construct_attn_mask_seq(seq_len: int) -> jnp.ndarray:
    # Construct attention mask. For a given observation, all actions up
    # to the observation's time step are masked out to all querying
    # actions, and each action can attend to previous actions and the
    # observation.
    attn_mask_seq = repeat(
        # Actions cannot attend to themselves, which is fine from a
        # nan perspective because there will always be an
        # observation to attend to. This also makes the
        # single-action case simpler.
        jnp.tril(jnp.ones((seq_len, seq_len)), k=-1),
        "seq_q seq_k -> seq_obs seq_q seq_k",
        seq_obs=seq_len,
    )

    # Mask out actions preceding the observation
    attn_mask_valid = jnp.arange(seq_len)[:, None] <= jnp.arange(seq_len)[None, :]
    attn_mask_valid = repeat(
        attn_mask_valid,
        "seq_obs seq_k -> seq_obs seq_q seq_k",
        seq_q=seq_len,
    )
    attn_mask_seq = attn_mask_seq * attn_mask_valid

    # Every action can attend to the observation
    attn_mask_seq = jnp.concatenate(
        [jnp.ones((seq_len, seq_len, 1)), attn_mask_seq],
        axis=2,
    )
    assert attn_mask_seq.shape == (seq_len, seq_len, seq_len + 1)
    return attn_mask_seq

if __name__ == "__main__":
    # Simple test
    seq_len = 4
    attn_mask_seq = construct_attn_mask_seq(seq_len)
    print(attn_mask_seq)