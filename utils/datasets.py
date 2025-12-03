from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

def get_pair_rel_utils(utils_to_terminals, times_to_terminals, discount: float):
    assert utils_to_terminals.shape == times_to_terminals.shape
    assert utils_to_terminals.ndim == 2
    batch_size, seq_len = utils_to_terminals.shape
    rel_times = times_to_terminals[:, :, None] - times_to_terminals[:, None, :]
    assert rel_times.shape == (batch_size, seq_len, seq_len)
    print("Relative times:")
    print(rel_times)
    later_coeffs = discount ** jnp.abs(rel_times)
    print("Later coefficients:")
    print(later_coeffs)
    util_diffs = jnp.where(
        rel_times > 0,
        utils_to_terminals[:, :, None] - later_coeffs * utils_to_terminals[:, None, :],
        0.0,
    )
    valid_mask = rel_times > 0
    print("Utility differences:")
    print(util_diffs)
    print("Valid mask:")
    print(valid_mask.astype(np.int32))
    return util_diffs, valid_mask

class Dataset(FrozenDict):
    """Dataset class."""

    @classmethod
    def create(cls, discount, freeze=True, **fields):
        """Create a dataset from the fields.

        Args:
            discount: Discount factor for computing rewards to go.
            freeze: Whether to freeze the arrays.
            **fields: Keys and values of the dataset.
        """
        data = fields

        # Set terminals = 1 where masks = 0
        data['terminals'] = np.where(data['masks'] == 0, 1.0, data['terminals']).astype(data['terminals'].dtype)

        # Set the final transition to a terminal
        data['terminals'][-1] = 1.0

        print("Terminals after adjustment:")
        print(data['terminals'])


        # Compute discounted sum of rewards to the next terminal. These are NOT
        # valid unless used to compute utils between transitions on a relative
        # basis, because these are relative to terminals, not episode
        # completions.
        rewards = data['rewards']
        utils_to_terminals = np.zeros_like(rewards)
        util_to_terminal = 0.0
        times_to_terminal = np.zeros_like(rewards)
        next_terminal_idx = len(rewards) - 1
        for t_idx in range(len(rewards) - 1, -1, -1):
            if data['terminals'][t_idx] > 0:
                next_terminal_idx = t_idx
            util_to_terminal = rewards[t_idx] + discount * util_to_terminal * (1 - data['terminals'][t_idx])
            utils_to_terminals[t_idx] = util_to_terminal
            times_to_terminal[t_idx] = next_terminal_idx - t_idx
        data['utils_to_terminals'] = utils_to_terminals
        print("Utils to terminals:")
        print(data['utils_to_terminals'])
        data['times_to_terminals'] = times_to_terminal
        print("Times to terminals:")
        print(data['times_to_terminals'])
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        return cls(data)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)

        # Store terminal locations for sampling within episodes
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        print("Terminal locations:")
        print(self.terminal_locs)

        # Store trajectory start locations
        self.start_locs = np.concatenate(([0], self.terminal_locs[:-1] + 1))
        print("Start locations:")
        print(self.start_locs)


    def sample_in_episodes(self, batch_size: int, sequence_length: int):
        """Sample transitions and return a batch of shape [batch_size, sequence_length, ...], 
        where all transitions within a sequence are from the same episode.
        Episodes are chosen proportionally to their length."""

        episode_lens = self.terminal_locs - self.start_locs + 1
        print("Episode lengths:")
        print(episode_lens)
        episode_probs = episode_lens / episode_lens.sum()
        print("Episode probabilities:")
        print(episode_probs)
        sampled_episodes = np.random.choice(len(episode_lens), size=batch_size, p=episode_probs)
        print("Sampled episodes:")
        print(sampled_episodes)

        # Get start locations and lengths for sampled episodes
        starts = self.start_locs[sampled_episodes]
        lens = episode_lens[sampled_episodes]

        # Sample random offsets within each episode: [batch_size, sequence_length]
        offsets = (np.random.random((batch_size, sequence_length)) * lens[:, None]).astype(np.int64)

        # Compute transition indices
        transition_idxs = starts[:, None] + offsets
        print("Transition indices:")
        print(transition_idxs)

        # Index into dataset
        return jax.tree_util.tree_map(lambda arr: arr[transition_idxs], self._dict)


# class ReplayBuffer(Dataset):
#     """Replay buffer class.

#     This class extends Dataset to support adding transitions.
#     """

#     @classmethod
#     def create(cls, transition, size):
#         """Create a replay buffer from the example transition.

#         Args:
#             transition: Example transition (dict).
#             size: Size of the replay buffer.
#         """

#         def create_buffer(example):
#             example = np.array(example)
#             return np.zeros((size, *example.shape), dtype=example.dtype)

#         buffer_dict = jax.tree_util.tree_map(create_buffer, transition)
#         return cls(buffer_dict)

#     @classmethod
#     def create_from_initial_dataset(cls, init_dataset, size):
#         """Create a replay buffer from the initial dataset.

#         Args:
#             init_dataset: Initial dataset.
#             size: Size of the replay buffer.
#         """

#         def create_buffer(init_buffer):
#             buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
#             buffer[: len(init_buffer)] = init_buffer
#             return buffer

#         buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
#         dataset = cls(buffer_dict)
#         dataset.size = get_size(init_dataset)
#         dataset.pointer = dataset.size % size
#         return dataset

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#         self.max_size = get_size(self._dict)
#         self.size = 0
#         self.pointer = 0
#         self.last_transition_terminal = True

#     def add_transition(self, transition):
#         """Add a transition to the replay buffer."""
#         # When adding new transitions, we will be overwriting old data and thus
#         # ruining the episode structure. To deal with this, we set the last
#         # added transition's terminal to 1 always, and if the true terminal was
#         # 0, set it back to 0 once we add the next transition. When sampling, we
#         # assume the episode that the pointer overlaps with is invalid.

#         self.last_transition_terminal = transition['terminals']
#         assert isinstance(self.last_transition_terminal, float) or isinstance(self.last_transition_terminal, int)

#         transition['terminals'] = max(1 - transition['masks'], transition['terminals'])

#         def set_idx(buffer, new_element):
#             if not self.last_transition_terminal:
#                 prev_pointer = (self.pointer - 1) % self.max_size
#                 buffer[prev_pointer]['terminals'] = 0.0
#             self.last_transition_terminal = new_element['terminals']
#             new_element['terminals'] = 1.0
#             buffer[self.pointer] = new_element

#         jax.tree_util.tree_map(set_idx, self._dict, transition)
#         self.pointer = (self.pointer + 1) % self.max_size
#         self.size = max(self.pointer, self.size)

#     def clear(self):
#         """Clear the replay buffer."""
#         self.size = self.pointer = 0

if __name__ == "__main__":
    np.random.seed(3)
    # Simple test
    data = {
        'observations': np.arange(6),
        'next_observations': np.arange(6) + 1,
        'actions': np.arange(6) * 2,
        'rewards': np.arange(6) * 0.5 + 1,
        'masks': np.array(
            [0, 1, 1, 0, 1, 1]
        ),
        'terminals': np.array(
            [1, 0, 0, 1, 0, 0]
        ),
    }
    print("Initial data:")
    for k, v in data.items():
        print(f"{k}:")
        print(v)
    print()

    dataset = Dataset.create(discount=0.9, **data)
    batch = dataset.sample_in_episodes(batch_size=3, sequence_length=4)
    print("Sampled batch:")
    print("Utils to terminals:")
    print(batch['utils_to_terminals'])
    print("Times to terminals:")
    print(batch['times_to_terminals'])

    pair_rel_utils, valid_mask = get_pair_rel_utils(batch["utils_to_terminals"], batch["times_to_terminals"], discount=0.9)