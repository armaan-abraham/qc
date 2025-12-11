from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from typing import Sequence


def get_size(data):
    """Return the size of the dataset."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))

def get_pair_rel_utils(utils_to_terminals, times_to_terminals, discount: float):
    # Valid for i, j where i occurs before j in the trajectory
    assert utils_to_terminals.shape == times_to_terminals.shape
    assert utils_to_terminals.ndim == 2
    batch_size, seq_len = utils_to_terminals.shape
    rel_times = times_to_terminals[:, :, None] - times_to_terminals[:, None, :]
    assert rel_times.shape == (batch_size, seq_len, seq_len)
    later_coeffs = discount ** jnp.abs(rel_times)
    util_diffs = jnp.where(
        rel_times > 0,
        utils_to_terminals[:, :, None] - later_coeffs * utils_to_terminals[:, None, :],
        0.0,
    )
    valid_mask = rel_times > 0
    return util_diffs, rel_times, valid_mask

def get_utils_and_times_to_terminals(rewards, terminals, discount: float):
    # Compute discounted sum of rewards to the next terminal. These are NOT
    # valid unless used to compute utils between transitions on a relative
    # basis, because these are relative to terminals, not episode
    # completions.
    utils_to_terminals = np.zeros_like(rewards, dtype=np.float32)
    util_to_terminal = 0.0
    times_to_terminals = np.zeros_like(rewards, dtype=np.int32)
    next_terminal_idx = len(rewards) - 1
    for t_idx in range(len(rewards) - 1, -1, -1):
        if terminals[t_idx] > 0:
            next_terminal_idx = t_idx
        util_to_terminal = rewards[t_idx] + discount * util_to_terminal * (1 - terminals[t_idx])
        utils_to_terminals[t_idx] = util_to_terminal
        times_to_terminals[t_idx] = next_terminal_idx - t_idx
    return utils_to_terminals, times_to_terminals


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
        if 'utils_to_terminals' in data:
            del data['utils_to_terminals']
        if 'times_to_terminals' in data:
            del data['times_to_terminals']

        # Set terminals = 1 where masks = 0
        data['terminals'] = np.where(data['masks'] == 0, 1.0, data['terminals']).astype(data['terminals'].dtype)

        # Set the final transition to a terminal
        data['terminals'][-1] = 1.0

        utils_to_terminals, times_to_terminals = get_utils_and_times_to_terminals(data['rewards'], data['terminals'], discount)

        data['utils_to_terminals'] = utils_to_terminals
        data['times_to_terminals'] = times_to_terminals
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), data)
        instance = cls(data)
        instance.discount = discount
        return instance
    
    def init_term_locs(self):
        # Store terminal locations for sampling within episodes
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        print("Num terminals in dataset:", len(self.terminal_locs))

        # Store trajectory start locations
        self.start_locs = np.concatenate(([0], self.terminal_locs[:-1] + 1))
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.size = get_size(self._dict)
        self.init_term_locs()



    def sample_in_trajectories(self, batch_size: int, sequence_length: int, discount: float, sample_method="contiguous"):
        """Sample transitions and return a batch of shape [batch_size, sequence_length, ...], 
        where all transitions within a sequence are from the same episode.
        Episodes are chosen proportionally to their length."""

        episode_lens = self.terminal_locs - self.start_locs + 1
        episode_probs = episode_lens / episode_lens.sum()
        sampled_episodes = np.random.choice(len(episode_lens), size=batch_size, p=episode_probs)

        # Get start locations and lengths for sampled episodes
        starts = self.start_locs[sampled_episodes]
        lens = episode_lens[sampled_episodes]


        if sample_method == "global_uniform":
            # Sample random offsets within each episode: [batch_size, seq_len]
            idxs = (np.random.random((batch_size, sequence_length)) * lens[:, None]).astype(np.int64)
        elif sample_method == "contiguous":

            # Sample random starting offsets within each episode: [batch_size]
            start_idxs = (np.random.random(batch_size) * lens).astype(np.int64)

            offsets = np.concatenate(
                [
                    np.zeros((batch_size, 1), dtype=np.int64),
                    np.ones((batch_size, sequence_length - 1), dtype=np.int64),
                ],
                axis=1
            )

            idxs = start_idxs[:, None] + np.cumsum(offsets, axis=1)

            # Wrap around indices that exceed episode length
            idxs = idxs % lens[:, None]
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")

        assert np.all(idxs < lens[:, None])

        # Compute transition indices
        global_idxs = starts[:, None] + idxs

        # Index into dataset
        return jax.tree_util.tree_map(lambda arr: arr[global_idxs], self._dict)


class ReplayBuffer(Dataset):
    """Replay buffer class.

    This class extends Dataset to support adding episodes of transitions.
    """

    @classmethod
    def create_from_initial_dataset(cls, init_dataset, size):
        """Create a replay buffer from the initial dataset.

        Args:
            init_dataset: Initial dataset.
            size: Size of the replay buffer.
        """

        def create_buffer(init_buffer):
            buffer = np.zeros((size, *init_buffer.shape[1:]), dtype=init_buffer.dtype)
            buffer[: len(init_buffer)] = init_buffer
            return buffer

        buffer_dict = jax.tree_util.tree_map(create_buffer, init_dataset)
        buffer = cls(buffer_dict)
        buffer.size = get_size(init_dataset)
        buffer.pointer = buffer.size % size
        return buffer

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.max_size = get_size(self._dict)
        self.size = 0
        self.pointer = 0

    def add_trajectory(self, trajectory: Sequence[dict], discount: float):
        """Add a trajectory of transitions to the replay buffer."""
        num_transitions = len(trajectory)

        # Convert trajectory to dict of arrays
        trajectory = jax.tree_util.tree_map(lambda *xs: np.stack(xs), *trajectory)
        assert isinstance(trajectory, dict)

        terminal_locs = np.nonzero(trajectory['terminals'] > 0)[0]
        assert terminal_locs.size == 1 and terminal_locs[0] == num_transitions - 1
        assert np.all(trajectory['terminals'][trajectory['masks'] == 0] == 1)

        # We populate the times and utils to terminals based on the entire
        # trajectory, despite the possibility of dividing the trajectory when
        # looping around the buffer. This is fine because these values are only
        # used on a relative basis.
        utils_to_terminals, times_to_terminals = get_utils_and_times_to_terminals(trajectory['rewards'], trajectory['terminals'], discount)
        trajectory['utils_to_terminals'] = utils_to_terminals
        trajectory['times_to_terminals'] = times_to_terminals

        for t_idx in range(num_transitions):
            transition = jax.tree_util.tree_map(lambda arr: arr[t_idx], trajectory)

            def set_idx(buffer, new_element):
                buffer[self.pointer] = new_element

            jax.tree_util.tree_map(set_idx, self._dict, transition)

            # If looping around, we set this terminal to 1
            if self.pointer == self.max_size - 1:
                self._dict['terminals'][self.pointer] = 1.0

            self.pointer = (self.pointer + 1) % self.max_size
            self.size = max(self.pointer, self.size)

        self.init_term_locs()

    def clear(self):
        """Clear the replay buffer."""
        self.size = self.pointer = 0

if __name__ == "__main__":
    np.random.seed(3)
    discount = 0.99
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

    dataset = Dataset.create(discount=discount, **data)
    batch = dataset.sample_in_trajectories(batch_size=3, sequence_length=4)
    print("Sampled batch:")
    print("Utils to terminals:")
    print(batch['utils_to_terminals'])
    print("Times to terminals:")
    print(batch['times_to_terminals'])

    pair_rel_utils, pair_rel_times, valid_mask = get_pair_rel_utils(batch["utils_to_terminals"], batch["times_to_terminals"], discount=0.9)
    print("Pairwise relative utils:")
    print(pair_rel_utils)
    print("Pairwise relative times:")
    print(pair_rel_times)
    print("Valid mask:")
    print(valid_mask)


    buffer = ReplayBuffer.create_from_initial_dataset(dict(dataset), size=7)
    # print("Initial buffer:")
    # for k, v in buffer._dict.items():
    #     print(f"{k}:")
    #     print(v)
    
    num_transitions = 3
    trajectory = [
        {
            'observations': np.array(i + 6),
            'next_observations': np.array(i + 7),
            'actions': np.array((i + 6) * 2),
            'rewards': (i + 6) * 0.5 + 1,
            'masks': 1 if i < num_transitions else 0,
            'terminals': 1 if i == num_transitions - 1 else 0,
        }
        for i in range(num_transitions)
    ]
    # print("Adding trajectory:")
    # print(trajectory)
    buffer.add_trajectory(trajectory, discount=discount)

    # print("Buffer after adding trajectory:")
    # for k, v in buffer._dict.items():
    #     print(f"{k}:")
    #     print(v)

