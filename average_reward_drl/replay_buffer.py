import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


def group_by_keys(lst: List[Dict], use_all_keys=True) -> Dict:
    """Convert a list of dictionaries to a dictionary of lists."""
    transposed = defaultdict(list)

    if use_all_keys:
        keys = set().union(*(d.keys() for d in lst))
    else:
        keys = set(lst[0].keys())

    for dct in lst:
        for key in keys:
            transposed[key].append(dct.get(key, None))

    return dict(transposed)


class ReplayBuffer:
    """Experience Replay Buffer

    Args:
        capacity (int): capacity in terms of number of transitions
        num_steps (int): Number of timesteps per stored transition
            (for N-step updates)
    """

    capacity: Optional[int] = None

    def __init__(self, capacity: Optional[int] = None):
        self.capacity = int(capacity)
        self.memory = []

    def append(
        self, state, next_state, action, reward, terminated, truncated, **kwargs
    ):
        transition = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
            **kwargs,
        )

        self.memory.append(transition)

        # Handle the buffer capacity
        if len(self.memory) > self.capacity:
            del self.memory[0]  # Remove the oldest element

    def sample(self, n):
        assert len(self.memory) >= n
        sampled_indices = random.sample(range(len(self.memory)), n)
        return group_by_keys([self.memory[i] for i in sampled_indices])

    def __len__(self):
        return len(self.memory)

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.memory, f)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.memory = pickle.load(f)


def convert_to_tensor(arr, device, dtype):
    """Convert a numpy array or a list to a torch tensor."""
    if isinstance(arr, np.ndarray):
        return torch.tensor(arr, device=device, dtype=dtype)
    elif isinstance(arr, list):
        return torch.tensor(np.array(arr), device=device, dtype=dtype)
    else:
        raise RuntimeError()


class Batch(object):
    """
    Batch of transitions.
    """

    def __init__(
        self,
        state,
        next_state,
        action,
        reward,
        terminated,
        truncated,
        device=None,
        **kwargs,
    ) -> None:
        """
        Args:
            state (torch.Tensor): State tensor.
            next_state (torch.Tensor): Next state tensor.
            action (torch.Tensor): Action tensor.
            reward (torch.Tensor): Reward tensor.
            terminated (torch.Tensor): terminated tensor.
            truncated (torch.Tensor): Truncated tensor.
            device (torch.device, optional): Device to send tensors to.
        """
        super().__init__()
        self.state = convert_to_tensor(state, device, dtype=torch.float32)
        self.next_state = convert_to_tensor(next_state, device, dtype=torch.float32)
        self.action = convert_to_tensor(action, device, dtype=torch.float32)
        self.reward = convert_to_tensor(reward, device, dtype=torch.float32)
        self.terminated = convert_to_tensor(terminated, device, torch.bool)
        self.truncated = convert_to_tensor(truncated, device, torch.bool)

    def __getitem__(self, idx):
        return Batch(
            state=self.state[idx],
            next_state=self.next_state[idx],
            action=self.action[idx],
            reward=self.reward[idx],
            terminated=self.terminated[idx],
            truncated=self.truncated[idx],
        )

    def __setitem__(self, idx, batch: "Batch"):
        self.state = batch.state[idx]
        self.next_state = batch.next_state[idx]
        self.action = batch.action[idx]
        self.reward = batch.reward[idx]
        self.terminated = batch.terminated[idx]
        self.truncated = batch.truncated[idx]

    def to(self, device):
        return Batch(
            state=self.state.to(device),
            next_state=self.next_state.to(device),
            action=self.action.to(device),
            reward=self.reward.to(device),
            terminated=self.terminated.to(device),
            truncated=self.truncated.to(device),
        )

    def __len__(self):
        batch_len = len(self.state)
        assert (
            batch_len
            == len(self.next_state)
            == len(self.action)
            == len(self.reward)
            == len(self.terminated)
            == len(self.truncated)
        )
        return batch_len

    def __str__(self) -> str:
        return (
            f"Batch(state={self.state}, next_state={self.next_state}, "
            f"action={self.action}, reward={self.reward}, "
            f"terminated={self.terminated}, truncated={self.truncated})"
        )
