import contextlib
from abc import ABCMeta, abstractmethod
from typing import Any
import numpy as np
from average_reward_drl.replay_buffer import ReplayBuffer


class AlgorithmBase(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True
    replay_buffer: ReplayBuffer

    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def observe(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        if self.training:
            self.replay_buffer.append(
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            self.update_if_dataset_is_ready()

    @abstractmethod
    def update_if_dataset_is_ready(self) -> Any:
        """
        Update the agent.(e.g. policy, q_function, ...)
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def eval_mode(self):
        orig_mode = self.training
        try:
            self.training = False
            yield
        finally:
            self.training = orig_mode
