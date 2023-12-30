import contextlib
from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from average_reward_drl.logger import Logger
from average_reward_drl.replay_buffer import ReplayBuffer

from typing import TypeVar, Type, Dict


class AlgorithmBase(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True
    just_updated = False
    replay_buffer: ReplayBuffer
    logs: Logger

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


AlgorithmType = TypeVar("AlgorithmType", bound=AlgorithmBase)

ALGORITHMS: Dict[str, Type[AlgorithmType]] = {}


def register_algorithm(id: str, algorithm: Type[AlgorithmType]) -> None:
    global ALGORITHMS
    if id in ALGORITHMS:
        raise ValueError(f"Cannot register duplicate algorithm ({id})")
    ALGORITHMS[id] = algorithm


def make_algorithm(
    id: str, dim_state: int, dim_action: int, **kwargs: Any
) -> AlgorithmType:
    if id not in ALGORITHMS:
        raise ValueError(f"Cannot find algorithm ({id})")
    return ALGORITHMS[id](
        dim_state=dim_state,
        dim_action=dim_action,
        **kwargs,
    )
