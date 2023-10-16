import contextlib
from abc import ABCMeta, abstractmethod
from typing import Any


class AlgorithmBase(object, metaclass=ABCMeta):
    """Abstract agent class."""

    training = True

    @abstractmethod
    def act(self, *args, **kwargs) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def observe(self, *args, **kwargs) -> None:
        """
        Observe consequences of the last action.(e.g. state, next_state, action, reward, terminated)
        """
        raise NotImplementedError()

    @abstractmethod
    def update_if_dataset_is_ready(self, *args, **kwargs) -> Any:
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
