import contextlib
import copy
from abc import ABCMeta, abstractmethod
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.modules import ConcatStateAction
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.utils import polyak_update


from typing import Callable


def get_reset_cost_network(dim_state: int, dim_action: int):
    return nn.Sequential(
        ConcatStateAction(),
        nn.Linear(dim_state + dim_action, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


class ResetCostBase(object, metaclass=ABCMeta):
    @abstractmethod
    def update(self, batch: Batch) -> None:
        raise NotImplementedError()


class FixedResetCost(ResetCostBase):
    def __init__(self) -> None:
        super().__init__()


class AutoTunedResetCost:
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
    ) -> None:
        self.reset_cost_q1 = get_reset_cost_network(dim_state, dim_action)
        self.reset_cost_q2 = get_reset_cost_network(dim_state, dim_action)
        self.reset_cost_q1_target = (
            copy.deepcopy(self.reset_cost_q1).eval().requires_grad_(False)
        )
        self.reset_cost_q2_target = (
            copy.deepcopy(self.reset_cost_q2).eval().requires_grad_(False)
        )

    def update(
        self, batch: Batch, get_next_action: Callable[[torch.Tensor], torch.Tensor]
    ) -> None:
        with torch.no_grad():
            next_actions = get_next_action(batch.next_state)
            next_reset_cost_q1 = self.reset_cost_q1_target(
                (batch.next_state, next_actions)
            )
            next_reset_cost_q2 = self.reset_cost_q2_target(
                (batch.next_state, next_actions)
            )
