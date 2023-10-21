import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.modules import ConcatStateAction
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.utils import polyak_update

import contextlib
from abc import ABCMeta, abstractmethod
from typing import Any


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
        self.reset_network1 = get_reset_cost_network(dim_state, dim_action)

    def update(self, batch: Batch) -> None:
        pass
