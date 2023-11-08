from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.independent import Independent
from torch.distributions.transforms import TanhTransform


class ConcatStateAction(nn.Module):
    def __init__(self):
        super(ConcatStateAction, self).__init__()

    def forward(self, state_action: Tuple[torch.Tensor, torch.Tensor]):
        return torch.cat(state_action, dim=-1)


class ScalarHolder(nn.Module):
    def __init__(self, value: float = 0.0):
        """
        A module that holds a scalar value.

        Args:
            value (float): Initial value of the scalar.
        """

        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self):
        return self.value


class ClippedScalarHolder(nn.Module):
    def __init__(
        self, value: float = 0.0, min_value: float = 0.0, max_value: float = 1.0
    ):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self):
        return torch.clamp(self.value, self.min_value, self.max_value)


class TemperatureHolder(nn.Module):
    def __init__(self, log_temperature: float = 0.0):
        """
        A module that holds a scalar value.

        Args:
            value (float): Initial value of the scalar.
        """

        super().__init__()
        self.log_temperature = nn.Parameter(
            torch.tensor(log_temperature, dtype=torch.float32)
        )

    def forward(self):
        return torch.exp(self.log_temperature)


class SquashedDiagonalGaussianHead(nn.Module):
    def forward(self, x):
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20.0, 2.0)
        std = torch.exp(log_std)
        base_distribution = Independent(Normal(loc=mean, scale=std), 1)
        squashed_distribution = TransformedDistribution(
            base_distribution, TanhTransform()
        )
        return squashed_distribution
