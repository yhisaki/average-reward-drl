from typing import Tuple

import numpy as np
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
    def __init__(
        self,
        value: float = 0.0,
        transform_fn=lambda x: x,
    ):
        """
        A module that holds a scalar value.

        Args:
            value (float): Initial value of the scalar.
        """

        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.transform_fn = transform_fn

    def forward(self):
        return self.transform_fn(self.value)


def ortho_init(layer: nn.Linear, gain: float = np.sqrt(1.0 / 3.0)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class MultiLinear(nn.Module):
    __constants__ = ["in_features", "out_features", "num_parallel"]

    in_features: int
    out_features: int
    num_parallel: int

    def __init__(
        self,
        num_parallel: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_parallel = num_parallel
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty((num_parallel, in_features, out_features), **factory_kwargs),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((num_parallel, 1, out_features), **factory_kwargs),
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for i in range(self.num_parallel):
            nn.init.orthogonal_(self.weight[i], gain=np.sqrt(1.0 / 3.0))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = input.matmul(self.weight)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, num_parallel={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.num_parallel,
            self.bias is not None,
        )


class SquashedDiagonalGaussianHead(nn.Module):
    deterministic: bool

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deterministic = False

    def forward(self, x):
        mean, log_scale = torch.chunk(x, 2, dim=x.dim() // 2)
        if self.deterministic:
            return torch.tanh(mean)
        log_scale = torch.clamp(log_scale, -20.0, 2.0)
        var = torch.exp(log_scale * 2)
        base_distribution = Independent(Normal(loc=mean, scale=torch.sqrt(var)), 1)
        squashed_distribution = TransformedDistribution(
            base_distribution, TanhTransform(cache_size=1)
        )
        return squashed_distribution
