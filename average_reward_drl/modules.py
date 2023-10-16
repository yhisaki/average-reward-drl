from numbers import Number
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Distribution, constraints


class ConcatStateAction(nn.Module):
    def __init__(self):
        super(ConcatStateAction, self).__init__()

    def forward(self, state_action: Tuple[torch.Tensor, torch.Tensor]):
        return torch.cat(state_action, dim=-1)


class Delta(Distribution):
    """Delta distribution.

    This is used

    Args:
        loc (float or Tensor): location of the distribution.
    """

    arg_constraints = {"loc": constraints.real}
    # mypy complains about the type of `support` since it is initialized
    # as None in `torch.distributions.Distribution` as of torch==1.5.0.
    support = constraints.real  # type: ignore
    has_rsample = True

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return torch.zeros_like(self.loc)

    @property
    def variance(self):
        return torch.zeros_like(self.loc)

    def __init__(self, loc, validate_args=None):
        self.loc = loc
        if isinstance(loc, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(Delta, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(Delta, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        super(Delta, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape).detach()

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        return self.loc.expand(shape)

    def log_prob(self, value):
        raise RuntimeError("Not defined")

    def entropy(self):
        raise RuntimeError("Not defined")


class DeterministicHead(nn.Module):
    """Head module for a deterministic policy."""

    def __init__(self):
        super().__init__()

    def forward_stochastic(self, loc):
        return torch.distributions.Independent(Delta(loc=loc), 1)

    def forward_determistic(self, loc) -> torch.Tensor:
        return loc
