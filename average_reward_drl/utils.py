import random
from typing import Iterable, Optional

import gymnasium
import numpy as np
import torch

RANDOM = None


def fix_seed(
    seed: int = 0,
    torch_seed: Optional[int] = None,
    random_seed: Optional[int] = None,
    np_seed: Optional[int] = None,
):
    """Fix the seed of the random number generators.

    Args:
        seed (int): Seed for the random number generators.
        torch_seed (int, optional): Seed for torch.
        random_seed (int, optional): Seed for random.
        np_seed (int, optional): Seed for numpy.
    """
    global RANDOM
    if seed is None:
        return
    torch.manual_seed(seed if torch_seed is None else torch_seed)
    random.seed(seed if random_seed is None else random_seed)
    np.random.seed(seed if np_seed is None else np_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gymnasium.Env._np_random, _ = gymnasium.utils.seeding.np_random(seed)

    RANDOM = np.random.RandomState(seed)


def get_random_state() -> np.random.RandomState:
    """Get the random state of the random number generators.

    Returns:
        np.random.RandomState: Random state of the random number generators.
    """
    return RANDOM


def polyak_update(
    params: Iterable[torch.Tensor],
    target_params: Iterable[torch.Tensor],
    tau: float,
) -> None:
    """Update the parameters of the target network.

    Args:
        params (Iterable[torch.Tensor]): Parameters of the network.
        target_params (Iterable[torch.Tensor]): Parameters of the target network.
        tau (float): Weight for the update.
    """
    with torch.no_grad():
        for param, target_param in zip(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)
