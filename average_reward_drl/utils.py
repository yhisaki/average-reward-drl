import random
from typing import Callable, Iterable, Optional

import gymnasium
import numpy as np
import torch
from dm_control.rl.control import Environment


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
    if seed is None:
        return
    torch.manual_seed(seed if torch_seed is None else torch_seed)
    random.seed(seed if random_seed is None else random_seed)
    np.random.seed(seed if np_seed is None else np_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    gymnasium.Env._np_random, _ = gymnasium.utils.seeding.np_random(seed)


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


ActorFunc = Callable[[np.ndarray], np.ndarray]


def eval_actor_gymnasium(env: gymnasium.Env, actor: ActorFunc, num_episodes: int = 10):
    reward_sums = []
    step_per_episodes = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        terminated, truncated = False, False

        step_per_episode = 0
        reward_sum = 0.0

        while not (terminated or truncated):
            action = actor(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            reward_sum += reward
            step_per_episode += 1

            if terminated or truncated:
                break
            else:
                state = next_state

        reward_sums.append(reward_sum)
        step_per_episodes.append(step_per_episode)

    return np.mean(reward_sums), np.mean(step_per_episodes)


def eval_actor_dm_control(env: Environment, actor: ActorFunc, num_episodes: int = 10):
    reward_sums = []
    step_per_episodes = []

    for _ in range(num_episodes):
        timestep = env.reset()
        terminated = False

        step_per_episode = 0
        reward_sum = 0.0

        while not terminated:
            action = actor(timestep.observation["observations"])
            timestep = env.step(action)
            reward_sum += timestep.reward
            step_per_episode += 1

            if timestep.last():
                break

        reward_sums.append(reward_sum)
        step_per_episodes.append(step_per_episode)

    return np.mean(reward_sums), np.mean(step_per_episodes)
