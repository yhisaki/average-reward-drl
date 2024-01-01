from logging import Logger
from typing import Callable

import colorama
import gymnasium
import numpy as np
from gymnasium.core import Env as GymnasiumEnv

import wandb
from average_reward_drl.algorithm import AlgorithmBase

ActorFunc = Callable[[np.ndarray], np.ndarray]


def evaluate(env: gymnasium.Env, actor: ActorFunc, num_episodes: int = 10):
    reward_sums = []
    step_per_episodes = []
    average_rewards = []

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
        average_rewards.append(reward_sum / step_per_episode)

    return {
        "returns": np.mean(reward_sums),
        "step_per_episode": np.mean(step_per_episodes),
        "average_rewards": np.mean(average_rewards),
    }


def train(
    agent: AlgorithmBase,
    env_train: GymnasiumEnv,
    env_eval: GymnasiumEnv,
    total_steps: int,
    log_interval: int,
    use_reset_scheme: bool,
    logger: Logger,
):
    state, _ = env_train.reset()
    terminated, truncated = False, False

    step_per_episode = 0
    total_returns = 0.0

    for step in range(total_steps):
        action = agent.act(state)

        next_state, reward, terminated, truncated, _ = env_train.step(action)
        total_returns += reward
        step_per_episode += 1
        reset_excuted = False

        if use_reset_scheme and terminated:  # Excute Reset if use_reset_scheme
            next_state, _ = env_train.reset()
            reset_excuted = True

        agent.observe(state, next_state, action, reward, terminated, truncated)

        if not reset_excuted and (terminated or truncated):
            state, _ = env_train.reset()
            reset_excuted = True
        else:
            state = next_state

        if reset_excuted:
            wandb.log({"step": step, "train/returns": total_returns})
            logger.info(
                colorama.Fore.GREEN
                + "Train: "
                + colorama.Style.RESET_ALL
                + f"step: {step}, returns: {total_returns:.3f}, step_per_episode: {step_per_episode}"
            )
            total_returns = 0.0
            step_per_episode = 0

        if step % log_interval == 0 and agent.just_updated:
            logs = agent.logs.flush()
            logger.info(
                colorama.Fore.BLUE + "AgentLog: " + colorama.Style.RESET_ALL + str(logs)
            )
            with agent.eval_mode():
                eval = evaluate(env_eval, agent.act, num_episodes=10)

            logs.update(
                {
                    "step": step,
                    "eval/returns": eval["returns"],
                    "eval/step_per_episode": eval["step_per_episode"],
                    "eval/average_rewards": eval["average_rewards"],
                }
            )
            wandb.log(logs)

            logger.info(
                colorama.Fore.RED
                + "Eval: "
                + colorama.Style.RESET_ALL
                + f"step: {step}, returns: {eval['returns']:.3f}, step_per_episode: {eval['step_per_episode']}"
            )
