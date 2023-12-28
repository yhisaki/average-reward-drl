import argparse

import colorama
import hydra
from dm_control import suite
from dm_control.rl.control import Environment
from gymnasium.wrappers.rescale_action import RescaleAction
from omegaconf import DictConfig, OmegaConf

import wandb
from average_reward_drl import eval_actor_dm_control, fix_seed
from average_reward_drl.algorithms.rvi_sac import RVI_SAC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="average-reward-drl")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--domain_name", type=str, default="walker")
    parser.add_argument("--task_name", type=str, default="run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=10**6)

    args = parser.parse_args()

    wandb.init(project=args.project, group=args.group)

    fix_seed(args.seed)

    env: Environment = suite.load(
        domain_name=args.domain_name,
        task_name=args.task_name,
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": args.seed},
    )

    env_eval: Environment = suite.load(
        domain_name=args.domain_name,
        task_name=args.task_name,
        environment_kwargs={"flat_observation": True},
        task_kwargs={"random": args.seed},
    )

    dim_state = env.observation_spec()["observations"].shape[0]
    dim_action = env.action_spec().shape[0]

    agent = RVI_SAC(
        dim_state=dim_state,
        dim_action=dim_action,
        use_reset=False,
    )

    timestep = env.reset()

    step_per_episode = 0
    reward_sum = 0.0

    for step in range(args.total_steps):
        action = agent.act(timestep.observation["observations"])
        next_timestep = env.step(action)
        reward_sum += next_timestep.reward
        step_per_episode += 1

        agent.observe(
            timestep.observation["observations"],
            next_timestep.observation["observations"],
            action,
            next_timestep.reward,
            next_timestep.discount == 0.0,
            next_timestep.last(),
        )

        timestep = next_timestep

        if next_timestep.last():
            timestep = env.reset()
            wandb.log({"step": step, "eval/mean": reward_sum})
            print(
                f"step: {step}, reward: {reward_sum:.3f}, step_per_episode: {step_per_episode}"
            )
            reward_sum = 0.0
            step_per_episode = 0

        if step % 5000 == 0 and agent.just_updated:
            print(agent.logs.flush())

            print(colorama.Fore.GREEN + "Evaluating..." + colorama.Style.RESET_ALL)
            with agent.eval_mode():
                returns, steps = eval_actor_dm_control(
                    env_eval, agent.act, num_episodes=10
                )
            print(
                colorama.Fore.RED
                + f"Average return: {returns.mean():.3f}, Average steps: {steps.mean()}"
                + colorama.Style.RESET_ALL
            )


if __name__ == "__main__":
    main()
