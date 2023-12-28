import argparse

import colorama
import gymnasium
import hydra
from gymnasium.wrappers.rescale_action import RescaleAction
from omegaconf import DictConfig, OmegaConf

import wandb
from average_reward_drl import eval_actor_gymnasium, fix_seed
from average_reward_drl.algorithms.rvi_sac import RVI_SAC


@hydra.main(version_base=None, config_name="rvi_sac")
def main(cfg: DictConfig) -> None:
    conf = OmegaConf.to_container(cfg, resolve=True)

    wandb.init(project=conf["project"], group=conf["group"], tags=[conf["env_id"]])

    # fix_seed(args.seed)

    # env = gymnasium.make(args.env_id)
    # env = RescaleAction(env, -1.0, 1.0)

    # env_eval = gymnasium.make(args.env_id, max_episode_steps=10000)
    # env_eval = RescaleAction(env_eval, -1.0, 1.0)

    # agent = RVI_SAC(
    #     dim_state=env.observation_space.shape[0],
    #     dim_action=env.action_space.shape[0],
    # )

    # state, _ = env.reset()
    # terminated, truncated = False, False

    # step_per_episode = 0
    # reward_sum = 0.0

    # for step in range(args.total_steps):
    #     action = agent.act(state)
    #     next_state, reward, terminated, truncated, _ = env.step(action)
    #     reward_sum += reward
    #     step_per_episode += 1

    #     if terminated:
    #         next_state, _ = env.reset()

    #     agent.observe(state, next_state, action, reward, terminated, truncated)

    #     if truncated:
    #         state, _ = env.reset()

    #     else:
    #         state = next_state

    #     if terminated or truncated:
    #         wandb.log({"step": step, "eval/mean": reward_sum})
    #         print(
    #             f"step: {step}, reward: {reward_sum:.3f}, step_per_episode: {step_per_episode}"
    #         )
    #         reward_sum = 0.0
    #         step_per_episode = 0

    #     if step % 5000 == 0 and agent.just_updated:
    #         print(agent.logs)

    #         print(colorama.Fore.GREEN + "Evaluating..." + colorama.Style.RESET_ALL)
    #         with agent.eval_mode():
    #             returns, steps = eval_actor(env_eval, agent.act, num_episodes=10)
    #         print(
    #             colorama.Fore.RED
    #             + f"Average return: {returns.mean():.3f}, Average steps: {steps.mean()}"
    #             + colorama.Style.RESET_ALL
    #         )


if __name__ == "__main__":
    main()
