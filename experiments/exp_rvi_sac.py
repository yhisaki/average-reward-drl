import argparse

import gymnasium
from gymnasium.wrappers.rescale_action import RescaleAction
import wandb

from average_reward_drl import fix_seed, eval_actor
from average_reward_drl.algorithms.rvi_sac import RVI_SAC


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="average_reward_drl")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--env_id", type=str, default="Humanoid-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=10**6)

    args = parser.parse_args()

    wandb.init(project=args.project, group=args.group, tags=[args.env_id, "RVI-SAC"])

    fix_seed(args.seed)

    env = gymnasium.make(args.env_id)
    env = RescaleAction(env, -1.0, 1.0)

    env_eval = gymnasium.make(args.env_id)
    env_eval = RescaleAction(env_eval, -1.0, 1.0)

    agent = RVI_SAC(
        dim_state=env.observation_space.shape[0],
        dim_action=env.action_space.shape[0],
    )

    state, _ = env.reset()
    terminated, truncated = False, False

    step_per_episode = 0
    reward_sum = 0.0

    for step in range(args.total_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        step_per_episode += 1
        agent.observe(state, next_state, action, reward, False, truncated)

        if step % 5000 == 0 and agent.just_updated:
            print(agent.logs)

            print("Evaluating...")
            with agent.eval_mode():
                returns, steps = eval_actor(env_eval, agent.act, num_episodes=10)
            print(f"step: {steps}, return: {returns:.3f}")

        if terminated or truncated:
            state, _ = env.reset()
            wandb.log({"step": step, "eval/mean": reward_sum})
            print(
                f"step: {step}, reward: {reward_sum:.3f}, step_per_episode: {step_per_episode}"
            )
            reward_sum = 0.0
            step_per_episode = 0
            if terminated:
                action = agent.act(next_state)
                agent.observe(next_state, state, action, float("nan"), False, False)

        else:
            state = next_state


if __name__ == "__main__":
    run_experiment()
