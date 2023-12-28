import argparse

import gymnasium

import wandb
from average_reward_drl import ARO_DDPG


def run_experiment():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="average-reward-drl")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--env_id", type=str, default="Swimmer-v4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_steps", type=int, default=10**6)

    args = parser.parse_args()

    wandb.init(project=args.project, group=args.group, tags=[args.env_id, "ARO-DDPG"])

    env = gymnasium.make(args.env_id)

    agent = ARO_DDPG(
        dim_state=env.observation_space.shape[0],
        dim_action=env.action_space.shape[0],
    )

    state, _ = env.reset()
    terminated, truncated = False, False
    reward_sum = 0.0

    for step in range(args.total_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        reward_sum += reward
        agent.observe(state, next_state, action, reward, terminated, truncated)

        if terminated or truncated:
            state, _ = env.reset()
            terminated, truncated = False, False
            wandb.log({"step": step, "reward": reward_sum})
            print(f"step: {step}, reward: {reward_sum}")
            reward_sum = 0.0
        else:
            state = next_state


if __name__ == "__main__":
    run_experiment()
