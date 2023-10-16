import gymnasium

import wandb
from average_reward_drl import ARO_DDPG

TOTAL_STEPS = 10**6

wandb.init(project="average_reward_drl")


env = gymnasium.make("Swimmer-v4")

agent = ARO_DDPG(
    dim_state=env.observation_space.shape[0],
    dim_action=env.action_space.shape[0],
)

state, _ = env.reset()
terminated, truncated = False, False
reward_sum = 0.0

for step in range(TOTAL_STEPS):
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
