import gymnasium

from average_reward_drl import Batch, ReplayBuffer

# Create a replay buffer
replay_buffer = ReplayBuffer(1000)

# Create an environment
env = gymnasium.make("Ant-v4")


def run_episode():
    """
    run an episode and store the transitions in the replay buffer
    """
    state, _ = env.reset()
    terminated, truncated = False, False
    while not terminated and not truncated:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        replay_buffer.append(state, next_state, action, reward, terminated, truncated)
        state = next_state


# Run 10 episodes
for _ in range(10):
    run_episode()

# Sample 5 transitions
samples = replay_buffer.sample(5)

# Create a batch
batch = Batch(**samples)

# Print the batch
print(batch)
