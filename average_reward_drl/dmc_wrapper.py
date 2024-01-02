from typing import Optional

from dm_control import suite
from dm_control.rl.control import Environment as DMCEnv
from gymnasium.core import Env as GymnasiumEnv
from gymnasium.spaces import Box

from average_reward_drl.utils import get_random_state


class DMCWrapper(GymnasiumEnv):
    def __init__(
        self,
        domain_name: str,
        task_name: str,
        max_episode_steps: Optional[int] = None,
    ) -> None:
        if max_episode_steps is not None:
            raise NotImplementedError("max_episode_steps is not implemented!!")

        super().__init__()
        self._env: DMCEnv = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            environment_kwargs={"flat_observation": True},
            task_kwargs={"random": get_random_state()},
        )

        self._max_episode_steps = max_episode_steps

        self.observation_space = Box(
            low=-float("inf"),
            high=float("inf"),
            shape=self._env.observation_spec()["observations"].shape,
            dtype=self._env.observation_spec()["observations"].dtype,
        )

        self.action_space = Box(
            low=self._env.action_spec().minimum,
            high=self._env.action_spec().maximum,
            shape=self._env.action_spec().shape,
            dtype=self._env.action_spec().dtype,
        )

    def step(self, action):
        timestep = self._env.step(action)
        return (
            timestep.observation["observations"],
            timestep.reward,
            timestep.discount == 0.0,
            timestep.last(),
            {},
        )

    def reset(self, **kwargs):
        timestep = self._env.reset()
        return timestep.observation["observations"], {}

    def render(self):
        return self._env.physics.render()


if __name__ == "__main__":
    from average_reward_drl.utils import fix_seed

    fix_seed(0)

    num_episodes = 10

    env = DMCWrapper(domain_name="walker", task_name="run")

    frames = []

    env.action_space.seed(0)

    print(f"observation_space: {env.observation_space}")
    print(f"action_space: {env.action_space}")

    for _ in range(num_episodes):
        state, _ = env.reset()

        terminated, truncated = False, False

        returns = 0.0
        steps = 0

        while not (terminated or truncated):
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
            returns += reward
            steps += 1

            if terminated:
                print("terminated")
                break
            elif truncated:
                print("truncated")
                break
            else:
                state = next_state

        print(f"returns: {returns}, steps: {steps}")
