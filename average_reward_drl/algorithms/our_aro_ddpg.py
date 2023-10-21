import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, cuda
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.modules import ConcatStateAction
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.utils import polyak_update


def get_q_network(dim_state: int, dim_action: int):
    return nn.Sequential(
        ConcatStateAction(),
        nn.Linear(dim_state + dim_action, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
    )


def get_policy_network(dim_state: int, dim_action: int):
    return nn.Sequential(
        nn.Linear(dim_state, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, dim_action),
        nn.Tanh(),
    )


class ARO_DDPG(AlgorithmBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        lr: float = 3e-4,
        batch_size: int = 256,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 1000,
        tau: float = 0.005,
        device: Union[str, torch.device] = torch.device(
            "cuda:0" if cuda.is_available() else "cpu"
        ),
    ) -> None:
        super().__init__()

        # define dimensions
        self.dim_state = dim_state
        self.dim_action = dim_action

        # define networks
        self.q1 = get_q_network(dim_state, dim_action).to(device)
        self.q2 = get_q_network(dim_state, dim_action).to(device)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).eval().requires_grad_(False)

        self.policy = get_policy_network(dim_state, dim_action).to(device)
        self.policy_target = copy.deepcopy(self.policy).eval().requires_grad_(False)

        # define optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        self.rho = torch.tensor([0.0], requires_grad=True, device=device)
        self.rho_optimizer = Adam([self.rho], lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size
        # replay_start_size should be larger than batch_size
        assert replay_start_size >= batch_size

        self.tau = tau

        self.device = device

    def observe(
        self,
        state: np.ndarray,
        next_state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        if self.training:
            self.replay_buffer.append(
                state=state,
                next_state=next_state,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            self.update_if_dataset_is_ready()

    def act(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self.training:
                if len(self.replay_buffer) < self.replay_start_size:
                    action = np.random.uniform(-1, 1, size=(self.dim_action,))
                else:
                    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action = self.policy(state).squeeze(
                        0
                    ).cpu().numpy() + np.random.normal(0, 0.1, size=(self.dim_action,))
                    action = np.clip(action, -1, 1)

            else:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.policy(state).squeeze(0).cpu().numpy()
                action = np.clip(action, -1, 1)

            return action

    def update_if_dataset_is_ready(self) -> Any:
        assert self.training

        self.just_updated = False

        if len(self.replay_buffer) < self.replay_start_size:
            return

        if len(self.replay_buffer) % self.update_interval != 0:
            return

        self.just_updated = True

        for update_step in range(self.update_interval):
            samples = self.replay_buffer.sample(self.batch_size)
            batch = Batch(**samples, device=self.device)

            if update_step % self.critic_update_freq == 0:
                self.update_critic(batch)

            if update_step % self.actor_update_freq == 0:
                self.update_actor(batch)
                self.update_target_networks()

    def update_critic(self, batch: Batch) -> Any:
        # Compute target Q value
        with torch.no_grad():
            next_actions = self.policy_target(batch.next_state)
            next_q1 = self.q1_target((batch.next_state, next_actions))
            next_q2 = self.q2_target((batch.next_state, next_actions))
            next_q = (1 - batch.terminated.to(torch.float32)) * torch.min(
                next_q1, next_q2
            ).flatten()

        # Compute Q loss
        q1 = self.q1((batch.state, batch.action)).flatten()
        q2 = self.q2((batch.state, batch.action)).flatten()

        critic_loss = (
            F.mse_loss(q1 + self.rho, batch.reward + next_q)
            + F.mse_loss(q2 + self.rho, batch.reward + next_q)
        ) / 2.0

        # Update Q function and rho
        self.q_optimizer.zero_grad()
        self.rho_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()
        self.rho_optimizer.step()

    def update_actor(self, batch: Batch) -> Any:
        actions = self.policy(batch.state)
        q1 = self.q1((batch.state, actions))
        q2 = self.q2((batch.state, actions))
        q = torch.min(q1, q2)

        actor_loss = -q.mean()

        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()

    def update_target_networks(self) -> Any:
        polyak_update(self.q1.parameters(), self.q1_target.parameters(), self.tau)
        polyak_update(self.q2.parameters(), self.q2_target.parameters(), self.tau)
        polyak_update(
            self.policy.parameters(), self.policy_target.parameters(), self.tau
        )
