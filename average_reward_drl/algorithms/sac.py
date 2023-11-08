import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.distributions import Distribution
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.modules import (
    ConcatStateAction,
    SquashedDiagonalGaussianHead,
    TemperatureHolder,
)
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
        SquashedDiagonalGaussianHead(),
    )


class SoftActorCritic(AlgorithmBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        gamma: float = 0.99,
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
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.gamma = gamma

        # define networks
        self.q1 = get_q_network(dim_state, dim_action).to(device)
        self.q2 = get_q_network(dim_state, dim_action).to(device)

        self.q1_target = copy.deepcopy(self.q1).eval().requires_grad_(False)
        self.q2_target = copy.deepcopy(self.q2).eval().requires_grad_(False)

        self.policy = get_policy_network(dim_state, dim_action).to(device)

        # define optimizers
        self.q_optimizer = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

        # define temperature
        self.temperature = TemperatureHolder().to(device)
        self.temperature_optimizer = Adam(self.temperature.parameters(), lr=lr)

        # define replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=replay_buffer_capacity,
            batch_size=batch_size,
            device=self.device,
        )
        self.replay_start_size = replay_start_size

        self.tau = tau

        self.device = device

    def act(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self.training:
                if len(self.replay_buffer) < self.replay_start_size:
                    action = np.random.uniform(-1, 1, size=(self.dim_action,))
                else:
                    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action_dist: Distribution = self.policy(state)
                    action = action_dist.sample().squeeze(0).cpu().numpy()

            else:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.policy(state).squeeze(0).cpu().numpy()

            action = np.clip(action, -1.0, 1.0)

            return action

    def update_if_dataset_is_ready(self) -> Any:
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) >= self.replay_start_size:
            self.just_updated = True
            samples = self.replay_buffer.sample()
            batch = Batch(**samples, device=self.device)
            self.update_critic(batch)
            self.update_actor_and_temperature(batch)
            self.update_target_networks()

    def update_critic(self, batch: Batch):
        with torch.no_grad():
            next_action_dist: Distribution = self.policy(batch.next_state)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action)
            next_q = torch.min(
                self.q1_target(batch.next_state, next_action),
                self.q2_target(batch.next_state, next_action),
            )
            entropy_term = self.temperature() * next_log_prob
            target_q = batch.reward + self.gamma * (1 - batch.terminated) * (
                next_q - entropy_term
            )

        q1_pred = torch.flatten(self.q1(batch.state, batch.action))
        q2_pred = torch.flatten(self.q2(batch.state, batch.action))

        q_loss = 0.5 * (F.mse_loss(q1_pred, target_q) + F.mse_loss(q2_pred, target_q))

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

    def update_actor_and_temperature(self, batch: Batch):
        action_dist: Distribution = self.policy(batch.state)
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action)
        q = torch.min(self.q1(batch.state, action), self.q2(batch.state, action))

        # L(θ) = E_π[α * log π(a|s) - Q(s, a)]
        policy_loss = torch.mean(self.temperature().detach() * log_prob - q)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # update temperature
        target_entropy = -self.dim_action

        # L(α) = E_π[α * log π(a|s)] + α * H
        temperature_loss = -torch.mean(
            self.temperature() * (log_prob.detach() + target_entropy)
        )

        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()

    def update_target_networks(self):
        polyak_update(self.q1, self.q1_target, self.tau)
        polyak_update(self.q2, self.q2_target, self.tau)
