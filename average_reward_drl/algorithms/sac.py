import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.distributions import Distribution
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.logger import Logger
from average_reward_drl.modules import (
    ConcatStateAction,
    MultiLinear,
    ScalarHolder,
    SquashedDiagonalGaussianHead,
    ortho_init,
)
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.utils import polyak_update


class SAC(AlgorithmBase):
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

        # define dimensions
        self.dim_state = dim_state
        self.dim_action = dim_action

        # define networks
        hidden_dim = 256
        num_parallel = 2
        init_gain = np.sqrt(1.0 / 3.0)

        self.critic = nn.Sequential(
            ConcatStateAction(),
            ortho_init(
                MultiLinear(num_parallel, dim_state + dim_action, hidden_dim),
                gain=init_gain,
            ),
            nn.ReLU(),
            ortho_init(
                MultiLinear(num_parallel, hidden_dim, hidden_dim), gain=init_gain
            ),
            nn.ReLU(),
            ortho_init(MultiLinear(num_parallel, hidden_dim, 1), gain=init_gain),
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)

        self.actor_gaussian_head = SquashedDiagonalGaussianHead()
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(dim_state, hidden_dim), gain=init_gain),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, hidden_dim), gain=init_gain),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, dim_action * 2), gain=init_gain),
            self.actor_gaussian_head,
        ).to(device)

        # define optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.actor.parameters(), lr=lr)

        # define temperature
        self.temperature = ScalarHolder(value=1.0, transform_fn=torch.exp).to(device)
        self.temperature_optimizer = Adam(self.temperature.parameters(), lr=lr)

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        # define other hyperparameters
        self.gamma = gamma  # discount factor
        self.tau = tau  # target network update rate

        # define device
        self.device = device

        self.logs = Logger()

    @torch.no_grad()
    def act(self, state: np.ndarray) -> np.ndarray:
        if self.training:
            if len(self.replay_buffer) < self.replay_start_size:
                action = np.random.uniform(-1, 1, size=(self.dim_action,))
            else:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action_dist: Distribution = self.actor(state)
                action = action_dist.sample().squeeze(0).cpu().numpy()

        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            self.actor_gaussian_head.deterministic = True
            action = self.actor(state).squeeze(0).cpu().numpy()
            self.actor_gaussian_head.deterministic = False

        action = np.clip(action, -1.0, 1.0)

        return action

    def update_if_dataset_is_ready(self) -> Any:
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) >= self.replay_start_size:
            self.just_updated = True
            samples = self.replay_buffer.sample(n=self.batch_size)
            batch = Batch(**samples, device=self.device)
            self.update_critic(batch)
            self.update_actor(batch)
            self.update_target_networks()

    def update_critic(self, batch: Batch):
        with torch.no_grad():
            next_action_dist: Distribution = self.actor(batch.next_state)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action)
            next_q1, next_q2 = self.critic_target(
                (batch.next_state.repeat(2, 1, 1), next_action.repeat(2, 1, 1))
            )
            next_q = torch.flatten(torch.min(next_q1, next_q2))
            entropy_term = self.temperature() * next_log_prob

            target_q = batch.reward + self.gamma * (1.0 - batch.truncated.float()) * (
                next_q - entropy_term
            )

        q1_pred, q2_pred = self.critic(
            (batch.state.repeat(2, 1, 1), batch.action.repeat(2, 1, 1))
        )

        critic_loss = 0.5 * (
            F.mse_loss(q1_pred.flatten(), target_q)
            + F.mse_loss(q2_pred.flatten(), target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("q1_pred", float(q1_pred.mean()))
        self.logs.log("q2_pred", float(q2_pred.mean()))

    def update_actor(self, batch: Batch):
        action_dist: Distribution = self.actor(batch.state)
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action)
        q1, q2 = self.critic((batch.state.repeat(2, 1, 1), action.repeat(2, 1, 1)))
        q = torch.flatten(torch.min(q1, q2))

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

        self.logs.log("policy_loss", float(policy_loss))
        self.logs.log("temperature", float(self.temperature()))

    def update_target_networks(self):
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
