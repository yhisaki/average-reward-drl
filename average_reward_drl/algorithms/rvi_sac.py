import copy
from typing import Union

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


class RVI_SAC(AlgorithmBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        target_reset_prob: float = 1e-3,
        fq_gain: float = 1e-3,
        lr: float = 3e-4,
        batch_size: int = 256,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 10**4,
        tau: float = 0.005,
        rho_update_tau: float = 1e-2,
        use_reset_scheme: bool = True,
        device: Union[str, torch.device] = torch.device(
            "cuda:0" if cuda.is_available() else "cpu"
        ),
        **kwargs,
    ) -> None:
        super().__init__()

        # define dimensions
        self.dim_state = dim_state
        self.dim_action = dim_action

        # define networks
        hidden_dim = 256
        num_parallel = 2
        # critic
        self.critic = nn.Sequential(
            ConcatStateAction(),
            MultiLinear(num_parallel, dim_state + dim_action, hidden_dim),
            nn.ReLU(),
            MultiLinear(num_parallel, hidden_dim, hidden_dim),
            nn.ReLU(),
            MultiLinear(num_parallel, hidden_dim, 1),
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)

        self.critic_reset = nn.Sequential(
            ConcatStateAction(),
            ortho_init(nn.Linear(dim_state + dim_action, 16)),
            nn.ReLU(),
            ortho_init(nn.Linear(16, 16)),
            nn.ReLU(),
            ortho_init(nn.Linear(16, 1)),
        ).to(device)
        self.critic_reset_target = (
            copy.deepcopy(self.critic_reset).eval().requires_grad_(False)
        )

        # define rho (average reward)
        self.rho = 0.0
        self.rho_reset = 0.0

        # actor
        self.actor_gaussian_head = SquashedDiagonalGaussianHead()
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(dim_state, hidden_dim)),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, dim_action * 2)),
            self.actor_gaussian_head,
        ).to(device)

        self.reset_cost = ScalarHolder(value=0.0, transform_fn=F.softplus).to(device)
        self.reset_cost_optimizer = Adam(self.reset_cost.parameters(), lr=lr)
        self.target_reset_prob = target_reset_prob

        self.use_reset_scheme = use_reset_scheme

        # define optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.critic_reset_optimizer = Adam(self.critic_reset.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.actor.parameters(), lr=lr)

        # define temperature
        self.temperature = ScalarHolder(value=0.0, transform_fn=torch.exp).to(device)
        self.temperature_optimizer = Adam(self.temperature.parameters(), lr=lr)

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        # define other hyperparameters
        self.fq_gain = fq_gain
        self.tau = tau
        self.rho_update_tau = rho_update_tau

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

    def update_if_dataset_is_ready(self):
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

            next_q1, next_q2 = self.critic_target((batch.next_state, next_action))
            next_q_reset = self.critic_reset_target((batch.next_state, next_action))

            next_q = torch.flatten(torch.min(next_q1, next_q2))

            entropy_term = self.temperature() * next_log_prob

            reset = batch.terminated.float()
            reward = batch.reward - float(self.reset_cost()) * reset

            target_q = reward - self.rho + (next_q - entropy_term)
            target_q_reset = reset - self.rho_reset + torch.flatten(next_q_reset)

            target_rho = torch.sum(next_q - entropy_term) * self.fq_gain
            target_rho_reset = torch.sum(next_q_reset) * self.fq_gain

        q1_pred, q2_pred = self.critic((batch.state, batch.action))
        q_reset_pred = self.critic_reset((batch.state, batch.action))

        critic_loss = 0.5 * (
            F.mse_loss(q1_pred.flatten(), target_q)
            + F.mse_loss(q2_pred.flatten(), target_q)
        )

        critic_reset_loss = F.mse_loss(q_reset_pred.flatten(), target_q_reset)

        self.critic_optimizer.zero_grad()
        self.critic_reset_optimizer.zero_grad()
        (critic_loss + critic_reset_loss).backward()
        self.critic_optimizer.step()
        self.critic_reset_optimizer.step()

        self.rho = (
            1 - self.rho_update_tau
        ) * self.rho + self.rho_update_tau * target_rho
        self.rho_reset = (
            1 - self.rho_update_tau
        ) * self.rho_reset + self.rho_update_tau * target_rho_reset

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("critic_reset_loss", float(critic_reset_loss))
        self.logs.log("q1_pred", float(q1_pred.mean()))
        self.logs.log("q2_pred", float(q2_pred.mean()))
        self.logs.log("q_reset_pred", float(q_reset_pred.mean()))
        self.logs.log("rho", float(self.rho))
        self.logs.log("rho_reset", float(self.rho_reset))

    def update_actor(self, batch: Batch):
        action_dist: Distribution = self.actor(batch.state)
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action)
        reset_cost = self.reset_cost()

        q1, q2 = self.critic((batch.state, action))

        q = torch.min(q1, q2)
        # L(θ) = E_π[α * log π(a|s) - Q(s, a)]
        policy_loss = torch.mean(self.temperature().detach() * log_prob - q.flatten())

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

        # update reset cost
        reset_cost_loss = -torch.mean(
            reset_cost * (self.rho_reset - self.target_reset_prob)
        )
        self.reset_cost_optimizer.zero_grad()
        reset_cost_loss.backward()
        self.reset_cost_optimizer.step()

        self.logs.log("policy_loss", float(policy_loss))
        self.logs.log("temperature", float(self.temperature()))
        self.logs.log("reset_cost", float(self.reset_cost()))

    def update_target_networks(self):
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        polyak_update(
            self.critic_reset.parameters(),
            self.critic_reset_target.parameters(),
            self.tau,
        )
