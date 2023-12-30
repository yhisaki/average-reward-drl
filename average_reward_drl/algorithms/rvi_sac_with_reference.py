import copy
from typing import Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.distributions import Distribution
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.logger import Logger
from average_reward_drl.modules import (ConcatStateAction, MultiLinear,
                                        ScalarHolder,
                                        SquashedDiagonalGaussianHead,
                                        ortho_init)
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.utils import polyak_update


class RVI_SAC_WITH_REFERENCE(AlgorithmBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        reference: Tuple[np.ndarray, np.ndarray],
        target_reset_prob: float = 1e-3,
        lr: float = 3e-4,
        batch_size: int = 250,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 10**4,
        tau: float = 0.005,
        rho_update_tau: float = 1e-2,
        use_reset_scheme: bool = True,
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
        num_parallel = 4
        init_gain = np.sqrt(1.0 / 3.0)

        # critic
        self.critic = nn.Sequential(
            ConcatStateAction(),
            ortho_init(
                MultiLinear(num_parallel, dim_state + dim_action, hidden_dim),
                gain=init_gain,
            ),
            nn.ReLU(),
            ortho_init(
                MultiLinear(num_parallel, hidden_dim, hidden_dim),
                gain=init_gain,
            ),
            nn.ReLU(),
            ortho_init(MultiLinear(num_parallel, hidden_dim, 1), gain=init_gain),
        ).to(device)

        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)

        # actor
        self.actor_gaussian_head = SquashedDiagonalGaussianHead()
        self.actor = nn.Sequential(
            ortho_init(nn.Linear(dim_state, hidden_dim), gain=np.sqrt(1.0 / 3.0)),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, hidden_dim), gain=np.sqrt(1.0 / 3.0)),
            nn.ReLU(),
            ortho_init(nn.Linear(hidden_dim, dim_action * 2), gain=np.sqrt(1.0 / 3.0)),
            self.actor_gaussian_head,
        ).to(device)

        self.reset_cost = ScalarHolder(value=0.0, transform_fn=F.softplus).to(device)
        self.reset_cost_optimizer = Adam(self.reset_cost.parameters(), lr=lr)
        self.target_reset_prob = target_reset_prob

        self.use_reset_scheme = use_reset_scheme

        # define optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.policy_optimizer = Adam(self.actor.parameters(), lr=lr)

        # define temperature
        self.temperature = ScalarHolder(value=0.0, transform_fn=torch.exp).to(device)
        self.temperature_optimizer = Adam(self.temperature.parameters(), lr=lr)

        # define replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.batch_size = batch_size
        self.replay_start_size = replay_start_size

        # define other hyperparameters
        self.reference = reference
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

            next_q1, next_q2, next_q1_reset, next_q2_reset = self.critic_target(
                (batch.next_state.repeat(4, 1, 1), next_action.repeat(4, 1, 1))
            )

            next_q = torch.min(next_q1, next_q2).flatten()
            next_q_reset = torch.min(next_q1_reset, next_q2_reset).flatten()

            reference_state, reference_action = self.reference
            # reference_q1, reference_q2, reference_q1_reset, reference_q2_reset = self.critic_target(

            entropy_term = self.temperature() * next_log_prob

            reset = -batch.terminated.float()

            target_q = batch.reward - self.rho + (next_q - entropy_term)
            target_q_reset = reset - self.rho_reset + (next_q_reset)

        q1_pred, q2_pred, q1_reset_pred, q2_reset_pred = self.critic(
            (batch.state.repeat(4, 1, 1), batch.action.repeat(4, 1, 1))
        )

        q_loss = 0.5 * (
            F.mse_loss(q1_pred.flatten(), target_q)
            + F.mse_loss(q2_pred.flatten(), target_q)
        )
        q_reset_loss = 0.5 * (
            F.mse_loss(q1_reset_pred.flatten(), target_q_reset)
            + F.mse_loss(q2_reset_pred.flatten(), target_q_reset)
        )
        critic_loss = q_loss + q_reset_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("q1_pred", float(q1_pred.mean()))
        self.logs.log("q2_pred", float(q2_pred.mean()))
        self.logs.log("q1_reset_pred", float(q1_reset_pred.mean()))
        self.logs.log("q2_reset_pred", float(q2_reset_pred.mean()))
        self.logs.log("rho", float(self.rho))
        self.logs.log("rho_reset", float(self.rho_reset))

    def update_actor(self, batch: Batch):
        action_dist: Distribution = self.actor(batch.state)
        action = action_dist.rsample()
        log_prob = action_dist.log_prob(action)
        reset_cost = self.reset_cost()

        q1, q2, q1_reset, q2_reset = self.critic(
            (batch.state.repeat(4, 1, 1), action.repeat(4, 1, 1))
        )

        q = torch.min(q1, q2) + self.use_reset_scheme * float(reset_cost) * torch.min(
            q1_reset, q2_reset
        )
        # q = torch.min(q1, q2)
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
            reset_cost * (-self.rho_reset - self.target_reset_prob)
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