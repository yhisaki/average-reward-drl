import copy
from typing import Any

import torch
from torch import cuda, nn
from torch.optim import Adam

from average_reward_drl.algorithms.sac import SAC
from average_reward_drl.modules import (
    ConcatStateAction,
    ScalarHolder,
    ortho_init,
)

from average_reward_drl.replay_buffer import Batch
from torch.distributions import Distribution

import torch.nn.functional as F


class SAC_WITH_RESET(SAC):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        critic_hidden_dim: int = 256,
        critic_reset_hidden_dim: int = 64,
        actor_hidden_dim: int = 256,
        target_reset_prob: float = 1e-3,
        gamma: float = 0.99,
        lr: float = 0.0003,
        batch_size: int = 256,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 1000,
        tau: float = 0.005,
        fq_update_tau: float = 1e-2,
        device: str
        | torch.device = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        **kwargs
    ) -> None:
        super().__init__(
            dim_state,
            dim_action,
            critic_hidden_dim,
            actor_hidden_dim,
            gamma,
            lr,
            batch_size,
            replay_buffer_capacity,
            replay_start_size,
            tau,
            device,
            **kwargs
        )

        self.critic_reset = nn.Sequential(
            ConcatStateAction(),
            ortho_init(nn.Linear(dim_state + dim_action, critic_reset_hidden_dim)),
            nn.ReLU(),
            ortho_init(nn.Linear(critic_reset_hidden_dim, critic_reset_hidden_dim)),
            nn.ReLU(),
            ortho_init(nn.Linear(critic_reset_hidden_dim, 1)),
        ).to(device)
        self.critic_reset_target = (
            copy.deepcopy(self.critic_reset).eval().requires_grad_(False)
        )
        self.critic_reset_optimizer = Adam(self.critic_reset.parameters(), lr=lr)
        self.fq_reset = 0.0
        self.fq_update_tau = fq_update_tau

        self.reset_cost = ScalarHolder(value=0.0).to(device)
        self.reset_cost_optimizer = Adam(self.reset_cost.parameters(), lr=lr)
        self.target_reset_prob = target_reset_prob

    def update_if_dataset_is_ready(self) -> Any:
        assert self.training
        self.just_updated = False
        if len(self.replay_buffer) >= self.replay_start_size:
            self.just_updated = True
            samples = self.replay_buffer.sample(n=self.batch_size)
            batch = Batch(**samples, device=self.device)
            self.update_critic(batch)
            self.update_reset_cost(batch)
            self.update_actor(batch)
            self.update_target_networks()

    def update_critic(self, batch: Batch):
        with torch.no_grad():
            next_action_dist: Distribution = self.actor(batch.next_state)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action)
            next_q1, next_q2 = self.critic_target((batch.next_state, next_action))
            next_q = torch.flatten(torch.min(next_q1, next_q2))
            entropy_term = self.temperature() * next_log_prob
            next_q_reset = self.critic_reset_target((batch.next_state, next_action))

            reset = batch.terminated.float()
            reward = batch.reward - float(self.reset_cost()) * reset

            target_q = reward + self.gamma * (next_q - entropy_term)
            target_q_reset = reset - self.fq_reset + torch.flatten(next_q_reset)

            target_fq_reset = torch.mean(next_q_reset)

        q1_pred, q2_pred = self.critic((batch.state, batch.action))
        q_reset_pred = self.critic_reset((batch.state, batch.action))

        critic_loss = F.mse_loss(q1_pred.flatten(), target_q) + F.mse_loss(
            q2_pred.flatten(), target_q
        )
        critic_reset_loss = F.mse_loss(q_reset_pred.flatten(), target_q_reset)

        self.critic_optimizer.zero_grad()
        self.critic_reset_optimizer.zero_grad()
        critic_loss.backward()
        critic_reset_loss.backward()
        self.critic_optimizer.step()
        self.critic_reset_optimizer.step()

        self.fq_reset = (
            1 - self.fq_update_tau
        ) * self.fq_reset + self.fq_update_tau * target_fq_reset

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("q1_pred_mean", float(q1_pred.mean()))
        self.logs.log("q1_pred_std", float(q1_pred.std()))
        self.logs.log("q2_pred_mean", float(q2_pred.mean()))
        self.logs.log("q2_pred_std", float(q2_pred.std()))
        self.logs.log("q_reset_pred_mean", float(q_reset_pred.mean()))
        self.logs.log("q_reset_pred_std", float(q_reset_pred.std()))
        self.logs.log("fq_reset", float(self.fq_reset))

    def update_reset_cost(self, _: Batch):
        # update reset cost
        reset_cost = self.reset_cost()
        reset_cost_loss = -torch.mean(
            reset_cost * (self.fq_reset - self.target_reset_prob)
        )
        self.reset_cost_optimizer.zero_grad()
        reset_cost_loss.backward()
        self.reset_cost_optimizer.step()
        self.reset_cost.value.data = torch.clamp(self.reset_cost.value.data, min=0.0)
        self.logs.log("reset_cost", float(self.reset_cost()))
