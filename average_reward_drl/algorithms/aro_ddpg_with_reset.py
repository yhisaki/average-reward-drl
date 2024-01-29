import copy
from typing import Any

import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.optim import Adam

from average_reward_drl.algorithms.aro_ddpg import ARO_DDPG
from average_reward_drl.modules import ConcatStateAction, ScalarHolder, ortho_init
from average_reward_drl.replay_buffer import Batch
from average_reward_drl.utils import polyak_update


class ARO_DDPG_WITH_RESET(ARO_DDPG):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        critic_hidden_dim: int = 128,
        critic_reset_hidden_dim: int = 64,
        actor_hidden_dim: int = 128,
        update_interval: int = 10,
        target_reset_prob: float = 1e-3,
        critic_update_freq: int = 1,
        actor_update_freq: int = 2,
        lr: float = 0.0003,
        batch_size: int = 256,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 1000,
        tau: float = 0.005,
        fq_update_tau: float = 1e-2,
        device: str
        | torch.device = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        **kwargs: Any
    ) -> None:
        super().__init__(
            dim_state,
            dim_action,
            critic_hidden_dim,
            actor_hidden_dim,
            update_interval,
            critic_update_freq,
            actor_update_freq,
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
                self.update_reset_cost(batch)

            if update_step % self.actor_update_freq == 0:
                self.update_actor(batch)
                self.update_target_networks()

    def update_critic(self, batch: Batch) -> Any:
        # Compute target Q value
        with torch.no_grad():
            next_actions = self.actor_target(batch.next_state)
            next_q1, next_q2 = self.critic_target((batch.next_state, next_actions))
            next_q_reset = self.critic_reset_target((batch.next_state, next_actions))
            next_q = torch.min(next_q1, next_q2).flatten()

            reset = batch.terminated.float()
            reward = batch.reward - float(self.reset_cost()) * reset

            target_q_reset = reset - self.fq_reset + torch.flatten(next_q_reset)
            target_fq_reset = torch.mean(next_q_reset)

        # Compute Q loss
        q1, q2 = self.critic((batch.state, batch.action))
        q_reset_pred = self.critic_reset((batch.state, batch.action))
        rho = self.rho()
        critic_loss = (
            F.mse_loss(q1.flatten() + rho, reward + next_q)
            + F.mse_loss(q2.flatten() + rho, reward + next_q)
        ) / 2.0

        critic_reset_loss = F.mse_loss(q_reset_pred.flatten(), target_q_reset)

        self.fq_reset = (
            1 - self.fq_update_tau
        ) * self.fq_reset + self.fq_update_tau * target_fq_reset

        # Update Q function and rho
        self.critic_optimizer.zero_grad()
        self.rho_optimizer.zero_grad()
        self.critic_reset_optimizer.zero_grad()
        critic_loss.backward()
        critic_reset_loss.backward()
        self.critic_optimizer.step()
        self.rho_optimizer.step()
        self.critic_reset_optimizer.step()

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("critic_reset_loss", float(critic_reset_loss))
        self.logs.log("rho", float(self.rho()))
        self.logs.log("q1_pred_mean", float(q1.mean()))
        self.logs.log("q2_pred_mean", float(q2.mean()))
        self.logs.log("q1_pred_std", float(q1.std()))
        self.logs.log("q2_pred_std", float(q2.std()))
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

    def update_target_networks(self) -> Any:
        super().update_target_networks()
        polyak_update(
            self.critic_reset.parameters(),
            self.critic_reset_target.parameters(),
            self.tau,
        )
