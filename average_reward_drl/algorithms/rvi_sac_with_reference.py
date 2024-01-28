import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda
from torch.distributions import Distribution

from average_reward_drl.algorithms.rvi_sac import RVI_SAC
from average_reward_drl.replay_buffer import Batch


class RVI_SAC_WITH_REFERENCE(RVI_SAC):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        reference_state: np.ndarray,
        reference_action: np.ndarray,
        critic_hidden_dim: int = 256,
        critic_reset_hidden_dim: int = 64,
        actor_hidden_dim: int = 256,
        target_reset_prob: float = 0.001,
        fq_gain: float = 0.1,
        fq_reset_gain: float = 0.1,
        lr: float = 0.0003,
        batch_size: int = 256,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 10**4,
        tau: float = 0.005,
        fq_update_tau: float = 0.01,
        device: str
        | torch.device = torch.device("cuda:0" if cuda.is_available() else "cpu"),
        **kwargs
    ) -> None:
        super().__init__(
            dim_state,
            dim_action,
            critic_hidden_dim,
            critic_reset_hidden_dim,
            actor_hidden_dim,
            target_reset_prob,
            fq_gain,
            fq_reset_gain,
            lr,
            batch_size,
            replay_buffer_capacity,
            replay_start_size,
            tau,
            fq_update_tau,
            device,
            **kwargs
        )
        # reference
        self.reference_state = (
            torch.tensor(reference_state).float().to(device).unsqueeze(0)
        )
        self.reference_action = (
            torch.tensor(reference_action).float().to(device).unsqueeze(0)
        )

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

            target_q = reward - self.fq + (next_q - entropy_term)
            target_q_reset = reset - self.fq_reset + torch.flatten(next_q_reset)

            next_q1_ref, next_q2_ref = self.critic_target(
                (self.reference_state, self.reference_action)
            )
            next_q_ref = torch.min(next_q1_ref, next_q2_ref) * self.fq_gain
            next_q_ref_reset = (
                self.critic_reset_target((self.reference_state, self.reference_action))
                * self.fq_reset_gain
            )

        q1_pred, q2_pred = self.critic((batch.state, batch.action))
        q_reset_pred = self.critic_reset((batch.state, batch.action))

        critic_loss = F.mse_loss(q1_pred.flatten(), target_q) + F.mse_loss(
            q2_pred.flatten(), target_q
        )

        critic_reset_loss = F.mse_loss(q_reset_pred.flatten(), target_q_reset)

        self.critic_optimizer.zero_grad()
        self.critic_reset_optimizer.zero_grad()
        (critic_loss + critic_reset_loss).backward()
        self.critic_optimizer.step()
        self.critic_reset_optimizer.step()

        self.fq = torch.mean(next_q_ref)
        self.fq_reset = torch.mean(next_q_ref_reset)

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("critic_reset_loss", float(critic_reset_loss))
        self.logs.log("q1_pred_mean", float(q1_pred.mean()))
        self.logs.log("q1_pred_std", float(q1_pred.std()))
        self.logs.log("q2_pred_mean", float(q2_pred.mean()))
        self.logs.log("q2_pred_std", float(q2_pred.std()))
        self.logs.log("q_reset_pred_mean", float(q_reset_pred.mean()))
        self.logs.log("q_reset_pred_std", float(q_reset_pred.std()))
        self.logs.log("fq", float(self.fq))
        self.logs.log("fq_reset", float(self.fq_reset))
