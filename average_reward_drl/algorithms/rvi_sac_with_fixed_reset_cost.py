import torch
import torch.nn.functional as F
from torch import cuda
from torch.distributions import Distribution

from average_reward_drl.algorithms.rvi_sac import RVI_SAC
from average_reward_drl.replay_buffer import Batch
from average_reward_drl.utils import polyak_update


class RVI_SAC_WITH_FIXED_RESET_COST(RVI_SAC):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        reset_cost: float,
        critic_hidden_dim: int = 256,
        actor_hidden_dim: int = 256,
        fq_gain: float = 1.0,
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
            dim_state=dim_state,
            dim_action=dim_action,
            critic_hidden_dim=critic_hidden_dim,
            actor_hidden_dim=actor_hidden_dim,
            fq_gain=fq_gain,
            lr=lr,
            batch_size=batch_size,
            replay_buffer_capacity=replay_buffer_capacity,
            replay_start_size=replay_start_size,
            tau=tau,
            fq_update_tau=fq_update_tau,
            device=device,
        )
        del self.critic_reset, self.critic_reset_target, self.critic_reset_optimizer
        del self.fq_reset, self.fq_reset_gain
        del self.reset_cost, self.reset_cost_optimizer

        self.reset_cost = reset_cost

    def update_critic(self, batch: Batch):
        with torch.no_grad():
            next_action_dist: Distribution = self.actor(batch.next_state)
            next_action = next_action_dist.sample()
            next_log_prob = next_action_dist.log_prob(next_action)

            next_q1, next_q2 = self.critic_target((batch.next_state, next_action))

            next_q = torch.flatten(torch.min(next_q1, next_q2))

            entropy_term = self.temperature() * next_log_prob

            reset = batch.terminated.float()
            reward = batch.reward - self.reset_cost * reset

            target_q = reward - self.fq + (next_q - entropy_term)

            target_fq = torch.mean(next_q - entropy_term) * self.fq_gain

        q1_pred, q2_pred = self.critic((batch.state, batch.action))

        critic_loss = F.mse_loss(q1_pred.flatten(), target_q) + F.mse_loss(
            q2_pred.flatten(), target_q
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.fq = (1 - self.fq_update_tau) * self.fq + self.fq_update_tau * target_fq

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("q1_pred_mean", float(q1_pred.mean()))
        self.logs.log("q1_pred_std", float(q1_pred.std()))
        self.logs.log("q2_pred_mean", float(q2_pred.mean()))
        self.logs.log("q2_pred_std", float(q2_pred.std()))
        self.logs.log("fq", float(self.fq))

    def update_reset_cost(self, _: Batch):
        pass

    def update_target_networks(self):
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
