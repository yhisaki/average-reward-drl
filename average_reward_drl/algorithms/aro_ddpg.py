import copy
from typing import Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import cuda, nn
from torch.optim import Adam

from average_reward_drl.algorithm import AlgorithmBase
from average_reward_drl.logger import Logger
from average_reward_drl.modules import ConcatStateAction, MultiLinear, ScalarHolder
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.utils import polyak_update


class OUNoise(object):
    def __init__(
        self,
        action_dim: int,
        mu=0.0,
        theta=0.15,
        max_sigma=0.2,
        min_sigma=0.2,
        decay_period=100000,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period
        )
        return np.clip(action + ou_state, -1.0, 1.0)


class ARO_DDPG(AlgorithmBase):
    def __init__(
        self,
        dim_state: int,
        dim_action: int,
        critic_hidden_dim: int = 128,
        actor_hidden_dim: int = 128,
        update_interval: int = 10,
        critic_update_freq: int = 1,
        actor_update_freq: int = 2,
        lr: float = 3e-4,
        batch_size: int = 256,
        replay_buffer_capacity: int = 10**6,
        replay_start_size: int = 1000,
        tau: float = 0.005,
        device: Union[str, torch.device] = torch.device(
            "cuda:0" if cuda.is_available() else "cpu"
        ),
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # define dimensions
        self.dim_state = dim_state
        self.dim_action = dim_action

        # define networks
        num_parallel = 2
        # critic
        self.critic = nn.Sequential(
            ConcatStateAction(),
            MultiLinear(num_parallel, dim_state + dim_action, critic_hidden_dim),
            nn.ReLU(),
            MultiLinear(num_parallel, critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            MultiLinear(num_parallel, critic_hidden_dim, 1),
        ).to(device)
        self.critic_target = copy.deepcopy(self.critic).eval().requires_grad_(False)

        self.rho = ScalarHolder(value=0.0).to(device)

        self.actor = nn.Sequential(
            nn.Linear(dim_state, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, dim_action),
            nn.Tanh(),
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor).eval().requires_grad_(False)

        self.ou_noise = OUNoise(dim_action)

        # define optimizers
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr)
        self.rho_optimizer = Adam(self.rho.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.replay_start_size = replay_start_size
        self.batch_size = batch_size

        # replay_start_size should be larger than batch_size
        assert replay_start_size >= batch_size

        self.update_interval = update_interval
        self.critic_update_freq = critic_update_freq
        self.actor_update_freq = actor_update_freq

        self.tau = tau

        self.device = device

        self.logs = Logger()

        self.just_updated = False

    def act(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if self.training:
                if len(self.replay_buffer) < self.replay_start_size:
                    action = np.random.uniform(-1, 1, size=(self.dim_action,))
                else:
                    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                    action = self.actor(state).squeeze(0).cpu().numpy()
                    action = self.ou_noise.get_action(action)

            else:
                state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                action = self.actor(state).squeeze(0).cpu().numpy()

            action = np.clip(action, -1.0, 1.0)

            return action

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

            if update_step % self.actor_update_freq == 0:
                self.update_actor(batch)
                self.update_target_networks()

    def update_critic(self, batch: Batch) -> Any:
        # Compute target Q value
        with torch.no_grad():
            next_actions = self.actor_target(batch.next_state)
            next_q1, next_q2 = self.critic_target((batch.next_state, next_actions))
            next_q = (1 - batch.terminated.to(torch.float32)) * torch.min(
                next_q1, next_q2
            ).flatten()

        # Compute Q loss
        q1, q2 = self.critic((batch.state, batch.action))
        rho = self.rho()

        critic_loss = (
            F.mse_loss(q1.flatten() + rho, batch.reward + next_q)
            + F.mse_loss(q2.flatten() + rho, batch.reward + next_q)
        ) / 2.0

        # Update Q function and rho
        self.critic_optimizer.zero_grad()
        self.rho_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.rho_optimizer.step()

        self.logs.log("critic_loss", float(critic_loss))
        self.logs.log("rho", float(self.rho()))
        self.logs.log("q1_pred_mean", float(q1.mean()))
        self.logs.log("q2_pred_mean", float(q2.mean()))
        self.logs.log("q1_pred_std", float(q1.std()))
        self.logs.log("q2_pred_std", float(q2.std()))

    def update_actor(self, batch: Batch) -> Any:
        actions = self.actor(batch.state)
        q1, q2 = self.critic((batch.state, actions))
        q = torch.min(q1, q2)

        policy_loss = -q.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.logs.log("policy_loss", float(policy_loss))

    def update_target_networks(self) -> Any:
        polyak_update(
            self.critic.parameters(), self.critic_target.parameters(), self.tau
        )
        polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
