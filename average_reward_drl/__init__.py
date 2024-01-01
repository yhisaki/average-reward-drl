from average_reward_drl.algorithm import make_algorithm, register_algorithm
from average_reward_drl.algorithms.rvi_sac import RVI_SAC
from average_reward_drl.algorithms.sac import SAC
from average_reward_drl.dmc_wrapper import DMCWrapper
from average_reward_drl.replay_buffer import Batch, ReplayBuffer
from average_reward_drl.train import train
from average_reward_drl.utils import fix_seed

__all__ = [
    "make_algorithm",
    "Batch",
    "ReplayBuffer",
    "train",
    "eval_actor",
    "fix_seed",
    "DMCWrapper",
]

# Register algorithms
register_algorithm("SAC", SAC)
register_algorithm("RVI_SAC", RVI_SAC)
