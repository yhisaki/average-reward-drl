import logging

import gymnasium
import hydra
from gymnasium.wrappers.rescale_action import RescaleAction
from omegaconf import DictConfig, OmegaConf

import wandb
from average_reward_drl import DMCWrapper, fix_seed, make_algorithm, train

# from pprint import pprint


log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    conf = OmegaConf.to_container(cfg, resolve=True)
    # pprint(conf)

    # Fix seed
    fix_seed(conf["seed"])

    # Create environments
    if conf["env"]["type"] == "gymnasium":
        env_id = conf["env"]["id"]
        env_train = gymnasium.make(env_id, **conf["env"]["train"])
        env_eval = gymnasium.make(env_id, **conf["env"]["eval"])
    elif conf["env"]["type"] == "dm_control":
        domain_name = conf["env"]["domain_name"]
        task_name = conf["env"]["task_name"]
        env_train = DMCWrapper(domain_name, task_name)
        env_eval = DMCWrapper(domain_name, task_name)

    env_train = RescaleAction(env_train, -1.0, 1.0)
    env_eval = RescaleAction(env_eval, -1.0, 1.0)

    dim_state = env_train.observation_space.shape[0]
    dim_action = env_train.action_space.shape[0]

    agent = make_algorithm(
        conf["algo"]["id"],
        dim_state=dim_state,
        dim_action=dim_action,
        **conf["algo"]["params"],
    )

    wandb.init(
        project=conf["project"],
        group=conf["group"],
        tags=conf["tags"],
        config=conf,
    )

    log.info(f"Environment: {conf['env']['name']}")
    log.info(f"Algorithm: {conf['algo']['id']}")

    train(
        agent=agent,
        env_train=env_train,
        env_eval=env_eval,
        total_steps=conf["total_steps"],
        log_interval=conf["log_interval"],
        use_reset_scheme=conf["use_reset_scheme"],
        logger=log,
    )


if __name__ == "__main__":
    main()
