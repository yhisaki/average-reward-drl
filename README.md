# Average Reward Deep RL

## Installation

### Prerequisites

- Make sure you have `poetry` installed on your system. If you don't have it yet, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation).

### Setting up the Environment

Run the following command to set up the environment using `poetry`.

```bash
poetry install
```

## Implemented Algorithms

- [x] (proposal) RVI-SAC
- [x] [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) (Original Implementation: [here]())
- [x] [ARO-DDPG](https://arxiv.org/abs/2305.12239) (Original Implementation: [here](https://github.com/namansaxena9/ARO-DDPG))
- [ ] [ATRPO](https://arxiv.org/abs/2305.12239)

## Running the Code

### RVI-SAC

- gymnasium

  ```bash
  poetry run python3 experiments/gymnasium/exp_rvi_sac.py env_id=Walker2d-v4
  ```

- dm_control

  ```bash
  poetry run python3 experiments/gymnasium/exp_rvi_sac.py env_id=walker_walk
  ```

### SAC


### ARO-DDPG

```bash
poetry run python3 experiments/aro_ddpg.py
```

## Results