#!/bin/sh
#$ -cwd
#$ -j y
#$ -o out/
#$ -l s_gpu=1
#$ -l h_rt=02:00:00
#$ -t 1:10

module load python/3.10.2 cuda/12.1.0 cudnn/8.8.1

ENV_ID=Walker2d-v4
SEED=$SGE_TASK_ID

poetry run python3 experiments/main.py \
  env="$ENV_ID" \
  project=average-reward-drl \
  seed="$SEED"
