#!/bin/sh
#$ -cwd
#$ -j y
#$ -o out/
#$ -l s_gpu=1
#$ -l h_rt=08:00:00
#$ -t 1:10

module load python/3.10.2 cuda/12.1.0 cudnn/8.8.1

ENV_ID=Walker2d-v4

poetry run python3 experiments/main.py \
  env="$ENV_ID" \
  project=average-reward-drl \
  group="$ENV_ID-$JOB_ID" \
  seed="$SGE_TASK_ID"
