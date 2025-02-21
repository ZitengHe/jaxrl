#!/bin/bash

log_dir="logs/mujoco/sacv1_attention"

mkdir -p "${log_dir}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export JAX_DISABLE_JIT=0

env_list=(
    "Ant-v3"
    "HalfCheetah-v3"
    "Hopper-v3"
    "Walker2d-v3"
    "Humanoid-v3"
    "Swimmer-v3"
    "HumanoidStandup-v2"
	)

for env in ${env_list[*]}; do

dir_name=$env

nohup python train.py \
  --config=configs/sac_v1_attention.py \
  --env_name=$env \
  --wandb_mode=online \
  > "${log_dir}/${dir_name}.log" 2>&1 &

sleep 2

done