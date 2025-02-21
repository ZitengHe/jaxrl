#!/bin/bash

log_dir="logs/mujoco/sacv1_two_head"

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

dynamics_path="./inv_dyna/${env}/lr_1e-4-dropout_0.1/pretrain_steps_500000/pretrained_inv_dyna_5.pickle"
dir_name=$env

nohup python train.py \
  --config=configs/sac_v1_two_head.py \
  --env_name=$env \
  --wandb_mode=online \
  > "${log_dir}/${dir_name}.log" 2>&1 &

sleep 2

done