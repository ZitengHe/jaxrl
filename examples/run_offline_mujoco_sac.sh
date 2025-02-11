#!/bin/bash

log_dir="logs/mujoco/sacv1"

mkdir -p "${log_dir}"

mode="sacv1"
# antmaze
actor_lr=3e-4
vf_lr=3e-4
qf_lr=3e-4
discount=0.99
expectile=0.9
temperature_a=10.0
temperature_s=10.0
dropout_rate=-1
value_layer_norm="False"
value_dropout_rate=-1
sample_strategy="por"  # por; iql; maxQ
inv_penalty=10.0
policy_type="gaussian"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1
export JAX_DISABLE_JIT=0

env_list=(
# #	"antmaze-umaze-v2"
# #	"antmaze-umaze-diverse-v2"
# 	"antmaze-medium-play-v2"
# 	"antmaze-medium-diverse-v2"
# 	"antmaze-large-play-v2"
# 	"antmaze-large-diverse-v2"
    "HalfCheetah-v1"
    # "halfcheetah-expert-v2"
    # "halfcheetah-medium-replay-v2"
    # "halfcheetah-medium-expert-v2"
    # "halfcheetah-medium-v2"
    # # "halfcheetah-random-v2"
    # # "hopper-expert-v2"
    # "hopper-medium-replay-v2"
    # "hopper-medium-expert-v2"
    # "hopper-medium-v2"
    # # # "hopper-random-v2"
    # # # "walker2d-expert-v2"
    # "walker2d-medium-replay-v2"
    # "walker2d-medium-expert-v2"
    # "walker2d-medium-v2"
    # # "walker2d-random-v2"
	)

for env in ${env_list[*]}; do

dynamics_path="./inv_dyna/${env}/lr_1e-4-dropout_0.1/pretrain_steps_500000/pretrained_inv_dyna_5.pickle"
dir_name=$env

if [ -n "$temperature_a" ]; then
    dir_name="${dir_name}_param_a_${temperature_a}"
fi

if [ -n "$temperature_s" ]; then
    dir_name="${dir_name}_param_s_${temperature_s}"
fi

nohup python train_offline.py \
  --config=configs/sac_v1_default.py \
  --mode=$mode \
  --env_name=$env \
  --wandb_mode=online \
  --actor_lr=$actor_lr \
  --vf_lr=$vf_lr \
  --qf_lr=$qf_lr \
  --discount=$discount \
  --expectile=$expectile \
  --temperature_a=$temperature_a \
  --temperature_s=$temperature_s \
  --dropout_rate=$dropout_rate \
  --value_layer_norm=$value_layer_norm \
  --value_dropout_rate=$value_dropout_rate \
  --inv_dyna_path=$dynamics_path \
  --inv_penalty=$inv_penalty \
  --sample_strategy=$sample_strategy \
  --policy_type=$policy_type \
  > "${log_dir}/${dir_name}.log" 2>&1 &

sleep 2

done



# env_list=(
# 	# # "antmaze-umaze-v2"
# 	# # "antmaze-umaze-diverse-v2"
# 	# # "antmaze-medium-play-v2"
# 	# # "antmaze-medium-diverse-v2"
# 	# "antmaze-large-play-v2"
# 	"antmaze-large-diverse-v2"
# 	)

# vf_lr=3e-4

# for env in ${env_list[*]}; do

# dynamics_path="./inv_dyna/${env}/lr_1e-4-dropout_0.1/pretrain_steps_500000/pretrained_inv_dyna_5.pickle"
# dir_name=$env

# if [ -n "$temperature_a" ]; then
#     dir_name="${dir_name}_param_a_${temperature_a}"
# fi

# if [ -n "$temperature_s" ]; then
#     dir_name="${dir_name}_param_s_${temperature_s}"
# fi

# nohup python train_offline.py \
#   --config=configs/antmaze_config_pro.py \
#   --mode=$mode \
#   --env_name=$env \
#   --wandb_mode=online \
#   --actor_lr=$actor_lr \
#   --vf_lr=$vf_lr \
#   --qf_lr=$qf_lr \
#   --discount=$discount \
#   --expectile=$expectile \
#   --temperature_a=$temperature_a \
#   --temperature_s=$temperature_s \
#   --dropout_rate=$dropout_rate \
#   --value_layer_norm=$value_layer_norm \
#   --value_dropout_rate=$value_dropout_rate \
#   --inv_dyna_path=$dynamics_path \
#   --inv_penalty=$inv_penalty \
#   --sample_strategy=$sample_strategy \
#   > "${log_dir}/${dir_name}.log" 2>&1 &

# sleep 2

# done



# dir_name="antmaze-umaze-diverse-v2"
# dynamics_path="./inv_dyna/antmaze-umaze-diverse-v2/lr_1e-4-dropout_-1/pretrain_steps_500000/pretrained_inv_dyna_5.pickle"
# nohup python train_offline.py \
#   --config=configs/antmaze_config_pro.py \
#   --mode=$mode \
#   --env_name="antmaze-umaze-diverse-v2" \
#   --wandb_mode=online \
#   --actor_lr=$actor_lr \
#   --vf_lr=$vf_lr \
#   --qf_lr=$qf_lr \
#   --discount=$discount \
#   --expectile=$expectile \
#   --temperature_a=$temperature_a \
#   --temperature_s=$temperature_s \
#   --dropout_rate=$dropout_rate \
#   --value_layer_norm=$value_layer_norm \
#   --value_dropout_rate=$value_dropout_rate \
#   --inv_dyna_path=$dynamics_path \
#   --inv_penalty=$inv_penalty \
#   --sample_strategy=$sample_strategy \
#   --policy_type=$policy_type \
#   > "${log_dir}/${dir_name}.log" 2>&1 &

# sleep 2
