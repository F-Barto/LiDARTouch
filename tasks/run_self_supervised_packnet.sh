#! /bin/bash

# Inria cluster specifics
source ~/.bashrc
source gpu_setVisibleDevices.sh

conda activate Depth_env

WANBKEY=$(<../wandb.key) # load the key from a file at root dir
wandb login $WANBKEY

CONFIG_DIR="../configs/"

python -u ../train.py \
--model_config_file $CONFIG_DIR"simpledepth.yml" \
--model_config_profile "default" \
--project_config_file $CONFIG_DIR"project.yml" \
--project_config_profile "default" \
"$@" # pass --fast_dev_run to the shell script and it will pass it to the python script
