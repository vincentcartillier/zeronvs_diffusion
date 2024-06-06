#!/bin/bash

cd /srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/dependencies/zeronvs_diffusion/zero123/

source ~/.bashrc

logdir="$1"
expename="$2"
config_file="$3"

rootdir="/srv/essa-lab/flash3/vcartillier3/egoexo-view-synthesis/dependencies/zeronvs_diffusion/zero123/"

echo "ROOTIR: $rootdir"
echo "LOGDIR: $logdir"
echo "EXPENAME: $expename"
echo "CONFIG FILE: $config_file"

conda activate zeronvs_diffusion


#trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM

python main_debug.py \
    -t \
    --base "$config_file" \
    --gpus "0,1,2,3,4,5,6,7" \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --finetune_from  ../../ZeroNVS/checkpoints/zeronvs_no_T.ckpt \
    --rootdir "$rootdir" \
    --logdir "$logdir" \
    --name "$expename" \
    --enable_look_for_checkpoints False 


# Set the PID var so that the trap can use it
#PID="$!"
#wait "${PID}"
