#!/bin/bash

# cd /nethome/abati7/flash/Work/recon/zeronvs_diffusion/zero123

source ~/.bashrc

logdir="$1"
expename="$2"
config_file="$3" 

rootdir="/nethome/abati7/flash/Work/recon/zeronvs_diffusion/zero123"

echo "ROOTIR: $rootdir"
echo "LOGDIR: $logdir"
echo "EXPENAME: $expename"
echo "CONFIG FILE: $config_file"

conda activate zero123


trap 'echo signal recieved in BATCH!; kill -15 "${PID}"; wait "${PID}";' SIGINT SIGTERM

python main_debug.py \
    --base "$config_file" \
    --gpus "0,1,2,3" \
    --scale_lr False \
    --num_nodes 1 \
    --seed 42 \
    --finetune_from /srv/essa-lab/flash3/vcartillier3/zeronvs_diffusion/zero123/runs/2024-08-28T13-47-48_exoego_preprocessed_data/checkpoints/trainstep_checkpoints/epoch=000000-step=000004999.ckpt \
    --rootdir "$rootdir" \
    --logdir "$logdir" \
    --name "$expename" \
    --enable_look_for_checkpoints False \
    --logdir_mode ""


# Set the PID var so that the trap can use it
PID="$!"
#wait "${PID}"
