#!/bin/bash

#SBATCH --job-name="ZeroNVS-EgoExo"

#SBATCH --output=/srv/essa-lab/flash3/vcartillier3/zeronvs_diffusion/zero123/batch_scripts/slurm_logs/sample-%j.out

#SBATCH --error=/srv/essa-lab/flash3/vcartillier3/zeronvs_diffusion/zero123/batch_scripts/slurm_logs/sample-%j.err

## number of nodes
#SBATCH --nodes=1

## number of tasks per node
#SBATCH --ntasks-per-node=1

#SBATCH --gpus-per-node=a40:8

#SBATCH --cpus-per-task=128

#SBATCH -p essa-lab

#SBATCH --qos short

#SBATCH --signal=B:TERM@120

srun $1 $2 $3 $4
