#!/bin/bash
# TODO: put your desired HPC preamble: partition, qos, time, gres, output, etc.

echo "Input to sbatch: " $@

module use /opt/insy/modulefiles
module load cuda/11.4 cudnn/11.4-8.2.2.26

# TODO: activate your own conda environment here

# TODO: insert your W&B API key
export WANDB_API_KEY="xxx"

# For multi-GPU, don't use P2P as it hangs
export NCCL_P2P_DISABLE=1

srun "$@"