#!/bin/bash
# Submit from the Trillium GPU login node (trillium-gpu.alliancecan.ca):
#   sbatch --account=<ACCOUNT> --time=<HH:MM:SS> --export=ALL,STAGE=<stage> tools/trillium_job.sbatch
# One H100 (Trillium schedules GPU jobs by 1 GPU or a whole node of 4).
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=galpy-h100
#SBATCH --output=%x-%j.out
set -u
source "$CONDA_SH" && conda activate "$CONDA_ENV"   # passed through --export=ALL
cd $SCRATCH/galpy-h100/galpy-jax
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
bash tools/trillium_stage.sh "$STAGE"
