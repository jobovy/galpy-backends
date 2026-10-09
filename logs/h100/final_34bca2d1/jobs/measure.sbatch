#!/bin/bash
# One final-campaign stage against the H100 copies: route pre -> repo stage -> route post.
#   sbatch --export=ALL,STAGE=<stage> measure.sbatch
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=final
set -u
J=$SCRATCH/galpy-h100/jobs
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate galpy-h100
cd $SCRATCH/galpy-h100/galpy-jax
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "galpy $(git -C $SCRATCH/galpy-h100/galpy-fb rev-parse HEAD)"
bash $J/route.sh pre $STAGE || { echo "!!! route pre failed"; exit 4; }
bash tools/trillium_stage.sh "$STAGE"
bash $J/route.sh post $STAGE || echo "!!! route post failed"
