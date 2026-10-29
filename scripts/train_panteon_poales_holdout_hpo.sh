#!/bin/bash
#SBATCH --job-name=glare_panteon_poales_hpo
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=n0005.es1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=glare/glare_panteon_poales_holdout_hpo_%j.out
#SBATCH --error=glare/glare_panteon_poales_holdout_hpo_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

pip install -e /global/scratch/users/nhaas/repeats/glare

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=4

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m glare.train_ddp \
  --config glare/configs/panteon_poales_holdout_hpo.yaml \
  --output-dir glare_panteon_poales_holdout_hpo_output
