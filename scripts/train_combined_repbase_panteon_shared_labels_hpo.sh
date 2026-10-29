#!/bin/bash
#SBATCH --job-name=glare_combined_shared_hpo
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=glare/glare_combined_repbase_panteon_shared_labels_hpo_%j.out
#SBATCH --error=glare/glare_combined_repbase_panteon_shared_labels_hpo_%j.err

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
  --nproc_per_node=1 \
  -m glare.train_ddp \
  --config glare/configs/combined_repbase_panteon_shared_labels_hpo.yaml \
  --output-dir glare_combined_repbase_panteon_shared_labels_hpo_output
