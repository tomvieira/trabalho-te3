#!/bin/bash
#SBATCH --job-name=glare_panteon_caduceus
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
#SBATCH --output=glare/glare_panteon_caduceus_%j.out
#SBATCH --error=glare/glare_panteon_caduceus_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glm_glare

# Force the active conda environment's runtime libraries to take precedence over
# the system toolchain so PyTorch resolves a compatible libstdc++.
export CONDA_PREFIX="${CONDA_PREFIX:-/global/scratch/users/nhaas/.conda/envs/glm_glare}"
export PATH="${CONDA_PREFIX}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd /global/scratch/users/nhaas/repeats

pip install -e /global/scratch/users/nhaas/repeats/glare

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# Caduceus uses mamba-ssm/Triton kernels; make Triton's packaged CUDA backends
# visible on this cluster, matching the existing PlantCAD2 runtime guard.
export TRITON_BACKENDS_IN_TREE=1
export OMP_NUM_THREADS=4

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m glare.train_ddp \
  --config glare/configs/panteon_caduceus.yaml \
  --output-dir glare_panteon_caduceus_output
