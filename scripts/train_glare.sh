#!/bin/bash
#SBATCH --job-name=glare_train
#SBATCH --account=pc_plantglm
#SBATCH --partition=es1
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=glare_train_v2_%j.out
#SBATCH --error=glare_train_v2_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

# Reduce CUDA memory fragmentation (helps avoid OOM at batch_size=16 with 80GB A100s)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /global/scratch/users/nhaas/repeats

echo "Starting GLARE training — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "GPUs per node: 4"

# ── torchrun (manages 4 DDP worker processes, one per GPU) ───────────────────
/global/scratch/users/nhaas/.conda/envs/glare-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m glare.train_ddp \
    --config glare/configs/default.yaml \
    --output-dir glare_output_v2

echo "Training finished — $(date)"
