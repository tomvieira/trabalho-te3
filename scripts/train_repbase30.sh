#!/bin/bash
#SBATCH --job-name=glare_repbase30
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=glare_repbase30_%j.out
#SBATCH --error=glare_repbase30_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

# Reduce CUDA memory fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /global/scratch/users/nhaas/repeats

# Ensure the editable install's glare/glare/ package takes precedence over the
# glare/ project directory, which Python would otherwise resolve as a namespace
# package for 'glare', causing glare/data/ to shadow glare/glare/data.py.
# Use ${PYTHONPATH:-} because this script runs with set -u.
export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

echo "Starting GLARE Repbase→PanTEon30 training — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "GPUs per node: 4"

/global/scratch/users/nhaas/.conda/envs/glare-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m glare.train_ddp \
    --config glare/configs/repbase30.yaml \
    --output-dir glare_repbase30_output

echo "Repbase→PanTEon30 training finished — $(date)"
