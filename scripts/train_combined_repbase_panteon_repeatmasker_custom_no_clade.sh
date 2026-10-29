#!/bin/bash
#SBATCH --job-name=glare_combined_rm_custom_no_clade
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=glare_combined_repbase_panteon_repeatmasker_custom_no_clade_%j.out
#SBATCH --error=glare_combined_repbase_panteon_repeatmasker_custom_no_clade_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

echo "Starting GLARE combined Repbase+PanTEon repeatmasker_custom no-clade training — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Nodes: ${SLURM_NODELIST}"
echo "GPUs per node: 4"

/global/scratch/users/nhaas/.conda/envs/glare-env/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    -m glare.train_ddp \
    --config glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade.yaml \
    --output-dir glare_combined_repbase_panteon_repeatmasker_custom_no_clade_output

echo "Combined Repbase+PanTEon repeatmasker_custom no-clade training finished — $(date)"
