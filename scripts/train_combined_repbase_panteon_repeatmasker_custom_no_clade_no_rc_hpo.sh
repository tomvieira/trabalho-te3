#!/bin/bash
#SBATCH --job-name=glare_combined_rm_custom_no_clade_no_rc_hpo
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=glare_combined_repbase_panteon_repeatmasker_custom_no_clade_no_rc_hpo_%j.out
#SBATCH --error=glare_combined_repbase_panteon_repeatmasker_custom_no_clade_no_rc_hpo_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

FINAL_CONFIG="glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_no_rc_train_hpo.yaml"
FINAL_OUTPUT_DIR="glare_combined_repbase_panteon_repeatmasker_custom_no_clade_no_rc_hpo_output"

echo "Starting GLARE combined Repbase+PanTEon repeatmasker_custom no-clade no-RC HPO training — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '
' ',' | sed 's/,$//')"
echo "Config: ${FINAL_CONFIG}"
echo "Output dir: ${FINAL_OUTPUT_DIR}"

/global/scratch/users/nhaas/.conda/envs/glare-env/bin/python -m glare.train_ddp     --config "${FINAL_CONFIG}"     --output-dir "${FINAL_OUTPUT_DIR}"

echo "GLARE combined Repbase+PanTEon repeatmasker_custom no-clade no-RC HPO training finished — $(date)"
echo "Checkpoint: ${FINAL_OUTPUT_DIR}/best_model.pt"
echo "Training log and automatic panteon_random_test/repbase_random_test metrics: ${FINAL_OUTPUT_DIR}/train_log.csv"
