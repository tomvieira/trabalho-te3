#!/bin/bash
#SBATCH --job-name=glare_calibrate
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=glare_calibrate_%j.out
#SBATCH --error=glare_calibrate_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Environment ──────────────────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

# ── Arguments ────────────────────────────────────────────────────────────────
MODEL_PATH="${GLARE_MODEL_PATH:-glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output/best_model.pt}"
CONFIG="${GLARE_CONFIG:-glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_final.yaml}"
OUTPUT_DIR="${GLARE_OUTPUT_DIR:-glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output}"

echo "Starting GLARE OOD calibration — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Model path  : ${MODEL_PATH}"
echo "Config      : ${CONFIG}"
echo "Output dir  : ${OUTPUT_DIR}"

# ── Run calibration ──────────────────────────────────────────────────────────
python -m glare.glare.calibrate \
    --model-path  "${MODEL_PATH}" \
    --config      "${CONFIG}" \
    --output-dir  "${OUTPUT_DIR}" \
    --device      cuda

echo "Calibration finished — $(date)"
echo "Artifacts written to: ${OUTPUT_DIR}/temperature_scaler.json"
echo "                      ${OUTPUT_DIR}/mahal_ood.npz"
