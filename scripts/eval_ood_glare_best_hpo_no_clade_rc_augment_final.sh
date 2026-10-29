#!/bin/bash
#SBATCH --job-name=glare_best_hpo_ood_eval
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=glare_best_hpo_ood_eval_%j.out
#SBATCH --error=glare_best_hpo_ood_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

FINAL_OUTPUT_DIR="glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output"
MODEL_PATH="${FINAL_OUTPUT_DIR}/best_model.pt"
CONFIG="glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_final.yaml"
TEMPERATURE_SCALER="${FINAL_OUTPUT_DIR}/temperature_scaler.json"
MAHAL_DETECTOR="${FINAL_OUTPUT_DIR}/mahal_ood.npz"
OUTPUT_DIR="glare_best_hpo_no_clade_rc_augment_ood_eval_results"

echo "Starting final GLARE best-HPO OOD evaluation — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG}"
echo "Output dir: ${OUTPUT_DIR}"

python -m glare.glare.eval_ood \
    --model-path         "${MODEL_PATH}" \
    --config             "${CONFIG}" \
    --output-dir         "${OUTPUT_DIR}" \
    --temperature-scaler "${TEMPERATURE_SCALER}" \
    --mahal-detector     "${MAHAL_DETECTOR}" \
    --device             cuda \
    --batch-size         8

echo "Final GLARE best-HPO OOD evaluation finished — $(date)"
echo "Results written to: ${OUTPUT_DIR}/"
echo "  Summary CSV:     ${OUTPUT_DIR}/ood_eval_summary.csv"
echo "  Thresholds JSON: ${OUTPUT_DIR}/ood_thresholds.json"
