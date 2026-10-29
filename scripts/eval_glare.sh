#!/bin/bash
#SBATCH --job-name=glare_eval
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=glare_eval_%j.out
#SBATCH --error=glare_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Environment ──────────────────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

# ── Arguments ────────────────────────────────────────────────────────────────
# Override CHECKPOINT path via environment variable or edit here.
CHECKPOINT="${GLARE_CHECKPOINT:-glare_output/best_model.pt}"
CONFIG="${GLARE_CONFIG:-glare/configs/default.yaml}"
OUTPUT_DIR="${GLARE_OUTPUT_DIR:-glare_results}"

echo "Starting GLARE evaluation — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Checkpoint: ${CHECKPOINT}"

# ── Run evaluation ───────────────────────────────────────────────────────────
python -m glare.cli evaluate \
    --checkpoint "${CHECKPOINT}" \
    --config     "${CONFIG}" \
    --output-dir "${OUTPUT_DIR}"

echo "Evaluation finished — $(date)"
echo "Results written to: ${OUTPUT_DIR}/glare_holdout_f1_results.csv"
