#!/bin/bash
#SBATCH --job-name=glare_ood_fasta_eval
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=glare_ood_fasta_eval_%j.out
#SBATCH --error=glare_ood_fasta_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

# ── Environment ──────────────────────────────────────────────────────────────
source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

# ── Arguments ────────────────────────────────────────────────────────────────
MODEL_PATH="glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output/best_model.pt"
CONFIG="glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_final.yaml"
TEMPERATURE_SCALER="glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output/temperature_scaler.json"
MAHAL_DETECTOR="glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output/mahal_ood.npz"
OUTPUT_DIR="glare_best_hpo_no_clade_rc_augment_ood_testsets_vs_non_te_results"

MAX_OOD_SEQUENCES_PER_FILE=10000

echo "Starting GLARE OOD vs FASTA evaluation — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Model: ${MODEL_PATH}"
echo "Config: ${CONFIG}"
echo "OOD FASTAs:"
echo "  glare/data/non-TE/arabidopsis_high_confidence_cds.fa"
echo "  glare/data/non-TE/transcripts.all.fa"
echo "  glare/data/non-TE/transcripts.nr.fa"
echo "Max OOD sequences per file: ${MAX_OOD_SEQUENCES_PER_FILE}"
echo "Output dir: ${OUTPUT_DIR}"

# ── Run true OOD evaluation: held-out test sets as ID vs non-TE FASTAs ───────
python -m glare.glare.eval_ood \
    --model-path        "${MODEL_PATH}" \
    --config            "${CONFIG}" \
    --temperature-scaler "${TEMPERATURE_SCALER}" \
    --mahal-detector    "${MAHAL_DETECTOR}" \
    --output-dir        "${OUTPUT_DIR}" \
    --eval-mode         testsets-fasta \
    --ood-fasta         glare/data/non-TE/arabidopsis_high_confidence_cds.fa \
    --ood-fasta         glare/data/non-TE/transcripts.all.fa \
    --ood-fasta         glare/data/non-TE/transcripts.nr.fa \
    --max-ood-sequences-per-file "${MAX_OOD_SEQUENCES_PER_FILE}" \
    --device            cuda \
    --batch-size        32

echo "OOD testsets vs FASTA evaluation finished — $(date)"
echo "Results written to: ${OUTPUT_DIR}/"
echo "  Summary CSV:       ${OUTPUT_DIR}/ood_testsets_vs_fasta_summary.csv"
echo "  Thresholds JSON:   ${OUTPUT_DIR}/ood_testsets_vs_fasta_thresholds.json"
echo "  ROC curves:        ${OUTPUT_DIR}/ood_testsets_vs_fasta_roc_curves.png"
echo "  Score dists:       ${OUTPUT_DIR}/ood_testsets_vs_fasta_score_distributions.png"
echo "  ID predictions:    ${OUTPUT_DIR}/ood_testsets_vs_fasta_predictions_id.csv"
echo "  OOD predictions:   ${OUTPUT_DIR}/ood_testsets_vs_fasta_predictions_ood.csv"
