#!/bin/bash
#SBATCH --job-name=glare_best_hpo_classic_eval
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4:00:00
#SBATCH --output=glare_best_hpo_classic_eval_%j.out
#SBATCH --error=glare_best_hpo_classic_eval_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

FINAL_OUTPUT_DIR="glare_combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment_best_hpo_output"
MODEL_PATH="${FINAL_OUTPUT_DIR}/best_model.pt"
PANTEON_CONFIG="glare/configs/combined_repeatmasker_custom_best_hpo_no_clade_rc_augment_eval_panteon_random_rc.yaml"
REPBASE_CONFIG="glare/configs/combined_repeatmasker_custom_best_hpo_no_clade_rc_augment_eval_repbase_random_rc.yaml"
PANTEON_RESULTS="glare_best_hpo_no_clade_rc_augment_on_panteon_random_rc_results"
REPBASE_RESULTS="glare_best_hpo_no_clade_rc_augment_on_repbase_random_rc_results"

echo "Starting final GLARE best-HPO classic evaluation — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "Model: ${MODEL_PATH}"
echo "PanTEon config: ${PANTEON_CONFIG}"
echo "Repbase config: ${REPBASE_CONFIG}"
echo "PanTEon results: ${PANTEON_RESULTS}"
echo "Repbase results: ${REPBASE_RESULTS}"

glare evaluate \
  --panteon \
  --fasta-path glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta \
  --checkpoint "${MODEL_PATH}" \
  --config "${PANTEON_CONFIG}" \
  --output-dir "${PANTEON_RESULTS}"

glare evaluate \
  --external-repbase \
  --repbase-fasta glare/data/repbase31_total \
  --toml-path glare/data/repbase-to-repeatmasker-custom.toml \
  --checkpoint "${MODEL_PATH}" \
  --config "${REPBASE_CONFIG}" \
  --output-dir "${REPBASE_RESULTS}"

echo "Final GLARE best-HPO classic evaluation finished — $(date)"
echo "PanTEon confusion matrix: ${PANTEON_RESULTS}/panteon_confusion_matrix.csv and ${PANTEON_RESULTS}/panteon_confusion_matrix.png"
echo "Repbase confusion matrix: ${REPBASE_RESULTS}/grasrep_confusion_matrix.csv and ${REPBASE_RESULTS}/grasrep_confusion_matrix.png"

