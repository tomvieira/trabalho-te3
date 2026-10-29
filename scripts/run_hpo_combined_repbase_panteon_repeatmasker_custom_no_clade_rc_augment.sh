#!/bin/bash
#SBATCH --job-name=glare_hpo_combined_rm_custom_no_clade_rc_augment
#SBATCH --account=pc_plantglm
#SBATCH --partition=es2
#SBATCH --qos=es2_normal
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=72:00:00
#SBATCH --output=glare/glare_hpo_combined_rm_custom_no_clade_rc_augment_%j.out
#SBATCH --error=glare/glare_hpo_combined_rm_custom_no_clade_rc_augment_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

echo "Starting GLARE HPO — $(date)"
echo "SLURM job id: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ',' | sed 's/,$//')"

STUDY_NAME="glare_hpo_combined_rm_custom_no_clade_rc_augment"
STORAGE_DB="${STUDY_NAME}.db"
BASE_CONFIG="glare/configs/combined_repbase_panteon_repeatmasker_custom_no_clade_rc_augment.yaml"
OUTPUT_DIR="glare_hpo_trials"
N_EPOCHS=10
# 4 workers × 10 trials = 40 total trials
N_TRIALS_PER_WORKER=10

# Run 4 parallel single-GPU workers; each picks up trials from the shared SQLite study.
# CUDA_VISIBLE_DEVICES restricts each worker to exactly one GPU.
for GPU_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -m glare.hpo \
        --base-config "${BASE_CONFIG}" \
        --n-trials "${N_TRIALS_PER_WORKER}" \
        --study-name "${STUDY_NAME}" \
        --storage "${STORAGE_DB}" \
        --output-dir "${OUTPUT_DIR}" \
        --n-epochs "${N_EPOCHS}" \
        --worker-seed $((42 + GPU_ID)) \
        2>&1 | sed "s/^/[GPU${GPU_ID}] /" &
done

# Wait for all background workers to finish
wait

echo "HPO finished — $(date)"

# Print best result from the study
echo ""
echo "=== Best trial summary ==="
python - <<'PYEOF'
import sys
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(
        study_name="glare_hpo_combined_rm_custom_no_clade_rc_augment",
        storage="sqlite:///glare_hpo_combined_rm_custom_no_clade_rc_augment.db",
    )
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"Completed trials: {len(completed)} / {len(study.trials)}")
    if completed:
        best = study.best_trial
        print(f"Best trial: #{best.number}  val_sf_f1={best.value:.4f}")
        print("Best hyperparameters:")
        for k, v in sorted(best.params.items()):
            print(f"  {k} = {v}")
    else:
        print("No completed trials found.")
except Exception as e:
    print(f"Could not load study: {e}", file=sys.stderr)
PYEOF
