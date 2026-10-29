#!/bin/bash
#SBATCH --job-name=glare_combined_panteon_eval
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=glare/glare_combined_on_panteon_poales_shared_labels_eval_%j.out
#SBATCH --error=glare/glare_combined_on_panteon_poales_shared_labels_eval_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

glare evaluate \
  --panteon \
  --fasta-path glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta \
  --checkpoint glare_combined_repbase_panteon_shared_labels_output/best_model.pt \
  --config glare/configs/combined_eval_panteon_poales_shared_labels.yaml \
  --output-dir glare_combined_on_panteon_poales_shared_labels_results

