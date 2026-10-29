#!/bin/bash
#SBATCH --job-name=glare_poales_hpo_grass_shared_eval
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=glare/glare_panteon_poales_holdout_eval_repbase_grass_shared_labels_hpo_eval_%j.out
#SBATCH --error=glare/glare_panteon_poales_holdout_eval_repbase_grass_shared_labels_hpo_eval_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

glare evaluate \
  --external-repbase \
  --repbase-fasta glare/data/repbase31_holdout \
  --toml-path glare/data/repbase-to-repeatmasker.toml \
  --checkpoint glare_panteon_poales_holdout_shared_labels_hpo_output/best_model.pt \
  --config glare/configs/panteon_poales_holdout_eval_repbase_grass_shared_labels_hpo.yaml \
  --output-dir glare_panteon_poales_holdout_eval_repbase_grass_shared_labels_hpo_results
