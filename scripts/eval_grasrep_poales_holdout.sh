#!/bin/bash
#SBATCH --job-name=glare_grasrep_poales_eval
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=glare/glare_grasrep_poales_holdout_eval_%j.out
#SBATCH --error=glare/glare_grasrep_poales_holdout_eval_%j.err

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

# Ensure the editable install's glare/glare/ package takes precedence over the
# glare/ project directory, which Python would otherwise resolve as a namespace
# package for 'glare', causing glare/data/ to shadow glare/glare/data.py.
export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare:${PYTHONPATH}

glare evaluate \
  --external-repbase \
  --repbase-fasta glare/data/repbase31_holdout/grasrep.ref \
  --toml-path glare/data/repbase-to-repeatmasker.toml \
  --checkpoint glare_panteon_poales_holdout_output/best_model.pt \
  --config glare/configs/panteon_poales_holdout.yaml \
  --output-dir glare_grasrep_poales_holdout_results
