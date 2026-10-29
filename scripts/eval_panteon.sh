#!/bin/bash
#SBATCH --job-name=glare_panteon
#SBATCH --partition=es1
#SBATCH --account=pc_plantglm
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=glare/glare_panteon_eval_LRfix_context4096_lambda_order_0.1_%j.out
#SBATCH --error=glare/glare_panteon_eval_LRfix_context4096_lambda_order_0.1_%j.err

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

# Ensure the editable install's glare/glare/ package takes precedence over the
# glare/ project directory, which Python would otherwise resolve as a namespace
# package for 'glare', causing glare/data/ to shadow glare/glare/data.py.
export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare:${PYTHONPATH}

glare evaluate \
  --panteon \
  --fasta-path glare/data/PanTEon_Database_v1.6.1_benchmark_edition.fasta \
  --checkpoint glare_panteon_output/best_model.pt \
  --config glare/configs/panteon_ntv3.yaml \
  --output-dir glare_panteon_results
