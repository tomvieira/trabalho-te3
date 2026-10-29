#!/bin/bash
#SBATCH --job-name=glare_combined_rm_custom_no_clade_no_rc_repbase_rc_hpo_eval
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --time=1:00:00
#SBATCH --output=glare/glare_combined_repeatmasker_custom_no_clade_no_rc_on_repbase_random_rc_hpo_eval_%j.out
#SBATCH --error=glare/glare_combined_repeatmasker_custom_no_clade_no_rc_on_repbase_random_rc_hpo_eval_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glare-env

cd /global/scratch/users/nhaas/repeats

export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

glare evaluate   --external-repbase   --repbase-fasta glare/data/repbase31_total   --toml-path glare/data/repbase-to-repeatmasker-custom.toml   --checkpoint glare_combined_repbase_panteon_repeatmasker_custom_no_clade_no_rc_hpo_output/best_model.pt   --config glare/configs/combined_repeatmasker_custom_no_clade_no_rc_train_eval_repbase_random_rc_hpo.yaml   --output-dir glare_combined_repeatmasker_custom_no_clade_no_rc_on_repbase_random_rc_hpo_results
