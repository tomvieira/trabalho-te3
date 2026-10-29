#!/bin/bash
#SBATCH --job-name=glare_panteon_plantcad2_train
#SBATCH --partition=es2
#SBATCH --account=pc_plantglm
#SBATCH --qos=es2_normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --exclude=n0005.es1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=glare/glare_panteon_plantcad2_full_fine_tune_%j.out
#SBATCH --error=glare/glare_panteon_plantcad2_full_fine_tune_%j.err

set -euo pipefail

source /global/software/rocky-8.x86_64/manual/modules/langs/miniforge3/25.9.1/etc/profile.d/conda.sh
conda activate glm_glare

# Force the active conda environment's runtime libraries to take precedence over
# the system toolchain so PyTorch resolves a compatible libstdc++.
export CONDA_PREFIX="${CONDA_PREFIX:-/global/scratch/users/nhaas/.conda/envs/glm_glare}"
export PATH="${CONDA_PREFIX}/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd /global/scratch/users/nhaas/repeats

# Repair any stale or partially upgraded HF package metadata/code in the job
# environment. The recent failures showed a mixed install where the active
# transformers package was 4.44.2, but stale 5.6.1 dist-info caused import-time
# dependency checks to require tokenizers>=0.22. A later failure showed
# huggingface_hub had been upgraded to a 1.x release whose httpx client no
# longer accepts the allow_redirects= kwarg used by transformers 4.44.x.
python -m pip install --upgrade --force-reinstall --no-deps \
  "transformers==4.44.2" \
  "tokenizers==0.19.1" \
  "huggingface_hub<1.0"

pip install -e /global/scratch/users/nhaas/repeats/glare

# Ensure the editable install's glare/glare/ package takes precedence over the
# glare/ project directory, which Python would otherwise resolve as a namespace
# package for 'glare', causing glare/data/ to shadow glare/glare/data.py.
export PYTHONPATH=/global/scratch/users/nhaas/repeats/glare${PYTHONPATH:+:${PYTHONPATH}}

# ── NCCL / DDP performance tuning ───────────────────────────────────────────
# Enable InfiniBand and P2P (NVLink) for fast inter-GPU communication.
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
# Triton 3.6 on this cluster can miss its in-tree CUDA backend when it only
# attempts entry-point discovery, which leaves triton.backends empty and makes
# triton.runtime.driver.active fail with "0 active drivers ([]). There should
# only be one." Force Triton to register its packaged backends directly.
export TRITON_BACKENDS_IN_TREE=1
# Limit OpenMP threads per process to avoid CPU oversubscription with 3 GPUs
# and 16 CPUs total (16 / 3 = 5.33 → use 4 to leave headroom for DataLoader workers).
export OMP_NUM_THREADS=4

echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "PATH=${PATH}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
echo "TRITON_BACKENDS_IN_TREE=${TRITON_BACKENDS_IN_TREE}"
echo "torchrun=$(command -v torchrun)"
echo "libstdc++=$(ldd "$(command -v python)" | grep libstdc++)"
strings "${CONDA_PREFIX}/lib/libstdc++.so.6" | grep 'GLIBCXX_3.4.30' | head -n 1
python - <<'PY'
import inspect
import os
import transformers, tokenizers, huggingface_hub
from transformers.utils.hub import get_session
print(f"transformers={transformers.__version__} ({transformers.__file__})")
print(f"tokenizers={tokenizers.__version__} ({tokenizers.__file__})")
print(f"huggingface_hub={huggingface_hub.__version__} ({huggingface_hub.__file__})")
print(f"hf_session={type(get_session()).__name__} head_signature={inspect.signature(get_session().head)}")
print(f"TRITON_BACKENDS_IN_TREE={os.environ.get('TRITON_BACKENDS_IN_TREE')}")
try:
    import torch
    import triton.backends
    from triton.runtime import driver

    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_device_count={torch.cuda.device_count()}")
    print(f"triton_backends={sorted(triton.backends.backends.keys())}")
    try:
        print(f"triton_driver_default={driver.default.__class__.__name__}")
    except Exception as exc:
        print(f"triton_driver_default_error={type(exc).__name__}: {exc}")
except Exception as exc:
    print(f"triton_preflight_error={type(exc).__name__}: {exc}")
PY

torchrun \
  --standalone \
  --nproc_per_node=4 \
  -m glare.train_ddp \
  --config glare/configs/panteon_plantcad2.yaml \
  --output-dir glare_panteon_plantcad2_output
