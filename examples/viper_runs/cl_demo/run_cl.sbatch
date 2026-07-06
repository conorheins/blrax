#!/bin/bash -l
# Split CIFAR-100 continual learning on frozen DINOv2 features — Viper-GPU.
# Extracts features once (skipped if npz exists), then runs the CL benchmark.
#SBATCH --job-name=cl_dino
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs
module purge
module load python-waterboa/2025.06 rocm/7.2
module load jax/0.8.2
source "$HOME/blrax-rocm/bin/activate"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTHONPATH=/ptmp/cheins/blrax/src:/ptmp/cheins/equimo/src:${PYTHONPATH:-}
export TMPDIR="/ptmp/$USER/tmp/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"

cd /ptmp/cheins/blrax/examples
echo "host $(hostname) | $(date -Iseconds)"

FEATS=/ptmp/cheins/data/cifar100_dinov2s14.npz
if [ ! -f "$FEATS" ]; then
  echo "=== extracting DINOv2 features ==="
  srun python extract_dino_features.py --out "$FEATS" || exit 1
fi

echo "=== CL benchmark ==="
srun python cl_split_cifar.py \
    --features "$FEATS" \
    --methods "${METHODS:-adamw,adamw-ewc,ivon,ivon-cl,evon-cl}" \
    --n-tasks 10 --epochs "${EPOCHS:-20}" --num-mc 32 \
    --seeds "${SEEDS:-0,1,2}" --hp "${HP:-{}}" \
    --out "/ptmp/cheins/blrax/examples/${OUT:-results_cl_dino.json}"
echo "finished $(date -Iseconds)"
