#!/bin/bash -l
# ivon-cl hparam grid on Split CIFAR-100 (features cached) — one run per config.
#SBATCH --job-name=cl_sweep
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
cd /ptmp/cheins/blrax/examples
FEATS=/ptmp/cheins/data/cifar100_dinov2s14.npz
for B2 in 0.99 0.999 0.99999; do
for LR in 3e-3 1e-2 3e-2; do
for TAU in 1 10; do
  HP="{\"ivon-cl\": {\"lr\": $LR, \"b2\": $B2, \"tau\": $TAU, \"h0\": 0.5, \"p0\": 1.0}}"
  echo "### b2=$B2 lr=$LR tau=$TAU"
  srun python cl_split_cifar.py --features "$FEATS" --methods ivon-cl \
      --n-tasks 10 --epochs 20 --num-mc 8 --seeds 0 --hp "$HP" \
      --out "/tmp/cl_sweep_tmp.json" 2>/dev/null | grep "seed0"
done; done; done
echo "finished $(date -Iseconds)"
