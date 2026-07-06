#!/bin/bash -l
#SBATCH --job-name=cl_sweep2
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
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
run () {  # lr clip epochs
  HP="{\"ivon-cl\": {\"lr\": $1, \"b2\": 0.99999, \"tau\": 10, \"h0\": 0.5, \"p0\": 1.0, \"clip\": $2}}"
  echo "### lr=$1 clip=$2 epochs=$3"
  srun python cl_split_cifar.py --features "$FEATS" --methods ivon-cl \
      --n-tasks 10 --epochs "$3" --num-mc 8 --seeds 0 --hp "$HP" \
      --out /tmp/cl_sweep2_tmp.json 2>/dev/null | grep "seed0"
}
run 3e-2 1e9 50
run 1e-1 1e9 20
run 1e-1 1e9 50
run 1e-1 0.003 20
run 1e-1 0.003 50
run 3e-2 0.003 20
run 1e-1 0.01 20
echo "finished $(date -Iseconds)"
