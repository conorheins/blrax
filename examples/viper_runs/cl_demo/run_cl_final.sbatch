#!/bin/bash -l
#SBATCH --job-name=cl_final
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
HP='{"ivon-cl": {"lr": 0.1, "b2": 0.99999, "tau": 10, "h0": 0.5, "p0": 1.0, "clip": 1e9},
     "evon-cl": {"lr": 0.1, "b2": 0.99999, "tau": 10, "h0": 0.5, "p0": 1.0, "clip": 1e9}}'
echo "=== FINAL benchmark: 5 methods x 3 seeds (tuned CL configs) ==="
srun python cl_split_cifar.py --features "$FEATS" \
    --methods adamw,adamw-ewc,ivon,ivon-cl,evon-cl \
    --n-tasks 10 --epochs 20 --num-mc 32 --seeds 0,1,2 --hp "$HP" \
    --out /ptmp/cheins/blrax/examples/results_cl_final.json
echo "=== JOINT upper bound (single task, all 100 classes) ==="
srun python cl_split_cifar.py --features "$FEATS" \
    --methods adamw,ivon --n-tasks 1 --epochs 20 --num-mc 32 --seeds 0 \
    --out /ptmp/cheins/blrax/examples/results_cl_joint.json
echo "finished $(date -Iseconds)"
