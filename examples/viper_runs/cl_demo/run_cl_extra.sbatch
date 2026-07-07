#!/bin/bash -l
# (a) rotated-CIFAR-10 DIL + (c) Split ImageNet-R, frozen DINOv2 features.
#SBATCH --job-name=cl_extra
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
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
HP='{"ivon-cl": {"lr": 0.1, "b2": 0.99999, "tau": 10, "h0": 0.5, "p0": 1.0, "clip": 1e9},
     "evon-cl": {"lr": 0.1, "b2": 0.99999, "tau": 10, "h0": 0.5, "p0": 1.0, "clip": 1e9}}'

if [ ! -f /ptmp/cheins/data/cifar10_rot_dinov2s14.npz ] || [ ! -f /ptmp/cheins/data/imagenet_r_dinov2s14.npz ]; then
  echo "=== feature extraction (rotcifar + imagenet-r) ==="
  srun python extract_dil_and_inr.py --which both || exit 1
fi

echo "=== (a) rotated-CIFAR-10 domain-incremental ==="
srun python cl_split_cifar.py --scenario dil \
    --features /ptmp/cheins/data/cifar10_rot_dinov2s14.npz \
    --methods adamw,adamw-ewc,ivon,ivon-cl,evon-cl \
    --epochs 20 --num-mc 32 --seeds 0,1,2 --hp "$HP" \
    --out /ptmp/cheins/blrax/examples/results_cl_dil.json

echo "=== (c) Split ImageNet-R 10x20 CIL ==="
srun python cl_split_cifar.py \
    --features /ptmp/cheins/data/imagenet_r_dinov2s14.npz \
    --methods adamw,adamw-ewc,ivon,ivon-cl,evon-cl \
    --n-tasks 10 --epochs 20 --num-mc 32 --seeds 0,1,2 --hp "$HP" \
    --out /ptmp/cheins/blrax/examples/results_cl_inr.json

echo "=== joint ceilings ==="
srun python cl_split_cifar.py --features /ptmp/cheins/data/imagenet_r_dinov2s14.npz \
    --methods adamw,ivon --n-tasks 1 --epochs 20 --num-mc 16 --seeds 0 \
    --out /ptmp/cheins/blrax/examples/results_cl_inr_joint.json
srun python cl_split_cifar.py --scenario dil --features /ptmp/cheins/data/cifar10_rot_dinov2s14.npz \
    --methods adamw,ivon --n-tasks 1 --epochs 20 --num-mc 16 --seeds 0 \
    --out /ptmp/cheins/blrax/examples/results_cl_dil_joint.json || true
echo "finished $(date -Iseconds)"
