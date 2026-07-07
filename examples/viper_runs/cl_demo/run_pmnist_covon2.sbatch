#!/bin/bash -l
#SBATCH --job-name=pmn_cov2
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
# CoVON lr semantics: official PyTorch IVON rescales lr by (h0+delta);
# blrax does not (rescale_lr=False), so feed pre-rescaled lrs:
# 0.02*(0.01) = 2e-4 on task 1, 0.003*(0.01) = 3e-5 after.
HP='{"ivon-cl": {"lr": 2e-4, "lr2": 3e-5, "h0": 0.01, "ess_abs": 1e7, "clip": 1e9, "b2": 0.99999, "p0": 1.0}}'
srun python cl_permuted_mnist.py \
    --data-dir /ptmp/cheins/blrax/examples/data \
    --n-tasks 10 --epochs 30 --hidden 100 --seeds 0,1,2 \
    --methods ivon-cl --hp "$HP" \
    --out /ptmp/cheins/blrax/examples/results_cl_pmnist100_covon2.json
echo "finished $(date -Iseconds)"
