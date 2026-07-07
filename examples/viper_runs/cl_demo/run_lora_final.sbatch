#!/bin/bash -l
#SBATCH --job-name=lora_fin
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
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
HP='{"ivon-cl": {"lr": 0.01, "clip": 0.01, "b2": 0.99999, "tau": 10, "h0": 0.5, "p0": 1.0},
     "evon-cl": {"lr": 0.01, "clip": 0.01, "b2": 0.99999, "tau": 10, "h0": 0.5, "p0": 1.0}}'
echo "=== ivon-cl full 10 tasks ==="
srun python cl_lora.py --methods ivon-cl --n-tasks 10 --epochs 10 --num-mc 8 \
    --seeds 0 --hp "$HP" --out /ptmp/cheins/blrax/examples/results_cl_lora_ivon.json
echo "=== evon-cl full 10 tasks ==="
srun python cl_lora.py --methods evon-cl --n-tasks 10 --epochs 10 --num-mc 8 \
    --seeds 0 --hp "$HP" --out /ptmp/cheins/blrax/examples/results_cl_lora_evon.json
echo "finished $(date -Iseconds)"
