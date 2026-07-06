#!/bin/bash -l
#SBATCH --job-name=confirm_vit
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
export TMPDIR="/ptmp/$USER/tmp/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"
cd /ptmp/cheins/blrax/examples
echo "host $(hostname) | $(date -Iseconds)"
srun python confirm_tuned_vit.py --epochs 60 --num-mc 32 \
    --out /ptmp/cheins/blrax/examples/results_vit_tuned.json
echo "finished $(date -Iseconds)"
