#!/bin/bash -l
#SBATCH --job-name=cl_lora
#SBATCH --partition=apu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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
srun python cl_lora.py --methods "${METHODS:-adamw}" \
    --n-tasks "${NTASKS:-10}" --epochs "${EPOCHS:-10}" --num-mc 8 --seeds 0 \
    --hp "${HP:-{}}" \
    --out "/ptmp/cheins/blrax/examples/${OUT:-results_cl_lora.json}"
echo "finished $(date -Iseconds)"
