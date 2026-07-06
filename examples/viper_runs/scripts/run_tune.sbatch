#!/bin/bash -l
# Optuna short-run sweeps: IVON/EVON vs tuned-AdamW early-training target.
# Submit: sbatch run_tune.sbatch
#SBATCH --job-name=tune_vit
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
export TMPDIR="/ptmp/$USER/tmp/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"

cd /ptmp/cheins/blrax/examples
echo "host $(hostname) | $(date -Iseconds)"
srun python tune_vit_optuna.py \
    --epochs "${EPOCHS:-2}" --eval-n 2000 --objective "${OBJ:-acc}" \
    --trials-adamw "${TA:-10}" --trials-ivon "${TI:-40}" --trials-evon "${TE:-25}" \
    --out "/ptmp/cheins/blrax/examples/${OUT:-optuna_vit_results.json}"
echo "finished $(date -Iseconds)"
