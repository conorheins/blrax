#!/bin/bash -l
# equimo ViT on CIFAR — Viper-GPU (AMD MI300A, ROCm).
# Smoke: sbatch --partition=apudev --time=00:15:00 --export=ALL,EPOCHS=1,NUMMC=4,OUT=smoke_eq.json run_vit_equimo.sbatch
# Real:  sbatch --export=ALL,EPOCHS=20,NUMMC=32,OUT=results_vit_equimo.json run_vit_equimo.sbatch
#SBATCH --job-name=vit_equimo
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
export TMPDIR="/ptmp/$USER/tmp/${SLURM_JOB_ID:-0}"; mkdir -p "$TMPDIR"

DATASET="${DATASET:-cifar100}"; EPOCHS="${EPOCHS:-20}"; NUMMC="${NUMMC:-32}"; OUT="${OUT:-results_vit_equimo.json}"

cd /ptmp/cheins/blrax/examples
echo "host $(hostname) | $(date -Iseconds) | dataset=$DATASET epochs=$EPOCHS"
srun python vit_cifar_equimo.py --dataset "$DATASET" --epochs "$EPOCHS" --num-mc "$NUMMC" \
    --out "/ptmp/cheins/blrax/examples/$OUT"
echo "=== RESULTS ($OUT) ==="
cat "/ptmp/cheins/blrax/examples/$OUT"
echo
echo "finished $(date -Iseconds)"
