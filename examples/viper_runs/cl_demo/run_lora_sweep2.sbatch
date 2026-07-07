#!/bin/bash -l
#SBATCH --job-name=lora_sw2
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
run () {  # h0 p0
  HP="{\"ivon-cl\": {\"lr\": 0.03, \"clip\": 0.003, \"b2\": 0.99999, \"tau\": 10, \"h0\": $1, \"p0\": $2}}"
  echo "### h0=$1 p0=$2"
  srun python cl_lora.py --methods ivon-cl --n-tasks 5 --epochs 10 --num-mc 4 \
      --seeds 0 --hp "$HP" --out /tmp/lora_sw2_tmp.json 2>/dev/null | grep -E "seed0 "
}
run 2.0 1.0
run 5.0 1.0
run 5.0 10.0
echo "finished $(date -Iseconds)"
