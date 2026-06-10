#!/bin/bash
#SBATCH -J afno_latency
#SBATCH -A ACD115083
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o logs/afno_latency_%j.out
#SBATCH -e logs/afno_latency_%j.err

set -eo pipefail

# Run from this script's directory instead of a hard-coded /work/r14525127 path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs

# Build the CUDA extension with the same kind of environment used by the working
# FourCastNet custom-kernel tests. PyTorch needs GCC >= 9, and this env uses cu121.
module load gcc10/10.2.1
module load cuda/12.8 || module load cuda/12.1 || module load cuda

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_BIN="$(dirname "$(command -v nvcc)")"
    export CUDA_HOME="$(dirname "$CUDA_BIN")"
  elif [[ -d /work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.8 ]]; then
    export CUDA_HOME=/work/HPC_SYS/twnia2/pkg-rocky8/nvidia/cuda/cuda-12.8
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  fi
fi

export CC="$(which gcc)"
export CXX="$(which g++)"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0}"
export MAX_JOBS="${MAX_JOBS:-2}"

python - <<'PYINFO'
import os, shutil, subprocess, torch
print('python torch:', torch.__version__)
print('torch cuda:', torch.version.cuda)
print('CUDA_HOME:', os.environ.get('CUDA_HOME'))
print('CC:', os.environ.get('CC'))
print('CXX:', os.environ.get('CXX'))
print('nvcc:', shutil.which('nvcc'))
if shutil.which('nvcc'):
    print(subprocess.check_output(['nvcc', '--version'], text=True).splitlines()[-1])
PYINFO

cd afno_extension
python setup.py build_ext --inplace

cd ..
python test_afno2d_latency.py   --warmup 20   --iters 100   --csv logs/afno2d_latency_${SLURM_JOB_ID:-manual}.csv
