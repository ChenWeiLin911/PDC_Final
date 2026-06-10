#!/bin/bash
#SBATCH -J afno_full_ncu
#SBATCH -A ACD115083
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 01:00:00
#SBATCH -o /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/afno_full_ncu_%j.out
#SBATCH -e /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/afno_full_ncu_%j.err

cd /work/r14525127/fourcastnet_project_integration/FourCastNet

mkdir -p logs/ncu_full

module load miniconda3/conda24.5.0_py3.9
module load cuda/11.7
conda activate fcn_baseline

cd afno_extension
python setup.py build_ext --inplace

cd ..

echo "========== Check ncu =========="
which ncu
ncu --version

echo "========== Profile Original Full AFNO2D =========="
ncu \
  --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --force-overwrite \
  -o logs/ncu_full/original_full_afno2d_${SLURM_JOB_ID} \
  python test_afno2d_full_ncu.py --mode original_full --repeat 1

echo "========== Profile CUDA Full AFNO2D Prototype =========="
ncu \
  --target-processes all \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --force-overwrite \
  -o logs/ncu_full/cuda_full_afno2d_${SLURM_JOB_ID} \
  python test_afno2d_full_ncu.py --mode cuda_full --repeat 1

echo "done"
