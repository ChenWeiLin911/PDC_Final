#!/bin/bash
#SBATCH -J afno_memory
#SBATCH -A ACD115083
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:20:00
#SBATCH -o /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/afno_memory_%j.out
#SBATCH -e /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/afno_memory_%j.err

cd /work/r14525127/fourcastnet_project_integration/FourCastNet

mkdir -p logs

module load miniconda3/conda24.5.0_py3.9
module load cuda/11.7
conda activate fcn_baseline

cd afno_extension
python setup.py build_ext --inplace

cd ..
python test_afno2d_memory.py
