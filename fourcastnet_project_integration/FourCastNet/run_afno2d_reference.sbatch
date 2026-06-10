#!/bin/bash
#SBATCH -J afno2d_ref
#SBATCH -A ACD115083
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:10:00
#SBATCH -o /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/afno2d_ref_%j.out
#SBATCH -e /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/afno2d_ref_%j.err

cd /work/r14525127/fourcastnet_project_integration/FourCastNet

mkdir -p logs

module load miniconda3/conda24.5.0_py3.9
conda activate fcn_baseline

python test_afno2d_reference.py
