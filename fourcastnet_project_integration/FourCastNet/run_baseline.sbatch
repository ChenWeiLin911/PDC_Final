#!/bin/bash
#SBATCH -J fcn_shape
#SBATCH -A ACD115083
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/fcn_shape_%j.out
#SBATCH -e /work/r14525127/fourcastnet_project_integration/FourCastNet/logs/fcn_shape_%j.err

cd /work/r14525127/fourcastnet_project_integration/FourCastNet

module load miniconda3/conda24.5.0_py3.9
conda activate fcn_baseline

mkdir -p /work/r14525127/fourcastnet_project_integration/FourCastNet/logs
mkdir -p /work/r14525127/fourcastnet_project_integration/output

python inference/inference.py \
  --yaml_config ./config/AFNO.yaml \
  --config afno_backbone \
  --run_num 0 \
  --override_dir /work/r14525127/fourcastnet_project_integration/output \
  --weights /work/r14525127/fourcastnet_project_integration/FCN_weights_v0/backbone.ckpt
