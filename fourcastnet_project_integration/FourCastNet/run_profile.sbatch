#!/bin/bash
#SBATCH -J fcn_nsys
#SBATCH -A ACD115083
#SBATCH -p gp2d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:30:00
#SBATCH -o /work/r14525127/fourcastnet_project/FourCastNet/logs/fcn_nsys_%j.out
#SBATCH -e /work/r14525127/fourcastnet_project/FourCastNet/logs/fcn_nsys_%j.err

cd /work/r14525127/fourcastnet_project/FourCastNet

module load miniconda3/conda24.5.0_py3.9
conda activate fcn_baseline

module load cuda/11.7

mkdir -p /work/r14525127/fourcastnet_project/FourCastNet/profiles
mkdir -p /work/r14525127/fourcastnet_project/output

nsys profile \
  --trace=cuda,nvtx,osrt,cublas,cudnn \
  --sample=none \
  --stats=true \
  --force-overwrite=true \
  -o /work/r14525127/fourcastnet_project/FourCastNet/profiles/fcn_baseline_${SLURM_JOB_ID} \
  python inference/inference.py \
    --yaml_config ./config/AFNO.yaml \
    --config afno_backbone \
    --run_num 0 \
    --override_dir /work/r14525127/fourcastnet_project/output \
    --weights /work/r14525127/fourcastnet_project/FCN_weights_v0/backbone.ckpt
