#!/usr/bin/env bash

srun -N 1 -n 1 --gpus-per-node 1 -A ACD115083 -t 00:20:00 python inference/inference.py \
  --yaml_config ./config/AFNO.yaml \
  --config afno_backbone_sep_bmm \
  --run_num 0 \
  --weights ../FCN_weights_v0/backbone.ckpt \
  --override_dir ./outputs/sep_bmm_compile_maxautotune_inference \
  --afno2d_impl sep_bmm \
  --compile_model \
  --compile_mode max-autotune \
  --warmup_model \
  --benchmark_model \
  --vis
