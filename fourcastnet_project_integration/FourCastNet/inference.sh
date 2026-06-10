python inference/inference.py \
  --yaml_config ./config/AFNO.yaml \
  --config afno_backbone \
  --run_num 0 \
  --weights ../FCN_weights_v0/backbone.ckpt \
  --override_dir ./outputs/inference_backbone \
  --vis \
  --benchmark_model