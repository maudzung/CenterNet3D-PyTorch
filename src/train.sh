#!/usr/bin/env bash

python train.py \
  --root-dir '../' \
  --saved_fn 'centernet3d' \
  --arch 'centernet3d' \
  --batch_size 2 \
  --num_workers 2 \
  --gpu_idx 0 \
  --no-val
