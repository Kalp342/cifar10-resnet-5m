#!/bin/bash
# Round 0: the 4.62M baseline (ResNet 64/128/256, blocks 3-4-3), 200 epochs.
#   baseline_val  : 45k train / 5k held-out split -> 97.64% held-out accuracy (flip TTA)
#   baseline_full : same recipe on all 50k images (also a round-1 teacher)
set -e; cd "$(dirname "$0")/.."; source scripts/common.sh
python train.py --data "$DATA" --out "$OUT/baseline_val"  --val_size 5000 --epochs 200 --seed 42
python train.py --data "$DATA" --out "$OUT/baseline_full" --val_size 0    --epochs 200 --seed 42
