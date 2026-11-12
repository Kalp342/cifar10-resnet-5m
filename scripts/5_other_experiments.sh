#!/bin/bash
# The other experiments in the report (not needed for the final model).
#   sam_val : baseline_val + SAM on the same split and seed -> 97.76% held-out (vs 97.64% without SAM)
#   highres : round 5, stage strides 1,1,2 -> same 4,918,602 params, 3.1x compute per image (public 0.903 / private 0.870)
set -e; cd "$(dirname "$0")/.."; source scripts/common.sh
python train.py --data "$DATA" --out "$OUT/sam_val" --val_size 5000 --epochs 200 --seed 42 --sam_rho 0.05
python train.py --data "$DATA" --out "$OUT/highres" --val_size 0 --epochs 300 --seed 41 --blocks 3,5,3 --strides 1,1,2 \
    --kd_teachers "$TEACHERS4" --kd_alpha 0.9 --kd_T 4 --tta10
