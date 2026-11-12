#!/bin/bash
# Round 3: the round-2 recipe + SAM (rho 0.05), with kd2_student as a fifth teacher. 500 epochs, 4,918,602 params.
# Public 0.903 / private 0.870. Its soup (scripts/4_model_soup.sh) is our selected final model.
# Needs scripts/1_teachers.sh and scripts/2_distill.sh to have finished.
set -e; cd "$(dirname "$0")/.."; source scripts/common.sh
python train.py --data "$DATA" --out "$OUT/kd3_sam" --val_size 0 --epochs 500 --seed 31 --blocks 3,5,3 \
    --kd_teachers "$TEACHERS5" --kd_alpha 0.9 --kd_T 4 --sam_rho 0.05 --tta10
