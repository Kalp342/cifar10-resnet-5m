#!/bin/bash
# Round 4: uniform model soups. Four copies of a student are fine-tuned from its final weights for 30 epochs each
# (peak learning rates 0.02-0.05, seeds 101-104, SAM rho 0.05, same distillation), then every weight and BatchNorm
# statistic is averaged into one network with the same 4,918,602 parameters (soup.py).
#   soup_kd3 : soup of the round-3 SAM student, 5 teachers -> public 0.905 / private 0.867   <- SELECTED FINAL MODEL
#   soup_kd2 : soup of the round-2 student, 4 teachers     -> public 0.905 / private 0.865   (starting point of round 6)
# Usage: bash scripts/4_model_soup.sh            (both soups)
#        bash scripts/4_model_soup.sh kd3        (only the final one)
set -e; cd "$(dirname "$0")/.."; source scripts/common.sh
make_soup () {   # $1 = base run, $2 = teachers, $3 = output name
  local LRS=(0.02 0.03 0.04 0.05) ING=""
  for k in 1 2 3 4; do
    python train.py --data "$DATA" --out "$OUT/${3}_ft$k" --init "$OUT/$1/model_final.pt" --val_size 0 \
        --epochs 30 --lr ${LRS[$((k-1))]} --warmup_epochs 1 --cutmix_off_epochs 10 --seed $((100+k)) --blocks 3,5,3 \
        --kd_teachers "$2" --kd_alpha 0.9 --kd_T 4 --sam_rho 0.05
    ING="$ING,$OUT/${3}_ft$k/model_final.pt"
  done
  python soup.py --data "$DATA" --out "$OUT/$3" --ingredients "${ING#,}" --base "$OUT/$1/model_final.pt"
}
make_soup kd3_sam "$TEACHERS5" soup_kd3
if [ "$1" != "kd3" ]; then make_soup kd2_student "$TEACHERS4" soup_kd2; fi
