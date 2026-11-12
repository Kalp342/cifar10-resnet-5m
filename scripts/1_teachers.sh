#!/bin/bash
# Teachers (allowed to exceed 5M parameters because they are never submitted), all on the 50k training images.
#   teacher_w1/w2   : ResNet 96/192/384, blocks 3-4-3, 10,393,834 params, 160 epochs, seeds 1 and 2
#   teacher_b11/b12 : ResNet 128/256/512, blocks 3-4-3, 18,470,538 params, 200 epochs, seeds 11 and 12
# On a machine with several GPUs, run the four lines in parallel with CUDA_VISIBLE_DEVICES=0/1/... and '&'.
set -e; cd "$(dirname "$0")/.."; source scripts/common.sh
python train.py --data "$DATA" --out "$OUT/teacher_w1"  --val_size 0 --epochs 160 --seed 1  --widths 96,192,384  --teacher
python train.py --data "$DATA" --out "$OUT/teacher_w2"  --val_size 0 --epochs 160 --seed 2  --widths 96,192,384  --teacher
python train.py --data "$DATA" --out "$OUT/teacher_b11" --val_size 0 --epochs 200 --seed 11 --widths 128,256,512 --teacher
python train.py --data "$DATA" --out "$OUT/teacher_b12" --val_size 0 --epochs 200 --seed 12 --widths 128,256,512 --teacher
