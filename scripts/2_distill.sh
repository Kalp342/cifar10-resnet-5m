#!/bin/bash
# Knowledge distillation: loss = 0.9 * KL(student || teachers' averaged softmax at T=4) + 0.1 * cross-entropy.
#   kd_student  (round 1): 4.62M student (3-4-3), 300 epochs, teachers w1, w2 + the two baselines  -> public 0.904
#   kd2_student (round 2): 4.92M student (3-5-3), 500 epochs, teachers b11, b12, w1, w2            -> public 0.904 (also a round-3 teacher)
# Needs scripts/0_baselines.sh and scripts/1_teachers.sh to have finished.
set -e; cd "$(dirname "$0")/.."; source scripts/common.sh
R1="$OUT/teacher_w1/model_final.pt@96,192,384@3,4,3;$OUT/teacher_w2/model_final.pt@96,192,384@3,4,3"
R1="$R1;$OUT/baseline_full/model_final.pt@64,128,256@3,4,3;$OUT/baseline_val/model_final.pt@64,128,256@3,4,3"
python train.py --data "$DATA" --out "$OUT/kd_student"  --val_size 0 --epochs 300 --seed 7 \
    --kd_teachers "$R1" --kd_alpha 0.9 --kd_T 4
python train.py --data "$DATA" --out "$OUT/kd2_student" --val_size 0 --epochs 500 --seed 21 --blocks 3,5,3 \
    --kd_teachers "$TEACHERS4" --kd_alpha 0.9 --kd_T 4 --tta10
