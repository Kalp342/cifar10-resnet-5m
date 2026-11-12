# Shared settings for the training scripts. Override on the command line, e.g.  DATA=/path/to/data bash scripts/2_distill.sh
DATA=${DATA:-data}      # competition folder: train_labels.csv, train_images/, test.csv, test_images/
OUT=${OUT:-runs}        # every run writes to $OUT/<name>/ (log.txt, model_final.pt, submission*.csv, curves.png)
mkdir -p "$OUT"
# the four teachers used from round 2 on (all trained from scratch by scripts/1_teachers.sh)
TEACHERS4="$OUT/teacher_b11/model_final.pt@128,256,512@3,4,3;$OUT/teacher_b12/model_final.pt@128,256,512@3,4,3"
TEACHERS4="$TEACHERS4;$OUT/teacher_w1/model_final.pt@96,192,384@3,4,3;$OUT/teacher_w2/model_final.pt@96,192,384@3,4,3"
# round 3 and its soup add the round-2 student as a fifth teacher
TEACHERS5="$TEACHERS4;$OUT/kd2_student/model_final.pt@64,128,256@3,5,3"
