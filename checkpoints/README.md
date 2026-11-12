# Checkpoints

`soup_kd3_model_final.pt` (19.7 MB) is one of our two final Kaggle picks: the soup of the round-3 SAM student,
4,918,602 trainable parameters, public 0.905 / private 0.867 with 10-view test-time augmentation. Its predictions
reproduce `results/submission_soup_kd3_tta10.csv` exactly. It is a plain PyTorch `state_dict` for
`model.ResNet(widths=(64,128,256), blocks=(3,5,3))`; `model.load_model()` builds the network and loads it, and
`predict.py` turns it into a submission.

The teachers and the other students are too large to keep in git comfortably (the 18.5M teachers are 74 MB each).
They are available here: **<add the Google Drive link to cifar10_checkpoints_final.zip>**

| file | architecture (widths / blocks) | params |
|---|---|---|
| `teacher_b11_model_final.pt`, `teacher_b12_model_final.pt` | 128,256,512 / 3,4,3 | 18,470,538 |
| `teacher_w1_model_final.pt`, `teacher_w2_model_final.pt` | 96,192,384 / 3,4,3 | 10,393,834 |
| `kd2_student_model_final.pt` (round 2) | 64,128,256 / 3,5,3 | 4,918,602 |
| `soup_kd2_model_final.pt` (soup of round 2, starting point of round 6) | 64,128,256 / 3,5,3 | 4,918,602 |

Normalization (not stored in the checkpoints): per-channel mean `(0.4914, 0.4822, 0.4465)` and std
`(0.2470, 0.2435, 0.2616)` of the 50,000 training images, applied after scaling pixels to [0, 1]. Full-precision
values are in `model.py`.
