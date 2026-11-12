# Checkpoints

| Selected submission | File | Inference |
|---|---|---|
| Compressed student, 0.906 public / 0.875 private | [Final EMA](tm_compress_seed51_ema_final_weights.pt) | `predict_compressed.py`, original and horizontal flip |
| Round-3 student soup, 0.905 public / 0.867 private | [KD3 soup](soup_kd3_model_final.pt) | `predict.py`, ten views |

Both predictors contain 4,918,602 trainable parameters. The compressed export stores the state dictionary and normalization together. The soup uses a plain EMA state dictionary with normalization defined in [model.py](../model.py). [Submitted predictions and SHA-256 hashes](../results/submissions.json)

The [round-6 training inputs](../training/teammate/) include teachers b11 and b12 (18,470,538 parameters each), w1 and w2 (10,393,834 each), and the 4,918,602-parameter round-2 KD2 soup. KD2 initializes the compressed student and contributes its predictions to the five-model teacher target. The selected soup checkpoint above is KD3.
