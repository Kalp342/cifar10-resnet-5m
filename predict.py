"""Write a Kaggle submission (id,label) from a trained checkpoint.

    python predict.py --checkpoint checkpoints/soup_kd3_model_final.pt --data data --out submission.csv

--data is the competition folder: it must contain test.csv (id,filename) and a test_images/ folder of 32x32 PNGs
(nested folders are fine; both are searched for). Runs on a GPU when one is available, otherwise on the CPU
(a few seconds on a GPU, 2-3 minutes on a CPU for the 2,000 test images with 10-view TTA).
"""
import argparse, glob, os, time
import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import load_model, predict, count_params, CONFIGS


def find(root, name, want_dir=False):
    hits = [p for p in glob.glob(os.path.join(root, "**", name), recursive=True) if os.path.isdir(p) == want_dir]
    if not hits:
        raise FileNotFoundError(f"could not find {name} under {root}")
    return sorted(hits, key=len)[-1] if want_dir else hits[0]   # deepest folder handles test_images/test_images


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/soup_kd3_model_final.pt")
    ap.add_argument("--arch", default=None, choices=list(CONFIGS),
                    help="architecture name; inferred from the checkpoint file name when omitted")
    ap.add_argument("--data", default="data", help="folder containing test.csv and test_images/")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--views", type=int, default=10, choices=[1, 2, 10],
                    help="test-time augmentation: 1 = none, 2 = + horizontal flip, 10 = flip x 5 shifts (submitted)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    net = load_model(args.checkpoint, name=args.arch, device=args.device)
    print(f"loaded {os.path.basename(args.checkpoint)}: {count_params(net):,} trainable parameters on {args.device}")

    test = pd.read_csv(find(args.data, "test.csv"))
    folder = find(args.data, "test_images", want_dir=True)
    images = np.stack([np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")) for f in test["filename"]])
    assert images.shape[1:] == (32, 32, 3), f"expected 32x32 RGB images, got {images.shape[1:]}"

    probs = predict(net, images, views=args.views)
    labels = probs.argmax(1)
    pd.DataFrame({"id": test["id"], "label": labels}).to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(labels)} predictions, class counts {np.bincount(labels, minlength=10).tolist()}, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
