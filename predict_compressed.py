"""Rebuild the competition submission with the final EMA and two-view inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from competition_data import ROOT, read_images, sha256
from student import TeammateResNet, parameter_counts, probs

STEM = "tm_compress_seed51_ema_final"
WEIGHTS = ROOT / "checkpoints" / f"{STEM}_weights.pt"
REFERENCE = ROOT / "results" / f"{STEM}_flip.csv"
RESULTS = ROOT / "results" / "submissions.json"


def load_student(checkpoint=WEIGHTS):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TeammateResNet().eval()
    model.load_state_dict(saved["state"], strict=True)
    if parameter_counts(model)["trainable"] != saved["parameters"] or saved["parameters"] != 4918602:
        raise ValueError("Unexpected student size")
    return model, saved["stats"]


def reproduce(data, checkpoint=WEIGHTS):
    model, stats = load_student(checkpoint)
    ids, images = read_images(Path(data), "test")
    probabilities = probs(model, images, stats)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1):
        raise ValueError("Invalid probability distribution")
    labels = probabilities.argmax(1)
    lines = ["id,label"] + [f"{index},{label}" for index, label in zip(ids, labels, strict=True)]
    content = ("\n".join(lines) + "\n").encode("ascii")
    if Path(checkpoint).resolve() == WEIGHTS.resolve():
        recorded = json.loads(RESULTS.read_text())["compressed"]
        if sha256(WEIGHTS) != recorded["weights_sha256"] or sha256(REFERENCE) != recorded["csv_sha256"]:
            raise ValueError("Published artifact checksum mismatch")
        if content != REFERENCE.read_bytes():
            raise ValueError("Regenerated CSV differs from the submitted file")
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / REFERENCE.name)
    args = parser.parse_args()
    torch.set_num_threads(4)
    content = reproduce(args.data, args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and args.output.read_bytes() != content:
        raise FileExistsError(f"Choose a new output path: {args.output}")
    args.output.write_bytes(content)
    print(f"Saved 2,000 predictions to {args.output}\nSHA-256 {sha256(args.output)}")


if __name__ == "__main__":
    main()
