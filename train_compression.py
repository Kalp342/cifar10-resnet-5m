"""Prepare the original compression trainer's input layout and run its seed-51 preset."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from competition_data import DATA_HASHES, ORIGINAL, ROOT, read_images, sha256


def prepare(data, check_only):
    for split, metadata, digest_key in (("train", "train_labels.csv", "train_labels_sha256"),
                                        ("test", "test.csv", "test_csv_sha256")):
        if sha256(data / metadata) != DATA_HASHES[digest_key]:
            raise ValueError(f"Original competition file required: {metadata}")
        _, pixels = read_images(data, split)
        if not check_only:
            shutil.copyfile(data / metadata, ORIGINAL / metadata)
            cache = ORIGINAL / "outputs" / f"{split}_images.npy"
            pending = cache.with_suffix(".pending")
            with pending.open("wb") as stream:
                np.save(stream, pixels)
            pending.replace(cache)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "compression")
    parser.add_argument("--check-inputs", action="store_true", help="Validate the inputs, preset and teacher states, then exit")
    args = parser.parse_args()
    os.environ["PROJECT1_ROOT"] = str(ORIGINAL)
    sys.path[:0] = [str(ORIGINAL / "outputs/replica_kd"), str(ORIGINAL)]
    import replica_train as trainer

    cfg = trainer.presets()["colab_compress"]
    if cfg != json.loads((ORIGINAL / "config.json").read_text()):
        raise ValueError("Preset differs from the config stored in the competition checkpoint")
    expected = json.loads((ORIGINAL / "teacher_hashes.json").read_text())
    for name, digest in expected.items():
        if sha256(ORIGINAL / "teammate" / name) != digest:
            raise ValueError(f"Teacher checkpoint digest mismatch: {name}")
    if not args.check_inputs and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError("Compression training requires a CUDA GPU with BF16 support.")
    prepare(args.data, args.check_inputs)
    if args.check_inputs:
        for name in cfg["compress_teachers"]:
            trainer.load_teachers([name], torch.device("cpu"))
        trainer.TeammateResNet().load_state_dict(trainer.teammate_student_state(cfg["init"]))
        print("Verified 50,000 labeled rows, 2,000 test images, five teacher states and student initialization.")
        return
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Use a fresh training output directory: {out}")
    subprocess.run([sys.executable, str(ORIGINAL / "outputs/replica_kd/replica_train.py"),
                    "--preset", "colab_compress", "--out", str(out)], check=True)
    final = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    if final["completed_epochs"] != 40:
        raise RuntimeError("The 40-epoch run did not complete.")
    torch.save({"state": final["ema"], "stats": final["stats"], "parameters": cfg["parameters_expected"],
                "architecture": cfg["architecture"]}, out / "ema_weights.pt")


if __name__ == "__main__":
    main()
