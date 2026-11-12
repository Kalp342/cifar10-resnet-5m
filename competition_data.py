"""Read the competition PNGs in CSV order and verify their decoded pixels."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "training"
DATA_HASHES = json.loads((ORIGINAL / "outputs/manifest.json").read_text())


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_images(directory, split):
    metadata = "test.csv" if split == "test" else "train_labels.csv"
    records = rows(directory / metadata)
    expected = 2000 if split == "test" else 50000
    ids = [int(row["id"]) for row in records]
    if len(ids) != expected or len(set(ids)) != expected:
        raise ValueError(f"Expected {expected} distinct IDs in {metadata}")
    pixels = np.empty((expected, 32, 32, 3), dtype=np.uint8)
    for index, row in enumerate(records):
        image_path = directory / f"{split}_images" / row["filename"]
        with Image.open(image_path) as image:
            if image.size != (32, 32):
                raise ValueError(f"Unexpected image size at {image_path}")
            pixels[index] = np.asarray(image.convert("RGB"))
    if hashlib.sha256(pixels).hexdigest() != DATA_HASHES[f"{split}_pixels_sha256"]:
        raise ValueError(f"{split} pixels or row order differ from the original run")
    if split == "test":
        sample_ids = [int(row["id"]) for row in rows(directory / "sample_submission.csv")]
        if ids != sample_ids:
            raise ValueError("Test IDs differ from sample submission order")
    return ids, pixels
