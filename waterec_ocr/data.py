"""Image/TXT data, label validation and plate-string-disjoint holdouts."""
from collections import defaultdict
from pathlib import Path
import hashlib
import json
import random
import shutil
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset, DataLoader
import torch

from .data_io import IMAGE_EXTENSIONS, normalize, scan_pairs


def validate_labels(records, tokenizer, max_length=25):
    supported = set(tokenizer._stoi)
    for row in records:
        unknown = set(row["label"]) - supported
        if unknown or len(row["label"]) > max_length:
            raise ValueError(f"Unsupported label in {row['image']}: unknown={sorted(unknown)}, "
                             f"length={len(row['label'])}, limit={max_length}. Labels are never truncated.")


def split_holdout(records, fraction=.1, seed=20261113):
    labels = sorted({r["label"] for r in records})
    if len(labels) < 2:
        raise ValueError("Need at least two distinct label strings for an internal holdout")
    random.Random(seed).shuffle(labels)
    held = set(labels[:max(1, min(len(labels)-1, int(len(labels)*fraction)))])
    training = [r for r in records if r["label"] not in held]
    holdout = [r for r in records if r["label"] in held]
    assert not {r["label"] for r in training} & {r["label"] for r in holdout}
    return training, holdout


def audit_splits(train, val):
    def pixels(path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            return hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    seen = defaultdict(list)
    for row in train:
        seen[pixels(row["path"])].append(row["image"])
    duplicates = [{"val": row["image"], "train": seen[h]} for row in val
                  if (h := pixels(row["path"])) in seen]
    labels = {r["label"] for r in train}
    return {"train_images": len(train), "val_images": len(val),
            "pixel_duplicates_across_splits": duplicates,
            "validation_labels_seen_in_train": sum(r["label"] in labels for r in val)}


class CropDataset(Dataset):
    def __init__(self, records, image_size=(64, 256), augment=False):
        self.records, self.image_size, self.augment = records, image_size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as im:
            image = im.convert("RGB")
        if self.augment:
            if random.random() < .5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(.7, 1.3))
            if random.random() < .5:
                image = ImageEnhance.Contrast(image).enhance(random.uniform(.7, 1.3))
            if random.random() < .25:
                image = image.rotate(random.uniform(-3, 3), Image.Resampling.BILINEAR,
                                     fillcolor=(127, 127, 127))
        h, w = self.image_size
        image = image.resize((w, h), Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()/127.5-1
        return torch.from_numpy(pixels), row["label"], row["image"]


def make_loader(records, config, augment=False):
    return DataLoader(CropDataset(records, config["image_size"], augment),
                      batch_size=config["batch_size"], shuffle=augment,
                      num_workers=config["num_workers"], pin_memory=True,
                      persistent_workers=config["num_workers"] > 0)


def save_manifest(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def prepare_from_labeled(source, destination, val_fraction=.2, seed=42,
                         normalization="upper_no_whitespace"):
    """Copy labeled crops into new train/val folders, grouping identical labels."""
    destination = Path(destination)
    if any((destination/split).exists() and any((destination/split).iterdir())
           for split in ("train", "val")):
        raise FileExistsError("Destination split is nonempty; choose a new dataset_root")
    records = scan_pairs(source, normalization)
    training, validation = split_holdout(records, val_fraction, seed)
    audit = audit_splits(training, validation)
    if audit["pixel_duplicates_across_splits"]:
        raise ValueError("Identical source pixels have conflicting labels across splits; fix labels first")
    for split, rows in (("train", training), ("val", validation)):
        folder = destination/split
        folder.mkdir(parents=True, exist_ok=True)
        for row in rows:
            shutil.copy2(row["path"], folder/row["image"])
            (folder/Path(row["image"]).with_suffix(".txt")).write_text(row["label"], encoding="utf-8")
    return audit
