"""Image/TXT reading and normalization without training-framework imports."""
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize(text, mode="upper_no_whitespace"):
    if mode == "upper_no_whitespace":
        return "".join(text.upper().split())
    if mode == "none":
        return text.rstrip("\r\n")
    raise ValueError(f"Unknown normalization: {mode}")


def scan_pairs(folder, normalization="upper_no_whitespace"):
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")
    records, stems = [], set()
    for image in sorted(folder.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if image.stem in stems:
            raise ValueError(f"Multiple image extensions share a label stem: {image.stem}")
        stems.add(image.stem)
        label = image.with_suffix(".txt")
        if not label.is_file():
            raise ValueError(f"Missing label: {label.name}")
        text = normalize(label.read_text(encoding="utf-8"), normalization)
        if not text:
            raise ValueError(f"Empty label: {label.name}")
        with Image.open(image) as im:
            im.verify()
        records.append({"image": image.name, "path": str(image), "label": text})
    if not records:
        raise ValueError(f"No image/TXT pairs found in {folder}")
    return records

