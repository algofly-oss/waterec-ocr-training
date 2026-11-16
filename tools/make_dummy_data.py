"""Generate fictional image/TXT examples without changing an existing dataset."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dummy_data")
    output = parser.parse_args().output.resolve()
    manifest = output / "DUMMY_DATA.json"
    if output.exists():
        if not manifest.is_file():
            raise FileExistsError("Existing folder is not this generated dummy dataset")
        saved = json.loads(manifest.read_text())
        for name, digest in saved["files"].items():
            if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Changed fixture: {name}; existing data was preserved")
        print(output)
        return
    output.mkdir(parents=True)
    files = {}
    for task in ("plates", "prices"):
        for split, count in (("train", 16), ("val", 4)):
            folder = output / task / split
            folder.mkdir(parents=True)
            for i in range(count):
                n = i + (100 if split == "val" else 0)
                if task == "plates":
                    label = f"AB{n:03d}CD{n + 123:03d}"
                    two_lines = i % 3 == 0
                    image = Image.new("RGB", (320, 128 if two_lines else 96), (245, 240, 210))
                    draw = ImageDraw.Draw(image)
                    if two_lines:
                        draw.text((75, 5), label[:5], font=font(42), fill="black")
                        draw.text((75, 61), label[5:], font=font(42), fill="black")
                    else:
                        draw.text((20, 21), label, font=font(38), fill="black")
                else:
                    whole, fraction = str(n + 7), f"{(n * 13 + 25) % 100:02d}"
                    suffix = " DH" if i % 2 == 0 else ""
                    label = f"{whole}.{fraction}{suffix}"
                    image = Image.new("RGB", (320, 160), (250, 235, 90))
                    draw = ImageDraw.Draw(image)
                    draw.text((25, 45), whole, font=font(64), fill="black")
                    right = 25 + draw.textlength(whole, font=font(64))
                    draw.text((right, 36 if i % 2 else 55), "." + fraction,
                              font=font(24 if i % 2 else 34), fill="black")
                    if suffix:
                        draw.text((240, 109), "DH", font=font(20), fill="black")
                image.save(folder / f"{i:03d}.png")
                (folder / f"{i:03d}.txt").write_text(label + "\n", encoding="utf-8")
                for path in (folder / f"{i:03d}.png", folder / f"{i:03d}.txt"):
                    files[str(path.relative_to(output))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps({
        "purpose": "Fictional workflow examples, never accuracy benchmarks",
        "annotation": "Image and same-stem UTF-8 expected text only; no boxes",
        "counts_per_task": {"train": 16, "val": 4},
        "examples": ["single/two-line plates", "mixed-size price digits and currency"],
        "files": files,
    }, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
