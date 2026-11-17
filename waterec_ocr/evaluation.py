"""The same crop list and metric for every inference backend."""
from pathlib import Path
import json
from PIL import Image
from .data_io import scan_pairs
from .metrics import score_pairs


def evaluate_folder(recognizer, folder, normalization="none", batch_size=8, output=None, price_numeric=False):
    if price_numeric:
        raise ValueError("Only literal-string exact reporting is supported")
    records = scan_pairs(folder, "none")
    predictions = []
    for start in range(0, len(records), batch_size):
        images = []
        for row in records[start:start+batch_size]:
            with Image.open(row["path"]) as image:
                images.append(image.convert("RGB"))
        predictions.extend(recognizer.infer(images))
    summary = score_pairs([r["label"] for r in records], predictions)
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output/"summary.json").write_text(json.dumps(summary, indent=2))
        (output/"predictions.jsonl").write_text("".join(json.dumps({"image": r["image"],
            "gt": r["label"], "prediction": pred})+"\n" for r, pred in zip(records, predictions)))
    return summary
