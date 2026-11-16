"""WATERec train-only selection, fixed full refit, and native evaluation.

Training reads only the training pool. Validation/OOD manifests are accepted by
`evaluate`, never by epoch selection or full refit.
"""
import contextlib
import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset
from .data_io import scan_pairs, normalize
from .data import split_holdout, audit_splits
from .metrics import score_pairs, score_price_values
from .waterec_models import (
    create_native,
    default_checkpoint,
    TorchRecognizer,
    ROOT,
    preprocess_image,
)
from .waterec_synthetic import generate_replay


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _path(value):
    value = Path(value).expanduser()
    return value if value.is_absolute() else ROOT / value


def read_manifest(path, normalization="none"):
    """JSON list (or {'rows':list}); path relative to manifest; label or gt."""
    path = _path(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    records = records["rows"] if isinstance(records, dict) else records
    result, identifiers = [], set()
    for record in records:
        image = record.get(
            "image", Path(record["path"]).name if "path" in record else None
        )
        if image is None or image in identifiers:
            raise ValueError("Manifest identifiers must be present and unique")
        identifiers.add(image)
        source = Path(record.get("path", image)).expanduser()
        source = source if source.is_absolute() else path.parent / source
        label = record.get("label", record.get("gt"))
        if not isinstance(label, str) or not label:
            raise ValueError(f"Missing/empty manifest label for {image}")
        if not source.is_file():
            raise FileNotFoundError(source)
        result.append(
            {
                "image": image,
                "path": str(source.resolve()),
                "label": normalize(label, normalization),
                "synthetic": bool(record.get("synthetic", False)),
            }
        )
    if not result:
        raise ValueError("Manifest is empty")
    return result


class ImageDataset(Dataset):
    def __init__(self, records, model, augment=False):
        self.records, self.tokens, self.augment = (
            records,
            model.config["tokens"],
            augment,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(row["path"]) as source:
            image = source.convert("RGB")
        if self.augment:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
        return preprocess_image(image, self.tokens), row["label"], row["image"]


def _collate(batch):
    return tuple(map(list, zip(*batch)))


def _loader(records, model, config, training=False):
    return DataLoader(
        ImageDataset(records, model, training),
        batch_size=config.get("batch_size", 16),
        shuffle=training,
        num_workers=config.get("num_workers", 4),
        collate_fn=_collate,
        pin_memory=str(config.get("device", "cpu")).startswith("cuda"),
    )


def _autocast(config):
    precision = config.get("precision", "fp32")
    if precision == "fp32":
        return contextlib.nullcontext()
    if (
        precision != "bf16"
        or config.get("device") != "cuda"
        or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError(
            "BF16 training requires a supporting CUDA device; use fp32 on CPU"
        )
    return torch.autocast("cuda", dtype=torch.bfloat16)


@torch.inference_mode()
def evaluate_records(model, records, config, output=None):
    """FP32 native evaluation, preserving raw predictions and decimal punctuation."""
    model.eval()
    device = next(model.parameters()).device
    rows = []
    for images, labels, names in _loader(records, model, config):
        texts = model.decode(model([image.to(device) for image in images]))
        rows.extend(
            {"image": name, "gt": gt, "prediction": prediction}
            for name, gt, prediction in zip(names, labels, texts)
        )
    labels, predictions = [r["gt"] for r in rows], [r["prediction"] for r in rows]
    summary = score_pairs(labels, predictions, config.get("normalization", "none"))
    summary["raw_correct"] = sum(gt == pred for gt, pred in zip(labels, predictions))
    if config.get("task") == "price":
        summary.update(score_price_values(labels, predictions))
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        (output / "predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    return summary


def evaluate(config, checkpoint=None, manifest=None, output=None):
    recognizer = TorchRecognizer(config, checkpoint)
    records = (
        read_manifest(manifest, config.get("normalization", "none"))
        if manifest
        else scan_pairs(
            Path(config["dataset_root"]) / "val", config.get("normalization", "none")
        )
    )
    return evaluate_records(recognizer.model, records, config, output)


def _seed(config):
    seed = config.get("seed", 20261114)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config.get("num_threads", 4))


def _checkpoint_config(config):
    # Model files need architecture/training policy, not private dataset locations.
    excluded = {
        "dataset_root",
        "train_manifest",
        "split_manifest",
        "synthetic_manifest",
        "synthetic_font_dir",
        "output_dir",
        "pretrained_dir",
    }
    return _json_safe({k: v for k, v in config.items() if k not in excluded})


def _fit(records, held, config, folder, epochs, resume):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _seed(config)
    model = create_native(config)
    for row in records + held:
        model.encode_labels([row["label"]])  # Strict alphabet/length; never truncate.
    model = model.to(config.get("device", "cpu"))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": config.get("encoder_learning_rate", 1e-5),
            },
            {
                "params": model.decoder.parameters(),
                "lr": config.get("decoder_learning_rate", 5e-5),
            },
        ],
        weight_decay=config.get("weight_decay", 0.01),
    )
    checkpoint = folder / ("best.pt" if held else "model.pt")
    last = folder / "last.pt"
    start_epoch, best, selected, history = 1, (-1.0, -1.0), 0, []
    if last.exists():
        if not resume:
            raise FileExistsError(
                f"Training state exists: {last}; use resume or a new output_dir"
            )
        # Only local resumable state uses weights_only=False; official downloads use True + SHA256.
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best, selected = (
            state["epoch"] + 1,
            tuple(state["best"]),
            state["selected_epoch"],
        )
        history = state["history"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    started = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        frozen = epoch <= config.get("warmup_epochs", 3)
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(not frozen)
        losses = []
        for images, labels, names in _loader(records, model, config, True):
            device = next(model.parameters()).device
            images = [image.to(device, non_blocking=True) for image in images]
            target = model.encode_labels(labels, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(config):
                memory = model.encoder(images)
                logits = model.decoder.forward_train(memory, target)
                loss = torch.nn.functional.cross_entropy(
                    logits.float().flatten(0, 1),
                    target[:, 1:].flatten(),
                    ignore_index=model.decoder.ignore_index,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            optimizer.step()
            losses.append(loss.detach().item())
        row = {
            "epoch": epoch,
            "encoder_frozen": frozen,
            "loss": float(np.mean(losses)),
            "seconds_this_process": time.perf_counter() - started,
        }
        if held:
            row["holdout"] = evaluate_records(model, held, config)
            metric = row["holdout"]
            score = (
                (metric["numeric_correct"], metric["raw_correct"])
                if config["task"] == "price"
                else (metric["correct"], 0)
            )
        else:
            score = (epoch, 0)  # Fixed full refit: final epoch, no benchmark selection.
        if score > best:
            best, selected = score, epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": _checkpoint_config(config),
                    "epoch": epoch,
                },
                checkpoint,
            )
        history.append(row)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best": best,
                "selected_epoch": selected,
                "history": history,
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if str(config.get("device")).startswith("cuda")
                else None,
            },
            last,
        )
        (folder / "metrics.json").write_text(json.dumps({"history": history}, indent=2))
        print(json.dumps(row), flush=True)
    summary = {
        "epochs": epochs,
        "selected_epoch": selected,
        "training_images": len(records),
        "holdout_images": len(held),
        "checkpoint": str(checkpoint),
        "parameters": sum(p.numel() for p in model.parameters()),
        "selection": "Internal real training holdout"
        if held
        else "Fixed full refit, final epoch",
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def split_training_records(records, config):
    """Measured split rules: ALPR floor10%; price ROUND15%, both grouped by GT."""
    fraction = config.get(
        "holdout_fraction", 0.15 if config["task"] == "price" else 0.1
    )
    seed = config.get("split_seed", 20261114 if config["task"] == "price" else 20261113)
    if not 0 < fraction < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if config["task"] != "price":
        return split_holdout(records, fraction, seed)
    labels = sorted({row["label"] for row in records})
    if len(labels) < 2:
        raise ValueError("Need at least two distinct price label strings for a holdout")
    random.Random(seed).shuffle(labels)
    count = max(1, min(len(labels) - 1, round(len(labels) * fraction)))
    held = set(labels[:count])
    return (
        [row for row in records if row["label"] not in held],
        [row for row in records if row["label"] in held],
    )


def _input_fingerprint(fit, held, replay):
    """Path-independent content identity, including actual fold and row order."""
    entries = []
    for membership, records in [("fit", fit), ("held", held), ("replay", replay)]:
        for position, row in enumerate(records):
            with Image.open(row["path"]) as source:
                image = source.convert("RGB")
                width, height = image.size
                pixels = hashlib.sha256(image.tobytes()).hexdigest()
            entries.append(
                {
                    "membership": membership,
                    "position": position,
                    "image": row["image"],
                    "label": row["label"],
                    "width": width,
                    "height": height,
                    "rgb_sha256": pixels,
                }
            )
    encoded = json.dumps(
        entries, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return {
        "version": 1,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "scope": "ImageIDs, normalizedlabels, RGBpixels/geometry, foldmembership and roworder; paths excluded",
        "fit_n": len(fit),
        "held_n": len(held),
        "replay_n": len(replay),
    }


def _verify_input_fingerprint(output, fingerprint):
    """Never bless an existing model with a newly computed input identity."""
    output = Path(output)
    path = output / "input_fingerprint.json"
    if path.exists():
        if json.loads(path.read_text()) != fingerprint:
            raise ValueError(
                "Training inputs or internal split changed in-place; choose a new output_dir. Existing split/checkpoints were preserved."
            )
        return
    previous = [
        folder
        for folder in [output / "selection", output / "full_fit"]
        if folder.exists()
        and (any(folder.glob("*.pt")) or any(folder.glob("*summary.json")))
    ]
    if previous:
        raise ValueError(
            "Existing training state lacks an input fingerprint; choose a new output_dir. Legacy checkpoints cannot be safely reused."
        )
    path.write_text(json.dumps(fingerprint, indent=2) + "\n")


def train(config, resume=True):
    config = dict(config)
    if (
        str(config.get("device", "cpu")).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA unavailable; use device cpu and precision fp32")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    snapshot = _json_safe(config)
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != snapshot:
        raise ValueError("Config differs from this run; choose a new output_dir")
    config_path.write_text(json.dumps(snapshot, indent=2))
    records = (
        read_manifest(config["train_manifest"], config["normalization"])
        if config.get("train_manifest")
        else scan_pairs(Path(config["dataset_root"]) / "train", config["normalization"])
    )
    real = [r for r in records if not r.get("synthetic")]
    replay = [r for r in records if r.get("synthetic")]
    if config.get("split_manifest"):
        split = json.loads(_path(config["split_manifest"]).read_text())
        train_ids, held_ids = set(split["train"]), set(split["holdout"])
        if train_ids & held_ids:
            raise ValueError("Training/holdout manifest overlap")

        def membership(row, ids):
            return row["image"] in ids or Path(row["image"]).name in ids

        fit = [r for r in real if membership(r, train_ids)]
        held = [r for r in real if membership(r, held_ids)]
        if len(fit) + len(held) != len(real) or not fit or not held:
            raise ValueError(
                "Split manifest must partition the entire real training pool"
            )
    else:
        fit, held = split_training_records(real, config)
    if {r["image"] for r in fit} & {r["image"] for r in held}:
        raise ValueError("Internal real split overlaps")
    if config.get("synthetic_manifest"):
        replay += read_manifest(config["synthetic_manifest"], config["normalization"])
    count = int(
        config.get("synthetic_plate_count", 2048 if config["task"] == "plate" else 0)
    )
    if count < 0:
        raise ValueError("synthetic_plate_count must be nonnegative")
    replay_note = {"provided_replay": len(replay)}
    if count and not replay:
        if config["task"] != "plate":
            raise ValueError("Synthetic plate replay is only supported for task plate")
        replay, replay_note = generate_replay(
            output / "synthetic_replay",
            count,
            config.get("synthetic_seed", 20261114),
            _path(config["synthetic_font_dir"])
            if config.get("synthetic_font_dir")
            else None,
        )
    if len({r["image"] for r in real + replay}) != len(real) + len(replay):
        raise ValueError(
            "Real and synthetic identifiers overlap; give synthetic rows unique prefixes"
        )
    fingerprint = _input_fingerprint(fit, held, replay)
    _verify_input_fingerprint(output, fingerprint)
    audit = audit_splits(fit + replay, held)
    if audit["pixel_duplicates_across_splits"]:
        raise ValueError(
            "Exact decoded-pixel duplicates across fit and internal holdout"
        )
    (output / "split.json").write_text(
        json.dumps(
            {
                "train": [r["image"] for r in fit],
                "holdout": [r["image"] for r in held],
                "replay_count": len(replay),
                "replay": replay_note,
                "audit": audit,
            },
            indent=2,
        )
    )
    selection_file = output / "selection/summary.json"
    selection = (
        json.loads(selection_file.read_text())
        if selection_file.exists()
        else _fit(
            fit + replay, held, config, output / "selection", config["epochs"], resume
        )
    )
    if config.get("refit_all_training", True):
        refit_file = output / "full_fit/summary.json"
        if not refit_file.exists():
            _fit(
                real + replay,
                [],
                config,
                output / "full_fit",
                selection["selected_epoch"],
                resume,
            )
    return selection
