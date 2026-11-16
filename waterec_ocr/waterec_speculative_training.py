"""Train the optional shared-cross-K/V drafter without updating the AR target.

Only the original training pool is read. Character supervision comes from the
frozen teacher's free greedy predictions, never from dataset ground-truth text.
The default staged recipe selects on an internal training holdout, then freshly
fits the corresponding full-data target for that fixed number of epochs.
"""
import contextlib
import gc
import hashlib
import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from .data import audit_splits
from .data_io import scan_pairs
from .waterec_export import project_cross_kv, _validate_checkpoint_config
from .waterec_models import default_checkpoint, load_native, sha256
from .waterec_speculative_export import DraftHead
from .waterec_synthetic import generate_replay
from .waterec_training import (
    _input_fingerprint,
    _json_safe,
    _path,
    read_manifest,
    split_training_records,
)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _settings(config):
    options = dict(config.get("speculative", {}))
    device = str(config.get("device", "cpu"))
    options.setdefault("epochs", 10)
    options.setdefault("batch_size", 128)
    options.setdefault("cache_batch_size", 16)
    options.setdefault("precision", "bf16" if device.startswith("cuda") else "fp32")
    options.setdefault("cache_mode", "ram")
    options.setdefault("seed", config.get("seed", 20261114))
    for key in ("epochs", "batch_size", "cache_batch_size"):
        options[key] = int(options[key])
        if options[key] <= 0:
            raise ValueError(f"speculative.{key} must be positive")
    if options["epochs"] > 10:
        raise ValueError("This recipe uses the original ten-epoch LR schedule")
    if options.get("fixed_epoch") is not None:
        options["fixed_epoch"] = int(options["fixed_epoch"])
        if not 1 <= options["fixed_epoch"] <= 10:
            raise ValueError("speculative.fixed_epoch must be in 1..10")
    if options["cache_mode"] not in ("ram", "disk"):
        raise ValueError("speculative.cache_mode must be ram or disk")
    if options["precision"] not in ("fp32", "bf16"):
        raise ValueError("Draft training supports fp32 or CUDA bf16")
    if options["precision"] == "bf16" and not device.startswith("cuda"):
        raise ValueError("BF16 draft training requires CUDA; use fp32 on CPU")
    return options


def _training_rows(config, fixed):
    records = (
        read_manifest(
            config["train_manifest"], config.get("normalization", "upper_no_whitespace")
        )
        if config.get("train_manifest")
        else scan_pairs(
            Path(config["dataset_root"]) / "train",
            config.get("normalization", "upper_no_whitespace"),
        )
    )
    real = [row for row in records if not row.get("synthetic")]
    replay = [row for row in records if row.get("synthetic")]
    if not real:
        raise ValueError("A nonempty real training pool is required")
    split_path = config.get("split_manifest")
    if not split_path and (Path(config["output_dir"]) / "split.json").exists():
        split_path = Path(config["output_dir"]) / "split.json"
    if fixed:
        fit, held = real, []
    elif split_path:
        split = json.loads(_path(split_path).read_text())
        fit_ids, held_ids = set(split["train"]), set(split["holdout"])
        if fit_ids & held_ids:
            raise ValueError("Training/holdout split overlaps")
        member = (
            lambda row, names: row["image"] in names or Path(row["image"]).name in names
        )
        fit = [row for row in real if member(row, fit_ids)]
        held = [row for row in real if member(row, held_ids)]
        if not fit or not held or len(fit) + len(held) != len(real):
            raise ValueError("Split must partition the whole real training pool")
    else:
        fit, held = split_training_records(real, config)
    if config.get("synthetic_manifest"):
        replay += read_manifest(
            config["synthetic_manifest"],
            config.get("normalization", "upper_no_whitespace"),
        )
    count = int(config.get("synthetic_plate_count", 2048))
    if count < 0:
        raise ValueError("synthetic_plate_count must be nonnegative")
    if count and not replay:
        replay, _ = generate_replay(
            Path(config["output_dir"]) / "synthetic_replay",
            count,
            config.get("synthetic_seed", 20261114),
            _path(config["synthetic_font_dir"])
            if config.get("synthetic_font_dir")
            else None,
        )
    identifiers = [row["image"] for row in real + replay]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Real/replay identifiers must be unique")
    if held and audit_splits(fit + replay, held)["pixel_duplicates_across_splits"]:
        raise ValueError("Decoded-pixel duplicate crosses fit and internal holdout")
    return fit, held, replay


def _verify_run(output, identity):
    """Reject changed inputs before reusing a draft or overwriting its metadata."""
    guard = output / "input_identity.json"
    if guard.exists():
        if json.loads(guard.read_text()) != identity:
            raise ValueError(
                "Draft target, settings or training inputs changed; choose a new speculative output_dir"
            )
    elif list(output.glob("*/draft.pt")) or list(output.glob("*/summary.json")):
        raise ValueError(
            "Legacy draft lacks an input identity; choose a new speculative output_dir"
        )
    else:
        output.mkdir(parents=True, exist_ok=True)
        guard.write_text(json.dumps(identity, indent=2))


def _selection_provenance(target, fingerprint, fit, held):
    if target.parent.name == "full_fit":
        raise ValueError(
            "A full-fit teacher cannot be used for internal-holdout draft selection"
        )
    for folder in (target.parent, target.parent.parent):
        guard, split_path = folder / "input_fingerprint.json", folder / "split.json"
        if guard.exists() and split_path.exists():
            if json.loads(guard.read_text()) != fingerprint:
                raise ValueError(
                    "Selection teacher training content/folds differ from draft inputs"
                )
            split = json.loads(split_path.read_text())
            if split["train"] != [r["image"] for r in fit] or split["holdout"] != [
                r["image"] for r in held
            ]:
                raise ValueError(
                    "Selection teacher split differs from draft fit/held rows"
                )
            return {
                "status": "Public teacher input fingerprint and fit/held split verified",
                "input_fingerprint_sha256": fingerprint["sha256"],
                "limitation": "Provenance metadata evidence, not proof of arbitrary file history",
            }
    return {
        "status": "Unverified legacy/custom teacher provenance",
        "limitation": "No matching public input fingerprint and split; caller must supply a fit-only teacher. No prior held-exposure guarantee is made.",
    }


def _profile(target):
    config = target.config
    if (
        config["task"] != "plate"
        or config["tokens"] != 512
        or config.get("max_decode_steps", 25) != 25
        or target.decoder.tgt_word_prj.out_features != 95
    ):
        raise ValueError(
            "Validated speculative profile requires plate, 512 patches, max length25 and95 output classes"
        )


def _state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def _allocate_cache(count, output, mode):
    shapes = {
        "keys": (count, 12, 512, 32),
        "values": (count, 12, 512, 32),
        "padding": (count, 512),
        "probabilities": (count, 25, 95),
    }
    dtypes = {name: np.bool_ if name == "padding" else np.float16 for name in shapes}
    size = sum(
        math.prod(shape) * np.dtype(dtypes[name]).itemsize
        for name, shape in shapes.items()
    )
    if mode == "disk":
        output.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output).free < size + 1024 * 1024:
            raise OSError(f"Teacher cache needs approximately {size} free bytes")
        bank = {
            name: np.lib.format.open_memmap(
                output / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape
            )
            for name, shape in shapes.items()
        }
    else:
        bank = {
            name: np.empty(shape, dtype=dtypes[name]) for name, shape in shapes.items()
        }
    for name, value in bank.items():
        value.fill(name == "padding")
    return bank, size


@contextlib.contextmanager
def _teacher_math(device):
    old_precision = torch.get_float32_matmul_precision()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with torch.autocast(device.type, enabled=False):
            yield
    finally:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
            torch.backends.cudnn.allow_tf32 = old_cudnn
            torch.set_float32_matmul_precision(old_precision)


def _cache_teacher(target, rows, output, options, device):
    bank, size = _allocate_cache(len(rows), output, options["cache_mode"])
    before = _state_digest(target)
    batch = options["cache_batch_size"]
    with torch.inference_mode(), _teacher_math(device):
        for start in range(0, len(rows), batch):
            chunk = rows[start : start + batch]
            images = []
            for row in chunk:
                with Image.open(row["path"]) as source:
                    images.append(target.preprocess(source.convert("RGB")).to(device))
            memory, padding = target.encoder(images)
            keys, values = project_cross_kv(memory, target.decoder.decoder.layers[:1])
            probabilities = target.decoder.forward_test((memory, padding))
            n, length = memory.shape[:2]
            if length > 512 or probabilities.shape[1] > 25:
                raise ValueError("Teacher outputs exceed the fixed speculative profile")
            end = start + n
            bank["keys"][start:end, :, :length] = (
                keys[0].float().cpu().numpy().astype(np.float16)
            )
            bank["values"][start:end, :, :length] = (
                values[0].float().cpu().numpy().astype(np.float16)
            )
            bank["padding"][start:end, :length] = padding.cpu().numpy()
            bank["probabilities"][start:end, : probabilities.shape[1]] = (
                probabilities.float().cpu().numpy().astype(np.float16)
            )
    if before != _state_digest(target) or any(
        p.grad is not None for p in target.parameters()
    ):
        raise RuntimeError("Frozen AR target changed during teacher caching")
    for value in bank.values():
        if isinstance(value, np.memmap):
            value.flush()
    return bank, size


def _targets(probabilities):
    hard = torch.from_numpy(np.asarray(probabilities).argmax(-1).copy())
    first = torch.where(hard.eq(0), torch.arange(25)[None], 25).amin(1)
    valid = torch.arange(25)[None] <= first[:, None]
    return hard, valid


def distillation_loss(logits, hard, probabilities, valid):
    """Original first-EOS-inclusive CE + .25*T² KL, T=2; no dataset GT."""
    ce = F.cross_entropy(
        logits.float().flatten(0, 1), hard.flatten(), reduction="none"
    ).reshape(-1, 25)
    softened = probabilities.float().clamp_min(1e-9).sqrt()
    softened = softened / softened.sum(-1, keepdim=True)
    kl = (
        F.kl_div(F.log_softmax(logits.float() / 2, -1), softened, reduction="none").sum(
            -1
        )
        * 4
    )
    return ((ce + 0.25 * kl) * valid).sum() / valid.sum()


def _cache_batch(bank, indices, device):
    indices = np.asarray(indices, dtype=np.int64)
    return tuple(
        torch.from_numpy(np.array(bank[name][indices], copy=True)).to(device)
        for name in ("keys", "values", "padding", "probabilities")
    )


@torch.inference_mode()
def _evaluate_prefix(head, bank, hard, valid, begin, batch, device):
    head.eval()
    prefixes, lengths, whole = [], [], 0
    for start in range(begin, len(hard), batch):
        indices = np.arange(start, min(start + batch, len(hard)))
        keys, values, padding, _ = _cache_batch(bank, indices, device)
        prediction = head(keys.float(), values.float(), padding).argmax(-1).cpu()
        target = hard[indices]
        size = valid[indices].sum(1)
        match = prediction.eq(target).long().cumprod(1).sum(1)
        accepted = torch.minimum(match + 1, size)
        prefixes.extend(accepted.tolist())
        lengths.extend(size.tolist())
        whole += int((match >= size).sum())
    return {
        "n": len(prefixes),
        "mean_verified_tokens": float(np.mean(prefixes)),
        "teacher_whole_match": whole,
        "teacher_mean_length": float(np.mean(lengths)),
        "selection": "Held mean verified prefix including first correction, earliest tie; no GT accuracy",
    }


def _fit_phase(
    config,
    options,
    target_path,
    fit,
    held,
    folder,
    epochs,
    fixed,
    provenance,
    expected_target_sha,
):
    if sha256(target_path) != expected_target_sha:
        raise ValueError("Original AR target changed before draft training")
    checkpoint = folder / "draft.pt"
    if (folder / "summary.json").exists():
        if not checkpoint.exists():
            raise ValueError(
                "Completed draft summary lacks checkpoint; choose a new output_dir"
            )
        summary = json.loads((folder / "summary.json").read_text())
        if summary.get("draft_checkpoint_sha256") != sha256(checkpoint):
            raise ValueError(
                "Completed draft checkpoint changed or lacks an integrity hash; choose a new output_dir"
            )
        if summary.get("target_checkpoint_sha256") != sha256(target_path):
            raise ValueError(
                "Completed draft target identity changed; choose a new output_dir"
            )
        return checkpoint, summary
    if checkpoint.exists():
        raise ValueError(
            "Interrupted draft training cannot safely resume; choose a new speculative output_dir"
        )
    folder.mkdir(parents=True, exist_ok=True)
    seed = int(options["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(config.get("device", "cpu"))
    target = load_native(target_path, device="cpu").float().eval()
    if sha256(target_path) != expected_target_sha:
        raise ValueError("Original AR checkpoint changed while it was loaded")
    _profile(target)
    _validate_checkpoint_config(config, target.config)
    target.requires_grad_(False)
    target.encoder.token_dropout_rate = 0
    # Deepcopied decoder modules inherit requires_grad=False unless explicitly
    # re-enabled. The original target remains separate and frozen.
    head = DraftHead(target.decoder).requires_grad_(True)
    if not all(p.requires_grad for p in head.parameters()):
        raise RuntimeError("Copied drafter parameters must all be trainable")
    target = target.to(device)
    head = head.to(device)
    rows = fit + held
    bank, cache_bytes = _cache_teacher(
        target, rows, folder / "teacher_cache", options, device
    )
    if sha256(target_path) != expected_target_sha:
        raise ValueError("Original AR checkpoint changed while teacher data was cached")
    target_sha = expected_target_sha
    target_profile = {
        key: target.config[key]
        for key in (
            "task",
            "variant",
            "tokens",
            "max_label_length",
            "include_space",
            "normalization",
        )
    }
    del target
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    hard, valid = _targets(bank["probabilities"])
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    selected, best, history, first_gradients = 0, -1.0, [], None
    started = time.time()
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(fit))
        lr = 1e-3 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (epoch - 1) / 9)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = []
        for start in range(0, len(order), options["batch_size"]):
            indices = order[start : start + options["batch_size"]].numpy()
            keys, values, padding, probabilities = _cache_batch(bank, indices, device)
            optimizer.zero_grad(set_to_none=True)
            amp = torch.autocast(
                device.type,
                dtype=torch.bfloat16,
                enabled=options["precision"] == "bf16",
            )
            with amp:
                logits = head(keys.float(), values.float(), padding)
            loss = distillation_loss(
                logits,
                hard[indices].to(device),
                probabilities.float(),
                valid[indices].to(device),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite draft distillation loss")
            loss.backward()
            if first_gradients is None:
                first_gradients = {}
                for name, parameter in head.named_parameters():
                    if (
                        parameter.grad is None
                        or not torch.isfinite(parameter.grad).all()
                    ):
                        raise RuntimeError(
                            f"Missing/nonfinite copied drafter gradient: {name}"
                        )
                    first_gradients[name] = float(parameter.grad.norm())
                if not any(first_gradients.values()):
                    raise RuntimeError("All drafter gradients are zero")
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metric = (
            {
                "n": 0,
                "mean_verified_tokens": None,
                "selection": "Fixed epoch; no held evaluation",
            }
            if fixed
            else _evaluate_prefix(
                head, bank, hard, valid, len(fit), options["batch_size"], device
            )
        )
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "learning_rate": lr,
                "held": metric,
                "training_order_sha256": _digest(order.tolist()),
            }
        )
        if (fixed and epoch == epochs) or (
            not fixed and metric["mean_verified_tokens"] > best
        ):
            selected, best = epoch, metric["mean_verified_tokens"]
            protocol = {
                "teacher": {"sha256": target_sha},
                "target_profile": target_profile,
                "epochs": epochs,
                "selected_epoch": selected,
                "full_refit": fixed,
                "teacher_provenance": provenance,
                "cache_precision": "FP32 teacher; FP16 CPU cross-K/V and greedy-probability cache",
                "loss": "Teacher free-greedy CE + .25*T^2 KL, T2, first EOS inclusive; no dataset GT targets",
                "optimizer": "AdamW1e-3, wd.01, clip1, original ten-epoch cosine floor.1 prefix",
                "training_precision": options["precision"],
                "target_frozen": True,
                "GT_character_supervision": False,
                "selection": metric["selection"],
            }
            torch.save(
                {
                    "state_dict": {
                        k: v.detach().cpu() for k, v in head.state_dict().items()
                    },
                    "target_checkpoint_sha256": target_sha,
                    "epoch": selected,
                    "protocol": protocol,
                    "held": metric,
                },
                checkpoint,
            )
    summary = {
        "selected_epoch": selected,
        "epochs": epochs,
        "fit_n": len(fit),
        "held_n": len(held),
        "target_checkpoint_sha256": target_sha,
        "draft_checkpoint_sha256": sha256(checkpoint),
        "target_frozen": True,
        "teacher_provenance": provenance,
        "parameters": sum(p.numel() for p in head.parameters()),
        "first_step_gradient_norms": first_gradients,
        "dataset_GT_character_supervision": False,
        "cache_mode": options["cache_mode"],
        "teacher_cache_bytes": cache_bytes,
        "history": history,
        "seconds": time.time() - started,
    }
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    return checkpoint, summary


def train_speculative(config, target_checkpoint=None):
    """Return a draft checkpoint bound to the exact original AR checkpoint SHA.

    ``speculative.fixed_epoch`` skips selection and fits only the supplied/final
    target for that fixed original ten-epoch schedule prefix. Otherwise selection
    expects the fit-only AR checkpoint (public fold fingerprints are checked when available; legacy provenance is explicitly unverified), and optional full refit starts freshly from
    the corresponding full-data AR checkpoint. Only plate/512/25/95 is validated.
    RAM cache is transient (~0.79 MB per crop); disk mode uses ignored output data.
    """
    config = dict(config)
    options = _settings(config)
    if (
        config.get("task", "plate") != "plate"
        or int(config.get("tokens", config.get("max_patch_tokens", 512))) != 512
        or int(config.get("max_decode_steps", 25)) != 25
    ):
        raise ValueError("Speculative training requires the plate/512/25 profile")
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select device cpu and fp32")
    if options["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("CUDA device does not support BF16")
    fixed_epoch = options.get("fixed_epoch")
    fixed = fixed_epoch is not None
    fit, held, replay = _training_rows(config, fixed)
    output = _path(
        options.get("output_dir", Path(config["output_dir"]) / "speculative")
    )
    refit = bool(config.get("refit_all_training", True)) and not fixed
    selection_target = _path(
        options.get(
            "selection_target_checkpoint",
            Path(config["output_dir"]) / "selection/best.pt",
        )
    )
    final_target = _path(target_checkpoint or default_checkpoint(config))
    if not refit and not fixed:
        selection_target = (
            _path(target_checkpoint) if target_checkpoint else selection_target
        )
    paths = (
        {"full_fit": final_target}
        if fixed
        else {
            "selection": selection_target,
            **({"full_fit": final_target} if refit else {}),
        }
    )
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Original AR checkpoint missing: {path}; train the original model first"
            )
    target_hashes = {name: sha256(path) for name, path in paths.items()}
    if refit and target_hashes["selection"] == target_hashes["full_fit"]:
        raise ValueError(
            "Selection/full-fit teacher hashes must differ; use explicit speculative.fixed_epoch for final-only training"
        )
    fingerprint = _input_fingerprint(fit, held, replay)
    selection_provenance = (
        None
        if fixed
        else _selection_provenance(selection_target, fingerprint, fit, held)
    )
    full_provenance = {
        "status": "Full-data/fixed teacher; no held selection in this phase"
    }
    implementations = {
        name: sha256(Path(__file__).parent / name)
        for name in (
            "waterec_speculative_training.py",
            "waterec_speculative_export.py",
            "waterec_export.py",
            "waterec_models.py",
            "_vendor/waterec/encoder.py",
            "_vendor/waterec/decoder.py",
        )
    }
    identity = {
        "version": 1,
        "settings": _json_safe(options),
        "config": _json_safe(config),
        "targets": target_hashes,
        "selection_teacher_provenance": selection_provenance,
        "input_content": fingerprint,
        "implementations": implementations,
        "torch": str(torch.__version__),
        "Pillow": Image.__version__,
    }
    _verify_run(output, identity)
    if fixed:
        result, _ = _fit_phase(
            config,
            options,
            final_target,
            fit + replay,
            [],
            output / "full_fit",
            fixed_epoch,
            True,
            full_provenance,
            target_hashes["full_fit"],
        )
        return result
    result, selection = _fit_phase(
        config,
        options,
        selection_target,
        fit + replay,
        held,
        output / "selection",
        options["epochs"],
        False,
        selection_provenance,
        target_hashes["selection"],
    )
    if refit:
        result, _ = _fit_phase(
            config,
            options,
            final_target,
            fit + held + replay,
            [],
            output / "full_fit",
            selection["selected_epoch"],
            True,
            full_provenance,
            target_hashes["full_fit"],
        )
    return result
