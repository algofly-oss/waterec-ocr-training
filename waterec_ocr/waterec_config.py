"""Configuration for the separate WATERec model family."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/waterec_alpr.yaml"):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("dataset_root", "output_dir", "pretrained_dir"):
        value = Path(config[key]).expanduser()
        config[key] = value if value.is_absolute() else ROOT / value
    if config.get("task") not in ("plate", "price"):
        raise ValueError("task must be plate or price")
    if config.get("variant") not in ("R", "RS"):
        raise ValueError("variant must be R or RS")
    if type(config.get("tokens")) is not int or config["tokens"] < 64:
        raise ValueError("tokens must be an integer >=64; deployment needs target-device validation")
    for key in ("epochs", "batch_size"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= config.get("warmup_epochs", 3) <= config["epochs"]:
        raise ValueError("warmup_epochs must be between zero and epochs")
    if not 0 < config.get("holdout_fraction", 0.1) < 1:
        raise ValueError("holdout_fraction must be between zero and one")
    if config.get("normalization") not in ("none", "upper_no_whitespace"):
        raise ValueError("Unsupported annotation-normalization metadata")
    if config.get("device") not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if config.get("precision") not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if not 1 <= config.get("max_label_length", 24) <= 25:
        raise ValueError("max_label_length must be between 1 and 25")
    if config.get("synthetic_plate_count", 0) < 0:
        raise ValueError("synthetic_plate_count must be nonnegative")
    config.setdefault("max_label_length", 24)
    config.setdefault("max_decode_steps", 25)
    if not 2 <= config["max_decode_steps"] <= 25:
        raise ValueError("max_decode_steps must be in 2..25, including EOS")
    return config
