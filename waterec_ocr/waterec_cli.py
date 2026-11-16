"""WATERec training and cached TensorRT deployment, separate from PARSeq."""
import argparse
import hashlib
import json
from pathlib import Path
from .waterec_config import load_config


def deployment_metadata(config, orientation=False):
    folder = config["output_dir"] / (
        "deployment_orientation" if orientation else "deployment"
    )
    metadata = json.loads((folder / "metadata.json").read_text())
    expected = {
        "task": config["task"],
        "variant": config["variant"],
        "max_tokens": config["tokens"],
        "max_len": config.get("max_decode_steps", 25),
        "normalization": config["normalization"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"Deployment/config mismatch for {key}: {metadata.get(key)!r} != {value!r}; "
                "use the matching preset and deployment"
            )
    if metadata.get("family") != "waterec" or not isinstance(metadata.get("max_tokens"), int) or metadata.get("max_tokens", 0) < 64:
        raise ValueError(
            "Cached WATERec deployment requires a patch budget>=64"
        )
    if orientation and not metadata.get("orientation"):
        raise ValueError(
            "Orientation was requested but this deployment has no orientation head"
        )
    return folder, metadata


def make_recognizer(config, backend, checkpoint=None):
    if config.get("speculative", False) and backend != "tensorrt":
        raise ValueError("--speculative requires --backend tensorrt")
    if config.get("gpu_preprocess", False) and backend != "tensorrt":
        raise ValueError("--gpu-preprocess requires --backend tensorrt")
    if config.get("orientation", False):
        if backend != "tensorrt":
            raise ValueError("--orientation currently requires --backend tensorrt")
        folder, metadata = deployment_metadata(config, orientation=True)
        verify_requested_checkpoint(metadata, checkpoint)
        from .waterec_orientation import OrientationRecognizer

        return OrientationRecognizer(
            folder,
            enabled=True,
            gpu_preprocess=config.get("gpu_preprocess", False),
            speculative_deployment=speculative_directory(config)
            if config.get("speculative", False)
            else None,
            speculative_shared_workspace=config.get(
                "speculative_shared_workspace", True
            ),
        )
    if backend == "tensorrt":
        folder, metadata = deployment_metadata(config)
        verify_requested_checkpoint(metadata, checkpoint)
        if config.get("speculative", False):
            from .waterec_speculative import SpeculativeRecognizer

            return SpeculativeRecognizer(
                folder,
                speculative_directory(config),
                gpu_preprocess=config.get("gpu_preprocess", False),
                share_workspace=config.get("speculative_shared_workspace", True),
            )
        from .waterec_runtime import TensorRTRecognizer

        if config.get("gpu_preprocess", False):
            from .waterec_gpu_preprocessing import GpuInputRecognizer

            return GpuInputRecognizer(folder)
        return TensorRTRecognizer(folder)
    if backend == "onnx":
        folder, metadata = deployment_metadata(config)
        verify_requested_checkpoint(metadata, checkpoint)
        from .waterec_onnx import ONNXRecognizer

        return ONNXRecognizer(folder)
    from .waterec_models import TorchRecognizer

    return TorchRecognizer(config, checkpoint)


def speculative_directory(config):
    return Path(
        config.get(
            "speculative_dir", config["output_dir"] / "deployment" / "speculative"
        )
    )


def verify_requested_checkpoint(metadata, checkpoint):
    if checkpoint is None:
        return
    digest = hashlib.sha256()
    with Path(checkpoint).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != metadata.get("checkpoint_sha256"):
        raise ValueError(
            "Requested checkpoint differs from the deployed engine; export/build it first"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "train",
            "infer",
            "export",
            "build",
            "evaluate",
            "benchmark",
            "train-orientation",
            "export-orientation",
            "build-orientation",
            "train-speculative",
            "export-speculative",
            "build-speculative",
        ],
    )
    parser.add_argument("--config", default="configs/waterec_alpr.yaml")
    parser.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--predecoded",
        action="store_true",
        help="Benchmark excludes image-file decoding; reported explicitly",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--gpu-preprocess",
        action="store_true",
        help="Use exact bicubic GPU resize/normalization with TensorRT; requires cc and nvcc",
    )
    parser.add_argument(
        "--orientation",
        action="store_true",
        help="Explicitly enable the separately built orientation pipeline",
    )
    parser.add_argument(
        "--fit-manifest",
        type=Path,
        help="JSON list of training-only image records with path fields",
    )
    parser.add_argument(
        "--held-manifest",
        type=Path,
        help="Disjoint training-only holdout records for orientation selection",
    )
    parser.add_argument("--orientation-head", type=Path)
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Opt in to experimental shared-KV speculative TensorRT decoding",
    )
    parser.add_argument(
        "--no-shared-workspace",
        action="store_true",
        help="Diagnostic: allocate separate speculative TensorRT workspaces",
    )
    parser.add_argument("--draft-checkpoint", type=Path)
    parser.add_argument("--speculative-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command in ("infer", "evaluate", "benchmark"):
        config["orientation"] = args.orientation
        config["gpu_preprocess"] = args.gpu_preprocess
        config["speculative"] = args.speculative
        config["speculative_shared_workspace"] = not args.no_shared_workspace
    if args.speculative_dir is not None:
        config["speculative_dir"] = args.speculative_dir
    if args.no_shared_workspace and not args.speculative:
        parser.error("--no-shared-workspace requires --speculative")
    if args.speculative and args.command not in ("infer", "evaluate", "benchmark"):
        parser.error("--speculative applies only to infer, evaluate and benchmark")
    if args.gpu_preprocess and args.command not in ("infer", "evaluate", "benchmark"):
        parser.error("--gpu-preprocess applies only to infer, evaluate and benchmark")
    if args.orientation and args.command not in ("infer", "evaluate", "benchmark"):
        parser.error("--orientation applies only to infer, evaluate and benchmark")
    if args.command == "train-speculative":
        from .waterec_speculative_training import train_speculative

        result = {"draft_checkpoint": str(train_speculative(config, args.checkpoint))}
    elif args.command == "export-speculative":
        if args.draft_checkpoint is None:
            parser.error("export-speculative requires --draft-checkpoint")
        from .waterec_speculative_export import export_speculative

        result = {
            "deployment": str(
                export_speculative(config, args.draft_checkpoint, args.checkpoint)
            )
        }
    elif args.command == "build-speculative":
        from .waterec_build import build_speculative

        result = {
            "deployment": str(
                build_speculative(speculative_directory(config), force=args.force)
            )
        }
    elif args.command == "train-orientation":
        if args.fit_manifest is None or args.held_manifest is None:
            parser.error(
                "train-orientation requires --fit-manifest and --held-manifest from training data"
            )
        from .waterec_orientation import train_head
        from .waterec_models import default_checkpoint

        def records(path):
            rows = json.loads(path.read_text())
            if not isinstance(rows, list) or not rows:
                raise ValueError("Orientation manifest must be a nonempty JSON list")
            for row in rows:
                image = Path(row["path"]).expanduser()
                row["path"] = str(
                    image
                    if image.is_absolute()
                    else (path.resolve().parent / image).resolve()
                )
            return rows

        result = {
            "head": str(
                train_head(
                    args.checkpoint or default_checkpoint(config),
                    records(args.fit_manifest),
                    records(args.held_manifest),
                    config["output_dir"] / "orientation",
                    device=config["device"],
                    batch_size=config["batch_size"],
                    seed=config.get("seed", 20261114),
                )
            )
        }
    elif args.command == "export-orientation":
        from .waterec_orientation import export_orientation

        folder, _ = deployment_metadata(config)
        result = {
            "deployment": str(
                export_orientation(
                    folder,
                    args.orientation_head
                    or config["output_dir"] / "orientation" / "orientation.pt",
                    config["output_dir"] / "deployment_orientation",
                )
            )
        }
    elif args.command == "build-orientation":
        from .waterec_build import build_engines

        folder, _ = deployment_metadata(config, orientation=True)
        result = {"deployment": str(build_engines(folder, force=args.force))}
    elif args.command == "train":
        from .waterec_training import train

        result = train(config)
    elif args.command == "export":
        from .waterec_export import export_onnx

        result = {"deployment": str(export_onnx(config, args.checkpoint))}
    elif args.command == "build":
        from .waterec_build import build_engines

        folder, _ = deployment_metadata(config)
        result = {"deployment": str(build_engines(folder, force=args.force))}
    elif args.command == "benchmark":
        if args.folder is None:
            parser.error("benchmark requires --folder")
        from .waterec_benchmark import benchmark

        result = benchmark(
            config,
            args.backend,
            args.folder,
            args.repeats,
            args.predecoded,
            args.checkpoint,
        )
    else:
        recognizer = make_recognizer(config, args.backend, args.checkpoint)
        try:
            if args.command == "infer":
                if args.image is None:
                    parser.error("infer requires --image")
                from PIL import Image

                with Image.open(args.image) as image:
                    result = {
                        "text": recognizer.infer([image.convert("RGB")])[0],
                        "backend": args.backend,
                        "decoder": "experimental_speculative"
                        if config.get("speculative", False)
                        else "original_greedy",
                    }
            else:
                if args.folder is None:
                    parser.error("evaluate requires --folder")
                from .evaluation import evaluate_folder

                result = evaluate_folder(
                    recognizer,
                    args.folder,
                    config["normalization"],
                    1,
                    args.output,
                    price_numeric=config["task"] == "price",
                )
        finally:
            if hasattr(recognizer, "close"):
                recognizer.close()
    if args.output and args.command == "benchmark":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
