"""Single-crop timing; TensorRT path does not import Torch or initialize its context."""
import os
import hashlib
import resource
import threading
import time
import numpy as np
from PIL import Image
from .data_io import scan_pairs


def benchmark(config, backend, folder, repeats=3, predecoded=False, checkpoint=None):
    from .waterec_cli import make_recognizer, deployment_metadata

    if config.get("gpu_preprocess", False) and backend != "tensorrt":
        raise ValueError("--gpu-preprocess requires --backend tensorrt")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rows = scan_pairs(folder, config["normalization"])
    if not rows:
        raise ValueError("No annotated images to benchmark")

    def digest(path):
        checksum = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        return checksum.hexdigest()

    identity = {
        "speculative_enabled": bool(config.get("speculative", False)),
        "speculative_shared_workspace": bool(
            config.get("speculative", False)
            and config.get("speculative_shared_workspace", True)
        ),
        "decoder": "experimental_speculative"
        if config.get("speculative", False)
        else "original_greedy",
        "task": config["task"],
        "variant": config["variant"],
        "tokens": config["tokens"],
        "orientation_enabled": bool(config.get("orientation", False)),
        "preprocessing": "gpu_exact_bicubic"
        if config.get("gpu_preprocess", False)
        else "cpu_pillow_bicubic",
    }
    if config.get("gpu_preprocess", False):
        from . import waterec_gpu_preprocessing as preprocessing

        identity["preprocessing_source_sha256"] = {
            "python": digest(preprocessing.__file__),
            "coefficients_native.c": digest(
                preprocessing.ASSETS / "coefficients_native.c"
            ),
            "kernels.cu": digest(preprocessing.ASSETS / "kernels.cu"),
        }
    if backend in ("tensorrt", "onnx"):
        deployment, metadata = deployment_metadata(
            config, identity["orientation_enabled"]
        )
        identity.update(
            checkpoint_sha256=metadata["checkpoint_sha256"],
            orientation_metadata=metadata.get("orientation"),
            metadata_sha256=digest(deployment / "metadata.json"),
            deployment_sha256={
                name: digest(deployment / metadata.get("assets", {}).get(name, name))
                for name in (("encoder_cross_kv.engine", "decoder_cached_step.engine") if backend == "tensorrt" else ("encoder_cross_kv.onnx", "decoder_cached_step.onnx"))
            },
        )
        identity["engine_sha256" if backend == "tensorrt" else "onnx_sha256"] = identity.pop("deployment_sha256")
        if config.get("speculative", False):
            from .waterec_cli import speculative_directory

            speculative = speculative_directory(config)
            identity["speculative_metadata_sha256"] = digest(
                speculative / "metadata.json"
            )
            identity["speculative_engine_sha256"] = {
                name: digest(speculative / name)
                for name in ("draft.engine", "verifier.engine")
            }
    else:
        from .waterec_models import default_checkpoint

        identity["checkpoint_sha256"] = digest(checkpoint or default_checkpoint(config))

    def read(row):
        with Image.open(row["path"]) as image:
            return image.convert("RGB")

    decoded = [read(row) for row in rows] if predecoded else None
    stop = threading.Event()
    memory = {"sampled_peak_bytes": None, "steady_bytes": None, "error": None}
    nv = None
    sampler = None
    try:
        import pynvml as nv

        nv.nvmlInit()
        # Runtime currently targets CUDA device zero; respect CUDA_VISIBLE_DEVICES.
        physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        handle = (
            nv.nvmlDeviceGetHandleByUUID(physical)
            if physical.startswith("GPU-")
            else nv.nvmlDeviceGetHandleByIndex(int(physical))
        )

        def sample():
            values = [
                int(p.usedGpuMemory)
                for p in nv.nvmlDeviceGetComputeRunningProcesses(handle)
                if p.pid == os.getpid() and 0 <= int(p.usedGpuMemory) < (1 << 50)
            ]
            value = sum(values)
            memory["steady_bytes"] = value
            memory["sampled_peak_bytes"] = max(memory["sampled_peak_bytes"] or 0, value)

        def run():
            while not stop.is_set():
                try:
                    sample()
                except Exception as exc:
                    memory["error"] = str(exc)
                    return
                stop.wait(0.02)

        sampler = threading.Thread(target=run, daemon=True)
        sampler.start()
    except Exception as exc:
        memory["error"] = str(exc)
    recognizer = None
    try:
        begin = time.perf_counter()
        recognizer = make_recognizer(config, backend, checkpoint)
        load_seconds = time.perf_counter() - begin
        first = decoded[0] if decoded is not None else read(rows[0])
        for _ in range(5):
            recognizer.infer([first])
        # infer returns host text, so every recorded call includes completion.
        timings = []
        for _ in range(repeats):
            for index, row in enumerate(rows):
                begin = time.perf_counter()
                image = decoded[index] if decoded is not None else read(row)
                recognizer.infer([image])
                timings.append((time.perf_counter() - begin) * 1000)
        if sampler and not memory["error"]:
            sample()
        return {
            "backend": backend,
            "batch_size": 1,
            "unique_images": len(rows),
            "pipeline": identity,
            "predecoded": bool(predecoded),
            "repeats": repeats,
            "timed_images": len(timings),
            "warmup": 5,
            "mean_ms": float(np.mean(timings)),
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "images_per_second": 1000 / float(np.mean(timings)),
            "load_seconds": load_seconds,
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            "gpu_memory": dict(memory),
            "memory_sample_interval_seconds": 0.02,
            "scope": (
                "Predecoded RGB to text; excludes image-file decoding"
                if predecoded
                else "Image-file decoding, preprocessing, inference and text decoding"
            ),
            "exclusions": "Network, detection, model loading and initial warmup",
        }
    finally:
        stop.set()
        if sampler:
            sampler.join(timeout=1)
        if nv:
            try:
                nv.nvmlShutdown()
            except Exception:
                pass
        if recognizer is not None and hasattr(recognizer, "close"):
            recognizer.close()
