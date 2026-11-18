"""Optional quarter-turn control from the frozen WATERec encoder.

Default OCR remains unchanged. Train a head for the EXACT final encoder; an
orientation head cannot be reused after encoder fine-tuning. Runtime inference
imports no PyTorch. No trained weights are bundled.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from PIL import Image


def spatial_weights(ph, pw, max_tokens=512):
    """Global + TL/TR/BL/BR mean weights, row-major valid patches only."""
    if ph < 1 or pw < 1 or ph * pw > max_tokens:
        raise ValueError("Invalid patch grid or token budget")
    yy, xx = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
    groups = [np.ones(ph * pw, dtype=bool)] + [
        (((yy >= ph / 2) == a) & ((xx >= pw / 2) == b)).ravel()
        for a, b in [(False, False), (False, True), (True, False), (True, True)]
    ]
    result = np.zeros((1, 5, max_tokens), np.float32)
    for i, mask in enumerate(groups):
        if mask.any():
            result[0, i, : ph * pw] = mask / mask.sum()
    return result


def create_head():
    from torch import nn

    return nn.Sequential(
        nn.LayerNorm(1920),
        nn.Linear(1920, 128),
        nn.GELU(),
        nn.Dropout(0.1),
        nn.Linear(128, 4),
    )


def _rotate(image, quarterturns):
    if isinstance(image, np.ndarray):
        return np.rot90(image, -quarterturns)
    return (
        image
        if quarterturns == 0
        else image.transpose(
            {
                1: Image.Transpose.ROTATE_270,
                2: Image.Transpose.ROTATE_180,
                3: Image.Transpose.ROTATE_90,
            }[quarterturns]
        )
    )


def train_head(
    checkpoint,
    fit_rows,
    held_rows,
    output,
    epochs=30,
    batch_size=16,
    device="cuda",
    seed=20261114,
):
    """Train-only orientation fit; rows contain local image ``path`` values.

    Encoder frozen in FP32. Four known rotations per original. Pooled features
    live in RAM only; best head and metrics are saved. Maximum held accuracy,
    earliest ties, selects the head. Existing OCR encoder training exposure is
    inherited: use a selection-only OCR encoder for strict encoder holdouts.
    No external evaluation rows may be supplied as fit/selection data.
    """
    import torch
    from torch import nn
    from .waterec_models import load_native, sha256

    fit_rows, held_rows = list(fit_rows), list(held_rows)
    if not fit_rows or not held_rows or epochs < 1 or batch_size < 1:
        raise ValueError("Nonempty train/holdout and positive epochs/batch required")
    paths = lambda rows: {str(Path(r["path"]).resolve()) for r in rows}
    if paths(fit_rows) & paths(held_rows):
        raise ValueError("Orientation train/holdout image paths overlap")
    torch.manual_seed(seed)
    model = load_native(checkpoint, device).float().eval().requires_grad_(False)
    if model.config["tokens"] != 512:
        raise ValueError("Validated orientation deployment requires512 tokens")
    cache = {}
    with torch.inference_mode():
        for name, rows in [("train", fit_rows), ("holdout", held_rows)]:
            vectors, labels = [], []
            for k in range(4):
                for start in range(0, len(rows), batch_size):
                    images = []
                    for row in rows[start : start + batch_size]:
                        with Image.open(row["path"]) as im:
                            images.append(
                                model.preprocess(_rotate(im.convert("RGB"), k)).to(
                                    device
                                )
                            )
                    features, mask = model.encoder(images)
                    for i, image in enumerate(images):
                        ph, pw = image.shape[-2] // 4, image.shape[-1] // 4
                        assert int((~mask[i]).sum()) == ph * pw
                        w = torch.from_numpy(
                            spatial_weights(ph, pw, features.shape[1])
                        ).to(device)
                        vectors.append((w @ features[i : i + 1]).flatten().cpu())
                        labels.append(k)
            cache[name] = (torch.stack(vectors), torch.tensor(labels))
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    head = create_head().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.01)
    x, y = cache["train"]
    vx, vy = cache["holdout"]
    history = []
    best = -1
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "encoder_checkpoint_name": Path(checkpoint).name,
        "encoder_checkpoint_sha256": sha256(checkpoint),
        "classes_clockwise_degrees": [0, 90, 180, 270],
        "parameters": sum(p.numel() for p in head.parameters()),
        "fit_images": len(fit_rows),
        "held_images": len(held_rows),
        "seed": seed,
        "precision": "FP32",
        "features": "pooledRAMonly",
        "selection": "maxorientationheldaccuracy;earliestties;noexternalcalibration",
        "inherited_exposure": "Encoder retains its prior OCR training exposure; use selection-only encoder for strictfold isolation.",
    }
    for epoch in range(1, epochs + 1):
        head.train()
        losses = []
        for ids in torch.randperm(len(y)).split(256):
            loss = nn.functional.cross_entropy(
                head(x[ids].to(device)), y[ids].to(device)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.inference_mode():
            pred = torch.cat(
                [head(t.to(device)).argmax(-1).cpu() for t in vx.split(512)]
            )
        correct = int((pred == vy).sum())
        conf = torch.bincount(vy * 4 + pred, minlength=16).reshape(4, 4).tolist()
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "correct": correct,
                "n": len(vy),
                "confusion_true_rows": conf,
            }
        )
        if correct > best:
            best = correct
            torch.save(
                {
                    **metadata,
                    "epoch": epoch,
                    "state_dict": {
                        k: v.detach().cpu() for k, v in head.state_dict().items()
                    },
                },
                output / "orientation.pt",
            )
        (output / "orientation_metrics.json").write_text(
            json.dumps({**metadata, "history": history, "best_correct": best}, indent=2)
            + "\n"
        )
    return output / "orientation.pt"


def train_embedded_orientation(config, checkpoint, fit, held):
    """Attach the selected classifier to the deployed OCR checkpoint atomically."""
    import tempfile
    import torch
    checkpoint = Path(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("orientation_head") is not None:
        return
    head = train_head(checkpoint, fit, held, config["output_dir"] / "orientation",
                      epochs=config.get("orientation_epochs", 30),
                      batch_size=config["batch_size"], device=config["device"])
    payload["orientation_head"] = head.read_bytes()
    with tempfile.NamedTemporaryFile(dir=checkpoint.parent, delete=False) as temp:
        path = Path(temp.name)
    try:
        torch.save(payload, path)
        path.replace(checkpoint)
    finally:
        path.unlink(missing_ok=True)


def export_orientation(deployment, head_checkpoint, output, *, embedded=None):
    """Clone portable deployment and append head to the encoder graph.

    Requires a head fitted for the exact deployment checkpoint SHA. Engines are
    NOT copied: build this new graph on the target GPU using build_engines().
    Baseline deployment remains unchanged; returned folder is explicitly opt-in.
    """
    import torch
    from torch import nn
    import onnx

    deployment, output = Path(deployment), Path(output)
    if embedded is None and deployment.resolve() == output.resolve():
        raise ValueError("Use a separate orientation deployment folder")
    metadata = json.loads((deployment / "metadata.json").read_text())
    import io
    head_bytes = embedded if embedded is not None else Path(head_checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(head_bytes), map_location="cpu", weights_only=True)
    if embedded is None and saved["encoder_checkpoint_sha256"] != metadata["checkpoint_sha256"]:
        raise ValueError("Orientation head and OCR encoder checkpoint SHA differ")
    head = create_head()
    head.load_state_dict(saved["state_dict"], strict=True)
    head.eval()

    class Pooled(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = head

        def forward(self, memory, pool_weights):
            return self.head(torch.matmul(pool_weights, memory).reshape(1, 1920))

    output.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        Pooled(),
        (torch.zeros(1, 512, 384), torch.zeros(1, 5, 512)),
        str(output / "orientation_head.onnx"),
        input_names=["memory", "pool_weights"],
        output_names=["orientation_logits"],
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(deployment / "encoder_cross_kv.onnx")
    final = [
        n.output[0]
        for n in graph.graph.node
        if n.op_type == "LayerNormalization"
        and n.name == "/encoder/norm/LayerNormalization"
    ]
    if len(final) != 1:
        raise ValueError("Expected pinned final encoder LayerNorm")
    h = onnx.load(output / "orientation_head.onnx")
    onnx.compose.add_prefix(h, "orientation/", inplace=True)
    for node in h.graph.node:
        for i, name in enumerate(node.input):
            if name == "orientation/memory":
                node.input[i] = final[0]
    graph.graph.node.extend(h.graph.node)
    graph.graph.initializer.extend(h.graph.initializer)
    graph.graph.input.extend(x for x in h.graph.input if x.name != "orientation/memory")
    graph.graph.output.extend(h.graph.output)
    onnx.checker.check_model(graph)
    onnx.save(graph, output / "encoder_cross_kv.onnx")
    for name in ([] if embedded is not None else ["decoder_cached_step.onnx", "rope.npz"]):
        shutil.copy2(deployment / name, output / name)
    metadata["orientation"] = {
        "default_enabled": embedded is not None,
        "pool_input": "orientation/pool_weights",
        "logits_output": "orientation/orientation_logits",
        "classes_clockwise_degrees": [0, 90, 180, 270],
        "head_sha256": hashlib.sha256(head_bytes).hexdigest(),
        "head_parameters": 250244,
        "note": "Optional; may harm upright crops. Corrections require second encoder pass.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


class OrientationRecognizer:
    """Explicit optional controller; default disabled, original greedy runtime.

    Use orientation-enabled deployment only with enabled=True. Native accuracy
    and research conditional-WHILE timings do not establish this public runtime's
    latency: benchmark the chosen input and decoder scope independently.
    """

    def __init__(
        self,
        deployment,
        enabled=False,
        capture=True,
        gpu_preprocess=False,
        speculative_deployment=None,
        speculative_shared_workspace=True,
        upscale_small_plates=False,
        upscale_very_small_plates=False,
    ):
        from .waterec_runtime import TensorRTRecognizer

        self.upscale_small_plates = upscale_small_plates
        self.upscale_very_small_plates = upscale_very_small_plates

        if gpu_preprocess:
            from .waterec_gpu_preprocessing import GpuInputRecognizer

            base_runtime = GpuInputRecognizer
        else:
            base_runtime = TensorRTRecognizer

        class Runtime(base_runtime):
            def preprocess(self, image):
                data = super().preprocess(image)
                if self.meta.get("orientation"):
                    data[self.meta["orientation"]["pool_input"]] = spatial_weights(
                        *self._last_grid
                    )
                return data

        if speculative_deployment is not None:
            from .waterec_speculative import SpeculativeRecognizer

            self.runtime = SpeculativeRecognizer(
                deployment,
                speculative_deployment,
                gpu_preprocess=gpu_preprocess,
                capture=capture,
                share_workspace=speculative_shared_workspace,
            )
        else:
            self.runtime = Runtime(deployment, capture=capture)
        self.enabled = enabled
        if enabled and not self.runtime.meta.get("orientation"):
            self.runtime.close()
            raise ValueError("Export/build an orientation deployment first")
        self.last_angles = []
        if self.runtime.meta.get("orientation"):
            from .waterec_runtime import cuda

            self._logits = cuda.pagelocked_empty((1, 4), np.float32)

    def _should_upscale(self, image):
        width = image.shape[1] if isinstance(image, np.ndarray) and image.ndim == 3 else (
            0 if isinstance(image, np.ndarray) else image.width
        )
        return (self.upscale_small_plates and 48 <= width <= 96) or (
            self.upscale_very_small_plates and 32 <= width <= 47
        )

    def _encode_upright(self, image):
        if not self._should_upscale(image):
            return self.runtime.encode(image)
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8 or image.shape[2] != 3 or image.shape[0] == 0:
                raise ValueError("Expected a nonempty H×W×3 uint8 RGB array")
            with Image.fromarray(image) as rgb:
                return self._encode_upright(rgb)
        with image.resize((2 * image.width, 2 * image.height), Image.Resampling.LANCZOS) as larger:
            return self.runtime.encode(larger)

    def infer(self, images):
        """Recognize PIL images or H×W×3 uint8 arrays in RGB channel order."""
        from .waterec_runtime import cuda

        result = []
        self.last_angles = []
        for image in images:
            if not isinstance(image, np.ndarray) and image.mode != "RGB":
                image = image.convert("RGB")
            angle = 0
            if self.enabled:
                prepared = self.runtime.encode(image)
                name = self.runtime.meta["orientation"]["logits_output"]
                cuda.memcpy_dtoh_async(
                    self._logits,
                    self.runtime.encoder.buffers[name],
                    self.runtime.stream,
                )
                self.runtime.stream.synchronize()
                angle = int(self._logits.argmax())
                if angle:
                    upright = _rotate(image, (-angle) % 4)
                    try:
                        prepared = self._encode_upright(upright)
                    finally:
                        if isinstance(upright, Image.Image):
                            upright.close()
                elif self._should_upscale(image):
                    prepared = self._encode_upright(image)
            else:
                prepared = self._encode_upright(image)
            result.append(self.runtime.decode_encoded(prepared))
            self.last_angles.append(angle * 90)
        return result

    def close(self):
        self.runtime.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
