"""Torch-free builders for existing WATERec ONNX deployment graphs.

Build policies and output barriers are preserved from the original exporters.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import hashlib
import json

def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_engines(deployment, force=False):
    """Build the proven static batch-one graphs for the current GPU.

    Keep the q/k barriers: removing them produced silently incorrect results on
    the measured RTX5090/TensorRT10.15 target, including an FP32 control.
    """
    import onnx
    import tensorrt as trt
    import pycuda.driver as cuda

    folder = Path(deployment)
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata.get("family") != "waterec" or metadata.get("max_tokens") < 64:
        raise ValueError("Expected WATERec staticB1 patch budget>=64 metadata")
    cuda.init()
    gpu = cuda.Device(0)
    n = metadata["max_tokens"]
    layers, heads, dim = (
        metadata[key] for key in ["decoder_layers", "decoder_heads", "cache_head_dim"]
    )
    shapes = {
        "patches": (1, n, 48),
        "cosine": (1, n, 32),
        "sine": (1, n, 32),
        "padding": (1, n),
        "token": (1, 1),
        "position": (1,),
        "cross_k": (layers, 1, heads, n, dim),
        "cross_v": (layers, 1, heads, n, dim),
        "past_k": (layers, 1, heads, 25, dim),
        "past_v": (layers, 1, heads, 25, dim),
    }
    if metadata.get("orientation"):
        shapes[
            metadata["orientation"].get("pool_input", "orientation/pool_weights")
        ] = (1, 5, n)
    logger = trt.Logger(trt.Logger.WARNING)
    for name in ["encoder_cross_kv", "decoder_cached_step"]:
        assets = metadata.get("assets", {})
        source = folder / assets.get(f"{name}.onnx", f"{name}.onnx")
        target = folder / assets.get(f"{name}.engine", f"{name}.engine")
        sidecar = target.with_suffix(".build.json")
        info = {
            "onnx_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "gpu": gpu.name(),
            "compute_capability": list(gpu.compute_capability()),
            "tensorrt_version": trt.__version__,
            "profile": f"static batch1,{n} memory tokens,25 self-cache slots",
            "precision": "FP16allowed,FP32normalization/softmax",
            "rope_barriers": name == "encoder_cross_kv",
        }
        if target.exists() and not force:
            if not sidecar.exists() or json.loads(sidecar.read_text()) != info:
                raise ValueError(
                    f"Cached {name} engine does not match source/target; rebuild with force=True"
                )
            continue
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(source.read_bytes()):
            raise RuntimeError(
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            )
        barriers = []
        if name == "encoder_cross_kv":
            model = onnx.load(source)
            wanted = {
                node.output[0]
                for node in model.graph.node
                if node.op_type == "Cast"
                and node.input[0].startswith("/encoder/Reshape")
            }
            for index in range(network.num_layers):
                layer = network.get_layer(index)
                for output in range(layer.num_outputs):
                    tensor = layer.get_output(output)
                    if tensor.name in wanted:
                        network.mark_output(tensor)
                        barriers.append(tensor.name)
            if len(barriers) != 2 * metadata["encoder_layers"]:
                raise RuntimeError(
                    "Expected two post-RoPE q/k barriers per encoder block"
                )
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.max_aux_streams = 0
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        if hasattr(trt.PreviewFeature, "RUNTIME_ACTIVATION_RESIZE_10_10"):
            config.set_preview_feature(
                trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True
            )
        profile = builder.create_optimization_profile()
        for index in range(network.num_inputs):
            tensor = network.get_input(index)
            profile.set_shape(
                tensor.name,
                shapes[tensor.name],
                shapes[tensor.name],
                shapes[tensor.name],
            )
        config.add_optimization_profile(profile)
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            if (
                layer.type
                in (
                    trt.LayerType.NORMALIZATION,
                    trt.LayerType.SOFTMAX,
                )
                or layer.name.startswith("orientation/")
            ):
                if layer.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
                    layer.precision = trt.float32
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError(f"Failed to build {name}")
        temporary = target.with_suffix(".engine.tmp")
        temporary.write_bytes(blob)
        temporary.replace(target)
        sidecar.write_text(json.dumps(info, indent=2) + "\n")
    return folder


def build_speculative(folder, force=False):
    """Build static B1 engines, retaining the tested draft attention barrier."""
    import tensorrt as trt
    import pycuda.driver as cuda

    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    if (
        metadata.get("family") != "waterec_speculative"
        or metadata.get("max_tokens") != 512
        or metadata.get("max_len") != 25
        or metadata.get("vocab_size") != 95
    ):
        raise ValueError("Unsupported speculative metadata/profile")
    cuda.init()
    device = cuda.Device(0)
    logger = trt.Logger(trt.Logger.WARNING)
    for name in ("draft", "verifier"):
        identity = dict(
            onnx_sha256=_digest(folder / f"{name}.onnx"),
            metadata_sha256=_digest(folder / "metadata.json"),
            tensorrt=trt.__version__,
            device=device.name(),
            capability=list(device.compute_capability()),
        )
        sidecar = folder / f"{name}.build.json"
        target = folder / f"{name}.engine"
        if not force and target.exists() and sidecar.exists():
            previous = json.loads(sidecar.read_text())
            if previous.get("identity") == identity and previous.get(
                "engine_sha256"
            ) == _digest(target):
                continue
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse((folder / f"{name}.onnx").read_bytes()):
            raise RuntimeError(
                "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            )
        barriers = []
        if name == "draft":
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.type == trt.LayerType.SOFTMAX:
                    tensor = layer.get_output(0)
                    network.mark_output(tensor)
                    barriers.append(tensor.name)
            if len(barriers) != 1:
                raise RuntimeError("Expected one draft attention output barrier")
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        config.max_aux_streams = 0
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        config.set_preview_feature(
            trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10, True
        )
        shapes = (
            {
                "tokens": (1, 25),
                "cross_k": (2, 1, 12, 512, 32),
                "cross_v": (2, 1, 12, 512, 32),
                "padding": (1, 512),
            }
            if name == "verifier"
            else {
                "keys": (1, 12, 512, 32),
                "values": (1, 12, 512, 32),
                "padding": (1, 512),
            }
        )
        profile = builder.create_optimization_profile()
        for key, shape in shapes.items():
            profile.set_shape(key, shape, shape, shape)
        config.add_optimization_profile(profile)
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type in (trt.LayerType.NORMALIZATION, trt.LayerType.SOFTMAX):
                layer.precision = trt.float32
        blob = builder.build_serialized_network(network, config)
        if blob is None:
            raise RuntimeError(f"Speculative {name} TensorRT build failed")
        target.write_bytes(bytes(blob))
        sidecar.write_text(
            json.dumps(
                dict(
                    identity=identity,
                    engine_sha256=_digest(target),
                    attention_output_barriers=barriers,
                    precision="FP16 allowed; normalization/softmax prefer FP32",
                    workspace_MiB=256,
                ),
                indent=2,
            )
            + "\n"
        )
    return folder
