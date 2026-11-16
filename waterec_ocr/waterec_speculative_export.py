"""Experimental shared-encoder draft export; original AR target stays unchanged.

The draft is an optimization hint. Verification uses original target weights.
Block and cached FP16 arithmetic can differ; validate deployment predictions.
"""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class DraftHead(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        layer = decoder.decoder.layers[0]
        c = 384
        heads = 12
        with torch.no_grad():
            ids = torch.full((1, 25), decoder.bos, dtype=torch.long)
            x = decoder.positional_encoding(decoder.embedding(ids))
            mask = torch.arange(25)[None, :] > torch.arange(25)[:, None]
            y = layer.self_attn(x, x, x, attn_mask=mask, need_weights=False)[0]
            x = layer.norm1(x + y)
            a = layer.multihead_attn
            q = (
                F.linear(x, a.in_proj_weight[:c], a.in_proj_bias[:c])
                .reshape(1, 25, heads, 32)
                .transpose(1, 2)
            )
        self.query = nn.Parameter(q.clone())
        self.residual = nn.Parameter(x.clone())
        self.output = copy.deepcopy(layer.multihead_attn.out_proj)
        self.norm2 = copy.deepcopy(layer.norm2)
        self.norm3 = copy.deepcopy(layer.norm3)
        self.linear1 = copy.deepcopy(layer.linear1)
        self.linear2 = copy.deepcopy(layer.linear2)
        self.head = copy.deepcopy(decoder.tgt_word_prj)
        self.requires_grad_(True)

    def forward(self, keys, values, padding):
        weights = (
            ((self.query * (32 ** -0.5)) @ keys.transpose(-2, -1))
            .masked_fill(padding[:, None, None, :], float("-inf"))
            .softmax(-1)
        )
        x = (weights @ values).transpose(1, 2).reshape(keys.shape[0], 25, 384)
        x = self.norm2(self.residual + self.output(x))
        x = self.norm3(x + self.linear2(F.relu(self.linear1(x))))
        return self.head(x)


class BlockVerifier(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.embedding = decoder.embedding
        self.layers = decoder.decoder.layers
        self.head = decoder.tgt_word_prj
        self.register_buffer(
            "positions", decoder.positional_encoding.pe[:, :25].detach().clone()
        )
        self.register_buffer(
            "causal", torch.arange(25)[None, :] > torch.arange(25)[:, None]
        )

    def forward(self, tokens, cross_k, cross_v, padding):
        x = self.embedding(tokens) + self.positions
        keys = []
        values = []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            c = a.embed_dim
            h = a.num_heads
            dim = c // h
            b, t, _ = x.shape
            qkv = (
                F.linear(x, a.in_proj_weight, a.in_proj_bias)
                .reshape(b, t, 3, h, dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            keys.append(k)
            values.append(v)
            weights = (
                ((q * (dim ** -0.5)) @ k.transpose(-2, -1))
                .masked_fill(self.causal[None, None], float("-inf"))
                .softmax(-1)
            )
            out = (weights @ v).transpose(1, 2).reshape(b, t, c)
            x = layer.norm1(x + F.linear(out, a.out_proj.weight, a.out_proj.bias))
            a = layer.multihead_attn
            q = F.linear(
                x,
                a.in_proj_weight[:c],
                None if a.in_proj_bias is None else a.in_proj_bias[:c],
            )
            q = q.reshape(b, t, h, dim).transpose(1, 2)
            weights = (
                ((q * (dim ** -0.5)) @ cross_k[i].transpose(-2, -1))
                .masked_fill(padding[:, None, None, :], float("-inf"))
                .softmax(-1)
            )
            out = (weights @ cross_v[i]).transpose(1, 2).reshape(b, t, c)
            x = layer.norm2(x + F.linear(out, a.out_proj.weight, a.out_proj.bias))
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        return self.head(x), torch.stack(keys), torch.stack(values)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_speculative(config, draft_checkpoint, target_checkpoint=None):
    """Export an opt-in ALPR drafter and original25-token causal verifier.

    A drafter is tied to the exact target checkpoint that generated its cache.
    It cannot be transferred to another fine-tuned encoder by matching shapes.
    """
    from .waterec_models import load_native, default_checkpoint
    from .waterec_export import CachedDecoderStep, _validate_checkpoint_config

    target_checkpoint = Path(target_checkpoint or default_checkpoint(config))
    draft_checkpoint = Path(draft_checkpoint)
    target_hash = _digest(target_checkpoint)
    saved = torch.load(draft_checkpoint, map_location="cpu", weights_only=True)
    expected = saved.get(
        "target_checkpoint_sha256",
        saved.get("protocol", {}).get("teacher", {}).get("sha256"),
    )
    if expected != target_hash:
        raise ValueError("Draft and target checkpoint SHA256 do not match")
    model = load_native(target_checkpoint, device="cpu").eval()
    actual = model.config
    _validate_checkpoint_config(config, actual)
    if actual.get("task", "plate") != "plate" or int(actual.get("tokens", 512)) != 512:
        raise ValueError("This speculative deployment supports plate512 checkpoints")
    if config.get("task", "plate") != "plate" or int(config.get("tokens", 512)) != 512:
        raise ValueError("Speculative configuration must be plate512")
    if (
        model.decoder.d_model != 384
        or model.decoder.nhead != 12
        or model.decoder.max_len != 25
        or len(model.decoder.decoder.layers) != 2
    ):
        raise ValueError("Expected original384-wide, two-layer25-position WATERec")
    vocab = model.decoder.tgt_word_prj.out_features
    if model.decoder.eos != 0 or model.decoder.bos != 95:
        raise ValueError("Speculative commit profile requires EOS0 and BOS95")
    if vocab != 95:
        raise ValueError(
            "This trained speculative ALPR profile requires95 output classes"
        )
    torch.set_num_threads(4)
    head = DraftHead(model.decoder).eval()
    head.load_state_dict(saved["state_dict"], strict=True)
    verifier = BlockVerifier(model.decoder).eval()
    step = CachedDecoderStep(model.decoder).eval()
    folder = Path(
        config.get(
            "speculative_dir", Path(config["output_dir"]) / "deployment" / "speculative"
        )
    )
    folder.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(20261114)
    keys = torch.randn((1, 12, 512, 32), generator=generator) * 0.5
    values = torch.randn((1, 12, 512, 32), generator=generator) * 0.5
    padding = torch.zeros((1, 512), dtype=torch.bool)
    padding[:, -7:] = True
    tokens = torch.randint(0, vocab, (1, 25), generator=generator)
    tokens[:, 0] = model.decoder.bos
    cross_k = torch.stack([keys, keys * 0.9])
    cross_v = torch.stack([values, values * 1.1])
    with torch.inference_mode():
        expected_head = head(keys, values, padding)
        expected_verifier = verifier(tokens, cross_k, cross_v, padding)
        caches = [torch.zeros((2, 1, 12, 25, 32)) for _ in range(2)]
        logits = []
        for position in range(25):
            result = step(
                tokens[:, position : position + 1],
                torch.tensor([position]),
                cross_k,
                cross_v,
                padding,
                *caches,
            )
            logits.append(result[0])
            caches = result[1:]
        step_logits = torch.cat(logits, 1)
        if not torch.equal(step_logits.argmax(-1), expected_verifier[0].argmax(-1)):
            raise RuntimeError("CPU block/cached verifier argmax parity failed")
    torch.onnx.export(
        head,
        (keys, values, padding),
        str(folder / "draft.onnx"),
        input_names=["keys", "values", "padding"],
        output_names=["draft_logits"],
        opset_version=17,
        dynamo=False,
    )
    torch.onnx.export(
        verifier,
        (tokens, cross_k, cross_v, padding),
        str(folder / "verifier.onnx"),
        input_names=["tokens", "cross_k", "cross_v", "padding"],
        output_names=["logits", "block_k", "block_v"],
        opset_version=17,
        dynamo=False,
    )
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    draft_ort = ort.InferenceSession(
        str(folder / "draft.onnx"), options, providers=["CPUExecutionProvider"]
    )
    verifier_ort = ort.InferenceSession(
        str(folder / "verifier.onnx"), options, providers=["CPUExecutionProvider"]
    )
    draft_result = draft_ort.run(
        None,
        dict(
            zip(
                ["keys", "values", "padding"],
                [x.numpy() for x in (keys, values, padding)],
            )
        ),
    )[0]
    verifier_result = verifier_ort.run(
        None,
        dict(
            zip(
                ["tokens", "cross_k", "cross_v", "padding"],
                [x.numpy() for x in (tokens, cross_k, cross_v, padding)],
            )
        ),
    )
    if not np.array_equal(
        draft_result.argmax(-1), expected_head.numpy().argmax(-1)
    ) or not np.array_equal(
        verifier_result[0].argmax(-1), expected_verifier[0].numpy().argmax(-1)
    ):
        raise RuntimeError("Speculative ONNX CPU argmax parity failed")
    metadata = dict(
        family="waterec_speculative",
        version=1,
        experimental=True,
        task="plate",
        static_batch=1,
        max_tokens=512,
        max_len=25,
        vocab_size=vocab,
        bos=model.decoder.bos,
        eos=model.decoder.eos,
        target_characters=model.characters,
        target_checkpoint_sha256=target_hash,
        draft_checkpoint_sha256=_digest(draft_checkpoint),
        draft_epoch=int(saved["epoch"]),
        attention_output_barrier=True,
        target_unchanged=True,
        numerical_claim="Greedy algorithm verifies original target; block and cached FP16 arithmetic can differ. Validate predictions, no universal lossless claim.",
        source="https://github.com/YesianRohn/OpenOCR-WATERec",
        source_license="Apache-2.0",
    )
    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (folder / "export_parity.json").write_text(
        json.dumps(
            dict(
                draft_ORT_maxabs=float(
                    np.max(np.abs(draft_result - expected_head.numpy()))
                ),
                verifier_ORT_maxabs=[
                    float(np.max(np.abs(a - b.numpy())))
                    for a, b in zip(verifier_result, expected_verifier)
                ],
                verifier_cached_maxabs=float(
                    (step_logits - expected_verifier[0]).abs().max()
                ),
                all25_argmax_equal=True,
                scope="CPU algebra only; target-device dataset parity required",
            ),
            indent=2,
        )
        + "\n"
    )
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
