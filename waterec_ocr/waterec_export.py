"""Cached WATERec ONNX/TensorRT deployment with original positional semantics.

SPDX-License-Identifier: Apache-2.0
Adapters derive from the pinned OpenOCR-WATERec model; see
_vendor/waterec/NOTICE. No pretrained or fine-tuned weights are bundled.
"""
from pathlib import Path
import hashlib
import json
import math
import torch
from torch import nn
from torch.nn import functional as F


def pack_images(images, encoder):
    """CPU/eager host preparation AFTER official NaRecTVResize, no resizing here.
    Keep native p[1] vertical/p[0] horizontal convention exactly (square in config).
    """
    p = encoder.patch_size
    seqs = []
    cs = []
    ss = []
    d = encoder.embed_dim // encoder.num_heads
    freq = 1 / (
        encoder.rope_theta ** (torch.arange(0, d, 4, dtype=torch.float32)[: d // 4] / d)
    )
    for image in images:
        h, w = image.shape[-2:]
        ph, pw = h // p[1], w // p[0]
        seqs.append(
            image.reshape(-1, ph, p[1], pw, p[0])
            .permute(1, 3, 0, 2, 4)
            .reshape(ph * pw, -1)
        )
        yy, xx = torch.meshgrid(
            torch.arange(ph, dtype=torch.float32),
            torch.arange(pw, dtype=torch.float32),
            indexing="ij",
        )
        angles = torch.cat([xx.reshape(-1, 1) * freq, yy.reshape(-1, 1) * freq], -1)
        cs.append(angles.cos())
        ss.append(angles.sin())
    patches = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    cosine = nn.utils.rnn.pad_sequence(cs, batch_first=True).to(patches.device)
    sine = nn.utils.rnn.pad_sequence(ss, batch_first=True).to(patches.device)
    lengths = torch.tensor([len(x) for x in seqs], device=patches.device)
    padding = (
        torch.arange(patches.shape[1], device=patches.device)[None, :]
        >= lengths[:, None]
    )
    return patches, cosine, sine, padding


def rotary_real(x, cosine, sine):
    dtype = x.dtype
    pair = x.float().reshape(*x.shape[:-1], -1, 2)
    a, b = pair[..., 0], pair[..., 1]
    c = cosine[:, None]
    s = sine[:, None]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2).to(dtype)


class DenseEncoder(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, patches, cosine, sine, padding):
        e = self.encoder
        x = e.pos_drop(e.patch_embed(patches))
        b, n, c = x.shape
        mask = padding[:, None, None, :]
        for block in e.blocks:
            a = block.attn
            qkv = (
                a.qkv(block.norm1(x))
                .reshape(b, n, 3, a.num_heads, c // a.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q = rotary_real(qkv[0], cosine, sine)
            k = rotary_real(qkv[1], cosine, sine)
            v = qkv[2]
            attn = (
                ((q * a.scale) @ k.transpose(-2, -1))
                .masked_fill(mask, float("-inf"))
                .softmax(-1)
            )
            y = (a.attn_drop(attn) @ v).transpose(1, 2).reshape(b, n, c)
            x = x + block.drop_path(a.proj_drop(a.proj(y)))
            x = x + block.drop_path(block.mlp(block.norm2(x)))
        return e.norm(x)


def explicit_mha(attn, query, key, value, mask=None):
    """Eval-only stock MultiheadAttention algebra with original learned weights."""
    c = attn.embed_dim
    w = attn.in_proj_weight
    b = attn.in_proj_bias
    q = F.linear(query, w[:c], None if b is None else b[:c])
    k = F.linear(key, w[c : 2 * c], None if b is None else b[c : 2 * c])
    v = F.linear(value, w[2 * c :], None if b is None else b[2 * c :])
    batch, t, _ = q.shape
    heads = attn.num_heads
    d = c // heads
    q = q.reshape(batch, t, heads, d).transpose(1, 2)
    k = k.reshape(batch, -1, heads, d).transpose(1, 2)
    v = v.reshape(batch, -1, heads, d).transpose(1, 2)
    scores = (q * (d ** -0.5)) @ k.transpose(-2, -1)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    y = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, t, c)
    return F.linear(y, attn.out_proj.weight, attn.out_proj.bias)


class PrefixDecoder(nn.Module):
    """One native AR full-prefix step, logits only last token; no Python loop.
    Host loop must use official bos/eos/max_len, preserving greedy semantics.
    No beam or KV cache implementation is claimed by this prototype.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, memory, padding, tokens):
        d = self.decoder
        x = d.positional_encoding(d.embedding(tokens))
        n = tokens.shape[1]
        ix = torch.arange(n, device=tokens.device)
        causal = (ix[None, :] > ix[:, None])[None, None]
        memorymask = padding[:, None, None, :]
        for layer in d.decoder.layers:
            if layer.norm_first:
                raise ValueError("Prototype expects native norm_first=False")
            x = layer.norm1(
                x + layer.dropout1(explicit_mha(layer.self_attn, x, x, x, causal))
            )
            x = layer.norm2(
                x
                + layer.dropout2(
                    explicit_mha(layer.multihead_attn, x, memory, memory, memorymask)
                )
            )
            x = layer.norm3(
                x
                + layer.dropout3(
                    layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
                )
            )
        if d.decoder.norm is not None:
            x = d.decoder.norm(x)
        return d.tgt_word_prj(x[:, -1:])


def project_cross_kv(memory, layers):
    keys = []
    values = []
    for layer in layers:
        a = layer.multihead_attn
        c = a.embed_dim
        heads = a.num_heads
        d = c // heads
        b = a.in_proj_bias
        k = F.linear(
            memory, a.in_proj_weight[c : 2 * c], None if b is None else b[c : 2 * c]
        )
        v = F.linear(
            memory, a.in_proj_weight[2 * c :], None if b is None else b[2 * c :]
        )
        keys.append(k.reshape(memory.shape[0], -1, heads, d).transpose(1, 2))
        values.append(v.reshape(memory.shape[0], -1, heads, d).transpose(1, 2))
    return torch.stack(keys), torch.stack(values)


class EncoderCrossKV(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = DenseEncoder(encoder)
        self.layers = decoder.decoder.layers

    def forward(self, patches, cosine, sine, padding):
        memory = self.encoder(patches, cosine, sine, padding)
        return project_cross_kv(memory, self.layers)


class CachedDecoderStep(nn.Module):
    """Exactly one native AR step; fixed25-slot caches, position0 starts BOS.
    All padded selfcache positions are masked, no dynamic cache allocation.
    """

    def __init__(self, decoder):
        super().__init__()
        self.embedding = decoder.embedding
        self.layers = decoder.decoder.layers
        self.head = decoder.tgt_word_prj
        self.register_buffer(
            "positions",
            decoder.positional_encoding.pe[:, : decoder.max_len].detach().clone(),
        )
        self.register_buffer("slots", torch.arange(decoder.max_len, dtype=torch.long))

    def forward(self, token, position, cross_k, cross_v, padding, past_k, past_v):
        x = self.embedding(token) + self.positions[:, position.flatten(), :]
        cachekeys = []
        cachevalues = []
        slotmask = (self.slots == position.flatten()[0])[None, None, :, None]
        invalid = (self.slots > position.flatten()[0])[None, None, None, :]
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            c = a.embed_dim
            h = a.num_heads
            dim = c // h
            b = x.shape[0]
            qkv = (
                F.linear(x, a.in_proj_weight, a.in_proj_bias)
                .reshape(b, 1, 3, h, dim)
                .permute(2, 0, 3, 1, 4)
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
            k = torch.where(slotmask, k, past_k[i])
            v = torch.where(slotmask, v, past_v[i])
            cachekeys.append(k)
            cachevalues.append(v)
            weights = (
                ((q * (dim ** -0.5)) @ k.transpose(-2, -1))
                .masked_fill(invalid, float("-inf"))
                .softmax(-1)
            )
            out = (weights @ v).transpose(1, 2).reshape(b, 1, c)
            x = layer.norm1(x + F.linear(out, a.out_proj.weight, a.out_proj.bias))
            a = layer.multihead_attn
            q = F.linear(
                x,
                a.in_proj_weight[:c],
                None if a.in_proj_bias is None else a.in_proj_bias[:c],
            )
            q = q.reshape(b, 1, h, dim).transpose(1, 2)
            weights = (
                ((q * (dim ** -0.5)) @ cross_k[i].transpose(-2, -1))
                .masked_fill(padding[:, None, None, :], float("-inf"))
                .softmax(-1)
            )
            out = (weights @ cross_v[i]).transpose(1, 2).reshape(b, 1, c)
            x = layer.norm2(x + F.linear(out, a.out_proj.weight, a.out_proj.bias))
            x = layer.norm3(x + layer.linear2(layer.activation(layer.linear1(x))))
        return self.head(x), torch.stack(cachekeys), torch.stack(cachevalues)


def _validate_checkpoint_config(config, actual):
    if int(actual.get("tokens", actual.get("max_patch_tokens", 512))) < 64:
        raise ValueError("Cached deployment patch budget must be >=64")
    if int(actual.get("max_decode_steps", 25)) != 25:
        raise ValueError("Cached deployment requires max_decode_steps=25")
    for key in ("task", "variant", "tokens", "max_label_length", "max_decode_steps"):
        if key in config and key in actual and config[key] != actual[key]:
            raise ValueError(f"Configuration {key} does not match checkpoint metadata")
    normalization = actual.get(
        "normalization",
        "upper_no_whitespace" if actual.get("task", "plate") == "plate" else "none",
    )
    if config.get("normalization", normalization) != normalization:
        raise ValueError(
            "Configuration normalization does not match checkpoint task/metadata"
        )
    return normalization


def export_onnx(config, checkpoint=None):
    """Export CPU graphs after checking dense-RoPE/cached-decoder algebra.

    The measured deployment uses a 512-patch budget, static batch one, and
    original greedy decoding. Dataset accuracy must be evaluated separately.
    """
    from .waterec_models import load_native, default_checkpoint

    checkpoint = Path(checkpoint) if checkpoint else default_checkpoint(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    output = Path(config["output_dir"]) / "deployment"
    output.mkdir(parents=True, exist_ok=True)
    if config.get("tokens", 512) < 64:
        raise ValueError("Cached deployment requires tokens>=64")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    model = load_native(checkpoint, device="cpu").eval()
    normalization = _validate_checkpoint_config(config, model.config)
    if (
        any(layer.norm_first for layer in model.decoder.decoder.layers)
        or model.decoder.decoder.norm is not None
    ):
        raise ValueError(
            "Expected the pinned post-norm WATERec decoder without an extra final norm"
        )
    if model.decoder.out_channels - 2 > 128:
        raise ValueError("Cached GPU greedy kernel supports at most 128 output classes")
    if model.decoder.max_len != 25:
        raise ValueError("The validated fixed-cache export requires max_len=25")
    torch.manual_seed(config.get("seed", 20261114))
    images = [torch.randn(3, 20, 44), torch.randn(3, 64, 128)]
    packed = pack_images(images, model.encoder)
    encoder = EncoderCrossKV(model.encoder, model.decoder).eval()
    decoder = CachedDecoderStep(model.decoder).eval()
    prefix = PrefixDecoder(model.decoder).eval()
    batch = len(images)
    heads = model.decoder.nhead
    dimension = model.decoder.d_model // heads
    layers = len(model.decoder.decoder.layers)
    shape = (layers, batch, heads, 25, dimension)
    with torch.no_grad():
        memory, padding = model.encoder(images)
        native_k, native_v = project_cross_kv(memory, model.decoder.decoder.layers)
        dense_k, dense_v = encoder(*packed)
        dense_error = max(
            float((native_k - dense_k).abs().max()),
            float((native_v - dense_v).abs().max()),
        )
        tokens = torch.randint(1, model.decoder.out_channels - 2, (batch, 25))
        tokens[:, 0] = model.decoder.bos
        past_k = torch.zeros(shape)
        past_v = torch.zeros_like(past_k)
        errors = []
        agreements = []
        for position in range(25):
            reference = prefix(memory, padding, tokens[:, : position + 1])
            actual, past_k, past_v = decoder(
                tokens[:, position : position + 1],
                torch.tensor([position]),
                native_k,
                native_v,
                padding,
                past_k,
                past_v,
            )
            errors.append(float((reference - actual).abs().max()))
            agreements.append(bool((reference.argmax(-1) == actual.argmax(-1)).all()))
    if dense_error > 0.001 or max(errors) > 0.001 or not all(agreements):
        raise RuntimeError(
            f"Cached export algebra check failed: dense={dense_error}, decoder={max(errors)}, argmax={all(agreements)}"
        )
    parity = {
        "scope": "CPU FP32 synthetic pixels and random prefixes, all25positions; not dataset accuracy or TensorRT parity",
        "encoder_cross_kv_max_abs": dense_error,
        "decoder_logit_max_abs": max(errors),
        "decoder_argmax_agreement": all(agreements),
    }
    (output / "export_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    metadata = {
        "family": "waterec",
        "graph_type": "waterec_cached_greedy",
        "static_batch": 1,
        "max_tokens": int(model.config["tokens"]),
        "min_tokens": 64,
        "patch": 4,
        "embedding_dim": model.decoder.d_model,
        "encoder_heads": model.encoder.num_heads,
        "encoder_layers": len(model.encoder.blocks),
        "decoder_heads": heads,
        "decoder_layers": layers,
        "cache_head_dim": dimension,
        "max_len": 25,
        "characters": model.characters,
        "bos": model.decoder.bos,
        "eos": model.decoder.eos,
        "normalization": normalization,
        "task": model.config["task"],
        "variant": model.config["variant"],
        "checkpoint_name": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source": "https://github.com/YesianRohn/OpenOCR-WATERec",
        "source_commit": "d895bc57dade600923015267c87df2a40a3147b5",
        "precision_policy": "FP16 allowed, normalization/softmax prefer FP32; validate target-device accuracy",
        "fusion_workaround": "Post-RoPE q/k output barriers, two reused physical buffers across sequential encoder blocks",
        "batch32": "not supported by this static batch-one deployment",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    frequency = 1 / (model.encoder.rope_theta ** (torch.arange(0, 64, 4).float() / 64))
    angles = torch.arange(metadata["max_tokens"]).float()[:, None] * frequency
    import numpy as np

    np.savez(
        output / "rope.npz", cosine=angles.cos().numpy(), sine=angles.sin().numpy()
    )
    torch.onnx.export(
        encoder,
        packed,
        str(output / "encoder_cross_kv.onnx"),
        opset_version=17,
        dynamo=False,
        input_names=["patches", "cosine", "sine", "padding"],
        output_names=["cross_k", "cross_v"],
        dynamic_axes={
            **{
                name: {0: "batch", 1: "memory_tokens"}
                for name in ["patches", "cosine", "sine", "padding"]
            },
            "cross_k": {1: "batch", 3: "memory_tokens"},
            "cross_v": {1: "batch", 3: "memory_tokens"},
        },
    )
    arguments = (
        torch.full((batch, 1), model.decoder.bos, dtype=torch.long),
        torch.tensor([0]),
        native_k,
        native_v,
        padding,
        torch.zeros(shape),
        torch.zeros(shape),
    )
    torch.onnx.export(
        decoder,
        arguments,
        str(output / "decoder_cached_step.onnx"),
        opset_version=17,
        dynamo=False,
        input_names=[
            "token",
            "position",
            "cross_k",
            "cross_v",
            "padding",
            "past_k",
            "past_v",
        ],
        output_names=["logits", "present_k", "present_v"],
        dynamic_axes={
            "token": {0: "batch"},
            "cross_k": {1: "batch", 3: "memory_tokens"},
            "cross_v": {1: "batch", 3: "memory_tokens"},
            "padding": {0: "batch", 1: "memory_tokens"},
            "past_k": {1: "batch"},
            "past_v": {1: "batch"},
            "logits": {0: "batch"},
            "present_k": {1: "batch"},
            "present_v": {1: "batch"},
        },
    )
    return output


def build_engines(deployment, force=False):
    """Compatibility wrapper for the Torch-free engine builder."""
    from .waterec_build import build_engines as build
    return build(deployment, force=force)
