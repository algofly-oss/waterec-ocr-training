"""Opt-in exact bicubic GPU preprocessing for the 512-patch TensorRT runtime.

Requires a C compiler and CUDA toolkit; compilation is cached by source hash.
CPU RGB decoding is retained. Runtime inference imports no PyTorch.
The resize algorithm follows Pillow; see _native/waterec/PILLOW_LICENSE.
"""
from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
import ctypes
import hashlib
import math
import os
import shlex
import subprocess
import tempfile
import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import compile as cuda_compile
from .waterec_preprocessing import rgb_array_size
from .waterec_runtime import TensorRTRecognizer, rt, check, _owned_context

ASSETS = Path(__file__).parent / "_native" / "waterec"


@lru_cache(maxsize=1)
def _coefficient_library():
    source = ASSETS / "coefficients_native.c"
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler:
        raise ValueError("CC must name a C compiler")
    identity = (
        source.read_bytes() + repr(compiler).encode() + os.uname().machine.encode()
    )
    digest = hashlib.sha256(identity).hexdigest()[:24]
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "parseq-ocr"
        / "resize"
    )
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / (digest + ".so")
    if not library.exists():
        with tempfile.TemporaryDirectory(prefix="build-", dir=cache) as directory:
            output = Path(directory) / "coefficients.so"
            try:
                subprocess.run(
                    compiler
                    + [
                        "-O3",
                        "-shared",
                        "-fPIC",
                        "-ffp-contract=off",
                        str(source),
                        "-lm",
                        "-o",
                        str(output),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                raise RuntimeError(
                    "GPU preprocessing needs a working C compiler (CC or cc); omit --gpu-preprocess to use CPU preprocessing"
                ) from exc
            os.replace(output, library)
    native = ctypes.CDLL(str(library))
    native.resize_coefficients.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    native.resize_coefficients.restype = ctypes.c_int
    return native


@lru_cache(maxsize=128)
def coefficients(insize, outsize):
    if insize < 1 or outsize < 1:
        raise ValueError("Resize dimensions must be positive")
    ksize = math.ceil(2 * max(insize / outsize, 1.0)) * 2 + 1
    bounds = np.empty((outsize, 2), np.int32)
    weights = np.empty((outsize, ksize), np.int32)
    if not _coefficient_library().resize_coefficients(
        insize, outsize, ksize, bounds.ctypes.data, weights.ctypes.data
    ):
        raise MemoryError("Coefficient buffer allocation failed")
    bounds.flags.writeable = weights.flags.writeable = False
    return bounds, weights


class GpuResize:
    def __init__(self, stream, max_tokens=512, cache_entries=64):
        if cache_entries < 2:
            raise ValueError("Coefficient cache requires at least two entries")
        self.stream = stream
        self.max_tokens = max_tokens
        self.cache_entries = cache_entries
        self.cache = OrderedDict()
        self.buffers = {}
        self.pinned = None
        self.copy_done = cuda.Event(flags=cuda.event_flags.DISABLE_TIMING)
        self.copy_pending = False
        capability = min(cuda.Context.get_device().compute_capability(), (8, 9))
        virtual_arch = f"compute_{capability[0]}{capability[1]}"
        module = cuda.module_from_buffer(
            cuda_compile(
                (ASSETS / "kernels.cu").read_text(),
                options=["--fmad=false"],
                no_extern_c=True,
                target="ptx",
                arch=virtual_arch,
            )
        )
        self.module = module
        self.horizontal = module.get_function("horizontal")
        self.pack = module.get_function("vertical_pack")

    def buffer(self, key, nbytes):
        if key not in self.buffers or self.buffers[key][1] < nbytes:
            if key in self.buffers:
                self.stream.synchronize()
                self.buffers[key][0].free()
            self.buffers[key] = (cuda.mem_alloc(nbytes), nbytes)
        return self.buffers[key][0]

    def coeff(self, a, b):
        key = (a, b)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        bounds, weights = coefficients(a, b)
        bp = cuda.mem_alloc(bounds.nbytes)
        wp = cuda.mem_alloc(weights.nbytes)
        cuda.memcpy_htod(bp, bounds)
        cuda.memcpy_htod(wp, weights)
        self.cache[key] = (bp, wp, weights.shape[1])
        if len(self.cache) > self.cache_entries:
            self.stream.synchronize()
            old = self.cache.popitem(last=False)[1]
            old[0].free()
            old[1].free()
        return self.cache[key]

    def __call__(self, rgb, size, output):
        rgb = np.asarray(rgb, dtype=np.uint8)
        h, w, c = rgb.shape
        ow, oh = size
        if not (
            c == 3
            and ow > 0
            and oh > 0
            and ow % 4 == 0
            and oh % 4 == 0
            and ow * oh <= self.max_tokens * 16
        ):
            raise ValueError(
                "Expected RGB and a positive patch-aligned size within the token budget"
            )
        if self.pinned is None or self.pinned.size < rgb.size:
            self.stream.synchronize()
            self.pinned = cuda.pagelocked_empty(rgb.size, np.uint8)
        if self.copy_pending:
            self.copy_done.synchronize()
        host = self.pinned[: rgb.size]
        np.copyto(host, rgb.reshape(-1))
        src = self.buffer("source", rgb.nbytes)
        tmp = self.buffer("horizontal", h * ow * 3)
        xb, xw, xk = self.coeff(w, ow)
        yb, yw, yk = self.coeff(h, oh)
        cuda.memcpy_htod_async(src, host, self.stream)
        self.copy_done.record(self.stream)
        self.copy_pending = True
        self.horizontal(
            src,
            tmp,
            xb,
            xw,
            np.int32(w),
            np.int32(h),
            np.int32(ow),
            np.int32(xk),
            block=(256, 1, 1),
            grid=((h * ow * 3 + 255) // 256, 1, 1),
            stream=self.stream,
        )
        self.pack(
            tmp,
            output,
            yb,
            yw,
            np.int32(ow),
            np.int32(h),
            np.int32(oh),
            np.int32(yk),
            np.int32(self.max_tokens),
            block=(256, 1, 1),
            grid=((self.max_tokens * 48 + 255) // 256, 1, 1),
            stream=self.stream,
        )

    def close(self):
        self.stream.synchronize()
        for bp, wp, _ in self.cache.values():
            bp.free()
            wp.free()
        self.cache.clear()
        for ptr, _ in self.buffers.values():
            ptr.free()
        self.buffers.clear()
        self.pinned = None


class _Geometry:
    def __init__(self, rope, max_tokens=512, cache_entries=64):
        self.rope = rope
        self.max_tokens = max_tokens
        self.cache_entries = cache_entries
        self.geometry = OrderedDict()
        self.cache_hits = self.cache_misses = 0

    def dimensions(self, image):
        w, h = rgb_array_size(image) if isinstance(image, np.ndarray) else image.size
        area = w * h
        if area > self.max_tokens * 16:
            scale = math.sqrt(self.max_tokens * 16 / area)
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
        elif area < 64 * 16:
            scale = math.sqrt(64 * 16 / area)
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
        return max(w // 4, 1) * 4, max(h // 4, 1) * 4

    def constants(self, ph, pw):
        key = (ph, pw)
        if key in self.geometry:
            self.cache_hits += 1
            self.geometry.move_to_end(key)
            return self.geometry[key]
        self.cache_misses += 1
        n = ph * pw
        if n > self.max_tokens:
            raise ValueError("Official resize exceeds the configured patch budget")
        xx = np.tile(np.arange(pw), ph)
        yy = np.repeat(np.arange(ph), pw)
        cosine = np.zeros((1, self.max_tokens, 32), np.float32)
        sine = np.zeros_like(cosine)
        cosine[0, :n] = np.concatenate(
            [self.rope["cosine"][xx], self.rope["cosine"][yy]], -1
        )
        sine[0, :n] = np.concatenate([self.rope["sine"][xx], self.rope["sine"][yy]], -1)
        padding = np.ones((1, self.max_tokens), bool)
        padding[0, :n] = False
        for value in (cosine, sine, padding):
            value.flags.writeable = False
        result = dict(cosine=cosine, sine=sine, padding=padding)
        self.geometry[key] = result
        if len(self.geometry) > self.cache_entries:
            self.geometry.popitem(last=False)
        return result


class GpuInputRecognizer(TensorRTRecognizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry = _Geometry(self.rope, max_tokens=self.meta["max_tokens"])
        try:
            self.gpu_resize = GpuResize(self.stream, max_tokens=self.meta["max_tokens"])
        except BaseException:
            self.close()
            raise

    def encode(self, image):
        if self._closed:
            raise RuntimeError("Recognizer has been closed")
        if not isinstance(image, np.ndarray) and image.mode != "RGB":
            image = image.convert("RGB")
        w, h = self.geometry.dimensions(image)
        ph, pw = h // 4, w // 4
        data = dict(self.geometry.constants(ph, pw))
        if self.meta.get("orientation"):
            from .waterec_orientation import spatial_weights

            data[self.meta["orientation"]["pool_input"]] = spatial_weights(ph, pw)
        self.gpu_resize(image, (w, h), self.encoder.buffers["patches"])
        for name, x in data.items():
            self.encoder.upload(name, x)
        if self.encoder_graph:
            check(
                rt.cudaGraphLaunch(
                    self.encoder_graph[1], rt.cudaStream_t(self.stream.handle)
                )
            )
        else:
            self.encoder.execute()
        self._last_grid = (ph, pw)
        self._generation += 1
        return {"inputs": data, "grid": self._last_grid, "generation": self._generation}

    def close(self):
        if not self._closed:
            with _owned_context(self.ctx):
                if hasattr(self, "gpu_resize"):
                    self.gpu_resize.close()
                super().close()
