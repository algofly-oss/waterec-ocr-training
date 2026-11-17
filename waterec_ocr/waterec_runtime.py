"""Torch-free, static batch-one WATERec TensorRT inference.

SPDX-License-Identifier: Apache-2.0
Original model provenance: _vendor/waterec/NOTICE. No weights are bundled.
A CUDA toolkit with nvcc is required to compile the small greedy PTX kernel.
Instances own one CUDA stream/context and are not thread-safe.
"""
import json, pathlib, math, resource
from contextlib import contextmanager
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
from pycuda.compiler import compile as cuda_compile
from cuda.bindings import runtime as rt, driver as cuda_driver

LOG = trt.Logger(trt.Logger.ERROR)


def check(x):
    if x[0] != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(str(x[0]))
    return x[1] if len(x) > 1 else None


@contextmanager
def _owned_context(context):
    """Temporarily suspend other wrappers without losing their stack entries.

    PyCUDA compares retained-primary wrappers by CUDA handle, while their hashes
    distinguish the wrapper objects. No extra own push is added or stranded.
    """
    identity = (context.handle, hash(context))
    suspended = []
    try:
        current = cuda.Context.get_current()
        while current is not None and (current.handle, hash(current)) != identity:
            cuda.Context.pop()
            suspended.append(current)
            current = cuda.Context.get_current()
        if current is None:
            raise RuntimeError("Owned CUDA context is not on this thread's stack")
        yield
    finally:
        for other in reversed(suspended):
            other.push()


def _release_context(context):
    """Pop the driver entry while PyCUDA still sees its primary wrapper.

    Primary-context release does not pop (CUDA Driver API). Detaching after a
    PyCUDA pop would make PyCUDA push the owner again, stranding a driver entry.
    """
    with _owned_context(context):
        status, current = cuda_driver.cuCtxGetCurrent()
        if int(status) or int(current) != context.handle:
            raise RuntimeError("Owned CUDA wrapper is not the physical current context")
        status, popped = cuda_driver.cuCtxPopCurrent()
        if int(status) or int(popped) != context.handle:
            raise RuntimeError("Failed to pop the owned CUDA driver context")
        context.detach()


class Engine:
    def __init__(self, path, stream, shapes, barrier_reuse=False, workspace_pool=None):
        self.context = None
        self.scratch = None
        self.buffers = {}
        self.host = {}
        try:
            self._initialize(path, stream, shapes, barrier_reuse, workspace_pool)
        except BaseException:
            self.close()
            raise

    def _initialize(
        self, path, stream, shapes, barrier_reuse=False, workspace_pool=None
    ):
        self.runtime = trt.Runtime(LOG)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        self.context = self.engine.create_execution_context(
            trt.ExecutionContextAllocationStrategy.USER_MANAGED
        )
        self.stream = stream
        self.buffers = {}
        self.shapes = {}
        self.types = {}
        self.host = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shapes[name])
        self.scratchbytes = self.context.update_device_memory_size_for_shapes()
        if workspace_pool is not None and self.engine.num_aux_streams:
            raise ValueError("Shared workspace requires zero auxiliary streams")
        self.scratch = (
            workspace_pool.acquire(self.scratchbytes, stream.handle)
            if workspace_pool is not None
            else cuda.mem_alloc(max(1, self.scratchbytes))
        )
        self.context.set_device_memory(int(self.scratch), self.scratchbytes)
        barrier_buffers = []
        barrier_count = 0
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.shapes[name] = shape
            self.types[name] = dtype
            if barrier_reuse and "/Cast_" in name:
                if len(barrier_buffers) < 2:
                    barrier_buffers.append(
                        cuda.mem_alloc(int(np.prod(shape)) * dtype.itemsize)
                    )
                self.buffers[name] = barrier_buffers[barrier_count % 2]
                barrier_count += 1
            else:
                self.buffers[name] = cuda.mem_alloc(
                    int(np.prod(shape)) * dtype.itemsize
                )
            self.context.set_tensor_address(name, int(self.buffers[name]))

    def execute(self):
        if not self.context.execute_async_v3(self.stream.handle):
            raise RuntimeError("TensorRTexecute failed")

    def upload(self, name, array):
        if name not in self.host:
            self.host[name] = cuda.pagelocked_empty(self.shapes[name], self.types[name])
        np.copyto(self.host[name], array)
        cuda.memcpy_htod_async(self.buffers[name], self.host[name], self.stream)

    def close(self):
        self.context = None
        unique = {int(buffer): buffer for buffer in self.buffers.values()}
        for buffer in unique.values():
            buffer.free()
        self.buffers.clear()
        self.host.clear()
        if self.scratch is not None:
            self.scratch.free()
            self.scratch = None
        self.engine = None
        self.runtime = None
        self.stream = None


KERNEL = """extern "C" __global__ void greedy(const float*logits,int n,long long*token,long long*position,long long*result){__shared__ float scores[128];__shared__ int ids[128];int i=threadIdx.x;scores[i]=i<n?logits[i]:-1.e30f;ids[i]=i;__syncthreads();for(int s=64;s;s>>=1){if(i<s){float a=scores[i],b=scores[i+s];int ai=ids[i],bi=ids[i+s];if(b>a||(b==a&&bi<ai)){scores[i]=b;ids[i]=bi;}}__syncthreads();}if(i==0){*token=ids[0];*result=ids[0];*position+=1;}}"""


class TensorRTRecognizer:
    def __init__(self, deployment, capture=True, workspace_pool=None):
        self.ctx = None
        self._closed = False
        try:
            self._initialize(deployment, capture, workspace_pool)
        except BaseException:
            if self.ctx is not None:
                self.close()
            raise

    def _initialize(self, deployment, capture, workspace_pool=None):
        self.folder = pathlib.Path(deployment)
        self._closed = False
        metadata = json.loads((self.folder / "metadata.json").read_text())
        from .waterec_charset import validate_deployment
        validate_deployment(metadata)
        if (
            metadata.get("family") != "waterec"
            or not isinstance(metadata.get("max_tokens"), int)
            or metadata.get("max_tokens", 0) < 64
            or metadata.get("static_batch") != 1
        ):
            raise ValueError("Expected WATERec static B1, patch budget>=64 metadata")
        self.assets = {
            name: self.folder / metadata.get("assets", {}).get(name, name)
            for name in [
                "encoder_cross_kv.engine",
                "decoder_cached_step.engine",
                "rope.npz",
            ]
        }
        for filename in self.assets:
            if not self.assets[filename].is_file():
                raise FileNotFoundError(
                    f"Missing deployment file: {filename}; export/build first"
                )
        cuda.init()
        self.ctx = cuda.Device(0).retain_primary_context()
        self.ctx.push()
        self.stream = cuda.Stream()
        self.meta = metadata
        self.max_tokens = int(metadata["max_tokens"])
        self.rope = dict(np.load(self.assets["rope.npz"]))
        self.shapes = {
            "patches": (1, self.max_tokens, 48),
            "cosine": (1, self.max_tokens, 32),
            "sine": (1, self.max_tokens, 32),
            "padding": (1, self.max_tokens),
            "token": (1, 1),
            "position": (1,),
            "cross_k": (2, 1, 12, self.max_tokens, 32),
            "cross_v": (2, 1, 12, self.max_tokens, 32),
            "past_k": (2, 1, 12, 25, 32),
            "past_v": (2, 1, 12, 25, 32),
        }
        if metadata.get("orientation"):
            self.shapes[
                metadata["orientation"].get("pool_input", "orientation/pool_weights")
            ] = (1, 5, self.max_tokens)
        self.encoder = Engine(
            self.assets["encoder_cross_kv.engine"],
            self.stream,
            self.shapes,
            barrier_reuse=True,
            workspace_pool=workspace_pool,
        )
        self.decoder = Engine(
            self.assets["decoder_cached_step.engine"],
            self.stream,
            self.shapes,
            workspace_pool=workspace_pool,
        )
        # Across sequential transformer blocks, earlier q/k lifetimes end before
        # the next block's q/k are produced. Keep distinct q and k buffers.
        barriers = [name for name in self.encoder.buffers if "/Cast_" in name]
        self.encoder_barrier_names = barriers
        # Alias crossKV outputs directly; decoder's duplicate buffers are released.
        for name in ["cross_k", "cross_v"]:
            self.decoder.buffers[name].free()
            self.decoder.buffers[name] = self.encoder.buffers[name]
            self.decoder.context.set_tensor_address(
                name, int(self.encoder.buffers[name])
            )
        capability = min(cuda.Device(0).compute_capability(), (8, 9))
        virtual_arch = f"compute_{capability[0]}{capability[1]}"
        self.module = cuda.module_from_buffer(
            cuda_compile(KERNEL, no_extern_c=True, target="ptx", arch=virtual_arch)
        )
        self.greedy = self.module.get_function("greedy")
        self.d_result = cuda.mem_alloc(8)
        self.h_result = cuda.pagelocked_empty((1,), np.int64)
        self.graph = None
        self.encoder_graph = None
        self._generation = 0
        self.vocab = self.decoder.shapes["logits"][-1]
        if self.vocab > 128:
            raise ValueError("greedykernel requires<=128classes")
        for name in self.decoder.buffers:
            if name not in ["cross_k", "cross_v"]:
                cuda.memset_d8(
                    self.decoder.buffers[name],
                    0,
                    int(np.prod(self.decoder.shapes[name]))
                    * self.decoder.types[name].itemsize,
                )
        for name in self.encoder.buffers:
            cuda.memset_d8(
                self.encoder.buffers[name],
                0,
                int(np.prod(self.encoder.shapes[name]))
                * self.encoder.types[name].itemsize,
            )
        if capture:
            for _ in range(3):
                self.encoder.execute()
            self.stream.synchronize()
            stream = rt.cudaStream_t(self.stream.handle)
            check(
                rt.cudaStreamBeginCapture(
                    stream, rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                )
            )
            self.encoder.execute()
            g = check(rt.cudaStreamEndCapture(stream))
            self.encoder_graph = (g, check(rt.cudaGraphInstantiate(g, 0)))
            for _ in range(3):
                self.decode_step()
            self.stream.synchronize()
            stream = rt.cudaStream_t(self.stream.handle)
            check(
                rt.cudaStreamBeginCapture(
                    stream, rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                )
            )
            self.decode_step()
            g = check(rt.cudaStreamEndCapture(stream))
            self.graph = (g, check(rt.cudaGraphInstantiate(g, 0)))

    def decode_step(self):
        d = self.decoder
        d.execute()
        for src, dst in [("present_k", "past_k"), ("present_v", "past_v")]:
            cuda.memcpy_dtod_async(
                d.buffers[dst],
                d.buffers[src],
                int(np.prod(d.shapes[dst])) * d.types[dst].itemsize,
                self.stream,
            )
        self.greedy(
            d.buffers["logits"],
            np.int32(self.vocab),
            d.buffers["token"],
            d.buffers["position"],
            self.d_result,
            block=(128, 1, 1),
            grid=(1, 1),
            stream=self.stream,
        )
        cuda.memcpy_dtoh_async(self.h_result, self.d_result, self.stream)

    def preprocess(self, image):
        from .waterec_preprocessing import pack_image
        data, self._last_grid = pack_image(image, self.rope, self.max_tokens)
        return data

    def encode(self, image):
        """Encode one PIL image; prepared memory remains valid until the next encode.

        This hook permits an optional external orientation controller to reuse an
        upright encoder result. The default OCR path has no orientation correction.
        """
        if self._closed:
            raise RuntimeError("Recognizer has been closed")
        data = self.preprocess(image)
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
        self._generation += 1
        return {"inputs": data, "grid": self._last_grid, "generation": self._generation}

    def decode_encoded(self, prepared):
        """Decode the current encoder memory using original cached greedy semantics."""
        if prepared["generation"] != self._generation:
            raise ValueError("Prepared memory was replaced by a later encode call")
        data = prepared["inputs"]
        self.decoder.upload("padding", data["padding"])
        self.decoder.upload("token", np.array([[self.meta["bos"]]], np.int64))
        self.decoder.upload("position", np.array([0], np.int64))
        for name in ["past_k", "past_v"]:
            cuda.memset_d8_async(
                self.decoder.buffers[name],
                0,
                int(np.prod(self.decoder.shapes[name]))
                * self.decoder.types[name].itemsize,
                self.stream,
            )
        ids = []
        for _ in range(self.meta["max_len"]):
            if self.graph:
                check(
                    rt.cudaGraphLaunch(
                        self.graph[1], rt.cudaStream_t(self.stream.handle)
                    )
                )
            else:
                self.decode_step()
            self.stream.synchronize()
            token = int(self.h_result[0])
            if token == self.meta["eos"]:
                break
            ids.append(token)
        return "".join(self.meta["characters"][i] for i in ids)

    def infer(self, images):
        """Return raw decoded text; each supplied image runs individually at batch1."""
        return [self.decode_encoded(self.encode(image)) for image in images]

    def close(self):
        if self._closed:
            return
        if self.ctx is None:
            self._closed = True
            return
        with _owned_context(self.ctx):
            stream = getattr(self, "stream", None)
            if stream is not None:
                stream.synchronize()
            for name in ("graph", "encoder_graph"):
                graph = getattr(self, name, None)
                if graph:
                    check(rt.cudaGraphExecDestroy(graph[1]))
                    check(rt.cudaGraphDestroy(graph[0]))
                    setattr(self, name, None)
            decoder = getattr(self, "decoder", None)
            encoder = getattr(self, "encoder", None)
            if decoder is not None:
                if encoder is not None:
                    for name in ("cross_k", "cross_v"):
                        if decoder.buffers.get(name) is encoder.buffers.get(name):
                            decoder.buffers.pop(name, None)
                decoder.close()
            if encoder is not None:
                encoder.close()
            self.greedy = None
            self.module = None
            result = getattr(self, "d_result", None)
            if result is not None:
                result.free()
                self.d_result = None
            self.h_result = None
            self.stream = None
            _release_context(self.ctx)
            self.ctx = None
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
