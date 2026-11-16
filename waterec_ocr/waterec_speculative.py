"""Experimental TensorRT speculative OCR with an unchanged AR target.

Verification is mathematically greedy under identical logits. Block/cached
FP16 arithmetic may differ; deployment accuracy and parity must be measured.
This module intentionally does not import Torch.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from .waterec_runtime import (
    _owned_context,
    Engine,
    TensorRTRecognizer,
    cuda,
    rt,
    check,
    cuda_compile,
)
from .waterec_gpu_preprocessing import GpuInputRecognizer

KERNEL = r"""
extern "C" __global__ void argmax25(const float*x,int n,int*y,long long*input,int bos,int draft){__shared__ float s[128];__shared__ int ids[128];int i=threadIdx.x,row=blockIdx.x;s[i]=i<n?x[row*n+i]:-1.e30f;ids[i]=i;__syncthreads();for(int d=64;d;d>>=1){if(i<d){float a=s[i],b=s[i+d];if(b>a||(b==a&&ids[i+d]<ids[i])){s[i]=b;ids[i]=ids[i+d];}}__syncthreads();}if(i==0){y[row]=ids[0];if(draft){if(row==0)input[0]=bos;if(row<24)input[row+1]=ids[0];}}}
extern "C" __global__ void commit25(const int*draft,const int*verify,long long*history,int*status,long long*token,long long*position){if(threadIdx.x==0){for(int i=0;i<25;i++)history[i]=0;int count=25,done=1;for(int i=0;i<25;i++){int y=verify[i];history[i]=y;if(y==0){count=i+1;done=1;break;}if(draft[i]!=y){count=i+1;done=count==25;break;}}status[0]=count;status[1]=done;*token=history[count-1];*position=count;}}
"""


class SpeculativeRecognizer:
    def __init__(
        self,
        base_deployment,
        speculative_deployment=None,
        gpu_preprocess=False,
        capture=True,
        share_workspace=True,
    ):
        self._closed = False
        self.model = None
        self.workspace_pool = None
        self.graph = None
        self.aliases = []
        self.draft = None
        self.verifier = None
        self.module = None
        try:
            self._initialize(
                base_deployment,
                speculative_deployment,
                gpu_preprocess,
                capture,
                share_workspace,
            )
        except BaseException:
            self.close()
            raise

    def _initialize(
        self,
        base_deployment,
        speculative_deployment=None,
        gpu_preprocess=False,
        capture=True,
        share_workspace=True,
    ):
        self._closed = False
        self.folder = (
            Path(speculative_deployment)
            if speculative_deployment
            else Path(base_deployment) / "speculative"
        )
        meta = json.loads((self.folder / "metadata.json").read_text())
        base_meta = json.loads((Path(base_deployment) / "metadata.json").read_text())
        if meta.get("eos") != 0 or meta.get("bos") != 95:
            raise ValueError("Speculative commit profile requires EOS0 andBOS95")
        if (
            meta.get("family") != "waterec_speculative"
            or meta.get("max_tokens") != 512
            or meta.get("max_len") != 25
            or meta.get("vocab_size") != 95
        ):
            raise ValueError("Unsupported speculative ALPR profile")
        if meta["target_checkpoint_sha256"] != base_meta.get("checkpoint_sha256"):
            raise ValueError("Draft and target checkpoint SHA256 differ")
        if (
            meta["target_characters"] != base_meta.get("characters")
            or meta["bos"] != base_meta.get("bos")
            or meta["eos"] != base_meta.get("eos")
        ):
            raise ValueError("Draft and target character mapping differ")
        if base_meta.get("task") != "plate" or base_meta.get("max_tokens") != 512:
            raise ValueError("Speculative inference requires plate512 target")
        for name in ["draft", "verifier"]:
            side = json.loads((self.folder / (name + ".build.json")).read_text())
            payload = self.folder / (name + ".engine")
            if (
                side.get("identity", {}).get("metadata_sha256")
                != hashlib.sha256(
                    (self.folder / "metadata.json").read_bytes()
                ).hexdigest()
            ):
                raise ValueError("Speculative engine metadata digest mismatch")
            if (
                side.get("engine_sha256")
                != hashlib.sha256(payload.read_bytes()).hexdigest()
            ):
                raise ValueError("Speculative engine/build digest mismatch")
            if name == "draft" and len(side.get("attention_output_barriers", [])) != 1:
                raise ValueError(
                    "Draft engine requires the validated attention output barrier"
                )
        Base = GpuInputRecognizer if gpu_preprocess else TensorRTRecognizer
        if not gpu_preprocess and base_meta.get("orientation"):

            class OrientationBase(Base):
                def preprocess(self, image):
                    from .waterec_orientation import spatial_weights

                    data = super().preprocess(image)
                    data[self.meta["orientation"]["pool_input"]] = spatial_weights(
                        *self._last_grid
                    )
                    return data

            Base = OrientationBase
        from .waterec_workspace import ScratchPool

        self.workspace_pool = ScratchPool(cuda.mem_alloc) if share_workspace else None
        self.model = Base(
            base_deployment, capture=capture, workspace_pool=self.workspace_pool
        )
        self.stream = self.model.stream
        self.speculative_metadata = meta
        self.verifier = Engine(
            self.folder / "verifier.engine",
            self.stream,
            {
                "tokens": (1, 25),
                "cross_k": (2, 1, 12, 512, 32),
                "cross_v": (2, 1, 12, 512, 32),
                "padding": (1, 512),
            },
            workspace_pool=self.workspace_pool,
        )
        self.draft = Engine(
            self.folder / "draft.engine",
            self.stream,
            {"keys": (1, 12, 512, 32), "values": (1, 12, 512, 32), "padding": (1, 512)},
            workspace_pool=self.workspace_pool,
        )
        self.aliases = []

        def alias(engine, name, buffer):
            engine.buffers[name].free()
            engine.buffers[name] = buffer
            engine.context.set_tensor_address(name, int(buffer))
            self.aliases.append((engine, name))

        for key in ["cross_k", "cross_v"]:
            alias(self.verifier, key, self.model.encoder.buffers[key])
        for name, key in [("keys", "cross_k"), ("values", "cross_v")]:
            alias(self.draft, name, self.model.encoder.buffers[key])
        for engine in [self.draft, self.verifier]:
            alias(engine, "padding", self.model.decoder.buffers["padding"])
        for source, target in [("block_k", "past_k"), ("block_v", "past_v")]:
            if (
                self.verifier.types[source] != self.model.decoder.types[target]
                or self.verifier.shapes[source] != self.model.decoder.shapes[target]
            ):
                raise ValueError("Verifier and target cache layout differ")
        capability = cuda.Device(0).compute_capability()
        architecture = "compute_" + str(min(capability[0] * 10 + capability[1], 89))
        self.module = cuda.module_from_buffer(
            cuda_compile(KERNEL, no_extern_c=True, target="ptx", arch=architecture)
        )
        self.argmax = self.module.get_function("argmax25")
        self.commit = self.module.get_function("commit25")
        self.d_draft = cuda.mem_alloc(100)
        self.d_verify = cuda.mem_alloc(100)
        self.d_history = cuda.mem_alloc(200)
        self.d_status = cuda.mem_alloc(8)
        self.h_history = cuda.pagelocked_empty(25, np.int64)
        self.h_status = cuda.pagelocked_empty(2, np.int32)
        self.graph = None
        self.last = None
        if capture:
            for _ in range(3):
                self.block()
            self.stream.synchronize()
            stream = rt.cudaStream_t(self.stream.handle)
            check(
                rt.cudaStreamBeginCapture(
                    stream, rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal
                )
            )
            self.block()
            graph = check(rt.cudaStreamEndCapture(stream))
            self.graph = (graph, check(rt.cudaGraphInstantiate(graph, 0)))

    @property
    def meta(self):
        return self.model.meta

    @property
    def encoder(self):
        return self.model.encoder

    def encode(self, image):
        if self._closed:
            raise RuntimeError("Recognizer is closed")
        return self.model.encode(image)

    def block(self):
        self.draft.execute()
        self.argmax(
            self.draft.buffers["draft_logits"],
            np.int32(95),
            self.d_draft,
            self.verifier.buffers["tokens"],
            np.int32(self.model.meta["bos"]),
            np.int32(1),
            block=(128, 1, 1),
            grid=(25, 1),
            stream=self.stream,
        )
        self.verify()

    def verify(self):
        self.verifier.execute()
        self.argmax(
            self.verifier.buffers["logits"],
            np.int32(95),
            self.d_verify,
            self.verifier.buffers["tokens"],
            np.int32(self.model.meta["bos"]),
            np.int32(0),
            block=(128, 1, 1),
            grid=(25, 1),
            stream=self.stream,
        )
        self.commit(
            self.d_draft,
            self.d_verify,
            self.d_history,
            self.d_status,
            self.model.decoder.buffers["token"],
            self.model.decoder.buffers["position"],
            block=(1, 1, 1),
            grid=(1, 1),
            stream=self.stream,
        )
        for source, target in [("block_k", "past_k"), ("block_v", "past_v")]:
            cuda.memcpy_dtod_async(
                self.model.decoder.buffers[target],
                self.verifier.buffers[source],
                int(np.prod(self.verifier.shapes[source]))
                * self.verifier.types[source].itemsize,
                self.stream,
            )
        cuda.memcpy_dtoh_async(self.h_history, self.d_history, self.stream)
        cuda.memcpy_dtoh_async(self.h_status, self.d_status, self.stream)

    def decode_encoded(self, prepared):
        if self._closed:
            raise RuntimeError("Recognizer is closed")
        if prepared["generation"] != self.model._generation:
            raise ValueError("Prepared memory was replaced by a later encode call")
        self.model.decoder.upload("padding", prepared["inputs"]["padding"])
        if self.graph:
            check(
                rt.cudaGraphLaunch(self.graph[1], rt.cudaStream_t(self.stream.handle))
            )
        else:
            self.block()
        self.stream.synchronize()
        count, done = map(int, self.h_status)
        if not 1 <= count <= 25:
            raise RuntimeError("Invalid speculative prefix length")
        ids = self.h_history[:count].tolist()
        fallback = 0
        if not done:
            for _ in range(count, 25):
                if self.model.graph:
                    check(
                        rt.cudaGraphLaunch(
                            self.model.graph[1], rt.cudaStream_t(self.stream.handle)
                        )
                    )
                else:
                    self.model.decode_step()
                self.stream.synchronize()
                token = int(self.model.h_result[0])
                ids.append(token)
                fallback += 1
                if token == self.model.meta["eos"]:
                    break
        self.last = dict(
            verified=count, fallback=fallback, done_without_fallback=bool(done)
        )
        chars = []
        for token in ids:
            if token == self.model.meta["eos"]:
                break
            chars.append(self.model.meta["characters"][token])
        return "".join(chars)

    def infer(self, images):
        return [self.decode_encoded(self.encode(image)) for image in images]

    def close(self):
        if self._closed:
            return
        if self.model is not None:
            with _owned_context(self.model.ctx):
                self.stream.synchronize()
                if self.graph:
                    check(rt.cudaGraphExecDestroy(self.graph[1]))
                    check(rt.cudaGraphDestroy(self.graph[0]))
                    self.graph = None
                for engine, name in self.aliases:
                    engine.buffers.pop(name, None)
                self.aliases = []
                for engine in (self.draft, self.verifier):
                    if engine is not None:
                        engine.close()
                for name in ("d_draft", "d_verify", "d_history", "d_status"):
                    buffer = getattr(self, name, None)
                    if buffer is not None:
                        buffer.free()
                        setattr(self, name, None)
                self.argmax = None
                self.commit = None
                self.module = None
                self.h_history = None
                self.h_status = None
                self.stream = None
                self.model.close()
        if self.workspace_pool is not None:
            self.workspace_pool.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
