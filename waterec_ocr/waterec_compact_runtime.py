"""Torch-free compact GPU preprocessing, geometry cache and deferred guard.

No orientation/batching support. Each instance owns one stream and is not
thread-safe. encode returns pending validation; standalone users must call
validate_encoded. decode checks the guard before returning any text.
SPDX-License-Identifier: Apache-2.0
"""
import hashlib,json
from pathlib import Path
import numpy as np
from .waterec_compact_runtime_base import TensorRTRecognizer,cuda,rt,check
from .waterec_gpu_preprocessing import _Geometry,GpuResize
from .waterec_workspace import ScratchPool
from pycuda.compiler import compile as cuda_compile

VALIDATOR=r'''#include <math.h>
extern "C" __global__ void validate(const float*s,const long long*ix,const bool*pad,int n,int keep,int budget,int*error){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n&&!isfinite(s[i]))atomicOr(error,1);if(i<keep){int k=n<keep?n:keep;long long j=ix[i];if(j<0||j>=budget||(i&&j<=ix[i-1]))atomicOr(error,2);bool expected=i>=k;if(pad[i]!=expected||(i<k&&j>=n)||(i>=k&&j<n))atomicOr(error,4);}}'''

def validator_virtual_arch(capability):
    """Virtual PTX compatible with the device; no claim of edge qualification."""
    major,minor=min(tuple(capability),(8,9))
    return f'compute_{major}{minor}'

class _GeometryUploads:
    def __init__(self):self.key=None;self.pending=None
    def ensure(self,key,data,upload,event,stream):
        if self.key==key:return False
        if self.pending is not None:self.pending.synchronize()
        for name,array in data.items():upload(name,array)
        self.pending=event();self.pending.record(stream);self.key=key;return True

class CompactRecognizer(TensorRTRecognizer):
    def __init__(self,deployment,capture=True):
        folder=Path(deployment);m=json.loads((folder/'metadata.json').read_text())
        if m.get('family')!='waterec_compact' or m.get('orientation'):raise ValueError('Expected compact deployment')
        if type(m.get('max_tokens')) is not int or not 64<=m['max_tokens']<=2048:raise ValueError('Compact FP16 TensorRT budget must be64..2048; native/CPU ONNX budgets are separate')
        build=json.loads((folder/'compact_build.json').read_text())
        if not build.get('pass') or build['metadata_sha256']!=hashlib.sha256((folder/'metadata.json').read_bytes()).hexdigest():raise ValueError('Missing/mismatched compact build')
        for name,record in build['records'].items():
            if record['engine_sha256']!=hashlib.sha256((folder/(name+'.engine')).read_bytes()).hexdigest():raise ValueError('Engine hash mismatch')
        self.workspace_pool=ScratchPool(cuda.mem_alloc);self.geometry_uploads=_GeometryUploads()
        super().__init__(folder,capture=capture,workspace_pool=self.workspace_pool)
        try:
            self.geometry=_Geometry(self.rope,max_tokens=self.input_tokens);self.gpu_resize=GpuResize(self.stream,max_tokens=self.input_tokens);self.validation_module=cuda.module_from_buffer(cuda_compile(VALIDATOR,no_extern_c=True,target='ptx',arch=validator_virtual_arch(cuda.Device(0).compute_capability())));self.validator=self.validation_module.get_function('validate');self.validation=cuda.mem_alloc(4);self.h_validation=cuda.pagelocked_empty((1,),np.int32)
        except BaseException:self.close();raise
    def encode(self,image):
        if self._closed:raise RuntimeError('Closed')
        if image.mode!='RGB':image=image.convert('RGB')
        w,h=self.geometry.dimensions(image);ph,pw=h//4,w//4;n=ph*pw
        if not 1<=n<=self.input_tokens:raise ValueError('Expected positive valid prefix within budget')
        data=dict(self.geometry.constants(ph,pw));self.gpu_resize(image,(w,h),self.encoder.buffers['patches']);self.geometry_uploads.ensure((ph,pw),data,self.encoder.upload,lambda:cuda.Event(flags=cuda.event_flags.DISABLE_TIMING),self.stream)
        if self.encoder_graph:check(rt.cudaGraphLaunch(self.encoder_graph[1],rt.cudaStream_t(self.stream.handle)))
        else:self.encoder.execute()
        cuda.memset_d8_async(self.validation,0,4,self.stream);self.validator(self.encoder.buffers['selection_scores'],self.encoder.buffers['selected_indices'],self.encoder.buffers['selected_padding'],np.int32(n),np.int32(self.memory_tokens),np.int32(self.input_tokens),self.validation,block=(256,1,1),grid=((self.input_tokens+255)//256,1),stream=self.stream);cuda.memcpy_dtoh_async(self.h_validation,self.validation,self.stream);self._generation+=1;self._last_grid=(ph,pw)
        return {'inputs':data,'grid':self._last_grid,'generation':self._generation,'validN':n,'validation_pending':True}
    def _handle(self,prepared):
        if self._closed:raise RuntimeError('Closed')
        if prepared['generation']!=self._generation:raise ValueError('Stale prepared memory')
    def _flag(self,prepared):
        if int(self.h_validation[0]):raise ValueError('Invalid selection flag: '+str(int(self.h_validation[0])))
        prepared['validation_pending']=False
    def validate_encoded(self,prepared):
        self._handle(prepared);self.stream.synchronize();self._flag(prepared);return prepared
    def decode_encoded(self,prepared):
        self._handle(prepared);text=super().decode_encoded(prepared);self._flag(prepared);return text
    def close(self):
        if getattr(self,'_closed',True):return
        stream=getattr(self,'stream',None)
        if stream is not None:stream.synchronize()
        if hasattr(self,'validation'):self.validation.free();del self.validation
        self.validator=None;self.validation_module=None
        if getattr(self,'gpu_resize',None) is not None:self.gpu_resize.close();self.gpu_resize=None
        super().close();self.workspace_pool.close()
