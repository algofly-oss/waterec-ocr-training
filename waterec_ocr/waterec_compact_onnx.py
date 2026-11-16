"""Torch-free experimental compact B1 ONNX inference; no host selection fallback.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .waterec_preprocessing import pack_image

class ONNXRecognizer:
    def __init__(self,deployment):
        folder=Path(deployment);self.meta=json.loads((folder/'metadata.json').read_text());m=self.meta
        if m.get('family')!='waterec_compact' or m.get('schema')!=1 or m.get('static_batch')!=1 or not 1<=m['memory_tokens']<=m['max_tokens'] or m['max_len']!=25:raise ValueError('Unsupported compact metadata')
        for n,h in m['graph_sha256'].items():
            if hashlib.sha256((folder/n).read_bytes()).hexdigest()!=h:raise ValueError('Graph hash mismatch: '+n)
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.encoder=ort.InferenceSession(str(folder/'encoder_cross_kv.onnx'),sess_options=options,providers=['CPUExecutionProvider']);self.decoder=ort.InferenceSession(str(folder/'decoder_cached_step.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
        self.rope=dict(np.load(folder/'rope.npz'));self.closed=False
    def infer_one(self,image):
        if self.closed:raise RuntimeError('Closed')
        m=self.meta;packed,grid=pack_image(image,self.rope,m['max_tokens']);n=grid[0]*grid[1]
        if not 1<=n<=m['max_tokens']:raise ValueError('Expected positive actual native grid within budget')
        k,v,indices,scores,padding=self.encoder.run(None,packed);valid=min(n,m['memory_tokens'])
        if not np.isfinite(scores[0,:n]).all() or not np.isfinite(k).all() or not np.isfinite(v).all() or not np.array_equal(padding,np.arange(m['memory_tokens'])[None]>=valid):raise ValueError('Invalid selection outputs')
        chosen=indices[:valid]
        if np.any(chosen<0) or np.any(chosen>=n) or np.any(chosen[1:]<=chosen[:-1]):raise ValueError('Invalid raster indices')
        cache=(m['decoder_layers'],1,m['decoder_heads'],25,m['cache_head_dim']);pk=np.zeros(cache,np.float32);pv=np.zeros_like(pk);token=np.array([[m['bos']]],np.int64);ids=[]
        for position in range(25):
            z,pk,pv=self.decoder.run(None,dict(token=token,position=np.array([position],np.int64),cross_k=k,cross_v=v,padding=padding,past_k=pk,past_v=pv))
            if not all(np.isfinite(x).all() for x in [z,pk,pv]):raise ValueError('Nonfinite decoder output')
            value=int(z.argmax(-1).item())
            if value==m['eos']:break
            if not 0<=value<len(m['characters']):raise ValueError('Invalid character ID')
            ids.append(value);token[0,0]=value
        return ''.join(m['characters'][i] for i in ids)
    def infer(self,images):return [self.infer_one(image) for image in images]
    def close(self):self.encoder=None;self.decoder=None;self.closed=True
