"""Torch-free fixedcapacity rowwise ONNX inference; no CPU selection fallback.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import numpy as np
from .waterec_compact_batch_assets import load_metadata,verify_graphs
from .waterec_compact_batch_runtime import pad_indices,EOSRows
from .waterec_preprocessing import pack_image
class BatchONNXRecognizer:
 def __init__(self,deployment):
  folder=Path(deployment);self.meta=load_metadata(folder);verify_graphs(folder,self.meta);self.capacity=self.meta['static_batch']
  import onnxruntime as ort
  options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
  self.encoder=ort.InferenceSession(str(folder/'encoder.onnx'),options,providers=['CPUExecutionProvider']);self.decoder=ort.InferenceSession(str(folder/'decoder.onnx'),options,providers=['CPUExecutionProvider']);self.rope=dict(np.load(folder/'rope.npz'));self.closed=False
  if self.decoder.get_inputs()[1].shape!=[1]:raise ValueError('Position must be global scalar[1]')
 def infer(self,images):
  if self.closed:raise RuntimeError('Closed')
  mapping=pad_indices(len(images),self.capacity);rows=[pack_image(im,self.rope,2048) for im in images];packed={name:np.concatenate([rows[i][0][name] for i in mapping],axis=0) for name in rows[0][0]};k,v,indices,scores,padding=self.encoder.run(None,packed)
  for row,i in enumerate(mapping):
   ph,pw=rows[i][1];n=ph*pw;valid=min(n,512);ix=indices[row]
   if not 1<=n<=2048 or not np.isfinite(scores[row,:n]).all() or not np.array_equal(padding[row],np.arange(512)>=valid):raise ValueError('Invalid selection outputs')
   if np.any(ix<0) or np.any(ix>=2048) or np.any(ix[1:]<=ix[:-1]) or np.any(ix[:valid]>=n) or np.any(ix[valid:]<n):raise ValueError('Invalid raster indices')
  if not np.isfinite(k).all() or not np.isfinite(v).all():raise ValueError('Nonfinite memory')
  B=self.capacity;pk=np.zeros((2,B,12,25,32),np.float32);pv=np.zeros_like(pk);token=np.full((B,1),self.meta['bos'],np.int64);status=EOSRows(B,len(images))
  for position in range(25):
   z,pk,pv=self.decoder.run(None,dict(token=token,position=np.array([position],np.int64),cross_k=k,cross_v=v,padding=padding,past_k=pk,past_v=pv))
   if not all(np.isfinite(x).all() for x in [z,pk,pv]):raise ValueError('Nonfinite decoder')
   token=z[:,0].argmax(-1).astype(np.int64)[:,None]
   if status.consume(token):break
  return [''.join(self.meta['characters'][i] for i in row) for row in status.ids[:len(images)]]
 infer_batch=infer
 def close(self):self.encoder=None;self.decoder=None;self.closed=True
