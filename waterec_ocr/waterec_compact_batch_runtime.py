"""Portable Torch-free fixedcapacity compact runtime.
SPDX-License-Identifier: Apache-2.0
Distinct and reused policies run SAME engines; original B1 API unchanged.
"""
from pathlib import Path
import hashlib,json
import numpy as np
from .waterec_compact_batch_assets import load_metadata
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pad_indices(real,capacity):
 if not 1<=real<=capacity or capacity not in [1,2,4,8]:raise ValueError('Expected positive real<=fixedcapacity1/2/4/8')
 return list(range(real))+[0]*(capacity-real)
def patch_offset(pointer,row,capacity):
 if not 0<=row<capacity:raise ValueError('Bad row')
 return np.uintp(int(pointer)+row*2048*48*np.dtype(np.float32).itemsize)
def barrier_plan(names,shapes,types,B):
 if len(names)!=24 or len(set(names))!=24:raise ValueError('Expected24uniqueorderedbarriers')
 caps=[0,0];slots={}
 for i,name in enumerate(names):
  expected=(B,6,2048 if i<12 else 512,64)
  if tuple(shapes[name])!=expected:raise ValueError('UnexpectedQKshape:'+name)
  slot=i%2;slots[name]=slot;caps[slot]=max(caps[slot],int(np.prod(expected))*np.dtype(types[name]).itemsize)
 for slot in [0,1]:
  if len({np.dtype(types[n]).str for n in names[slot::2]})!=1:raise ValueError('Unexpectedmixedslotdtype')
 return caps,slots
class EOSRows:
 def __init__(self,capacity,real):self.real=real;self.finished=np.zeros(capacity,bool);self.ids=[[] for _ in range(capacity)]
 def consume(self,tokens):
  for row,value in enumerate(np.asarray(tokens).reshape(-1).tolist()):
   if not self.finished[row] and value!=0:self.ids[row].append(int(value))
   if value==0:self.finished[row]=True
  return bool(self.finished[:self.real].all())
class Engine:
 def __init__(self,path,stream,cuda,trt,shapes,barriers=(),policy='distinct',borrowed=None,B=1):
  self.cuda,self.trt,self.stream=cuda,trt,stream;self.context=None;self.engine=None;self.runtime=None;self.buffers={};self.host={};self.owned={};self.borrowed=set();self.shapes={};self.types={};self.closed=False;self.barrier_layout={}
  try:
   self.runtime=trt.Runtime(trt.Logger(trt.Logger.ERROR));self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes());assert self.engine is not None and self.engine.num_aux_streams==0;self.context=self.engine.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
   for i in range(self.engine.num_io_tensors):
    name=self.engine.get_tensor_name(i)
    if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
     wanted=tuple(shapes[name]);ret=self.context.set_input_shape(name,wanted);assert ret is not False and tuple(self.context.get_tensor_shape(name))==wanted
   assert not self.context.infer_shapes();self.workspace_bytes=self.context.update_device_memory_size_for_shapes()
   for i in range(self.engine.num_io_tensors):
    name=self.engine.get_tensor_name(i);shape=tuple(self.context.get_tensor_shape(name));assert all(v>0 for v in shape);assert self.engine.get_tensor_format(name)==trt.TensorFormat.LINEAR;self.shapes[name]=shape;self.types[name]=np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
   def allocate(size):
    pointer=cuda.mem_alloc(max(1,size));self.owned[int(pointer)]=pointer;return pointer
   shared=[];slots={}
   if barriers:
    assert all(n in self.shapes for n in barriers);caps,slots=barrier_plan(barriers,self.shapes,self.types,B)
    if policy=='reused':shared=[allocate(c) for c in caps]
    self.barrier_layout={'names':list(barriers),'slot_capacities_bytes':caps,'policy':policy}
   for name,shape in self.shapes.items():
    if name in (borrowed or {}):
     assert name in ['cross_k','cross_v'] and self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT;pointer,other_shape,other_type=borrowed[name];assert shape==tuple(other_shape) and self.types[name]==other_type;self.buffers[name]=pointer;self.borrowed.add(name)
    elif name in slots and policy=='reused':self.buffers[name]=shared[slots[name]]
    else:self.buffers[name]=allocate(int(np.prod(shape))*self.types[name].itemsize)
    self.context.set_tensor_address(name,int(self.buffers[name]))
   if barriers:
    unique=len({int(self.buffers[n]) for n in barriers});assert unique==(2 if policy=='reused' else 24);self.barrier_layout['unique_physical_buffers']=unique;self.barrier_layout['actual_output_types']={n:self.types[n].str for n in barriers}
  except BaseException:self.close();raise
 def bind_workspace(self,pointer,size):
  if size<self.workspace_bytes:raise ValueError('Insufficientworkspace')
  self.context.set_device_memory(int(pointer),size)
 def host_array(self,name):
  if name not in self.host:self.host[name]=self.cuda.pagelocked_empty(self.shapes[name],self.types[name])
  return self.host[name]
 def upload(self,name,array):
  np.copyto(self.host_array(name),array);self.cuda.memcpy_htod_async(self.buffers[name],self.host[name],self.stream)
 def execute(self):
  if not self.context.execute_async_v3(self.stream.handle):raise RuntimeError('TRTexecutionfailed')
 def close(self):
  if self.closed:return
  self.closed=True;self.context=None
  for pointer in self.owned.values():pointer.free()
  self.owned.clear();self.borrowed.clear();self.buffers.clear();self.host.clear();self.engine=None;self.runtime=None
GREEDY=r'''extern "C" __global__ void greedy(const float*z,int vocab,long long*token,long long*position,long long*result){__shared__ float score[128];__shared__ int ix[128];int lane=threadIdx.x,row=blockIdx.x;score[lane]=lane<vocab?z[row*vocab+lane]:-1.e30f;ix[lane]=lane;__syncthreads();for(int s=64;s;s>>=1){if(lane<s){float a=score[lane],b=score[lane+s];int ai=ix[lane],bi=ix[lane+s];if(b>a||(b==a&&bi<ai)){score[lane]=b;ix[lane]=bi;}}__syncthreads();}if(lane==0){token[row]=ix[0];result[row]=ix[0];if(row==0)*position+=1;}}'''
VALIDATOR=r'''#include <math.h>
extern "C" __global__ void validate(const float*s,const long long*ix,const bool*pad,const int*counts,int B,int*error){int i=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y;if(row>=B)return;int n=counts[row],k=n<512?n:512;if(i<n&&!isfinite(s[row*2048+i]))atomicOr(error+row,1);if(i<512){long long j=ix[row*512+i];if(j<0||j>=2048||(i&&j<=ix[row*512+i-1]))atomicOr(error+row,2);if(pad[row*512+i]!=(i>=k)||(i<k&&j>=n)||(i>=k&&j<n))atomicOr(error+row,4);}}'''
class Recognizer:
 def __init__(self,folder,policy='reused',capture=True):
  if policy not in ['distinct','reused']:raise ValueError('Unknownpolicy')
  self.closed=False;self.ctx=None;self.stream=None;self.encoder=None;self.decoder=None;self.resizers=[];self.extra=[];self.workspaces=[];self.graph=None;self.encoder_graph=None;self.geometry_key=None;self.geometry_event=None;self.generation=0
  folder=Path(folder);meta=load_metadata(folder);build=json.loads((folder/'build.json').read_text());assert build['pass'] and build['metadata_sha256']==sha(folder/'metadata.json') and build['batch']==meta['static_batch'];assert sha(folder/'rope.npz')==meta['graph_sha256']['rope.npz']
  for name,x in build['records'].items():assert sha(folder/(name+'.engine'))==x['engine_sha256']
  import pycuda.driver as cuda
  import tensorrt as trt
  from pycuda.compiler import compile as compile_cuda
  from cuda.bindings import runtime as rt
  from waterec_ocr.waterec_compact_runtime_base import check,_release_context
  from waterec_ocr.waterec_gpu_preprocessing import GpuResize,_Geometry
  self.cuda,self.rt,self.check,self.release=cuda,rt,check,_release_context;self.meta=meta;self.capacity=B=meta['static_batch'];self.policy=policy
  for name,x in build['records'].items():assert sha(folder/(name+'.engine'))==x['engine_sha256']
  cuda.init();self.ctx=cuda.Device(0).retain_primary_context();self.ctx.push();self.stream=cuda.Stream()
  try:
   self.geometry=_Geometry(dict(np.load(folder/'rope.npz')),max_tokens=2048);self.encoder=Engine(folder/'encoder.engine',self.stream,cuda,trt,meta['shapes']['encoder'],meta['QK_barriers'],policy,B=B);borrowed={n:(self.encoder.buffers[n],self.encoder.shapes[n],self.encoder.types[n]) for n in ['cross_k','cross_v']};self.decoder=Engine(folder/'decoder.engine',self.stream,cuda,trt,meta['shapes']['decoder'],borrowed=borrowed,B=B)
   assert self.decoder.shapes['position']==(1,) and self.decoder.shapes['token']==(B,1) and self.decoder.shapes['logits']==(B,1,len(meta['characters'])-2);assert self.decoder.types['logits']==np.dtype('float32')
   for n in ['past_k','past_v','present_k','present_v']:assert self.decoder.shapes[n]==(2,B,12,25,32) and self.decoder.types[n]==np.dtype('float32')
   sizes=[self.encoder.workspace_bytes,self.decoder.workspace_bytes]
   if policy=='reused':
    pointer=cuda.mem_alloc(max(1,max(sizes)));self.workspaces.append(pointer)
    for engine in [self.encoder,self.decoder]:engine.bind_workspace(pointer,max(1,max(sizes)))
   else:
    for engine,size in zip([self.encoder,self.decoder],sizes):pointer=cuda.mem_alloc(max(1,size));self.workspaces.append(pointer);engine.bind_workspace(pointer,max(1,size))
   self.resizers=[GpuResize(self.stream,max_tokens=2048) for _ in range(B)]
   arch=min(cuda.Device(0).compute_capability(),(8,9));virtual=f'compute_{arch[0]}{arch[1]}';self.module=cuda.module_from_buffer(compile_cuda(GREEDY+'\n'+VALIDATOR,no_extern_c=True,target='ptx',arch=virtual));self.greedy=self.module.get_function('greedy');self.validator=self.module.get_function('validate');self.result=cuda.mem_alloc(B*8);self.extra.append(self.result);self.validation=cuda.mem_alloc(B*4);self.extra.append(self.validation);self.counts=cuda.mem_alloc(B*4);self.extra.append(self.counts);self.h_result=cuda.pagelocked_empty((B,),np.int64);self.h_validation=cuda.pagelocked_empty((B,),np.int32);self.h_counts=cuda.pagelocked_empty((B,),np.int32)
   for engine in [self.encoder,self.decoder]:
    for name,ptr in engine.buffers.items():
     if name not in engine.borrowed:cuda.memset_d8_async(ptr,0,int(np.prod(engine.shapes[name]))*engine.types[name].itemsize,self.stream)
   self.decoder.upload('token',np.full((B,1),meta['bos'],np.int64));self.decoder.upload('position',np.array([0],np.int64))
   if capture:
    for _ in range(3):self.encoder.execute()
    self.stream.synchronize();st=rt.cudaStream_t(self.stream.handle);check(rt.cudaStreamBeginCapture(st,rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal));self.encoder.execute();graph=check(rt.cudaStreamEndCapture(st));self.encoder_graph=(graph,check(rt.cudaGraphInstantiate(graph,0)))
    for _ in range(3):self._step()
    self.stream.synchronize();check(rt.cudaStreamBeginCapture(st,rt.cudaStreamCaptureMode.cudaStreamCaptureModeGlobal));self._step();graph=check(rt.cudaStreamEndCapture(st));self.graph=(graph,check(rt.cudaGraphInstantiate(graph,0)))
  except BaseException:self.close();raise
 def _step(self):
  d=self.decoder;d.execute()
  for source,target in [('present_k','past_k'),('present_v','past_v')]:self.cuda.memcpy_dtod_async(d.buffers[target],d.buffers[source],int(np.prod(d.shapes[target]))*d.types[target].itemsize,self.stream)
  self.greedy(d.buffers['logits'],np.int32(len(self.meta['characters'])-2),d.buffers['token'],d.buffers['position'],self.result,block=(128,1,1),grid=(self.capacity,1),stream=self.stream);self.cuda.memcpy_dtoh_async(self.h_result,self.result,self.stream)
 def encode(self,images):
  if self.closed:raise RuntimeError('Closed')
  mapping=pad_indices(len(images),self.capacity);prepared=[]
  for image in images:
   image=image if image.mode=='RGB' else image.convert('RGB');w,h=self.geometry.dimensions(image);n=(h//4)*(w//4)
   if not 1<=n<=2048:raise ValueError('Badnativegrid')
   prepared.append((image,(w,h),(h//4,w//4)))
  grids=tuple(prepared[i][2] for i in mapping)
  if grids!=self.geometry_key:
   if self.geometry_event is not None:self.geometry_event.synchronize()
   constants=[self.geometry.constants(*grid) for grid in grids]
   for name in ['cosine','sine','padding']:self.encoder.upload(name,np.concatenate([c[name] for c in constants],axis=0))
   np.copyto(self.h_counts,np.array([h*w for h,w in grids],np.int32));self.cuda.memcpy_htod_async(self.counts,self.h_counts,self.stream);self.geometry_event=self.cuda.Event(flags=self.cuda.event_flags.DISABLE_TIMING);self.geometry_event.record(self.stream);self.geometry_key=grids
  for row,(image,size,_) in enumerate(prepared):self.resizers[row](image,size,patch_offset(self.encoder.buffers['patches'],row,self.capacity))
  rowbytes=2048*48*4
  for row in range(len(images),self.capacity):self.cuda.memcpy_dtod_async(int(patch_offset(self.encoder.buffers['patches'],row,self.capacity)),self.encoder.buffers['patches'],rowbytes,self.stream)
  if self.encoder_graph:self.check(self.rt.cudaGraphLaunch(self.encoder_graph[1],self.rt.cudaStream_t(self.stream.handle)))
  else:self.encoder.execute()
  self.cuda.memset_d8_async(self.validation,0,self.capacity*4,self.stream);self.validator(self.encoder.buffers['selection_scores'],self.encoder.buffers['selected_indices'],self.encoder.buffers['selected_padding'],self.counts,np.int32(self.capacity),self.validation,block=(256,1,1),grid=(8,self.capacity),stream=self.stream);self.cuda.memcpy_dtoh_async(self.h_validation,self.validation,self.stream);self.generation+=1
  return {'generation':self.generation,'real':len(images),'mapping':mapping,'grids':grids,'validation_pending':True}
 def _handle(self,p):
  if self.closed:raise RuntimeError('Closed')
  if p['generation']!=self.generation:raise ValueError('Stalehandle')
 def _flag(self,p):
  if np.any(self.h_validation):raise ValueError('Invalidselectionflags:'+str(self.h_validation.tolist()))
  p['validation_pending']=False
 def validate_encoded(self,p):self._handle(p);self.stream.synchronize();self._flag(p);return p
 def decode_encoded(self,p):
  self._handle(p);d=self.decoder;self.cuda.memcpy_dtod_async(d.buffers['padding'],self.encoder.buffers['selected_padding'],self.capacity*512,self.stream);d.upload('token',np.full((self.capacity,1),self.meta['bos'],np.int64));d.upload('position',np.array([0],np.int64))
  for n in ['past_k','past_v']:self.cuda.memset_d8_async(d.buffers[n],0,int(np.prod(d.shapes[n]))*d.types[n].itemsize,self.stream)
  status=EOSRows(self.capacity,p['real']);trace=[]
  for position in range(25):
   if self.graph:self.check(self.rt.cudaGraphLaunch(self.graph[1],self.rt.cudaStream_t(self.stream.handle)))
   else:self._step()
   self.stream.synchronize();self._flag(p);before=status.finished.copy();done=status.consume(self.h_result);trace.append({'position':position,'tokens':self.h_result.tolist(),'finished_before':before.tolist(),'position_shape':[1]})
   if done:break
  texts=[''.join(self.meta['characters'][v] for v in row) for row in status.ids];self.trace={'physical_batch':self.capacity,'real_batch':p['real'],'mapping':p['mapping'],'grids':p['grids'],'encoder_calls':1,'decoder_steps':trace,'all_physical_texts':texts,'barrier_layout':self.encoder.barrier_layout,'workspace_bytes_requested':[self.encoder.workspace_bytes,self.decoder.workspace_bytes],'workspace_allocations':len(self.workspaces),'policy':self.policy};return texts[:p['real']]
 def infer(self,images):return self.decode_encoded(self.encode(images))
 def close(self):
  if self.closed:return
  self.closed=True
  if self.ctx is None:return
  if self.stream is not None:self.stream.synchronize()
  for name in ['graph','encoder_graph']:
   graph=getattr(self,name,None)
   if graph:self.check(self.rt.cudaGraphExecDestroy(graph[1]));self.check(self.rt.cudaGraphDestroy(graph[0]));setattr(self,name,None)
  for resizer in self.resizers:resizer.close();resizer.horizontal=None;resizer.pack=None;resizer.module=None;resizer.copy_done=None
  self.resizers=[]
  if self.decoder is not None:self.decoder.close();self.decoder=None
  if self.encoder is not None:self.encoder.close();self.encoder=None
  for ptr in self.workspaces+self.extra:ptr.free()
  self.workspaces=[];self.extra=[];self.greedy=None;self.validator=None;self.module=None;self.h_result=None;self.h_validation=None;self.h_counts=None;self.stream=None;self.release(self.ctx);self.ctx=None

Recognizer.infer_batch=Recognizer.infer
BatchCompactRecognizer=Recognizer
