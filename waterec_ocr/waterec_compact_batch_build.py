"""Torch-free portable fixedbatch builder; actualGPUqualification still required.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import hashlib,json
from .waterec_compact_batch_assets import load_metadata,verify_graphs,sha

def build_compact_batch(deployment):
 folder=Path(deployment);m=load_metadata(folder);verify_graphs(folder,m);B=m['static_batch']
 if any((folder/(n+'.engine')).exists() for n in ['encoder','decoder']):raise FileExistsError('Use a new outputdirectory; engines alreadyexist')
 records={}
 import tensorrt as trt
 for name in ['encoder','decoder']:
  graph=folder/(name+'.onnx');logger=trt.Logger(trt.Logger.WARNING);b=trt.Builder(logger);net=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(net,logger);parsed=parser.parse(graph.read_bytes());errors=[str(parser.get_error(i)) for i in range(parser.num_errors)];(folder/(name+'_parser.json')).write_text(json.dumps({'parsed':parsed,'errors':errors},indent=2)+'\n');assert parsed,'Parserfailurepreserved'
  barriers=[];scorer=[]
  for i in range(net.num_layers):
   layer=net.get_layer(i)
   for j in range(layer.num_outputs):
    t=layer.get_output(j)
    if name=='encoder' and t.name in m['QK_barriers']:net.mark_output(t);barriers.append({'name':t.name,'shape':list(t.shape),'dtype':str(t.dtype)})
   if layer.type in (trt.LayerType.NORMALIZATION,trt.LayerType.SOFTMAX) or '/scorer/' in layer.name:
    if layer.type not in (trt.LayerType.CONSTANT,trt.LayerType.SHAPE):layer.precision=trt.float32
   if '/scorer/' in layer.name:scorer.append(layer.name)
  if name=='encoder':assert len(barriers)==24 and scorer and sorted(r['shape'][2] for r in barriers)==[512]*12+[2048]*12
  cfg=b.create_builder_config();cfg.set_flag(trt.BuilderFlag.FP16);cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS);cfg.max_aux_streams=0;cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256<<20);cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
  if hasattr(trt.PreviewFeature,'RUNTIME_ACTIVATION_RESIZE_10_10'):cfg.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10,True)
  profile=b.create_optimization_profile()
  for i in range(net.num_inputs):
   t=net.get_input(i);s=tuple(m['shapes'][name][t.name]);ret=profile.set_shape(t.name,s,s,s);assert ret is not False and tuple(tuple(x) for x in profile.get_shape(t.name))==(s,s,s)
  cfg.add_optimization_profile(profile);blob=b.build_serialized_network(net,cfg);assert blob is not None;target=folder/(name+'.engine');target.write_bytes(blob);rt=trt.Runtime(logger);eng=rt.deserialize_cuda_engine(bytes(blob));assert eng is not None and eng.num_aux_streams==0;ins=folder/(name+'_inspector.json');ins.write_text(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON));io=[{'name':eng.get_tensor_name(i),'shape':list(eng.get_tensor_shape(eng.get_tensor_name(i))),'dtype':str(eng.get_tensor_dtype(eng.get_tensor_name(i))),'mode':str(eng.get_tensor_mode(eng.get_tensor_name(i)))} for i in range(eng.num_io_tensors)];records[name]={'engine_sha256':sha(target),'graph_sha256':sha(graph),'inspector_sha256':sha(ins),'bytes':target.stat().st_size,'aux_streams':eng.num_aux_streams,'workspace_bytes':eng.device_memory_size_v2,'IO':io,'QK_barriers':barriers};eng=None;rt=None
 (folder/'build.json').write_text(json.dumps({'pass':True,'batch':B,'metadata_sha256':sha(folder/'metadata.json'),'records':records,'scope':'Buildonly; target-deviceFIT/output/task/resourcequalification required beforedeployment claims.'},indent=2)+'\n');return folder
