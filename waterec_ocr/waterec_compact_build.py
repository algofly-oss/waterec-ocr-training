"""Torch-free dedicated compact TensorRT builder; no import-time GPU work.

Preserves the measured FP16-allowed/PREFER FP32 norm/softmax/scorer and24
post-RoPE Q/K barriers. Flags are requests, not accumulator-precision proof.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import hashlib,json

def build_compact(deployment):
    folder=Path(deployment);m=json.loads((folder/'metadata.json').read_text())
    if m.get('family')!='waterec_compact' or m.get('static_batch')!=1 or m['max_tokens']<64 or not 1<=m['memory_tokens']<=m['max_tokens']:raise ValueError('Expected compact static B1 assets')
    if type(m.get('max_tokens')) is not int or not 64<=m['max_tokens']<=2048:raise ValueError('Compact FP16 TensorRT budget must be64..2048; native/CPU ONNX budgets are separate')
    import tensorrt as trt
    n=m['max_tokens'];k=m['memory_tokens'];cache=(m['decoder_layers'],1,m['decoder_heads'],25,m['cache_head_dim'])
    shapes={'encoder_cross_kv':{'patches':(1,n,48),'cosine':(1,n,32),'sine':(1,n,32),'padding':(1,n)},'decoder_cached_step':{'token':(1,1),'position':(1,),'cross_k':(cache[0],1,cache[2],k,cache[4]),'cross_v':(cache[0],1,cache[2],k,cache[4]),'padding':(1,k),'past_k':cache,'past_v':cache}}
    for name in shapes:
        if (folder/(name+'.engine')).exists():raise FileExistsError('Existing engine; use a separate output directory')
    records={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,inputs in shapes.items():
        graph=folder/(name+'.onnx')
        if sha(graph)!=m['graph_sha256'][graph.name]:raise ValueError('Graph binding mismatch')
        logger=trt.Logger(trt.Logger.WARNING);builder=trt.Builder(logger);net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH));parser=trt.OnnxParser(net,logger);parsed=parser.parse(graph.read_bytes());errors=[str(parser.get_error(i)) for i in range(parser.num_errors)];(folder/(name+'_parser.json')).write_text(json.dumps({'parsed':parsed,'errors':errors},indent=2)+'\n')
        if not parsed:raise RuntimeError('TensorRT parser failed; details preserved')
        barriers=[];scorer=[]
        for i in range(net.num_layers):
            layer=net.get_layer(i)
            for j in range(layer.num_outputs):
                tensor=layer.get_output(j)
                if name=='encoder_cross_kv' and tensor.name in m['QK_barriers']:net.mark_output(tensor);barriers.append({'name':tensor.name,'shape':list(tensor.shape),'dtype':str(tensor.dtype)})
            if layer.type in (trt.LayerType.NORMALIZATION,trt.LayerType.SOFTMAX) or '/scorer/' in layer.name:
                if layer.type not in (trt.LayerType.CONSTANT,trt.LayerType.SHAPE):layer.precision=trt.float32
            if '/scorer/' in layer.name:scorer.append(layer.name)
        if name=='encoder_cross_kv' and (len(barriers)!=24 or not scorer):raise RuntimeError('Missing connected Q/K barriers or scorer')
        cfg=builder.create_builder_config();cfg.set_flag(trt.BuilderFlag.FP16);cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS);cfg.max_aux_streams=0;cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,256<<20);cfg.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        if hasattr(trt.PreviewFeature,'RUNTIME_ACTIVATION_RESIZE_10_10'):cfg.set_preview_feature(trt.PreviewFeature.RUNTIME_ACTIVATION_RESIZE_10_10,True)
        profile=builder.create_optimization_profile()
        for i in range(net.num_inputs):
            tensor=net.get_input(i);shape=inputs[tensor.name];ret=profile.set_shape(tensor.name,shape,shape,shape)
            if ret is False or tuple(tuple(v) for v in profile.get_shape(tensor.name))!=(shape,shape,shape):raise ValueError('Invalid input profile: '+tensor.name)
        cfg.add_optimization_profile(profile);blob=builder.build_serialized_network(net,cfg)
        if blob is None:raise RuntimeError('TensorRT build failed')
        target=folder/(name+'.engine');target.write_bytes(blob);runtime=trt.Runtime(logger);engine=runtime.deserialize_cuda_engine(bytes(blob));inspector=folder/(name+'_inspector.json');inspector.write_text(engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON));records[name]={'graph_sha256':sha(graph),'engine_sha256':sha(target),'inspector_sha256':sha(inspector),'bytes':target.stat().st_size,'workspace_bytes':engine.device_memory_size_v2,'QK_barriers':barriers,'scorer_FP32_requested':scorer};engine=None;runtime=None
    (folder/'compact_build.json').write_text(json.dumps({'pass':True,'records':records,'tensorrt':trt.__version__,'metadata_sha256':sha(folder/'metadata.json'),'scope':'Build only, target-device numerical/output parity and task qualification required before timing.'},indent=2)+'\n');return folder
