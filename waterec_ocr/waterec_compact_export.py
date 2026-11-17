"""Dedicated experimental compact cached ONNX export; standard R/RS paths unchanged.

Algebra follows the measured rank-MLP/image-TopK WATE deployment. CPU export
is not TensorRT or task-accuracy qualification. No weights are bundled.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import copy,hashlib,json
import numpy as np
import torch
from torch import nn
from torch.onnx import symbolic_helper
from .waterec_compact_models import load_compact_checkpoint
from .waterec_export import rotary_real,project_cross_kv,pack_images,CachedDecoderStep

class StableRaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx,scores,keep,budget):
        return torch.argsort(scores[0],descending=True,stable=True)[:keep].sort().values
    @staticmethod
    @symbolic_helper.parse_args('v','i','i')
    def symbolic(g,scores,keep,budget):
        def c(v):return g.op('Constant',value_t=torch.as_tensor(v))
        axes=c([0]);s=g.op('Squeeze',scores,axes);K=c([keep])
        values,_=g.op('TopK',s,K,axis_i=0,largest_i=1,sorted_i=1,outputs=2)
        cutoff=g.op('Gather',values,c([keep-1]),axis_i=0)
        above=g.op('Greater',s,cutoff);ties=g.op('Equal',s,cutoff)
        ai=g.op('Cast',above,to_i=6);ti=g.op('Cast',ties,to_i=6)
        count=g.op('ReduceSum',ai,axes,keepdims_i=1)
        needed=g.op('Sub',c(torch.tensor([keep],dtype=torch.int32)),count)
        rank=g.op('CumSum',ti,c(torch.tensor(0,dtype=torch.int32)),exclusive_i=0,reverse_i=0)
        chosen=g.op('Or',above,g.op('And',ties,g.op('LessOrEqual',rank,needed)))
        keys=g.op('Where',chosen,c(torch.arange(budget,dtype=torch.float32)),c(torch.tensor(16777216.,dtype=torch.float32)))
        indices,_=g.op('TopK',keys,K,axis_i=0,largest_i=0,sorted_i=1,outputs=2)
        return g.op('Cast',indices,to_i=7)

class SelectedDenseEncoder(nn.Module):
    def __init__(self,model):
        super().__init__();self.body=model.native.encoder;self.scorer=model.scorer
        self.after=model.after_blocks;self.keep=model.keep_tokens;self.budget=model.config['tokens']
    def forward(self,patches,cosine,sine,padding):
        e=self.body;x=e.pos_drop(e.patch_embed(patches));b,n,c=x.shape;mask=padding[:,None,None,:]
        for i,block in enumerate(e.blocks):
            a=block.attn;qkv=a.qkv(block.norm1(x)).reshape(b,n,3,a.num_heads,c//a.num_heads).permute(2,0,3,1,4)
            q=rotary_real(qkv[0],cosine,sine);k=rotary_real(qkv[1],cosine,sine);v=qkv[2]
            attn=(((q*a.scale)@k.transpose(-2,-1)).masked_fill(mask,float('-inf')).softmax(-1))
            y=(a.attn_drop(attn)@v).transpose(1,2).reshape(b,n,c)
            x=x+block.drop_path(a.proj_drop(a.proj(y)));x=x+block.drop_path(block.mlp(block.norm2(x)))
            if i==self.after-1:
                scores=self.scorer(x.float()).squeeze(-1)
                ix=StableRaster.apply(scores.masked_fill(padding,float('-inf')),self.keep,self.budget)
                x=x.index_select(1,ix);cosine=cosine.index_select(1,ix);sine=sine.index_select(1,ix)
                padding=padding.index_select(1,ix);mask=padding[:,None,None,:];n=self.keep
        return e.norm(x),padding,ix,scores

class EncoderCrossKV(nn.Module):
    def __init__(self,model):
        super().__init__();self.encoder=SelectedDenseEncoder(model);self.decoder_layers=model.native.decoder.decoder.layers
    def forward(self,patches,cosine,sine,padding):
        memory,pad,ix,scores=self.encoder(patches,cosine,sine,padding)
        k,v=project_cross_kv(memory,self.decoder_layers)
        return k,v,ix,scores,pad

def padded_pack(images,encoder,budget):
    if len(images)!=1:raise ValueError('Compact deployment is static batch one')
    values=pack_images(images,encoder);n=values[0].shape[1]
    if not 1<=n<=budget:raise ValueError('Expected positive actual patches within budget')
    return tuple(torch.nn.functional.pad(a,(0,0,0,budget-n)) if a.ndim==3 else torch.nn.functional.pad(a,(0,budget-n),value=True) for a in values)

def export_compact(checkpoint,output):
    """Strict typed checkpoint -> separate B1 encoder/new decoder/metadata assets."""
    import onnx
    model=load_compact_checkpoint(checkpoint,'cpu').eval();budget=model.config['tokens'];keep=model.keep_tokens
    if model.native.decoder.out_channels-2>128:raise ValueError('Current cached TensorRT greedy supports <=128 output classes')
    if budget<64 or budget>=16777216:raise ValueError('Configured budget must be64..16777215 for exact raster float keys')
    if model.native.decoder.decoder.norm is not None or any(l.norm_first for l in model.native.decoder.decoder.layers):raise ValueError('Unsupported decoder normalization')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    for name in ['encoder_cross_kv.onnx','decoder_cached_step.onnx','metadata.json','rope.npz']:
        if (output/name).exists():raise FileExistsError(output/name)
    enc=EncoderCrossKV(model).eval();dec=CachedDecoderStep(copy.deepcopy(model.native.decoder)).eval()
    args=(torch.zeros(1,budget,48),torch.ones(1,budget,32),torch.zeros(1,budget,32),torch.zeros(1,budget,dtype=torch.bool))
    layers=len(model.native.decoder.decoder.layers);heads=model.native.decoder.nhead;dim=model.native.decoder.d_model//heads;past=torch.zeros(layers,1,heads,25,dim);cross=torch.zeros(layers,1,heads,keep,dim)
    options=dict(opset_version=17,dynamo=False)
    torch.onnx.export(enc,args,output/'encoder_cross_kv.onnx',input_names=['patches','cosine','sine','padding'],output_names=['cross_k','cross_v','selected_indices','selection_scores','selected_padding'],**options)
    torch.onnx.export(dec,(torch.tensor([[model.native.decoder.bos]]),torch.tensor([0]),cross,cross.clone(),torch.zeros(1,keep,dtype=torch.bool),past,past.clone()),output/'decoder_cached_step.onnx',input_names=['token','position','cross_k','cross_v','padding','past_k','past_v'],output_names=['logits','present_k','present_v'],**options)
    graph=onnx.load(output/'encoder_cross_kv.onnx');onnx.checker.check_model(graph)
    barriers=[n.output[0] for n in graph.graph.node if n.op_type=='Cast' and n.input[0].startswith('/encoder/Reshape')]
    if len(barriers)!=24:raise RuntimeError('Expected24 post-real-RoPE Q/K barriers')
    onnx.checker.check_model(onnx.load(output/'decoder_cached_step.onnx'))
    meta={'family':'waterec_compact','schema':1,'graph_type':'rank_mlp_topk_cached_greedy','compact_architecture':model.architecture_metadata(),'static_batch':1,'max_tokens':budget,'memory_tokens':keep,'min_tokens':64,'min_actual_tokens':1,'patch':4,'encoder_layers':12,'decoder_layers':layers,'decoder_heads':heads,'cache_head_dim':dim,'max_len':25,'characters':model.native.characters,'bos':model.native.decoder.bos,'eos':model.native.decoder.eos,'task':model.config['task'],'variant':model.config['variant'],'QK_barriers':barriers,'normalization':model.config.get('normalization','none'),'parameters':sum(p.numel() for p in model.parameters()),'qualification':'CPU export only; target-device parity required','precision_policy':'FP16 allowed; normalization/softmax/scorer prefer FP32, not internal accumulator guarantee'}
    from .waterec_charset import metadata
    meta.update(metadata(model.native))
    if isinstance(checkpoint,(str,Path)):meta['checkpoint_sha256']=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    frequency=1/(model.native.encoder.rope_theta**(torch.arange(0,64,4).float()/64));angles=torch.arange(budget).float()[:,None]*frequency;np.savez(output/'rope.npz',cosine=angles.cos().numpy(),sine=angles.sin().numpy())
    meta['graph_sha256']={n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in ['encoder_cross_kv.onnx','decoder_cached_step.onnx']};(output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');return output
