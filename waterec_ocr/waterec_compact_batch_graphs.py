"""Portable fixed-B row-wise compact export math.
SPDX-License-Identifier: Apache-2.0"""
import torch
from torch import nn
from torch.onnx import symbolic_helper
from waterec_ocr.waterec_export import rotary_real,project_cross_kv,CachedDecoderStep

class StableRasterRows(torch.autograd.Function):
 @staticmethod
 def forward(ctx,scores):return torch.argsort(scores,dim=1,descending=True,stable=True)[:,:512].sort(dim=1).values
 @staticmethod
 def symbolic(g,scores):
  def c(v):return g.op('Constant',value_t=torch.as_tensor(v))
  axes=c([1]);K=c([512]);values,_=g.op('TopK',scores,K,axis_i=1,largest_i=1,sorted_i=1,outputs=2);cutoff=g.op('Gather',values,c([511]),axis_i=1);above=g.op('Greater',scores,cutoff);ties=g.op('Equal',scores,cutoff);ai=g.op('Cast',above,to_i=6);ti=g.op('Cast',ties,to_i=6);count=g.op('ReduceSum',ai,axes,keepdims_i=1);needed=g.op('Sub',c(torch.tensor([512],dtype=torch.int32)),count);rank=g.op('CumSum',ti,c(torch.tensor(1,dtype=torch.int32)),exclusive_i=0,reverse_i=0);chosen=g.op('Or',above,g.op('And',ties,g.op('LessOrEqual',rank,needed)));keys=g.op('Where',chosen,c(torch.arange(2048,dtype=torch.float32)[None]),c(torch.tensor(16777216.,dtype=torch.float32)));indices,_=g.op('TopK',keys,K,axis_i=1,largest_i=0,sorted_i=1,outputs=2);return g.op('Cast',indices,to_i=7)

def gather(values,indices):
 if values.ndim==3:return torch.gather(values,1,indices[:,:,None].expand(-1,-1,values.shape[-1]))
 return torch.gather(values,1,indices)

class SelectedDenseRows(nn.Module):
 def __init__(self,m):super().__init__();self.body=m.native.encoder;self.scorer=m.scorer
 def forward(self,patches,cosine,sine,padding):
  e=self.body;x=e.pos_drop(e.patch_embed(patches));b,n,c=x.shape;mask=padding[:,None,None,:]
  for i,block in enumerate(e.blocks):
   a=block.attn;qkv=a.qkv(block.norm1(x)).reshape(b,n,3,a.num_heads,c//a.num_heads).permute(2,0,3,1,4);q=rotary_real(qkv[0],cosine,sine);k=rotary_real(qkv[1],cosine,sine);v=qkv[2];attn=(((q*a.scale)@k.transpose(-2,-1)).masked_fill(mask,float('-inf')).softmax(-1));y=(a.attn_drop(attn)@v).transpose(1,2).reshape(b,n,c);x=x+block.drop_path(a.proj_drop(a.proj(y)));x=x+block.drop_path(block.mlp(block.norm2(x)))
   if i==5:
    scores=self.scorer(x.float()).squeeze(-1);indices=StableRasterRows.apply(scores.masked_fill(padding,float('-inf')));x=gather(x,indices);cosine=gather(cosine,indices);sine=gather(sine,indices);padding=gather(padding,indices);mask=padding[:,None,None,:];n=512
  return e.norm(x),padding,indices,scores
class EncoderCrossKVRows(nn.Module):
 def __init__(self,m):super().__init__();self.encoder=SelectedDenseRows(m);self.layers=m.native.decoder.decoder.layers
 def forward(self,patches,cosine,sine,padding):
  mem,pad,ix,scores=self.encoder(patches,cosine,sine,padding);k,v=project_cross_kv(mem,self.layers);return k,v,ix,scores,pad

def pack_rows(images,encoder):
 from waterec_ocr.waterec_export import pack_images
 values=pack_images(images,encoder);n=values[0].shape[1]
 if not 1<=n<=2048:raise ValueError('Positive actual row lengths within2048 required')
 return tuple(torch.nn.functional.pad(v,(0,0,0,2048-n)) if v.ndim==3 else torch.nn.functional.pad(v,(0,2048-n),value=True) for v in values)
