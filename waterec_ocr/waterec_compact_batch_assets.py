"""Torch-free fixed-batch compact asset contract. SPDX-License-Identifier: Apache-2.0"""
from pathlib import Path
import hashlib,json

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_metadata(folder):
 m=json.loads((Path(folder)/'metadata.json').read_text())
 if not isinstance(m,dict):raise ValueError('Expected fixed-batch metadata dictionary')
 B=m.get('static_batch')
 if m.get('family')!='waterec_compact_batch' or type(B) is not int or B not in [1,2,4,8] or m.get('batch')!=B:raise ValueError('Expected dedicated compact fixedbatch1/2/4/8 assets')
 if m.get('max_tokens')!=2048 or m.get('memory_tokens')!=512 or m.get('patch')!=4 or m.get('max_len')!=25:raise ValueError('Batch deployment currently supports2048input/512memory/25steps only')
 if (m.get('decoder_layers'),m.get('decoder_heads'),m.get('cache_head_dim'))!=(2,12,32):raise ValueError('Unsupported cached decoder architecture')
 if 'character_vocabulary' in m:raise ValueError('Unicode fixed-batch ONNX/TensorRT is not qualified; use the separate static B1 export')
 a=m.get('compact_architecture')
 if not isinstance(a,dict) or a.get('family')!='waterec_compact' or a.get('architecture')!='rank_mlp_image_topk' or a.get('schema')!=1:raise ValueError('Expected original typed compact architecture')
 if a.get('after_blocks')!=6 or a.get('target_keep')!=512:raise ValueError('Unsupported compact selection architecture')
 chars=m.get('characters')
 if not isinstance(chars,list) or len(chars) not in [97,98] or not all(isinstance(c,str) for c in chars) or len(set(chars))!=len(chars) or chars[0]!='</s>' or any(not isinstance(c,str) or len(c)!=1 for c in chars[1:-2]):raise ValueError('Expected original95/96 output characters')
 original=['</s>']+(Path(__file__).parent/'_vendor/waterec/characters.txt').read_text().splitlines()
 if a.get('characters')!=chars:raise ValueError('Architecture and deployment alphabets differ')
 if chars!=original+([' '] if ' ' in chars else [])+['<s>','<pad>']:raise ValueError('Fixed-batch assets require the original ordered ASCII alphabet; Unicode uses separate static B1 export')
 if m.get('output_classes')!=len(chars)-2:raise ValueError('Outputhead/codec count mismatch')
 if m.get('eos')!=0 or m.get('bos')!=len(chars)-2 or chars[-2:]!=['<s>','<pad>']:raise ValueError('Unsupported native EOS/BOS policy')
 if not isinstance(m.get('QK_barriers'),list) or not all(isinstance(n,str) for n in m['QK_barriers']) or len(m.get('QK_barriers',[]))!=24 or len(set(m['QK_barriers']))!=24:raise ValueError('Expected24orderedQK outputs')
 if not isinstance(m.get('checkpoint_sha256'),str) or len(m['checkpoint_sha256'])!=64 or any(c not in '0123456789abcdef' for c in m['checkpoint_sha256']):raise ValueError('Missing typedcheckpoint binding')
 cache=[2,B,12,25,32];cross=[2,B,12,512,32];expected={'encoder':{'patches':[B,2048,48],'cosine':[B,2048,32],'sine':[B,2048,32],'padding':[B,2048]},'decoder':{'token':[B,1],'position':[1],'cross_k':cross,'cross_v':cross,'padding':[B,512],'past_k':cache,'past_v':cache}}
 if m.get('shapes')!=expected:raise ValueError('Invalid fixedbatch shape contract')
 return m

def verify_graphs(folder,m):
 for name in ['encoder.onnx','decoder.onnx','rope.npz']:
  if sha(Path(folder)/name)!=m['graph_sha256'][name]:raise ValueError('Asset hash mismatch:'+name)

def load_batch_checkpoint_payload(checkpoint):
 """Inspect typed metadata before importing/constructing model classes."""
 import torch
 try:payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
 except Exception as error:raise ValueError('Expected a standalone typed compact checkpoint') from error
 if not isinstance(payload,dict) or not isinstance(payload.get('compact_architecture'),dict):raise ValueError('Explicit typed compact checkpoint required')
 a=payload['compact_architecture'];config=a.get('config')
 if a.get('family')!='waterec_compact' or a.get('architecture')!='rank_mlp_image_topk' or not isinstance(config,dict):raise ValueError('Explicit typed compact checkpoint required')
 if a.get('schema')==2 or config.get('additional_characters') or 'character_vocabulary' in payload:raise ValueError('Unicode fixed-batch export is not qualified; omit --static-batch for the separate static B1 path')
 if a.get('schema')!=1 or config.get('tokens')!=2048 or a.get('target_keep')!=512 or a.get('after_blocks')!=6:raise ValueError('Fixed-batch export requires original ASCII2048input/512keep/after6 architecture')
 chars=a.get('characters');original=['</s>']+(Path(__file__).parent/'_vendor/waterec/characters.txt').read_text().splitlines()
 if not isinstance(chars,list) or chars!=original+([' '] if ' ' in chars else [])+['<s>','<pad>']:raise ValueError('Fixed-batch export requires original ordered ASCII alphabet')
 return payload
