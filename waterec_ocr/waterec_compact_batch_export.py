"""Typed checkpoint -> separate fixedB rowwise encoder and new decoder.
Current portable batch profile is input2048→memory512, B1/2/4/8.
SPDX-License-Identifier: Apache-2.0
"""
from pathlib import Path
import copy,json
import numpy as np
from .waterec_compact_batch_assets import sha,load_metadata,load_batch_checkpoint_payload

def export_compact_batch(checkpoint,output,static_batch=1):
 if type(static_batch) is not int or static_batch not in [1,2,4,8]:raise ValueError('Expected fixedbatch1/2/4/8')
 payload=load_batch_checkpoint_payload(checkpoint)
 import torch
 from .waterec_compact_models import load_compact_checkpoint
 from .waterec_compact_batch_graphs import EncoderCrossKVRows,CachedDecoderStep
 model=load_compact_checkpoint(payload,'cpu').eval()
 if model.config.get('additional_characters'):raise ValueError('Unicode fixed-batch export is not qualified; omit --static-batch for the separate static B1 path')
 if model.architecture_metadata()['after_blocks']!=6:raise ValueError('Batch export requires selection after block6')
 if model.config['tokens']!=2048 or model.keep_tokens!=512:raise ValueError('Batch export currently requires2048input/512keep typedcheckpoint')
 if model.native.decoder.decoder.norm is not None or any(l.norm_first for l in model.native.decoder.decoder.layers):raise ValueError('Unsupported decoder normalization')
 if model.native.decoder.tgt_word_prj.out_features!=len(model.native.characters)-2:raise ValueError('Decoderhead/codec count mismatch')
 B=static_batch;output=Path(output);output.mkdir(parents=True,exist_ok=True)
 for name in ['encoder.onnx','decoder.onnx','metadata.json','rope.npz','build.json']:
  if (output/name).exists():raise FileExistsError(output/name)
 layers=len(model.native.decoder.decoder.layers);heads=model.native.decoder.nhead;dim=model.native.decoder.d_model//heads
 if (layers,heads,dim)!=(2,12,32):raise ValueError('Unsupported decoder architecture')
 enc=EncoderCrossKVRows(model).eval();dec=CachedDecoderStep(copy.deepcopy(model.native.decoder)).eval();args=(torch.zeros(B,2048,48),torch.ones(B,2048,32),torch.zeros(B,2048,32),torch.zeros(B,2048,dtype=torch.bool));past=torch.zeros(2,B,12,25,32);cross=torch.zeros(2,B,12,512,32)
 with torch.inference_mode():
  torch.onnx.export(enc,args,output/'encoder.onnx',input_names=['patches','cosine','sine','padding'],output_names=['cross_k','cross_v','selected_indices','selection_scores','selected_padding'],opset_version=17,dynamo=False)
  torch.onnx.export(dec,(torch.full((B,1),model.native.decoder.bos,dtype=torch.long),torch.tensor([0]),cross,cross.clone(),torch.zeros(B,512,dtype=torch.bool),past,past.clone()),output/'decoder.onnx',input_names=['token','position','cross_k','cross_v','padding','past_k','past_v'],output_names=['logits','present_k','present_v'],opset_version=17,dynamo=False)
 import onnx
 eg=onnx.load(output/'encoder.onnx');onnx.checker.check_model(eg);onnx.checker.check_model(onnx.load(output/'decoder.onnx'));barriers=[n.output[0] for n in eg.graph.node if n.op_type=='Cast' and n.input[0].startswith('/encoder/Reshape')]
 if len(barriers)!=24 or sum(n.op_type=='TopK' for n in eg.graph.node)!=2 or sum(n.op_type=='CumSum' for n in eg.graph.node)!=1:raise RuntimeError('Unexpected stable-selection/barrier graph')
 frequency=1/(model.native.encoder.rope_theta**(torch.arange(0,64,4).float()/64));angles=torch.arange(2048).float()[:,None]*frequency;np.savez(output/'rope.npz',cosine=angles.cos().numpy(),sine=angles.sin().numpy())
 cache=[2,B,12,25,32];kv=[2,B,12,512,32];m={'family':'waterec_compact_batch','schema':1,'static_batch':B,'batch':B,'max_tokens':2048,'memory_tokens':512,'min_actual_tokens':1,'min_tokens':64,'patch':4,'max_len':25,'decoder_layers':2,'decoder_heads':12,'cache_head_dim':32,'output_classes':model.native.decoder.tgt_word_prj.out_features,'characters':list(model.native.characters),'bos':model.native.decoder.bos,'eos':model.native.decoder.eos,'task':model.config['task'],'variant':model.config['variant'],'normalization':model.config.get('normalization','none'),'parameters':sum(p.numel() for p in model.parameters()),'compact_architecture':model.architecture_metadata(),'checkpoint_sha256':sha(checkpoint),'QK_barriers':barriers,'shapes':{'encoder':{'patches':[B,2048,48],'cosine':[B,2048,32],'sine':[B,2048,32],'padding':[B,2048]},'decoder':{'token':[B,1],'position':[1],'cross_k':kv,'cross_v':kv,'padding':[B,512],'past_k':cache,'past_v':cache}},'graph_sha256':{n:sha(output/n) for n in ['encoder.onnx','decoder.onnx','rope.npz']},'qualification':'CPU export only; actualtargetGPUparity required. Fixedcapacitynotdynamicbatch; tail duplicates firstreal and discardsdummyoutputs.'}
 (output/'metadata.json').write_text(json.dumps(m,indent=2)+'\n');load_metadata(output);return output
