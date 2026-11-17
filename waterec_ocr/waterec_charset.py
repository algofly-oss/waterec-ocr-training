"""Explicit append-only Unicode alphabet migration. Default vocabulary unchanged."""
import hashlib,json,unicodedata

def vocabulary_hash(chars):return hashlib.sha256(json.dumps(list(chars),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def validate_additions(additional,existing=()):
 if not isinstance(additional,(list,tuple)):raise ValueError('additional_characters must be an ordered list')
 out=[]
 for c in additional:
  if not isinstance(c,str) or len(c)!=1 or unicodedata.normalize('NFC',c)!=c or c.isspace() or unicodedata.category(c).startswith('C'):raise ValueError('Additional characters must be single NFC non-control/non-whitespace codepoints')
  if c in out or c in existing:raise ValueError('Duplicate or already supported additional character')
  out.append(c)
 return out

def extend_characters(model,additional,provenance=None):
 import torch
 from torch import nn
 old=list(model.characters);extra=validate_additions(additional,old)
 if not extra:return
 chars=old[:-2]+extra+old[-2:];d=model.decoder;dev=d.embedding.embedding.weight.device
 with torch.random.fork_rng(devices=[dev.index] if dev.type=='cuda' else []):
  emb=nn.Embedding(len(chars),d.d_model,padding_idx=d.embedding.embedding.padding_idx,device=dev,dtype=d.embedding.embedding.weight.dtype)
  head=nn.Linear(d.d_model,len(chars)-2,bias=False,device=dev,dtype=d.tgt_word_prj.weight.dtype)
 with torch.no_grad():
  for i,c in enumerate(old):
   j=chars.index(c);emb.weight[j].copy_(d.embedding.embedding.weight[i])
   if i<len(old)-2:head.weight[j].copy_(d.tgt_word_prj.weight[i])
  for c in extra:
   j=chars.index(c);emb.weight[j].copy_(d.embedding.embedding.weight[1:-2].mean(0));head.weight[j].copy_(d.tgt_word_prj.weight[1:].mean(0))
 model.characters=chars;d.embedding.embedding=emb;d.tgt_word_prj=head;d.out_channels=len(chars);d.bos=len(chars)-2;d.ignore_index=len(chars)-1
 previous=[c for c in model.config.get('additional_characters',[]) if c in old]
 model.config['additional_characters']=previous+extra
 model._character_provenance=provenance or {'source':'fresh_initialization','optimizer':'fresh; no optimizer-state migration'}

def metadata(model):
 if not model.config.get('additional_characters'):return {}
 return {'character_vocabulary':{'schema':1,'characters':list(model.characters),'sha256':vocabulary_hash(model.characters),'additional_characters':list(model.config['additional_characters']),'bos':model.decoder.bos,'pad':model.decoder.ignore_index,'eos':model.decoder.eos,'output_classes':model.decoder.out_channels-2,'initialization':'mean existing ordinary rows; no zero-shot or unchanged-argmax guarantee','provenance':model._character_provenance}}

def restore_metadata(model,payload):
 extra=model.config.get('additional_characters',[]);meta=payload.get('character_vocabulary')
 if not extra:
  if meta is not None:raise ValueError('Unexpected character metadata for legacy/default alphabet')
  return
 if not isinstance(meta,dict) or meta.get('schema')!=1:raise ValueError('Explicit character_vocabulary metadata required for extended checkpoint')
 model._character_provenance=meta.get('provenance')
 if not isinstance(model._character_provenance,dict) or metadata(model)['character_vocabulary']!=meta:raise ValueError('Ordered alphabet/hash/special-token/provenance metadata mismatch')

def validate_deployment(meta):
    from pathlib import Path
    if not isinstance(meta, dict):
        raise ValueError('Deployment metadata must be a dictionary')
    a=meta.get('character_vocabulary')
    if 'character_vocabulary' in meta and not isinstance(a, dict):
        raise ValueError('character_vocabulary must be a dictionary')
    chars=meta.get('characters')
    if not isinstance(chars, list) or not all(isinstance(c, str) for c in chars):
        raise ValueError('characters must be a list of strings')
    original=['</s>']+(Path(__file__).parent/'_vendor/waterec/characters.txt').read_text().splitlines()
    if a is None:
        if meta.get('family') in ('waterec','waterec_compact') and any(any(ord(c)>127 for c in token) for token in meta.get('characters', [])):
            raise ValueError('Nonlegacy alphabet requires versioned character metadata')
        return
    chars=meta.get('characters');bos=meta.get('bos');eos=meta.get('eos')
    additional=validate_additions(a.get('additional_characters'),original+[' ','<s>','<pad>'])
    expected=original+([' '] if ' ' in chars else [])+additional+['<s>','<pad>']
    if chars!=expected:raise ValueError('Declared extension order differs from characters')
    if a.get('schema')!=1 or chars!=a.get('characters') or vocabulary_hash(chars)!=a.get('sha256') or bos!=len(chars)-2 or eos!=0 or a.get('bos')!=bos or a.get('eos')!=eos or a.get('output_classes')!=len(chars)-2 or a.get('pad')!=len(chars)-1:
        raise ValueError('Deployment ordered alphabet/hash/shape metadata mismatch')
    if len(chars)-2>128:raise ValueError('Current TensorRT greedy supports <=128 classes')
