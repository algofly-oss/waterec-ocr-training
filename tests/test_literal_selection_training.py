"""Exercise the actual training selector/resume path with tiny CPU fake modules."""
from pathlib import Path
import pytest
import torch
from waterec_ocr import waterec_training as training
from waterec_ocr.metrics import score_pairs, EXACT_POLICY

class Encoder(torch.nn.Linear):
 def __init__(self):super().__init__(1,1)
 def forward(self,images):return super().forward(torch.stack(images).reshape(len(images),1))
class Decoder(torch.nn.Linear):
 ignore_index=2
 def __init__(self):super().__init__(1,2)
 def forward_train(self,memory,target):return super().forward(memory).unsqueeze(1)
class Fake(torch.nn.Module):
 def __init__(self):super().__init__();self.encoder=Encoder();self.decoder=Decoder()
 def encode_labels(self,labels,device=None):return torch.tensor([[1,0]]*len(labels),device=device)

def setup(monkeypatch,scores):
 monkeypatch.setattr(training,'create_native',lambda config:Fake())
 monkeypatch.setattr(training,'alphabet_metadata',lambda model:{})
 monkeypatch.setattr(training,'_loader',lambda *args:[([torch.ones(1,1)],['A'],['a.png'])])
 it=iter(scores)
 def evaluate(*args):
  result=score_pairs(['A','B'],['A','B' if next(it)==2 else 'X'])
  result.update(cer=0,numeric_correct=999)  # These must never break an exact tie.
  return result
 monkeypatch.setattr(training,'evaluate_records',evaluate)
 return dict(seed=1,precision='fp32',device='cpu',task='price',normalization='none',warmup_epochs=0)

def test_actual_fit_exact_ties_and_current_resume(monkeypatch,tmp_path):
 cfg=setup(monkeypatch,[1,1]);summary=training._fit([{'label':'A'}],[{'label':'B'}],cfg,tmp_path,2,False)
 assert summary['selected_epoch']==1 and summary['selection_policy']==EXACT_POLICY
 setup(monkeypatch,[2]);summary=training._fit([{'label':'A'}],[{'label':'B'}],cfg,tmp_path,3,True)
 assert summary['selected_epoch']==3

def test_legacy_resume_rejected_before_model_allocation(monkeypatch,tmp_path):
 torch.save({'best':[1,0]},tmp_path/'last.pt')
 monkeypatch.setattr(training,'create_native',lambda _:pytest.fail('Model must not be allocated'))
 with pytest.raises(ValueError,match='legacy/incompatible'):
  training._fit([],[],{},tmp_path,1,True)
