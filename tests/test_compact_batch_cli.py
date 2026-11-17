"""Lightweight fixedcapacity dispatch/guards; no realweights, CUDA or engine."""
import copy,contextlib,io,json,os,subprocess,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
from waterec_ocr import waterec_compact_cli as cli
from waterec_ocr.waterec_compact_batch_assets import load_metadata
ROOT=Path(__file__).resolve().parents[1]
def metadata(B=2):
 chars=['</s>']+(ROOT/'waterec_ocr/_vendor/waterec/characters.txt').read_text().splitlines()+[' ','<s>','<pad>'];cache=[2,B,12,25,32];cross=[2,B,12,512,32]
 return {'family':'waterec_compact_batch','schema':1,'static_batch':B,'batch':B,'max_tokens':2048,'memory_tokens':512,'patch':4,'max_len':25,'decoder_layers':2,'decoder_heads':12,'cache_head_dim':32,'characters':chars,'output_classes':len(chars)-2,'bos':len(chars)-2,'eos':0,'task':'price','normalization':'none','checkpoint_sha256':'0'*64,'compact_architecture':{'family':'waterec_compact','schema':1,'architecture':'rank_mlp_image_topk','after_blocks':6,'target_keep':512,'characters':chars},'QK_barriers':['qk'+str(i) for i in range(24)],'shapes':{'encoder':{'patches':[B,2048,48],'cosine':[B,2048,32],'sine':[B,2048,32],'padding':[B,2048]},'decoder':{'token':[B,1],'position':[1],'cross_k':cross,'cross_v':cross,'padding':[B,512],'past_k':cache,'past_v':cache}}}
class BatchCLITests(unittest.TestCase):
 def test_fixed_metadata_rejects_unicode_and_corruption_before_GPU(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for kind in ['family','EOS','BOS','PAD','shape','vocab','Unicode','badchars']:
    m=copy.deepcopy(metadata())
    if kind=='family':m['family']='waterec'
    elif kind=='EOS':m['eos']=1
    elif kind=='BOS':m['bos']+=1
    elif kind=='PAD':m['characters'][-1],m['characters'][-2]=m['characters'][-2],m['characters'][-1]
    elif kind=='shape':m['shapes']['decoder']['position']=[2]
    elif kind=='vocab':m['output_classes']=100
    elif kind=='Unicode':m['character_vocabulary']={'schema':1}
    else:m['characters']=[{}]*98
    (root/'metadata.json').write_text(json.dumps(m))
    with self.subTest(kind=kind),self.assertRaises(ValueError):load_metadata(root)
   self.assertNotIn('tensorrt',sys.modules);self.assertNotIn('pycuda.driver',sys.modules)
 def test_unicode_checkpoint_batch_rejects_before_model_or_GPU_import(self):
  code="""import sys,tempfile,torch
from pathlib import Path
from waterec_ocr.waterec_compact_batch_export import export_compact_batch
from waterec_ocr.waterec_compact_cli import main
assert 'waterec_ocr.waterec_compact_models' not in sys.modules
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'unicode.pt';torch.save({'compact_architecture':{'family':'waterec_compact','architecture':'rank_mlp_image_topk','schema':2,'config':{'tokens':2048,'additional_characters':['€']}}},p)
 for B in [1,2,4,8]:
  out=Path(d)/('B'+str(B))
  try:export_compact_batch(p,out,B)
  except ValueError as e:assert 'Unicode fixed-batch' in str(e)
  else:raise AssertionError('Unicode batch accepted')
  assert not out.exists()
 try:main(['export','--checkpoint',str(p),'--deployment',str(Path(d)/'cli'),'--static-batch','2'])
 except SystemExit as e:assert e.code==2
 else:raise AssertionError('CLI accepted Unicode batch')
assert not any(x in sys.modules for x in ['waterec_ocr.waterec_compact_models','tensorrt','pycuda.driver'])
"""
  result=subprocess.run([sys.executable,'-c',code],env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True)
  self.assertEqual(result.returncode,0,result.stderr)
 def test_batch_export_and_legacy_export_dispatch(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cp=root/'model.pt';cp.write_bytes(b'fixture only');calls=[];batch=types.ModuleType('waterec_ocr.waterec_compact_batch_export');legacy=types.ModuleType('waterec_ocr.waterec_compact_export');batch.export_compact_batch=lambda checkpoint,out,B:(calls.append(('batch',B)) or out);legacy.export_compact=lambda checkpoint,out:(calls.append(('legacy',1)) or out)
   with patch.dict(sys.modules,{batch.__name__:batch,legacy.__name__:legacy}),contextlib.redirect_stdout(io.StringIO()):
    self.assertEqual(cli.main(['export','--checkpoint',str(cp),'--deployment',str(root/'B4'),'--static-batch','4']),0)
    self.assertEqual(cli.main(['export','--checkpoint',str(cp),'--deployment',str(root/'B1')]),0)
   self.assertEqual(calls,[('batch',4),('legacy',1)])
 def test_true_batch_dispatch_tail_counts_and_console_return(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);deploy=root/'deployment';deploy.mkdir();m=metadata();(deploy/'metadata.json').write_text(json.dumps(m));data=root/'images';data.mkdir()
   for i in range(3):Image.new('RGB',(2,2),(i,0,0)).save(data/f'{i}.png')
   calls=[];mod=types.ModuleType('waterec_ocr.waterec_compact_batch_runtime')
   class Fake:
    def __init__(self,folder):self.meta=m
    def infer_batch(self,images):calls.append(len(images));return [str(im.getpixel((0,0))[0]) for im in images]
    def infer(self,images):raise AssertionError('B1 loop/fallback used')
    def close(self):pass
   class NoGPU:
    def close(self):return {'scope':'fake CPU test only'}
   mod.BatchCompactRecognizer=Fake;output=root/'result.json'
   with patch.dict(sys.modules,{mod.__name__:mod}),patch.object(cli,'_MemorySampler',NoGPU),contextlib.redirect_stdout(io.StringIO()):
    rc=cli.main(['benchmark','--backend','tensorrt','--deployment',str(deploy),'--folder',str(data),'--batch-size','2','--repeats','3','--predecoded','--output',str(output)])
   self.assertEqual(rc,0);self.assertEqual(calls[:5],[2]*5);self.assertEqual(calls[5:],[2,1]*3);result=json.loads(output.read_text());self.assertEqual(result['timed_images'],9);self.assertEqual(result['physical_timed_slots'],12);self.assertEqual(result['static_deployment_batch'],2);self.assertEqual(result['full_batch_ms']['count'],3);self.assertEqual(len(result['partial_batches']),3);self.assertFalse(result['static_deployment_B1'])
 def test_capacity_rejects_before_backend_import(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'metadata.json').write_text(json.dumps(metadata(2)));image=root/'unused.png';image.write_bytes(b'not decoded')
   with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:cli.main(['infer','--backend','tensorrt','--deployment',str(root),'--image',str(image),'--batch-size','4'])
   self.assertEqual(error.exception.code,2);self.assertNotIn('tensorrt',sys.modules);self.assertNotIn('pycuda.driver',sys.modules)
if __name__=='__main__':unittest.main()
