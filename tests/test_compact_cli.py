"""Lightweight CLI contracts: no trained weights, CUDA device or model forward."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
MODULE='waterec_ocr.waterec_compact_cli'

class CompactCLITests(unittest.TestCase):
    def run_python(self,arguments):
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='')
        return subprocess.run([sys.executable]+arguments,env=env,text=True,capture_output=True)

    def test_help_all_commands_needs_no_training_or_GPU_modules(self):
        for command in [[],['train'],['infer'],['export'],['build'],['evaluate'],['benchmark']]:
            with self.subTest(command=command):
                result=self.run_python(['-m',MODULE]+command+['--help'])
                self.assertEqual(result.returncode,0,result.stderr)
        code="import sys;import waterec_ocr.waterec_compact_cli;assert not any(n in sys.modules for n in ['torch','onnxruntime','tensorrt','pycuda.driver'])"
        result=self.run_python(['-c',code]);self.assertEqual(result.returncode,0,result.stderr)

    def test_standard_metadata_is_rejected_before_GPU_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'metadata.json').write_text(json.dumps({'family':'waterec'}))
            code="""import sys
from waterec_ocr.waterec_compact_cli import main
try:main(['build','--deployment',sys.argv[1]])
except SystemExit as error:assert error.code==2
else:raise AssertionError('Standard metadata accepted')
assert not any(n in sys.modules for n in ['torch','onnxruntime','tensorrt','pycuda.driver'])
"""
            result=self.run_python(['-c',code,str(root)])
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('typed compact metadata',result.stderr)

    def test_checkpoint_hash_guard_precedes_backend_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);checkpoint=root/'model.pt';checkpoint.write_bytes(b'not a model')
            metadata={'family':'waterec_compact','schema':1,'compact_architecture':{'family':'waterec_compact','architecture':'rank_mlp_image_topk'},'static_batch':1,'max_tokens':64,'memory_tokens':64,'max_len':25,'checkpoint_sha256':'0'*64}
            (root/'metadata.json').write_text(json.dumps(metadata))
            code="""import sys
from waterec_ocr.waterec_compact_cli import _metadata
try:_metadata(sys.argv[1],sys.argv[2])
except ValueError as error:assert 'checkpoint SHA' in str(error)
else:raise AssertionError('Mismatched checkpoint accepted')
assert not any(n in sys.modules for n in ['torch','onnxruntime','tensorrt','pycuda.driver'])
"""
            result=self.run_python(['-c',code,str(root),str(checkpoint)])
            self.assertEqual(result.returncode,0,result.stderr)

    def test_static_batch_and_missing_paths_fail_before_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'unused.png';image.write_bytes(b'not decoded by guard')
            result=self.run_python(['-m',MODULE,'infer','--backend','tensorrt','--deployment',str(root),'--image',str(image),'--batch-size','2'])
            self.assertEqual(result.returncode,2);self.assertIn('static B1',result.stderr)
            result=self.run_python(['-m',MODULE,'infer','--backend','tensorrt','--deployment',str(root),'--image',str(root/'missing.png')])
            self.assertEqual(result.returncode,2);self.assertIn('missing.png',result.stderr)
            result=self.run_python(['-m',MODULE,'export','--checkpoint',str(root/'missing.pt'),'--deployment',str(root/'out')])
            self.assertEqual(result.returncode,2);self.assertIn('missing.pt',result.stderr)
            self.assertFalse((root/'out').exists())

    def test_TRT_budget_cap_rejects_before_backend_and_keeps_ONNX_metadata_flexible(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);image=root/'unused.png';image.write_bytes(b'not decoded by guard')
            metadata={'family':'waterec_compact','schema':1,'compact_architecture':{'family':'waterec_compact','architecture':'rank_mlp_image_topk'},'static_batch':1,'max_tokens':4096,'memory_tokens':512,'max_len':25}
            (root/'metadata.json').write_text(json.dumps(metadata))
            code="""import sys
from waterec_ocr.waterec_compact_cli import main,_metadata
from waterec_ocr.waterec_compact_build import build_compact
assert _metadata(sys.argv[1])['max_tokens']==4096
try:build_compact(sys.argv[1])
except ValueError as error:assert '64..2048' in str(error)
else:raise AssertionError('Builder accepted unsupported budget')
for args in [['build','--deployment',sys.argv[1]],['infer','--backend','tensorrt','--deployment',sys.argv[1],'--image',sys.argv[2]]]:
 try:main(args)
 except SystemExit as error:assert error.code==2
 else:raise AssertionError('CLI accepted unsupported TRT budget')
assert not any(n in sys.modules for n in ['torch','onnxruntime','tensorrt','pycuda.driver'])
"""
            result=self.run_python(['-c',code,str(root),str(image)])
            self.assertEqual(result.returncode,0,result.stderr)

    def test_successful_console_sys_exit_main_returns_zero_not_result_dict(self):
        with tempfile.TemporaryDirectory() as directory:
            config=Path(directory)/'config.yaml';config.write_text('dataset_root: data\noutput_dir: output\n')
            code="""import sys,types
workflow=types.ModuleType('waterec_ocr.waterec_compact_workflow')
workflow.train_compact=lambda config:{'pass':True,'checkpoint':'fake.pt','scope':'Lightweight console exit regression, no training'}
sys.modules[workflow.__name__]=workflow
from waterec_ocr.waterec_compact_cli import main
sys.exit(main(['train','--config',sys.argv[1]]))
"""
            result=self.run_python(['-c',code,str(config)])
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(json.loads(result.stdout)['pass'],True)
            self.assertEqual(result.stderr,'')

    def test_training_paths_resolve_relative_to_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config=root/'config.yaml';config.write_text('dataset_root: data\noutput_dir: output\npretrained_dir: weights\nvalidation_folder: val\nteacher_checkpoint: null\n')
            code="""import sys
from pathlib import Path
from waterec_ocr.waterec_compact_cli import _config
path=Path(sys.argv[1]);config=_config(path)
for key,folder in [('dataset_root','data'),('output_dir','output'),('pretrained_dir','weights'),('validation_folder','val')]:assert config[key]==str(path.parent/folder)
assert config['teacher_checkpoint'] is None and 'torch' not in sys.modules
"""
            result=self.run_python(['-c',code,str(config)]);self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
