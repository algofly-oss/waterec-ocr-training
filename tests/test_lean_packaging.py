"""CPU checks for dependency boundaries and unchanged engine-build policies."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LeanPackagingTests(unittest.TestCase):
    def test_extracted_builders_keep_original_AST(self):
        reference = json.loads((ROOT / 'tests/engine_build_ast_reference.json').read_text())
        tree = ast.parse((ROOT / 'waterec_ocr/waterec_build.py').read_text())
        for name, expected in reference['functions'].items():
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(), expected)

    def test_lean_CLI_dispatches_all_builders_without_framework_imports(self):
        code = '''
import sys, importlib.abc
from pathlib import Path
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch','onnx','onnxruntime','tensorrt','pycuda','cuda','jupyterlab','ipykernel'}:
            raise RuntimeError('Forbidden import during CPU dispatch: '+fullname)
sys.meta_path.insert(0, Block())
from waterec_ocr import waterec_build as build, waterec_cli as cli
import waterec_ocr.waterec_benchmark, waterec_ocr.evaluation
seen=[]
build.build_engines=lambda folder,force=False: seen.append(('cached',force)) or folder
build.build_speculative=lambda folder,force=False: seen.append(('speculative',force)) or folder
cli.deployment_metadata=lambda config,orientation=False: (Path('deployment'),{})
cli.speculative_directory=lambda config: Path('speculative')
for command in ('build','build-orientation','build-speculative'):
    sys.argv=['waterec-ocr',command,'--config','configs/waterec_alpr.yaml','--force']
    cli.main()
assert seen==[('cached',True),('cached',True),('speculative',True)]
'''
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, capture_output=True, text=True)

    def test_dependency_profiles(self):
        try:
            import tomllib
        except ImportError:
            from pip._vendor import tomli as tomllib
        project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
        extras = project['optional-dependencies']
        self.assertFalse(any(x.startswith('torch') for x in project['dependencies']))
        for name in ('training', 'notebooks', 'onnx', 'test'):
            self.assertIn('torch==2.10.0', extras[name])
        lean = project['dependencies'] + extras['tensorrt'] + extras['build']
        self.assertFalse(any(x.split('=')[0] in {'torch','onnxruntime','jupyterlab','ipykernel'} for x in lean))
        self.assertEqual(extras['build'], ['onnx==1.21.0'])


if __name__ == '__main__':
    unittest.main()
