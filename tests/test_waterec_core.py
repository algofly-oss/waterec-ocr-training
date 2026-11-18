"""CPU-only standalone import, native policy and real ORT cached-loop tests."""
import json,os,subprocess,sys,tempfile,unittest,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

class CoreTests(unittest.TestCase):
    def test_optional_plate_upscale_width_boundaries(self):
        from waterec_ocr.waterec_orientation import OrientationRecognizer

        reader = OrientationRecognizer.__new__(OrientationRecognizer)
        for standard, tiny, expected in (
            (False, False, set()),
            (True, False, {48, 96}),
            (False, True, {32, 47}),
            (True, True, {32, 47, 48, 96}),
        ):
            reader.upscale_small_plates = standard
            reader.upscale_very_small_plates = tiny
            for width in (31, 32, 47, 48, 96, 97):
                with Image.new("RGB", (width, 16)) as image:
                    self.assertEqual(reader._should_upscale(image), width in expected)
                pixels = np.zeros((16, width, 3), np.uint8)
                self.assertEqual(reader._should_upscale(pixels), width in expected)

    def test_rgb_arrays_preserve_patch_packing_and_rotation(self):
        from waterec_ocr.waterec_orientation import _rotate
        from waterec_ocr.waterec_preprocessing import pack_image

        rgb = np.random.default_rng(7).integers(0, 256, (31, 77, 3), np.uint8)
        image = Image.fromarray(rgb)
        rope = {key: np.zeros((512, 16), np.float32) for key in ("cosine", "sine")}
        for turns in range(4):
            pil = _rotate(image, turns)
            arr = _rotate(rgb, turns)
            np.testing.assert_array_equal(arr, np.asarray(pil))
            packed, grid = pack_image(pil, rope, 512)
            packed_array, grid_array = pack_image(arr, rope, 512)
            self.assertEqual(grid, grid_array)
            for key in packed:
                np.testing.assert_array_equal(packed[key], packed_array[key])
        for bad in (rgb.astype(np.float32), rgb[:, :, 0], rgb[:0]):
            with self.assertRaisesRegex(ValueError, "uint8 RGB array"):
                pack_image(bad, rope, 512)

    def test_help_is_light_and_standalone(self):
        env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='')
        with tempfile.TemporaryDirectory() as d:
            result=subprocess.run([sys.executable,'-m','waterec_ocr','--help'],cwd=d,env=env,capture_output=True,text=True,check=True)
            self.assertIn('onnx',result.stdout)
            code="import sys; import waterec_ocr.waterec_cli; assert not any(x in sys.modules for x in ['torch','tensorrt','parseq_ocr','pytorch_lightning','timm'])"
            subprocess.run([sys.executable,'-c',code],cwd=d,env=env,check=True)

    def test_vendor_integrity(self):
        folder=ROOT/'waterec_ocr/_vendor/waterec'
        pin=json.loads((folder/'source_pin.json').read_text())
        for name,entry in pin['files'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),entry['vendored_sha256'])

    def test_budgets_native_pixels_and_EOS(self):
        import torch
        from waterec_ocr.waterec_models import create_native,preprocess_image
        from waterec_ocr.waterec_preprocessing import pack_image
        torch.set_num_threads(2)
        rng=np.random.default_rng(123)
        for budget in [512,1024,2048]:
            freq=1/(10000**(np.arange(0,64,4,dtype=np.float32)/64))
            angle=np.arange(budget,dtype=np.float32)[:,None]*freq
            rope={'cosine':np.cos(angle),'sine':np.sin(angle)}
            for w,h in [(320,160),(77,31)]:
                im=Image.fromarray(rng.integers(0,256,(h,w,3),np.uint8));packed,(ph,pw)=pack_image(im,rope,budget);n=ph*pw
                restored=packed['patches'][0,:n].reshape(ph,pw,3,4,4).transpose(2,0,3,1,4).reshape(3,ph*4,pw*4)
                self.assertTrue(np.array_equal(restored,preprocess_image(im,budget).numpy()))
                self.assertEqual(packed['padding'].sum(),budget-n)
        m=create_native(dict(task='plate',variant='R',tokens=512,max_label_length=24,max_decode_steps=25),pretrained=False)
        self.assertEqual(sum(p.numel()for p in m.parameters()),26120544)
        self.assertEqual(tuple(m.encode_labels(['A'*24]).shape),(1,26))
        with self.assertRaisesRegex(ValueError,'EOS'):m.encode_labels(['A'*25])
        self.assertFalse(torch.cuda.is_initialized())

    def test_mean_space_preserves_rows_and_rng(self):
        import torch
        from torch import nn
        from types import SimpleNamespace
        from waterec_ocr.waterec_models import NativeWATERec
        m=NativeWATERec.__new__(NativeWATERec);nn.Module.__init__(m)
        m.characters=['</s>','A','B','<s>','<pad>'];m.config={'include_space':False}
        old_embedding=nn.Embedding(5,4);old_head=nn.Linear(4,3,bias=False)
        with torch.no_grad():
            old_embedding.weight.copy_(torch.arange(20).reshape(5,4))
            old_head.weight.copy_(torch.arange(12).reshape(3,4)*2)
        embedding_rows=old_embedding.weight.detach().clone();head_rows=old_head.weight.detach().clone()
        m.decoder=SimpleNamespace(d_model=4,embedding=SimpleNamespace(embedding=old_embedding),tgt_word_prj=old_head)
        rng=torch.get_rng_state().clone();m.add_space()
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        for i,char in enumerate(['</s>','A','B','<s>','<pad>']):
            self.assertTrue(torch.equal(m.decoder.embedding.embedding.weight[m.characters.index(char)],embedding_rows[i]))
            if i<3:self.assertTrue(torch.equal(m.decoder.tgt_word_prj.weight[m.characters.index(char)],head_rows[i]))
        self.assertTrue(torch.equal(m.decoder.embedding.embedding.weight[3],embedding_rows[1:-2].mean(0)))
        self.assertTrue(torch.equal(m.decoder.tgt_word_prj.weight[3],head_rows[1:].mean(0)))
        self.assertEqual((m.decoder.bos,m.decoder.ignore_index),(4,5))
        self.assertTrue(m.config['include_space']);m.add_space()
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_real_onnx_cached_inference(self):
        try:
            import onnx,onnxruntime
        except ImportError:
            self.skipTest('Install the onnx extra for this test')
        from onnx import helper as h,TensorProto as T,numpy_helper as nh
        from waterec_ocr.waterec_onnx import ONNXRecognizer
        from waterec_ocr.waterec_cli import deployment_metadata
        def vi(name,t,shape):return h.make_tensor_value_info(name,t,shape)
        def save(graph,path):
            model=h.make_model(graph,opset_imports=[h.make_opsetid('',17)]);model.ir_version=8;onnx.checker.check_model(model);onnx.save(model,path)
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)/'deployment';folder.mkdir();N=64;cache=[2,1,12,25,32];cross=[2,1,12,N,32]
            enc=h.make_graph([h.make_node('Constant',[],['cross_k'],value=nh.from_array(np.zeros(cross,np.float32))),h.make_node('Constant',[],['cross_v'],value=nh.from_array(np.zeros(cross,np.float32)))],'encoder',[vi('patches',T.FLOAT,[1,N,48]),vi('cosine',T.FLOAT,[1,N,32]),vi('sine',T.FLOAT,[1,N,32]),vi('padding',T.BOOL,[1,N])],[vi('cross_k',T.FLOAT,cross),vi('cross_v',T.FLOAT,cross)]);save(enc,folder/'encoder_cross_kv.onnx')
            table=nh.from_array(np.array([[[0,10,-1]],[[10,0,-1]]],np.float32),'table')
            dec=h.make_graph([h.make_node('Gather',['table','position'],['logits'],axis=0),h.make_node('Identity',['past_k'],['present_k']),h.make_node('Identity',['past_v'],['present_v'])],'decoder',[vi('token',T.INT64,[1,1]),vi('position',T.INT64,[1]),vi('cross_k',T.FLOAT,cross),vi('cross_v',T.FLOAT,cross),vi('padding',T.BOOL,[1,N]),vi('past_k',T.FLOAT,cache),vi('past_v',T.FLOAT,cache)],[vi('logits',T.FLOAT,[1,1,3]),vi('present_k',T.FLOAT,cache),vi('present_v',T.FLOAT,cache)],[table]);save(dec,folder/'decoder_cached_step.onnx')
            meta=dict(family='waterec',static_batch=1,max_tokens=N,max_len=25,decoder_layers=2,decoder_heads=12,cache_head_dim=32,characters=['</s>','A',' ','<s>','<pad>'],bos=3,eos=0,task='plate',variant='R',normalization='upper_no_whitespace')
            (folder/'metadata.json').write_text(json.dumps(meta));np.savez(folder/'rope.npz',cosine=np.ones((N,16),np.float32),sine=np.zeros((N,16),np.float32))
            r=ONNXRecognizer(folder)
            try:self.assertEqual(r.infer([Image.new('RGB',(32,32))]*2),['A','A'])
            finally:r.close()
            with self.assertRaises(RuntimeError):r.infer([Image.new('RGB',(32,32))])
            bad=dict(output_dir=Path(d),task='plate',variant='R',tokens=512,normalization='upper_no_whitespace',max_decode_steps=25)
            with self.assertRaisesRegex(ValueError,'max_tokens'):deployment_metadata(bad)

if __name__=='__main__':unittest.main()
