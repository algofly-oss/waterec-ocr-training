"""Cached WATE ONNX Runtime inference; no Torch or TensorRT imports."""
import json
from pathlib import Path
import numpy as np
from .waterec_preprocessing import pack_image

class ONNXRecognizer:
    """Static B1 execution, looping over supplied images; not true batching."""
    def __init__(self, deployment):
        folder = Path(deployment)
        self.meta = json.loads((folder / 'metadata.json').read_text())
        from .waterec_charset import validate_deployment
        validate_deployment(self.meta)
        m = self.meta
        if m.get('family') != 'waterec' or m.get('static_batch') != 1 or type(m.get('max_tokens')) is not int or m['max_tokens'] < 64:
            raise ValueError('Expected static B1 WATE cached deployment metadata')
        if m.get('orientation'):
            raise ValueError('ONNX backend uses standard AR graphs; orientation is TensorRT-only')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        assets = m.get('assets', {})
        self.encoder = ort.InferenceSession(str(folder / assets.get('encoder_cross_kv.onnx', 'encoder_cross_kv.onnx')), sess_options=options, providers=['CPUExecutionProvider'])
        self.decoder = ort.InferenceSession(str(folder / assets.get('decoder_cached_step.onnx', 'decoder_cached_step.onnx')), sess_options=options, providers=['CPUExecutionProvider'])
        self.rope = dict(np.load(folder / assets.get('rope.npz', 'rope.npz')))
        self.closed = False

    def infer_one(self, image):
        if self.closed:
            raise RuntimeError('Recognizer is closed')
        m = self.meta
        packed, _ = pack_image(image, self.rope, m['max_tokens'])
        cross_k, cross_v = self.encoder.run(None, packed)
        cache = (m['decoder_layers'], 1, m['decoder_heads'], m['max_len'], m['cache_head_dim'])
        past_k = np.zeros(cache, np.float32)
        past_v = np.zeros_like(past_k)
        token = np.array([[m['bos']]], np.int64)
        ids = []
        for position in range(m['max_len']):
            logits, past_k, past_v = self.decoder.run(None, dict(token=token, position=np.array([position], np.int64), cross_k=cross_k, cross_v=cross_v, padding=packed['padding'], past_k=past_k, past_v=past_v))
            if not np.isfinite(logits).all():
                raise RuntimeError('Nonfinite ONNX decoder logits')
            token = logits.argmax(-1).astype(np.int64)
            value = int(token[0, 0])
            if value == m['eos']:
                break
            if value < len(m['characters']) - 2:
                ids.append(value)
        return ''.join(m['characters'][i] for i in ids)

    def infer(self, images):
        return [self.infer_one(im) for im in images]

    def close(self):
        self.encoder = self.decoder = None
        self.closed = True
