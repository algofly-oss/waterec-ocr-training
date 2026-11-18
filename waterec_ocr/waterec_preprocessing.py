"""NumPy/Pillow WATE patch packing, shared by ONNX and TensorRT CPU input."""
import math
import numpy as np
from PIL import Image

def rgb_array_size(image):
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or not image.shape[0] or not image.shape[1]):
        raise ValueError("Expected a nonempty H×W×3 uint8 RGB array")
    return image.shape[1], image.shape[0]

def pack_image(image, rope, max_tokens):
    if isinstance(image, np.ndarray):
        rgb_array_size(image)
        image = Image.fromarray(image)
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    w, h = image.size
    area = w * h
    if area > max_tokens * 16:
        scale = math.sqrt(max_tokens * 16 / area)
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
    elif area < 64 * 16:
        scale = math.sqrt(64 * 16 / area)
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
    w, h = max(w // 4, 1) * 4, max(h // 4, 1) * 4
    pw, ph = w // 4, h // 4
    n = pw * ph
    if n > max_tokens:
        raise ValueError(f'Official resize exceeds {max_tokens} patches: {n}')
    arr = np.asarray(image.resize((w, h), Image.Resampling.BICUBIC), dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    arr = (arr - np.float32(.5)) / np.float32(.5)
    seq = arr.reshape(3, ph, 4, pw, 4).transpose(1, 3, 0, 2, 4).reshape(n, 48)
    patches = np.zeros((1, max_tokens, 48), np.float32)
    patches[0, :n] = seq
    xx, yy = np.tile(np.arange(pw), ph), np.repeat(np.arange(ph), pw)
    cosine = np.zeros((1, max_tokens, 32), np.float32)
    sine = np.zeros_like(cosine)
    cosine[0, :n] = np.concatenate([rope['cosine'][xx], rope['cosine'][yy]], -1)
    sine[0, :n] = np.concatenate([rope['sine'][xx], rope['sine'][yy]], -1)
    padding = np.ones((1, max_tokens), bool)
    padding[0, :n] = False
    return dict(patches=patches, cosine=cosine, sine=sine, padding=padding), (ph, pw)
