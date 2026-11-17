"""Portable WATERec model and native inference, with pinned official weights.

The TensorRT runtime is a separate module and never imports this PyTorch module.
"""
from pathlib import Path
import hashlib
import math
import urllib.request
import numpy as np
from PIL import Image
import torch
from torch import nn
from ._vendor.waterec.encoder import NaViT_ROPE
from ._vendor.waterec.decoder import NaARDecoder

ROOT = Path(__file__).resolve().parents[1]
VENDOR = Path(__file__).resolve().parent / '_vendor' / 'waterec'
REVISION = '678247a94c09c462811647c87f338cb2169d94c0'
WEIGHTS = {
    'R': ('WATERec-R.pth', 'ca03d07e6b0a7eebf97ade6bbefd48a1c2deed7c13593f085d32e2196bd875d9'),
    'RS': ('WATERec-RS.pth', 'cdc3e33a61a08a1fb7f00cb3054cd74a9259863830869d7546ec5e2bc8714801'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download_pretrained(variant, folder):
    if variant not in WEIGHTS:
        raise ValueError('Official WATERec variants are R and RS')
    name, expected = WEIGHTS[variant]
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if not target.exists():
        partial = target.with_suffix('.partial')
        url = f'https://huggingface.co/Yesianrohn/WATERec-Models/resolve/{REVISION}/{name}'
        with urllib.request.urlopen(url, timeout=60) as source, partial.open('wb') as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                output.write(chunk)
        if sha256(partial) != expected:
            raise ValueError('Official WATERec checkpoint SHA256 mismatch')
        partial.replace(target)
    if sha256(target) != expected:
        raise ValueError(f'Official WATERec checkpoint SHA256 mismatch: {target}')
    return target


def _canonical(config):
    config = dict(config)
    from .waterec_charset import validate_additions
    if 'additional_characters' in config:
        additional = validate_additions(config['additional_characters'])
        if additional:
            config['additional_characters'] = additional
        else:
            config.pop('additional_characters', None)
    config.setdefault('task', 'plate')
    config.setdefault('variant', 'RS')
    budget = config.get('tokens', config.get('max_patch_tokens', 512))
    if type(budget) is not int or budget < 64:
        raise ValueError('Patch budget must be an integer >=64')
    config['tokens'] = budget
    config.setdefault('include_space', config['task'] == 'price')
    config.setdefault('max_label_length', 24)
    config.setdefault('max_decode_steps', 25)
    config.setdefault('normalization', 'none' if config['task'] == 'price' else 'upper_no_whitespace')
    if config['variant'] not in WEIGHTS or config['task'] not in ('plate', 'price'):
        raise ValueError('WATERec requires task plate/price and variant R/RS')
    if config['tokens'] < 64 or not 1 <= config['max_label_length'] <= 25:
        raise ValueError('Patch budget must be >=64 and maximum text length in1..25')
    if not 2 <= config['max_decode_steps'] <= 25:
        raise ValueError('max_decode_steps must be in2..25 including EOS')
    return config



def preprocess_image(image, tokens=512):
    """Official aspect-preserving 4x4-patch resize; no EXIF autotranspose."""
    if not isinstance(image, Image.Image):
        raise TypeError('preprocess expects a PIL image')
    image = image.convert('RGB')
    width, height = image.size
    area = width * height
    maximum, minimum = tokens * 16, 64 * 16
    if area > maximum:
        scale = math.sqrt(maximum / area)
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    elif area < minimum:
        scale = math.sqrt(minimum / area)
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
    width, height = max(width // 4, 1) * 4, max(height // 4, 1) * 4
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).contiguous().float()
    return pixels.div_(255).sub_(.5).div_(.5)


class NativeWATERec(nn.Module):
    """Official 12-layer384 encoder and two-layer384 AR reader."""
    def __init__(self, config):
        super().__init__()
        self.config = _canonical(config)
        self.characters = ['</s>'] + (VENDOR / 'characters.txt').read_text().splitlines() + ['<s>', '<pad>']
        self.encoder = NaViT_ROPE(patch_size=[4, 4], embed_dim=384, depth=12,
                                 num_heads=6, mlp_ratio=4)
        self.decoder = NaARDecoder(in_channels=384, out_channels=len(self.characters),
                                   beam_size=0, num_decoder_layers=2, nhead=12,
                                   max_len=self.config['max_decode_steps'])

    def forward(self, images, data=None):
        return self.decoder(self.encoder(images), data=data)

    def add_space(self):
        """Append space while preserving every original character and special token."""
        if ' ' in self.characters:
            return
        old = list(self.characters)
        self.characters = old[:-2] + [' '] + old[-2:]
        d = self.decoder
        # Match the generic mean-SPACE recipe: ordinary printable character rows
        # only, excluding EOS (row0) and the final BOS/PAD embedding rows.
        # Constructors sample temporary weights, all of which are overwritten.
        # Keep this extension RNG-neutral on CPU and the existing CUDA device.
        devices = [d.embedding.embedding.weight.device.index] if d.embedding.embedding.weight.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            embedding = nn.Embedding(len(self.characters), d.d_model, padding_idx=0,
                                     device=d.embedding.embedding.weight.device,
                                     dtype=d.embedding.embedding.weight.dtype)
            head = nn.Linear(d.d_model, len(self.characters) - 2, bias=False,
                             device=d.tgt_word_prj.weight.device, dtype=d.tgt_word_prj.weight.dtype)
        with torch.no_grad():
            for i, token in enumerate(old):
                j = self.characters.index(token)
                embedding.weight[j].copy_(d.embedding.embedding.weight[i])
                if i < len(old) - 2:
                    head.weight[j].copy_(d.tgt_word_prj.weight[i])
            space = self.characters.index(' ')
            embedding.weight[space].copy_(d.embedding.embedding.weight[1:-2].mean(0))
            head.weight[space].copy_(d.tgt_word_prj.weight[1:].mean(0))
        d.embedding.embedding = embedding
        d.tgt_word_prj = head
        d.out_channels = len(self.characters)
        d.ignore_index = len(self.characters) - 1
        d.bos = len(self.characters) - 2
        self.config['include_space'] = True

    def preprocess(self, image):
        return preprocess_image(image, self.config['tokens'])

    def encode_labels(self, labels, device=None):
        lookup = {token: i for i, token in enumerate(self.characters)}
        labels = list(labels)
        if not labels:
            raise ValueError('Label batch is empty')
        for text in labels:
            if not text or len(text) > min(self.config['max_label_length'], self.decoder.max_len - 1) or any(c not in lookup for c in text):
                raise ValueError(f'Empty, unsupported or oversized label: {text!r}; no truncation; one decoding step is reserved for EOS')
        target = torch.full((len(labels), max(map(len, labels)) + 2), self.decoder.ignore_index,
                            dtype=torch.long, device=device)
        target[:, 0] = self.decoder.bos
        for i, text in enumerate(labels):
            target[i, 1:len(text) + 1] = torch.tensor([lookup[c] for c in text], device=device)
            target[i, len(text) + 1] = self.decoder.eos
        return target

    def decode(self, logits):
        texts = []
        for ids in logits.argmax(-1).tolist():
            text = []
            for token in ids:
                if token == self.decoder.eos:
                    break
                if token < len(self.characters) - 2:
                    text.append(self.characters[token])
            texts.append(''.join(text))
        return texts


def default_checkpoint(config):
    output = Path(config['output_dir'])
    return output / ('full_fit/model.pt' if config.get('refit_all_training', True)
                     else 'selection/best.pt')


def _payload(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Checkpoint must contain a model state dictionary')
    return payload


def create_native(config, checkpoint=None, pretrained=True):
    config = _canonical(config)
    payload = _payload(checkpoint) if checkpoint is not None else None
    if payload is not None:
        recorded = payload.get('config', payload.get('configuration'))
        if not isinstance(recorded, dict) or 'task' not in recorded:
            raise ValueError('Fine-tuned checkpoint needs task metadata; initialize official weights with pretrained=True')
        recorded = _canonical(recorded)
        keys = ('task', 'variant', 'tokens', 'include_space', 'max_label_length', 'max_decode_steps', 'normalization')
        differences = {key: (recorded[key], config[key]) for key in keys if recorded[key] != config[key]}
        if recorded.get('additional_characters', []) != config.get('additional_characters', []):
            differences['additional_characters'] = (recorded.get('additional_characters', []), config.get('additional_characters', []))
        if differences:
            raise ValueError(f'Checkpoint/config profile mismatch (saved, requested): {differences}')
    model = NativeWATERec(config)
    from .waterec_charset import extend_characters, restore_metadata
    if payload is not None:
        if config['include_space']:
            model.add_space()
        extend_characters(model, config.get('additional_characters', []))
        restore_metadata(model, payload)
        model.load_state_dict(payload['state_dict'], strict=True)
    else:
        if pretrained:
            folder = config.get('pretrained_dir', ROOT / 'weights' / 'waterec')
            payload = _payload(download_pretrained(config['variant'], folder))
            model.load_state_dict(payload.get('state_dict', payload), strict=True)
        if config['include_space']:
            model.add_space()
        provenance = {'source': 'official_pretrained' if pretrained else 'fresh_initialization', 'source_checkpoint_sha256': WEIGHTS[config['variant']][1] if pretrained else None, 'optimizer': 'fresh; no optimizer-state migration'}
        extend_characters(model, config.get('additional_characters', []), provenance)
    return model


def load_native(checkpoint, device='cpu'):
    payload = _payload(checkpoint)
    config = payload.get('config', payload.get('configuration'))
    if not isinstance(config, dict) or 'task' not in config:
        raise ValueError('Fine-tuned checkpoint needs config metadata; use create_native for official weights')
    model = NativeWATERec(config)
    if model.config['include_space']:
        model.add_space()
    from .waterec_charset import extend_characters, restore_metadata
    extend_characters(model, model.config.get('additional_characters', []))
    restore_metadata(model, payload)
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval().to(device)


class TorchRecognizer:
    def __init__(self, config, checkpoint=None):
        self.config = dict(config)
        checkpoint = Path(checkpoint) if checkpoint else default_checkpoint(config)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'WATERec checkpoint not found: {checkpoint}; run training first')
        self.model = create_native(config, checkpoint).float().eval().to(config.get('device', 'cpu'))
        self.device = next(self.model.parameters()).device

    @torch.inference_mode()
    def infer(self, images):
        images = list(images)
        result = []
        batch = int(self.config.get('inference_batch_size', self.config.get('batch_size', 16)))
        for start in range(0, len(images), batch):
            tensors = [self.model.preprocess(image).to(self.device) for image in images[start:start + batch]]
            result.extend(self.model.decode(self.model(tensors)))
        return result


def migrate_native_checkpoint(checkpoint, additional_characters):
    """Explicit append-only task-checkpoint migration; fresh optimizer required."""
    from .waterec_charset import extend_characters, metadata, vocabulary_hash
    model=load_native(checkpoint)
    provenance={'source':'explicit_task_checkpoint_migration','source_checkpoint_sha256':sha256(checkpoint),'source_vocabulary_sha256':vocabulary_hash(model.characters),'optimizer':'fresh; optimizer state not migrated'}
    extend_characters(model,additional_characters,provenance)
    return model,dict(config=dict(model.config),state_dict=model.state_dict(),**metadata(model))
