"""Experimental native rank-MLP / image-only token-selection WATERec.

Not a supported CLI preset. No weights are bundled. Training consumes images and
transcriptions; selection at inference depends only on image features.
SPDX-License-Identifier: Apache-2.0
"""
import copy
from collections.abc import Mapping
from pathlib import Path

from .waterec_charset import metadata as alphabet_metadata
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .waterec_models import create_native
from ._vendor.waterec.encoder import compute_axial_cis


class FactorLinear(nn.Module):
    """Two linear factors, with the original bias on the second factor only."""
    def __init__(self, linear, rank=128, initialize_svd=False):
        super().__init__()
        if type(rank) is not int or not 1 <= rank <= min(linear.in_features, linear.out_features):
            raise ValueError('Invalid factor rank')
        self.a = nn.Linear(linear.in_features, rank, bias=False,
                           device=linear.weight.device, dtype=linear.weight.dtype)
        self.b = nn.Linear(rank, linear.out_features, bias=linear.bias is not None,
                           device=linear.weight.device, dtype=linear.weight.dtype)
        if initialize_svd:
            u, singular, vh = torch.linalg.svd(linear.weight.detach().double(), full_matrices=False)
            root = singular[:rank].sqrt()
            with torch.no_grad():
                self.a.weight.copy_((root[:, None] * vh[:rank]).float())
                self.b.weight.copy_((u[:, :rank] * root[None, :]).float())
                if linear.bias is not None:
                    self.b.bias.copy_(linear.bias)

    def forward(self, x):
        return self.b(self.a(x))


def selection_head(seed=15001):
    # CPU generator only: constructing a CPU module must not reset CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 128),
                             nn.GELU(), nn.Linear(128, 1))


class CompactWATERec(nn.Module):
    """Preserve twelve native visual blocks and the native two-layer AR reader.

    ``initialize_svd=False`` constructs factor shapes for strict checkpoint
    loading, not a useful pretrained initialization. Use ``from_teacher`` for
    actual SVD initialization from an existing standard WATERec model.
    """
    def __init__(self, config, *, native=None, rank=128, after_blocks=6,
                 target_keep=512, initialize_svd=False):
        super().__init__()
        if type(after_blocks) is not int or not 1 <= after_blocks < 12:
            raise ValueError('Selection must follow an interior encoder block')
        if type(target_keep) is not int or target_keep < 1:
            raise ValueError('target_keep must be positive')
        self.native = create_native(config, pretrained=False) if native is None else native
        if len(self.native.encoder.blocks) != 12 or len(self.native.decoder.decoder.layers) != 2:
            raise ValueError('Expected the native twelve-block/two-layer WATERec')
        self.config = dict(self.native.config)
        self.rank, self.after_blocks = rank, after_blocks
        self.target_keep = target_keep
        self.keep_tokens = min(target_keep, self.config['tokens'])
        for block in self.native.encoder.blocks:
            for name in ('fc1', 'fc2'):
                linear = getattr(block.mlp, name)
                if not isinstance(linear, nn.Linear):
                    raise ValueError('Expected unfactorized teacher MLP linears')
                setattr(block.mlp, name, FactorLinear(linear, rank, initialize_svd))
        self.scorer = selection_head().to(device=next(self.native.parameters()).device)
        self.native.encoder.token_dropout_rate = 0.

    @classmethod
    def from_teacher(cls, teacher, **kwargs):
        """Copy an existing standard native model; leave its weights unchanged."""
        native = copy.deepcopy(teacher)
        return cls(native.config, native=native, initialize_svd=True, **kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.native.encoder.token_dropout_rate = 0.
        return self

    def preprocess(self, image):
        return self.native.preprocess(image)

    def encode_labels(self, labels, device=None):
        return self.native.encode_labels(labels, device=device)

    def decode(self, logits):
        return self.native.decode(logits)

    def memory(self, images, prune=True):
        """Return ((features, key-padding mask), raster indices, image scores).

        Images are preprocessed normalized CHW tensors. Native-sized per-image
        grids retain their own original 2D RoPE coordinates after selection.
        No targets, boxes or teacher features enter this method.
        """
        images = list(images)
        if not images:
            raise ValueError('Image batch must be nonempty')
        enc = self.native.encoder
        device = images[0].device
        seq, freq = [], []
        for x in images:
            if x.ndim != 3 or x.shape[0] != 3 or x.device != device:
                raise ValueError('Expected same-device normalized RGB CHW tensors')
            h, w = x.shape[-2:]
            if h % 4 or w % 4 or not 1 <= h * w // 16 <= self.config['tokens']:
                raise ValueError('Expected 4-divisible native grid within 1..budget actual patches (configured budget >=64)')
            ph, pw = h // 4, w // 4
            seq.append(x.view(-1, ph, 4, pw, 4).permute(1, 3, 0, 2, 4).reshape(ph * pw, -1))
            freq.append(compute_axial_cis(dim=enc.embed_dim // enc.num_heads,
                                         end_x=pw, end_y=ph, theta=enc.rope_theta))
        lengths = [len(x) for x in seq]

        def pack(a, f):
            lens = torch.tensor([len(x) for x in a], device=device)
            mask = torch.arange(int(lens.max()), device=device)[None, :] >= lens[:, None]
            return (pad_sequence(a, batch_first=True).to(device),
                    pad_sequence(f, batch_first=True).to(device), mask,
                    torch.where(mask, float('-inf'), 0.)[:, None, None, :])

        x, fc, pad, attention_mask = pack(seq, freq)
        x = enc.pos_drop(enc.patch_embed(x))
        for i, block in enumerate(enc.blocks):
            x = block(x, fc, attention_mask)
            if i == self.after_blocks - 1:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    scores = self.scorer(x.float()).squeeze(-1)
                chosen = [torch.argsort(scores[j, :n], descending=True, stable=True)[:min(self.keep_tokens, n)].sort().values
                          if prune else torch.arange(n, device=device)
                          for j, n in enumerate(lengths)]
                if prune:
                    x, fc, pad, attention_mask = pack(
                        [x[j].index_select(0, ix) for j, ix in enumerate(chosen)],
                        [fc[j].index_select(0, ix) for j, ix in enumerate(chosen)])
        return (enc.norm(x), pad), chosen, scores

    def forward_train(self, images, tokens):
        memory, _, _ = self.memory(images)
        return self.native.decoder.forward_train(memory, tokens)

    def forward(self, images, tokens=None):
        if tokens is not None:
            return self.forward_train(images, tokens)
        memory, _, _ = self.memory(images)
        return self.native.decoder.forward_test(memory)

    def architecture_metadata(self):
        return {'family': 'waterec_compact', 'schema': 2 if self.config.get('additional_characters') else 1,
                'architecture': 'rank_mlp_image_topk', 'rank': self.rank,
                'after_blocks': self.after_blocks, 'target_keep': self.target_keep,
                'config': dict(self.config), 'characters': list(self.native.characters)}

    def checkpoint_payload(self):
        return {'compact_architecture': self.architecture_metadata(),
                'state_dict': self.state_dict(),
                **alphabet_metadata(self.native)}


def load_compact_checkpoint(checkpoint, device='cpu'):
    """Strict explicit compact metadata; never guess from a standard checkpoint."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False) if isinstance(checkpoint, (str, Path)) else checkpoint
    if not isinstance(payload, Mapping) or not isinstance(payload.get('compact_architecture'), dict):
        raise ValueError('Explicit compact architecture metadata is required')
    meta = payload['compact_architecture']
    if meta.get('family') != 'waterec_compact' or meta.get('schema') not in (1, 2) or meta.get('architecture') != 'rank_mlp_image_topk':
        raise ValueError('Unsupported compact checkpoint architecture')
    model = CompactWATERec(meta['config'], rank=meta['rank'],
                          after_blocks=meta['after_blocks'], target_keep=meta['target_keep'])
    from .waterec_charset import restore_metadata
    if bool(meta['config'].get('additional_characters')) != (meta['schema'] == 2):
        raise ValueError('Extended alphabet requires architecture schema2; legacy requires schema1')
    restore_metadata(model.native, payload)
    if model.architecture_metadata() != meta:
        raise ValueError('Compact architecture/character metadata mismatch')
    model.load_state_dict(payload['state_dict'], strict=True)
    return model.eval().to(device)


def migrate_compact_checkpoint(checkpoint, additional_characters):
    """Explicit alphabet migration, never infer labels from state tensor shapes."""
    from .waterec_charset import extend_characters,vocabulary_hash
    from .waterec_models import sha256
    model=load_compact_checkpoint(checkpoint)
    provenance={'source':'explicit_task_checkpoint_migration','source_checkpoint_sha256':sha256(checkpoint),'source_vocabulary_sha256':vocabulary_hash(model.native.characters),'optimizer':'fresh; optimizer state not migrated'}
    extend_characters(model.native,additional_characters,provenance)
    model.config=dict(model.native.config)
    return model,model.checkpoint_payload()
