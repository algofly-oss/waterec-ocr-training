"""Experimental image/transcript-only compact WATERec training primitives.

No dataset orchestrator, private assets, boxes, CLI preset or GPU policy. Teacher
responses are raw logits; temperature scaling occurs only in the loss. Hard
selection needs explicit image-importance KL, not a fictitious index gradient.
SPDX-License-Identifier: Apache-2.0
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .waterec_compact_models import CompactWATERec


@dataclass(frozen=True)
class TeacherTargets:
    raw_logits: torch.Tensor
    importance: torch.Tensor | None
    valid_patches: torch.Tensor
    valid_tokens: torch.Tensor
    tokens: torch.Tensor
    characters: tuple
    response_format: str = 'raw_logits'

    def subset(self, start, stop):
        return TeacherTargets(self.raw_logits[start:stop],
                              None if self.importance is None else self.importance[start:stop],
                              self.valid_patches[start:stop], self.valid_tokens[start:stop],
                              self.tokens[start:stop], self.characters, self.response_format)


def freeze_teacher(teacher):
    """Explicit caller-owned freeze; target extraction never adds an optimizer."""
    teacher.requires_grad_(False)
    return teacher.eval()


@contextmanager
def _temporary_eval(module):
    modes = [(m, m.training) for m in module.modules()]
    try:
        module.eval()
        yield
    finally:
        # Restore exact flags, including intentionally frozen child modules.
        for child, mode in modes:
            child.training = mode


def _attention_map(module, args, kwargs):
    """Reconstruct unaveraged probabilities from the ORIGINAL call inputs.

    Forward is never replaced and its need_weights argument is never changed.
    This preserves the normal teacher response path while exposing supervision.
    """
    query, key, _ = args[:3]
    c, heads = module.embed_dim, module.num_heads
    if not module.batch_first or not module._qkv_same_embed_dim:
        raise ValueError('Expected native batch-first, shared-dimension attention')
    b, t, _ = query.shape
    n, d = key.shape[1], c // heads
    weight, bias = module.in_proj_weight, module.in_proj_bias
    q = F.linear(query, weight[:c], None if bias is None else bias[:c])
    k = F.linear(key, weight[c:2*c], None if bias is None else bias[c:2*c])
    q = q.reshape(b, t, heads, d).transpose(1, 2)
    k = k.reshape(b, n, heads, d).transpose(1, 2)
    score = (q / (d ** .5)) @ k.transpose(-2, -1)
    padding = kwargs.get('key_padding_mask')
    if padding is not None:
        mask = torch.where(padding, float('-inf'), 0.) if padding.dtype == torch.bool else padding
        score = score + mask[:, None, None, :]
    mask = kwargs.get('attn_mask')
    if mask is not None:
        mask = torch.where(mask, float('-inf'), 0.) if mask.dtype == torch.bool else mask
        if mask.ndim == 3:
            mask = mask.reshape(b, heads, t, n)
        score = score + mask
    return score.softmax(-1)


def _validate_tokens(decoder, tokens, batch):
    if tokens.ndim != 2 or tokens.dtype != torch.long or tokens.shape[0] != batch or not 3 <= tokens.shape[1] <= decoder.max_len + 1:
        raise ValueError('Expected native BOS/text/EOS/PAD target batch')
    if not bool((tokens[:, 0] == decoder.bos).all()):
        raise ValueError('Missing native BOS')
    expected = tokens[:, 1:]
    valid = expected.ne(decoder.ignore_index)
    lengths = valid.sum(-1)
    contiguous = torch.arange(expected.shape[1], device=tokens.device)[None] < lengths[:, None]
    if not torch.equal(valid, contiguous) or not bool((lengths >= 2).all()):
        raise ValueError('Expected nonempty text followed by EOS and trailing PAD only')
    if bool(((expected < 0) | ((expected >= decoder.bos) & valid)).any()):
        raise ValueError('Unsupported target character/special token')
    last = expected.gather(1, (lengths - 1)[:, None]).squeeze(1)
    if not bool((last == decoder.eos).all()) or not bool(((expected == decoder.eos) & valid).sum(-1).eq(1).all()):
        raise ValueError('Exactly one EOS must follow each transcript')
    return valid


@torch.no_grad()
def teacher_targets(teacher, images, tokens, *, include_importance=True,
                    validate_attention=False):
    """Frozen teacher full-memory responses and training-only image importance.

    Use a standard full WATERec teacher for rank recovery. Use a frozen compact
    rank parent WITHOUT pruning for scorer warmup/joint targets. Mean over native
    decoder layers and heads, then MAX over valid GT-character+EOS query rows;
    exclude image padding and normalize the resulting nonnegative vector once.
    There is no second softmax over teacher importance and no mean over queries.
    """
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('Call freeze_teacher before extracting targets')
    native = teacher.native if isinstance(teacher, CompactWATERec) else teacher
    decoder = native.decoder
    valid_tokens = _validate_tokens(decoder, tokens, len(images))
    captured, handles = [], []
    with _temporary_eval(teacher):
        try:
            if include_importance:
                for layer in decoder.decoder.layers:
                    def hook(module, args, kwargs):
                        captured.append((module, args, dict(kwargs)))
                    handles.append(layer.multihead_attn.register_forward_pre_hook(hook, with_kwargs=True))
            memory = teacher.memory(images, prune=False)[0] if isinstance(teacher, CompactWATERec) else native.encoder(images)
            raw = decoder.forward_train(memory, tokens)
        finally:
            for handle in handles:
                handle.remove()
        valid_patches = ~memory[1]
        importance = None
        if include_importance:
            if len(captured) != len(decoder.decoder.layers):
                raise ValueError('Expected one cross-attention call per decoder layer')
            maps = []
            for module, args, kwargs in captured:
                attention = _attention_map(module, args, kwargs)
                if not bool(torch.isfinite(attention).all()):
                    raise ValueError('Nonfinite teacher attention')
                if validate_attention:
                    reference_kwargs = dict(kwargs, need_weights=True, average_attn_weights=False)
                    reference = module(*args, **reference_kwargs)[1]
                    if not torch.allclose(attention, reference, atol=1e-5, rtol=1e-5):
                        raise ValueError('Teacher probability reconstruction mismatch')
                maps.append(attention)
            average = torch.stack(maps).mean((0, 2))
            importance = average.masked_fill(~valid_tokens[:, :, None], 0.).max(1).values
            importance = importance.masked_fill(~valid_patches, 0.)
            total = importance.sum(-1, keepdim=True)
            if not bool((total > 0).all()):
                raise ValueError('Empty image importance')
            importance = importance / total
    if not bool(torch.isfinite(raw).all()):
        raise ValueError('Nonfinite teacher logits')
    return TeacherTargets(raw.detach(), None if importance is None else importance.detach(),
                          valid_patches.detach(), valid_tokens.detach(), tokens.detach().clone(),
                          tuple(native.characters))


def response_sums(logits, tokens, teacher, pad_id, temperature=2.):
    """Token-summed CE and T² KL, to divide by the EFFECTIVE batch count."""
    if temperature <= 0 or teacher.response_format != 'raw_logits':
        raise ValueError('Require positive temperature and raw teacher logits')
    expected = tokens[:, 1:]
    valid = expected.ne(pad_id)
    raw = teacher.raw_logits
    if logits.shape != raw.shape or logits.shape[:2] != expected.shape or not torch.equal(valid, teacher.valid_tokens) or not torch.equal(tokens, teacher.tokens):
        raise ValueError('Student/teacher target shapes or token masks differ')
    if raw.requires_grad or not bool(torch.isfinite(raw).all()):
        raise ValueError('Teacher must be finite and detached')
    ce = F.cross_entropy(logits.float().flatten(0, 1), expected.flatten(),
                         ignore_index=pad_id, reduction='sum')
    probability = F.softmax(raw.float() / temperature, dim=-1)
    kl = F.kl_div(F.log_softmax(logits.float() / temperature, dim=-1),
                  probability, reduction='none').sum(-1)
    return ce, kl[valid].sum() * temperature ** 2, int(valid.sum())


def importance_sum(scores, target, valid):
    """Sum of per-image KL; teacher target is ALREADY sum-normalized."""
    if scores.shape != target.shape or scores.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError('Importance shapes/mask differ')
    if target.requires_grad or not bool(torch.isfinite(target).all()) or bool((target < 0).any()) or bool(target[~valid].any()):
        raise ValueError('Invalid/differentiable teacher importance')
    if not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-5, rtol=1e-5):
        raise ValueError('Teacher importance must sum to one per image')
    logp = scores.float().masked_fill(~valid, float('-inf')).log_softmax(-1)
    terms = target.float() * (target.float().clamp_min(1e-30).log() - logp.masked_fill(~valid, 0.))
    return terms.masked_fill(~valid, 0.).sum()


def configure_stage(model, stage, *, training=True):
    """Set stage gradients/modes explicitly; no hidden teacher or optimizer reuse."""
    if stage not in ('recovery', 'scorer', 'joint'):
        raise ValueError('Stage must be recovery, scorer or joint')
    model.requires_grad_(False)
    model.train(training)
    if stage == 'scorer':
        model.native.eval()
        model.scorer.requires_grad_(True)
    else:
        model.native.requires_grad_(True)
        if stage == 'joint':
            model.scorer.requires_grad_(True)
        else:
            model.scorer.eval()
    return model


def make_stage_optimizer(model, stage, *, encoder_lr=1e-5, decoder_lr=5e-5,
                         scorer_lr=None, weight_decay=.01):
    """Fresh AdamW for EACH stage; caller must not carry moments across stages."""
    configure_stage(model, stage)
    groups = []
    if stage != 'scorer':
        groups += [{'params': list(model.native.encoder.parameters()), 'lr': encoder_lr, 'name': 'encoder'},
                   {'params': list(model.native.decoder.parameters()), 'lr': decoder_lr, 'name': 'decoder'}]
    if stage != 'recovery':
        groups.append({'params': list(model.scorer.parameters()),
                       'lr': (1e-3 if stage == 'scorer' else 1e-4) if scorer_lr is None else scorer_lr,
                       'name': 'scorer'})
    parameters = [p for group in groups for p in group['params']]
    if len({id(p) for p in parameters}) != len(parameters) or any(not p.requires_grad for p in parameters):
        raise ValueError('Duplicate or frozen optimizer parameters')
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def backward_effective_batch(model, images, tokens, teacher, stage, *, microbatch=1,
                             kd_weight=.5, importance_weight=.1, temperature=2.):
    """Accumulate gradients; caller zeroes once, clips once and steps once.

    CE/KD divide by all valid characters+EOS in the effective batch. Image KL
    divides by all images. Microbatch losses never receive equal averaging when
    transcript lengths differ. Dropout/RNG policy is caller-owned; only a
    dropout-disabled CPU proof is microbatch-equivalent, not arbitrary training.
    """
    if stage not in ('recovery', 'scorer', 'joint') or type(microbatch) is not int or microbatch < 1:
        raise ValueError('Invalid stage or microbatch')
    expected_flags = {'encoder': stage != 'scorer', 'decoder': stage != 'scorer', 'scorer': stage != 'recovery'}
    for name, module in [('encoder', model.native.encoder), ('decoder', model.native.decoder), ('scorer', model.scorer)]:
        if any(p.requires_grad != expected_flags[name] for p in module.parameters()):
            raise ValueError('Configure the stage before accumulating gradients')
    if stage == 'scorer' and model.native.training:
        raise ValueError('Scorer warmup requires the frozen native body in eval mode')
    images = list(images)
    if not images or tokens.shape[0] != len(images) or tuple(model.native.characters) != teacher.characters:
        raise ValueError('Image/transcript/vocabulary batch mismatch')
    total_tokens = int(tokens[:, 1:].ne(model.native.decoder.ignore_index).sum())
    if not torch.equal(tokens, teacher.tokens):
        raise ValueError('Teacher/student transcripts differ')
    if total_tokens == 0:
        raise ValueError('No valid target tokens')
    metrics = {'CE': 0., 'KD_T2_scaled': 0., 'image_KL': 0.,
               'valid_tokens': total_tokens, 'images': len(images)}
    for start in range(0, len(images), microbatch):
        stop = min(start + microbatch, len(images))
        target = teacher.subset(start, stop)
        memory, _, scores = model.memory(images[start:stop], prune=stage == 'joint')
        loss = scores.new_zeros(())
        if stage != 'scorer':
            logits = model.native.decoder.forward_train(memory, tokens[start:stop])
            ce, kd, _ = response_sums(logits, tokens[start:stop], target,
                                      model.native.decoder.ignore_index, temperature)
            loss = (ce + kd_weight * kd) / total_tokens
            metrics['CE'] += float(ce.detach()) / total_tokens
            metrics['KD_T2_scaled'] += float(kd.detach()) / total_tokens
        if stage != 'recovery':
            if target.importance is None:
                raise ValueError('Scorer/joint stage needs teacher image importance')
            n = scores.shape[1]
            valid = target.valid_patches[:, :n]
            expected_lengths = torch.tensor([x.shape[-2] * x.shape[-1] // 16 for x in images[start:stop]], device=scores.device)
            if not torch.equal(valid.sum(-1), expected_lengths):
                raise ValueError('Teacher/student native grids differ')
            kl = importance_sum(scores, target.importance[:, :n], valid)
            weight = 1. if stage == 'scorer' else importance_weight
            loss = loss + weight * kl / len(images)
            metrics['image_KL'] += float(kl.detach()) / len(images)
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite training loss')
        loss.backward()
    metrics['total'] = (metrics['image_KL'] if stage == 'scorer' else
                        metrics['CE'] + kd_weight * metrics['KD_T2_scaled'] +
                        (importance_weight * metrics['image_KL'] if stage == 'joint' else 0.))
    return metrics
