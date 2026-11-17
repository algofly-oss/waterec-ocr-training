"""Experimental portable compact training workflow, not a CLI/model preset.

Only paired images and expected text enter training. This conservative workflow
is not a byte-for-byte reproduction of the private benchmark schedule. Existing
R/RS training and deployment are unchanged. Every invocation starts a fresh run;
durable stage checkpoints retain optimizer/RNG information for audit, but this
prototype does not implement automatic resume.
SPDX-License-Identifier: Apache-2.0
"""
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch

from .data_io import scan_pairs
from .data import audit_splits, split_holdout
from .metrics import score_pairs, selection_score, EXACT_POLICY
from .waterec_models import create_native
from .waterec_compact_models import CompactWATERec, load_compact_checkpoint
from .waterec_compact_training import (
    TeacherTargets, freeze_teacher, teacher_targets, configure_stage,
    make_stage_optimizer, backward_effective_batch,
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _state(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    return value


def _atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(_safe(data), stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_torch(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(data, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _rng(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch_CPU': torch.get_rng_state(),
            'torch_CUDA': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}


def _seed(seed, device):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _model_config(config):
    task = config.get('task', 'price')
    result = {'task': task, 'variant': config.get('variant', 'R' if task == 'price' else 'RS'),
            'tokens': config.get('tokens', 512), 'include_space': config.get('include_space', task == 'price'),
            'max_label_length': config.get('max_label_length', 24),
            'max_decode_steps': config.get('max_decode_steps', 25),
            'normalization': config.get('normalization', 'none')}
    if 'additional_characters' in config:
        from .waterec_charset import validate_additions
        additional=validate_additions(config['additional_characters'])
        if additional:result['additional_characters']=additional
    return result


def _fingerprints(rows):
    result = []
    for row in rows:
        path = Path(row['path'])
        with Image.open(path) as image:
            rgb = image.convert('RGB')
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        result.append({'image': row['image'], 'path': str(path.resolve()),
                       'label': row['label'], 'label_file_sha256': _sha(path.with_suffix('.txt')),
                       'file_sha256': _sha(path), 'RGB_sha256': pixels})
    return result


def _verify_inputs(manifest):
    for group in ('fit', 'validation'):
        if _fingerprints(manifest[group]) != manifest[group]:
            raise ValueError('Input files, decoded pixels or labels changed during the run')


def _pixels(native, rows, device, seed=None, augment=False):
    generator = random.Random(seed)
    tensors = []
    for row in rows:
        with Image.open(row['path']) as source:
            image = source.convert('RGB')
        if augment:
            image = ImageEnhance.Contrast(image).enhance(generator.uniform(.8, 1.2))
        tensors.append(native.preprocess(image).to(device))
    return tensors


def _evaluate(model, rows, device, batch_size, prune):
    if not rows:
        return None
    model.eval()
    records = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            pixels = _pixels(model.native, part, device)
            memory, _, _ = model.memory(pixels, prune=prune)
            text = model.decode(model.native.decoder.forward_test(memory))
            records.extend({'image': row['image'], 'gt': row['label'], 'prediction': prediction}
                           for row, prediction in zip(part, text))
    labels, predictions = [r['gt'] for r in records], [r['prediction'] for r in records]
    metrics = score_pairs(labels, predictions, normalization='none')
    return {'metrics': metrics, 'records': records, 'image_only': True,
            'prune': prune, 'selection': 'literal exact count only; earliest tie'}


def _memory_guard(config, device):
    if device.type == 'cuda':
        limit = config.get('max_gpu_memory_mib', 4096) * (1 << 20)
        if torch.cuda.max_memory_allocated(device) > limit:
            raise MemoryError('Compact training GPU allocation cap exceeded; no automatic batch/resolution fallback')


def _teacher(config, model_config, fit, output):
    """Provided task teacher, auto standard fine-tuning, or explicit upstream opt-in."""
    checkpoint = config.get('teacher_checkpoint')
    mode = 'provided' if checkpoint else config.get('teacher_mode', 'finetune')
    if mode == 'provided':
        checkpoint = Path(checkpoint).expanduser().resolve()
        try:
            teacher = create_native(model_config, checkpoint=checkpoint, pretrained=False)
        except ValueError as error:
            raise ValueError('Provided teacher checkpoint/config/ordered alphabet mismatch. Use automatic matching teacher finetuning (teacher_mode=finetune without teacher_checkpoint), or provide an explicitly migrated matching checkpoint. '+str(error)) from error
    elif mode == 'finetune':
        from .waterec_training import train as train_standard
        teacher_folder = output / 'standard_teacher'
        manifest = output / 'teacher_FIT_only.json'
        _atomic_json(manifest, [{'image': r['image'], 'path': r['path'], 'label': r['label']} for r in fit])
        tc = dict(model_config, output_dir=teacher_folder, train_manifest=manifest,
                  pretrained_dir=config.get('pretrained_dir', Path('weights/waterec').resolve()),
                  epochs=config.get('teacher_epochs', 1), warmup_epochs=0,
                  batch_size=config.get('teacher_batch_size', 1), num_workers=0,
                  precision=config.get('precision', 'fp32'), device=config.get('device', 'cpu'),
                  encoder_learning_rate=config.get('encoder_learning_rate', 1e-5),
                  decoder_learning_rate=config.get('decoder_learning_rate', 5e-5),
                  weight_decay=config.get('weight_decay', .01),
                  holdout_fraction=config.get('teacher_holdout_fraction', .15),
                  split_seed=config.get('seed', 15001), seed=config.get('seed', 15001),
                  synthetic_plate_count=0, refit_all_training=True)
        train_standard(tc, resume=False)
        checkpoint = teacher_folder / 'full_fit/model.pt'
        try:
            teacher = create_native(model_config, checkpoint=checkpoint, pretrained=False)
        except ValueError as error:
            raise ValueError('Provided teacher checkpoint/config/ordered alphabet mismatch. Use automatic matching teacher finetuning (teacher_mode=finetune without teacher_checkpoint), or provide an explicitly migrated matching checkpoint. '+str(error)) from error
    elif mode == 'upstream' and config.get('allow_upstream_teacher') is True:
        tc = dict(model_config, pretrained_dir=config.get('pretrained_dir', Path('weights/waterec').resolve()))
        teacher = create_native(tc, pretrained=True)
    else:
        raise ValueError('Use a task teacher, auto finetune, or explicit allow_upstream_teacher=True')
    binding = {'mode': mode, 'checkpoint_sha256': _sha(checkpoint) if checkpoint else None,
               'upstream_without_task_finetuning': mode == 'upstream'}
    if model_config.get('additional_characters'):
        from .waterec_charset import metadata, vocabulary_hash
        binding.update(teacher_state_sha256=_state(teacher), ordered_characters=list(teacher.characters),
                       vocabulary_sha256=vocabulary_hash(teacher.characters), alphabet_metadata=metadata(teacher))
    return freeze_teacher(teacher), binding


def _targets(teacher, images, tokens, microbatch, include_importance):
    # Bound teacher activation memory independently of the effective batch.
    parts = [teacher_targets(teacher, images[start:start + microbatch], tokens[start:start + microbatch],
                             include_importance=include_importance)
             for start in range(0, len(images), microbatch)]
    width = max(x.valid_patches.shape[1] for x in parts)
    pad = torch.nn.functional.pad
    return TeacherTargets(torch.cat([x.raw_logits for x in parts]),
                          None if not include_importance else torch.cat([pad(x.importance, (0, width - x.importance.shape[1])) for x in parts]),
                          torch.cat([pad(x.valid_patches, (0, width - x.valid_patches.shape[1]), value=False) for x in parts]),
                          torch.cat([x.valid_tokens for x in parts]), tokens.detach().clone(), parts[0].characters)


def _run_stage(model, teacher, stage, name, epochs, fit, validation, config, output, binding, stage_index):
    binding = copy.deepcopy(binding)
    folder = output / name
    folder.mkdir()
    device = next(model.parameters()).device
    optimizer = make_stage_optimizer(model, stage,
                                     encoder_lr=config.get('encoder_learning_rate', 1e-5),
                                     decoder_lr=config.get('decoder_learning_rate', 5e-5),
                                     scorer_lr=config.get('scorer_learning_rate', 1e-3) if stage == 'scorer' else config.get('joint_scorer_learning_rate', 1e-4),
                                     weight_decay=config.get('weight_decay', .01))
    assert not optimizer.state
    teacher_state, initial_state = _state(teacher), _state(model)
    history, stream = [], []
    selected_epoch, best_score, steps = 0, None, 0
    batch_size, micro = config.get('batch_size', 1), config.get('microbatch', 1)
    precision = config.get('precision', 'fp32')
    best_path, last_path = folder / 'best.pt', folder / 'last.pt'
    for epoch in range(epochs + 1):
        epoch_metrics = []
        if epoch:
            configure_stage(model, stage)
            indexes = list(range(len(fit)))
            random.Random(config.get('seed', 15001) + stage_index * 10000 + epoch).shuffle(indexes)
            for offset in range(0, len(indexes), batch_size):
                selected = indexes[offset:offset + batch_size]
                rows = [fit[i] for i in selected]
                seed = config.get('seed', 15001) + stage_index * 100000 + epoch * 1000 + offset
                _seed(seed, device)
                images = _pixels(model.native, rows, device, seed, config.get('augment_contrast', False))
                tokens = model.encode_labels([r['label'] for r in rows], device=device)
                target = _targets(teacher, images, tokens, config.get('teacher_microbatch', micro), include_importance=stage != 'recovery')
                optimizer.zero_grad(set_to_none=True)
                autocast = torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else contextlib.nullcontext()
                with autocast:
                    metrics = backward_effective_batch(model, images, tokens, target, stage,
                                                        microbatch=micro, kd_weight=.5,
                                                        importance_weight=.1, temperature=2.)
                metrics['preclip_norm'] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('clip_norm', 5.), error_if_nonfinite=True))
                optimizer.step()
                steps += 1
                stream.append({'step': steps, 'epoch': epoch, 'seed': seed,
                               'FIT_indices': selected, 'images': [r['image'] for r in rows],
                               'pixel_sha256': [hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest() for x in images],
                               'valid_tokens': metrics['valid_tokens'], 'image_count': len(rows)})
                epoch_metrics.append(metrics)
                _memory_guard(config, device)
        evaluation = _evaluate(model, validation, device, config.get('evaluation_batch_size', 1), prune=stage != 'recovery')
        score = None if evaluation is None else selection_score(evaluation['metrics'])
        improved = score is not None and (best_score is None or score > best_score)
        if improved:
            best_score, selected_epoch = score, epoch
        if evaluation is None and epoch == epochs:
            selected_epoch = epoch
        history.append({'epoch': epoch, 'updates': steps, 'training_batches': epoch_metrics,
                        'validation': evaluation, 'selected_epoch_so_far': selected_epoch})
        if _state(teacher) != teacher_state:
            raise ValueError('Frozen teacher changed')
        payload = model.checkpoint_payload()
        payload.update(optimizer=optimizer.state_dict(), rng=_rng(device),
                       compact_training={'schema': 1, 'selection_policy': EXACT_POLICY, 'stage': stage, 'name': name, 'epoch': epoch,
                                         'steps': steps, 'initial_state': initial_state, 'model_state': _state(model),
                                         'teacher_state': teacher_state, 'binding': binding,
                                         'selected_epoch': selected_epoch, 'best_score': best_score,
                                         'selection': 'image-only literal exact count only, earliest tie, including epoch0' if validation else 'fixed terminal; no validation',
                                         'target_policy': 'Frozen teacher RAW T2 once; meanlayers/heads thenMAXvalidGTchar+EOS importance normalized once',
                                         'prune_training': stage == 'joint', 'prune_validation': stage != 'recovery',
                                         'history': history, 'stream': stream,
                                         'optimizer_groups': [{'name': g['name'], 'lr': g['lr'], 'parameters': sum(p.numel() for p in g['params'])} for g in optimizer.param_groups]})
        _atomic_torch(last_path, payload)
        if improved or (evaluation is None and epoch == epochs):
            _atomic_torch(best_path, payload)
        _atomic_json(folder / 'history.json', history)
        _memory_guard(config, device)
    selected = torch.load(best_path, map_location='cpu', weights_only=False)
    # Stage entry remains the same object; retain stage teacher separately.
    model.load_state_dict(selected['state_dict'], strict=True)
    if _state(model) != selected['compact_training']['model_state']:
        raise ValueError('Selected stage checkpoint failed strict reload')
    summary = {'stage': stage, 'name': name, 'epochs': epochs, 'steps': steps,
               'exposures': epochs * len(fit), 'selected_epoch': selected_epoch,
               'selected_checkpoint_sha256': _sha(best_path), 'last_checkpoint_sha256': _sha(last_path),
               'teacher_state': teacher_state, 'initial_state': initial_state, 'selected_state': _state(model),
               'fresh_optimizer_at_entry': True, 'optimizer_state_retained_in_checkpoints': True,
               'history_sha256': _sha(folder / 'history.json'), 'binding': binding,
               'scope': 'Portable image/text-only stage; not benchmark recipe/trajectory parity'}
    _atomic_json(folder / 'completion.json', summary)
    return summary


def train_compact(config):
    """Run a fresh portable staged experiment; return final checkpoint and records.

    Required: dataset_root with train/{image, matching UTF-8 .txt}, output_dir.
    Optional val/ folder (never teacher training), or validation_fraction=0 for a
    fixed terminal. Default batch/micro/evaluation batch sizes are all one.
    A task-adapted standard teacher can be supplied; otherwise the existing
    standard trainer fine-tunes official weights in a distinct output directory.
    Explicit teacher_mode='upstream', allow_upstream_teacher=True skips that step.
    """
    config = dict(config)
    if config.get('normalization', 'none') != 'none':
        raise ValueError('Literal training labels require normalization: none')
    mc = _model_config(config)
    device = torch.device(config.get('device', 'cpu'))
    if config.get('precision', 'fp32') not in ('fp32', 'bf16'):
        raise ValueError('Use fp32 or supported CUDA bf16')
    if config.get('precision', 'fp32') == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 training requires a supporting CUDA device')
    for name in ('batch_size', 'microbatch', 'evaluation_batch_size', 'teacher_microbatch'):
        if type(config.get(name, 1)) is not int or config.get(name, 1) < 1:
            raise ValueError('Batch sizes must be positive integers')
    stages = [('recovery', 'rank_recovery', config.get('recovery_epochs', 1)),
              ('scorer', 'scorer_warmup', config.get('scorer_epochs', 1)),
              ('joint', 'joint', config.get('joint_epochs', 1))]
    restart = config.get('restart_epochs', 0)
    if restart:
        stages.append(('joint', 'restart', restart))
    if any(type(n) is not int or n < 0 for _, _, n in stages):
        raise ValueError('Stage epoch counts must be nonnegative integers')
    if not config.get('teacher_checkpoint') and config.get('teacher_mode', 'finetune') == 'finetune':
        if type(config.get('teacher_epochs', 1)) is not int or config.get('teacher_epochs', 1) < 1:
            raise ValueError('Automatic standard-teacher finetuning needs at least one epoch')
    fraction = config.get('validation_fraction', .15)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError('validation_fraction must be finite and in [0, 1)')
    root = Path(config['dataset_root']).expanduser().resolve()
    explicit_validation = config.get('validation_folder')
    if 'validation_folder' in config and config.get('use_validation_folder', True):
        if not isinstance(explicit_validation, (str, Path)) or not str(explicit_validation).strip():
            raise ValueError('validation_folder must name an existing directory when enabled')
        requested = Path(explicit_validation).expanduser()
        if not requested.exists():
            raise FileNotFoundError('Explicit validation_folder does not exist: ' + str(requested))
        if not requested.is_dir():
            raise NotADirectoryError('Explicit validation_folder is not a directory: ' + str(requested))
    checkpoint = config.get('teacher_checkpoint')
    teacher_mode = config.get('teacher_mode', 'finetune')
    if teacher_mode not in ('provided', 'finetune', 'upstream'):
        raise ValueError('teacher_mode must be provided, finetune or upstream')
    if teacher_mode == 'provided' and not checkpoint:
        raise ValueError('teacher_mode=provided requires teacher_checkpoint')
    if checkpoint and not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError('teacher_checkpoint is not an existing file: ' + str(checkpoint))
    if not checkpoint and teacher_mode == 'upstream' and config.get('allow_upstream_teacher') is not True:
        raise ValueError('Upstream teacher requires explicit allow_upstream_teacher=True')
    output = Path(config['output_dir']).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a fresh output directory; automatic resume is not implemented')
    output.mkdir(parents=True, exist_ok=True)
    _seed(config.get('seed', 15001), device)
    fit = scan_pairs(root / 'train', mc['normalization'])
    val_folder = Path(config.get('validation_folder', root / 'val')).expanduser() if config.get('use_validation_folder', True) else root / 'val'
    if val_folder.is_dir() and config.get('use_validation_folder', True):
        validation = scan_pairs(val_folder, mc['normalization'])
    elif config.get('validation_fraction', .15) > 0:
        fit, validation = split_holdout(fit, config.get('validation_fraction', .15), config.get('seed', 15001))
    else:
        validation = []
    split_audit = audit_splits(fit, validation)
    if split_audit['pixel_duplicates_across_splits']:
        raise ValueError('Decoded RGB overlap across FIT and validation')
    manifest = {'fit': _fingerprints(fit), 'validation': _fingerprints(validation), 'audit': split_audit}
    _atomic_json(output / 'inputs.json', manifest)
    package = Path(__file__).resolve().parent
    sources = {name: _sha(package / name) for name in
               ['waterec_compact_models.py', 'waterec_compact_training.py', 'waterec_compact_workflow.py',
                'waterec_models.py', 'waterec_training.py', 'data_io.py', 'data.py', 'metrics.py', 'waterec_synthetic.py', '_vendor/waterec/encoder.py', '_vendor/waterec/decoder.py', '_vendor/waterec/characters.txt']}
    if mc.get('additional_characters'):
        sources['waterec_charset.py'] = _sha(package / 'waterec_charset.py')
    _atomic_json(output / 'run_protocol.json', {'schema': 1, 'config': _safe(config), 'model': mc,
                                               'sources': sources, 'input_manifest_sha256': _sha(output / 'inputs.json'),
                                               'stages': stages, 'loss': 'CE+.5*T²KL(T2) token mean; joint+.1 imageKL image mean',
                                               'teacher_target_policy': 'No labels from validation; no boxes; freeze full-memory teacher per stage',
                                               'fresh_run_only': True, 'no_benchmark_or_runtime_claim': True})
    started = time.perf_counter()
    teacher, teacher_binding = _teacher(config, mc, fit, output)
    teacher = teacher.to(device)
    for row in fit + validation:
        teacher.encode_labels([row['label']])
    model = CompactWATERec.from_teacher(teacher, rank=config.get('rank', 128),
                                        after_blocks=config.get('after_blocks', 6),
                                        target_keep=config.get('target_keep', 512)).to(device)
    standard_state = _state(teacher)
    binding = {'run_protocol_sha256': _sha(output / 'run_protocol.json'),
               'input_manifest_sha256': _sha(output / 'inputs.json'),
               'standard_teacher': dict(teacher_binding, state=standard_state), 'sources': sources}
    results = []
    rank_teacher = None
    for index, (stage, name, epochs) in enumerate(stages):
        current_teacher = teacher if stage == 'recovery' else rank_teacher
        if current_teacher is None:
            raise ValueError('Rank recovery must establish the frozen full-memory parent')
        results.append(_run_stage(model, current_teacher, stage, name, epochs, fit, validation,
                                  config, output, binding, index))
        if stage == 'recovery':
            rank_teacher = freeze_teacher(copy.deepcopy(model))
            binding['rank_parent_teacher_state'] = _state(rank_teacher)
        _verify_inputs(manifest)
        for source, expected in sources.items():
            if _sha(package / source) != expected:
                raise ValueError('Training source changed during the run')
    if _state(teacher) != standard_state:
        raise ValueError('Standard teacher changed')
    model.eval()
    final = model.checkpoint_payload()
    final.update(compact_training={'schema': 1, 'selection_policy': EXACT_POLICY, 'stage': 'final_inference', 'binding': binding,
                                   'stages': results, 'model_state': _state(model),
                                   'selection': 'Selected stage states; literal exact only, earliest ties, or fixed terminal without validation'},
                 rng=_rng(device))
    _atomic_torch(output / 'model.pt', final)
    fresh = load_compact_checkpoint(output / 'model.pt', device=device)
    if _state(fresh) != _state(model):
        raise ValueError('Fresh final checkpoint reload mismatch')
    _memory_guard(config, device)
    result = {'pass': True, 'checkpoint': str(output / 'model.pt'), 'checkpoint_sha256': _sha(output / 'model.pt'),
              'model_state': _state(model), 'parameters': sum(p.numel() for p in model.parameters()),
              'stages': results, 'binding': binding, 'fit_images': len(fit), 'validation_images': len(validation),
              'training_seconds_including_teacher_setup_validation_serialization_reload': time.perf_counter() - started,
              'scope': 'Portable workflow validation, not pretrained quality, benchmark reproduction or inference resource qualification'}
    _atomic_json(output / 'completion.json', result)
    return result
