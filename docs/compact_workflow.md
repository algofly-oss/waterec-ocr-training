# Experimental compact workflow

This is a separate WATERec-derived workflow, not a new R/RS pretrained option and
not the one-step Florence training recipe. It uses **multistage distillation**:
a standard teacher, rank recovery, scorer warmup, joint token selection and an
optional fresh-optimizer restart. Training reads images and expected text only.
No boxes or region annotations are required or accepted as model inputs.

The standard `waterec-ocr` R/RS commands remain unchanged. After installing the
updated package, `waterec-compact` is the separate console entry point. The
module form below also works from a source checkout:

```bash
python -m waterec_ocr.waterec_compact_cli --help
python -m waterec_ocr.waterec_compact_cli train \
  --config configs/experimental/compact_dummy_price.yaml
```

Install the `training` extra for native training/inference. Install the existing
`onnx` extra for CPU export/ONNX Runtime, and `build,tensorrt` for target engine
building and TensorRT inference; see [lean deployment](lean_deployment.md).
Training/export import Torch; the TensorRT runtime and builder have separate
lazy import paths. No trained compact weights or engines are bundled.

## Data, paths and teachers

Use `train/image.png` plus the same-stem UTF-8 `train/image.txt`. An optional
`val/` folder is validation only. The dedicated CLI resolves dataset, output,
pretrained, teacher-checkpoint and validation-folder paths **relative to the YAML
file**, whereas direct `train_compact(dict)` calls resolve paths against the
current working directory. Inference/export/build paths are explicit CLI paths.

An explicitly configured validation folder must exist when enabled. Otherwise a
validation fraction in `[0, 1)` creates a deterministic text-grouped holdout if no
validation folder is present. To train to a fixed terminal without validation,
set `use_validation_folder: false` and `validation_fraction: 0`. Validation
selection uses image-only decoding, raw exact then mean normalized Levenshtein
similarity, retaining the earliest exact tie including epoch zero. Validation is
never used to construct teacher training targets.

By default, the existing standard trainer fine-tunes official R/RS weights on a
FIT-only manifest in `standard_teacher/`; its internal selection/refit never
receives the outer compact validation images. Existing standard teacher-training
policy is retained. Official weights can be downloaded if absent. Alternatively,
set `teacher_mode: provided` and `teacher_checkpoint` to a trusted task-adapted
**standard** WATERec checkpoint with matching task, variant, budget, SPACE and
charset metadata. Supplied-teacher pretraining, task-data provenance and absence
of validation/test leakage are the caller's responsibility; a compatible state
dictionary cannot establish those facts. To deliberately skip task fine-tuning,
use `teacher_mode: upstream` and `allow_upstream_teacher: true`.

The native alphabet is the original printable character set, EOS/BOS/PAD and the
configured SPACE extension. Unsupported characters, oversized labels or missing
EOS alignment are rejected, never truncated or transliterated. This is not a
universal multilingual reader.

## Stages and checkpoint semantics

1. Copy the standard teacher and initialize 24 visual MLP linears by balanced
   rank-128 SVD. Keep all 12 visual blocks and both autoregressive decoder layers.
2. Recover encoder/decoder with full image memory, CE plus `0.5 × T² KL` at `T=2`.
   Teacher responses are **raw logits**, temperature-scaled once in the loss.
3. Freeze the learned rank parent as a full-memory teacher. Warm up only the
   50,177-parameter image scorer using explicit image-importance KL. Importance is
   decoder-attention mean over layers/heads, then **maximum** over valid training
   character-plus-EOS queries, masked and sum-normalized once. It is not query
   averaging or another softmax of teacher probabilities.
4. Jointly train encoder, decoder and scorer with CE + `0.5 × T² KL` + `0.1 × image
   KL`. Keep up to 512 patches after six blocks, in raster order with original
   two-dimensional RoPE coordinates. Inference selects from the image only.
5. An optional restart uses fresh AdamW moments and the same frozen rank parent.

Every phase uses a fresh optimizer. Effective-batch CE/KD denominators count all
valid characters plus EOS; image KL uses the image count. Student and teacher
microbatch sizes bound activation work separately. Conservative defaults are one
image per effective/micro/evaluation batch. The supplied one-epoch stages are
**runnable demonstrations, not the measured accuracy recipe**. The budget-64
dummy example keeps all available patches; it does not demonstrate 2048-to-512
compression. A separate audited two-FIT high-resolution smoke tests that path,
without claiming a new trained accuracy result.

Checkpoints and history are written atomically every epoch. Stage `best.pt` and
`last.pt` contain typed compact architecture/charset metadata, weights, optimizer,
RNG, split/source/teacher hashes, selection history and input-order/pixel hashes.
`model.pt` is a typed inference checkpoint with stage provenance. Automatic
resume is **not implemented**: use a fresh output directory. Standard R/RS
checkpoints cannot be guessed or renamed into compact checkpoints.

## Native, ONNX and TensorRT

[Notebook 9](../notebooks/9_experimental_compact_workflow.ipynb) provides the
optional sequence. Example commands after training (select an existing fictional image):

```bash
IMAGE=$(python -c 'from pathlib import Path; print(next(Path("dummy_data/prices/val").glob("*.png")))')
python -m waterec_ocr.waterec_compact_cli infer --backend torch \
  --checkpoint artifacts/compact_dummy_price/model.pt \
  --image "$IMAGE"
python -m waterec_ocr.waterec_compact_cli export \
  --checkpoint artifacts/compact_dummy_price/model.pt \
  --deployment artifacts/compact_dummy_price/deployment
python -m waterec_ocr.waterec_compact_cli infer --backend onnx \
  --deployment artifacts/compact_dummy_price/deployment \
  --image "$IMAGE"
# On a compatible NVIDIA/CUDA target, opt in to engine building:
python -m waterec_ocr.waterec_compact_cli build \
  --deployment artifacts/compact_dummy_price/deployment
```

Use an image that exists in your dataset; the notebook selects one rather than
assuming its name. Native lists may batch, but the current ONNX/TensorRT graphs
are fixed to the exported budget and **static batch 1**. Changing native training
resolution does not dynamically resize an existing engine. Build and qualify
new target engines for each intended profile. Exporting a profile does not prove
accuracy or GPU behavior at a different resolution. Native budgets are configurable above 64 subject to resources; that does not
imply arbitrary TensorRT support. CPU ONNX tests cover budgets 64 and 2048; the
trained GPU profile is 2048 with 512 retained tokens. The current compact TensorRT builder/runtime accepts configured budgets only in
64–2048; this is a numerical selection-ordering bound, not an accuracy guarantee
for every budget in that range. Higher budgets require a separately validated
implementation.
The budget-64 workflow passed CPU and GPU software smokes; its actual CLI
export/build/inference on six fictional FIT images and evaluation on two VAL
images matched CPU outputs. This is functional proof only: it cannot inherit the
trained 2048 profile’s accuracy or resource measurements.
NVCC and a C compiler are required by GPU preprocessing/kernel setup. There is no
Jetson/edge-device or rotation qualification for this compact checkpoint.

## Qualified checkpoint versus a newly trained model

For the separately trained and frozen 17,913,953-parameter price checkpoint, the
current standalone public runtime v3 preserves all 206 native raw strings and
all 1,236 timed outputs. It measured **1.356083 ms RGB / 1.620985 ms encoded** and
**708 MiB** sampled process GPU usage on RTX 5090/static B1, five warmups and three
206-image passes per fresh process. Its input budget is 2048 (typically 128×256),
with 512 selected tokens after six blocks and two AR layers. Accuracy remains
**161/206 raw (78.16%), 187/206 numeric (90.78%), 132/144 AI-reviewed numeric**;
mean normalized Levenshtein similarity is 90.18%, not 90% raw exact. AI144 is an
uncertified AI-assisted subset of 206, not human ground truth. Florence's 133/144
AI result remains one case higher.

Current host RSS was 576.5 MiB in the RGB process and 545.3 MiB in the encoded process; libraries/metadata and the RGB cache are included. OS high-watermark fields are separate, not minimum RAM.

These numbers describe that exact audited checkpoint/runtime snapshot, **not**
weights produced by the demonstration schedule. The earlier strict 0.05 numeric
failure remains recorded; production output-fidelity/margin checks are a
separate policy, not numerical, routing or calibration equivalence. Driver GPU
usage includes CUDA context; host RSS includes libraries/metadata/cache and is
not minimum RAM. Runtime v3 is a distinct sequential measurement from earlier
private/cache trials, not a clock-controlled causal speedup. See
[research history](compact_research.md) for those immutable older records.


## External price-photo limitation

The same qualified checkpoint was also tested on an existing **eight-photo
licensed external probe**. Native FP32 and TensorRT each recovered **2/8 primary
amounts**, both under strict decimal precision and under numeric equivalence.
Visible currency was correct on **0/6** references that contain currency. The
combined amount-and-currency count is 1/8, but that sole match has no visible
currency; it is not evidence of successful currency recognition.

Native and TensorRT raw strings agreed on **7/8**, with both differing outputs
wrong by the frozen primary-price policy. The earlier **206/206 raw-string
agreement remains a qualification of that specific cohort**, not a promise of
backend identity on arbitrary photos. Read-only localization is tracked
separately; no unique cause is claimed here.

The unchanged PP-OCRv6 price engine also scored 2/8 amounts on these inputs; its
native checkpoint result is not measured. Florence now also has a **separate,
source-closed native rerun: 3/8 amounts and 0/6 visible currencies**. Its sole
combined amount-and-currency match also has no visible currency. The exact
previously missing helper was recovered from archived source snapshots. With the
same base and decoder checkpoint, 384×384 processing, BF16 autocast, beam 5 and a
32-token cap, the new run reproduced all eight historical output strings. All
eight ended with EOS before the cap. There was no expected-score acceptance gate.

The older 3/8 record remains unchanged: recovering source and reproducing strings
does not retroactively certify its original dependency environment. The fresh run
has its own pinned current implementation, merged-weight proof and artifact
audit. No latency or memory result is claimed for this rerun. These are eight
independent source photos, all obtained as licensed published display derivatives. The old probe has influenced research,
so this repeat is exploratory and post-development, **not a blind population
benchmark**. Dual-currency and primary/secondary-price ambiguities remain under
the predeclared policy. Some currency glyphs are unsupported; all eight photos
remain in the denominator.

No crop, reference, checkpoint, precision or selection policy was changed for
this repeat. Labels were not parsed or used to condition inference; provenance
checks did binary-hash the reference files. Strict amount scoring preserves zeros
and decimal precision; numeric equivalence is separate, with no substring search
or decimal insertion. The result is negative evidence for broad price-photo
capability despite the stronger in-domain 206 results. Aggregate metrics and the
completed artifact-audit hash are in
[the workflow record](../results/public_compact_workflow.json).
