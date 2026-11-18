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
selection uses image-only decoding and literal whole-string exact count only,
retaining the earliest tied count including epoch zero. Validation is
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

The default alphabet is the original printable character set, EOS/BOS/PAD and the
configured SPACE extension. An optional ordered `additional_characters` list
appends NFC codepoints, for example `["€", "£", "¥", "₹"]`. Omit the key or
use `[]` to retain the original alphabet and checkpoint format. New rows use
mean initialization and require training; they do not provide zero-shot
recognition or preserve previous predictions automatically. Extended checkpoints
and deployment assets carry an ordered alphabet/hash and explicit EOS/BOS/PAD
metadata. A matching extended teacher is required for compact distillation. Unsupported characters, oversized labels or missing
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
assuming its name. Native lists may batch. The default export remains fixed to
the exported budget and **static batch 1**; its existing commands are unchanged. Changing native training
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

## Optional fixed-capacity batching

For an **ASCII** compact checkpoint trained with budget 2048 and target keep
512, export a separate fixed-capacity deployment. This does not change training:

```bash
waterec-compact export --checkpoint artifacts/compact_price/model.pt \
  --deployment artifacts/compact_price/batch2 --static-batch 2
waterec-compact infer --backend onnx --deployment artifacts/compact_price/batch2 \
  --folder dataset_price/val --batch-size 2
# Build on the intended NVIDIA/CUDA target, then qualify its outputs:
waterec-compact build --deployment artifacts/compact_price/batch2
waterec-compact benchmark --backend tensorrt \
  --deployment artifacts/compact_price/batch2 --folder dataset_price/val \
  --batch-size 2 --repeats 3 --predecoded
```

`--static-batch` accepts 1, 2, 4 or 8 and creates dedicated rowwise graphs and
engines. Build/infer/evaluate/benchmark dispatch from the metadata family;
`--batch-size` is the number of real images per call and must not exceed the
fixed capacity. One batched encoder and cached decoder process the rows together.
A partial tail duplicates the first real row to capacity and discards dummy
outputs. Benchmark JSON reports physical slots, real-image throughput, full
batch latency and partial batches separately. Batch latency is not single-image
latency. These are fixed profiles, not a dynamic-batch or edge qualification.

Extended Unicode checkpoints currently use the **original static B1 path**:
omit `--static-batch` and keep `--batch-size 1`. Unicode fixed-batch export and
metadata are rejected explicitly, including `--static-batch 1`; the available
100-class B1 proof does not qualify Unicode B2/B4/B8. The 128-lane greedy kernel
is an implementation limit, not evidence that every alphabet is qualified.

The existing 206-image qualification is bounded to that cohort. On a separate,
previously examined eight-photo probe, B1/B2 output `4` for p02 while B4/B8 output
`PRICE`; both are wrong under the frozen primary-price policy. Other seven
outputs agree, and order/policy changes agree within each capacity. The result
remains 2/8 strict amounts and 0/6 actual visible currencies. This is not
universal native/batch equivalence or generic price-photo capability.

### Measured fixed-capacity profiles

RTX 5090, the same 17,913,953-parameter checkpoint, aspect-preserved budget
2048 (typically 128×256 pixels), keeping 512 tokens after six blocks. These
are the explicit `--static-batch` profiles, using reused barrier allocations.

| Capacity | RGB full-batch ms | Encoded full-batch ms | RGB real images/s | GPU MiB | Host RSS MiB, RGB / encoded |
|---|---:|---:|---:|---:|---:|
| 1 | 1.390 | 1.662 | 719.5 | 708 | 574.4 / 539.9 |
| 2 | 2.333 | 2.804 | 857.1 | 732 | 574.5 / 541.1 |
| 4 | 3.537 | 4.518 | 1120.2 | 778 | 577.2 / 544.3 |
| 8 | 5.497 | 7.247 | 1443.0 | 878 | 581.3 / 544.9 |

All capacities reproduced the same 206 native outputs: **161/206 literal exact (78.16%)**.
This repeatedly examined cohort is post-development evidence, not blind accuracy.

Each scope used a fresh process, five warm batch calls and three complete passes
(618 real images). B4/B8 used 624 physical slots including padded tails. The
full-batch means exclude partial calls; real-image throughput includes them.
RGB starts with decoded images; encoded includes file/PIL decoding. Host RSS
includes libraries, metadata and the decoded-image cache where applicable; GPU
memory is the sampled process peak including the CUDA context. Neither value is
a minimum RAM requirement or model-weight size. These are single measured trials,
not controlled speedup estimates. Detailed timing, memory and evidence hashes are
in [the result record](../results/public_compact_workflow.json). Unicode software
fixtures do not establish new-symbol recognition accuracy or these resource costs.

## Qualified checkpoint versus a newly trained model

For the separately trained and frozen 17,913,953-parameter price checkpoint, the
current standalone public runtime v3 preserves all 206 native raw strings and
all 1,236 timed outputs. It measured **1.356083 ms RGB / 1.620985 ms encoded** and
**708 MiB** sampled process GPU usage on RTX 5090/static B1, five warmups and three
206-image passes per fresh process. Its input budget is 2048 (typically 128×256),
with 512 selected tokens after six blocks and two AR layers. Accuracy remains
**161/206 literal exact (78.16%)**. Every symbol, space and sequence position matters. Historical training/selection ancestry differs from the current exact-only workflow; these results are not evidence that old runs used today’s policy.

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
licensed external probe**. The frozen external-photo probe exposed difficult price and currency errors. Its older amount-parser scores are not literal full-string accuracy and are not reported here; the private evidence is preserved.
Native and TensorRT raw strings agreed on **7/8**, with both differing outputs
wrong by the frozen primary-price policy. The earlier **206/206 raw-string
agreement remains a qualification of that specific cohort**, not a promise of
backend identity on arbitrary photos. Read-only localization is tracked
separately; no unique cause is claimed here.

A separate source-closed Florence native rerun reproduced all eight historical output strings with the same base/decoder checkpoint,384×384 processing, BF16 autocast, beam5 and a32-token cap. All ended with EOS before the cap. The previously missing helper was recovered from archived source. This does not retroactively certify the earlier environment, and no latency/memory claim is made.

These eight licensed source photos had already influenced research, so this is exploratory post-development evidence, not a blind population benchmark. Crops, references, weights, precision and output policy were unchanged; reference hashes were checked, but labels did not condition inference. This demonstrates the limits of the in-domain206 output agreement, not broad price-photo capability. Private historical scoring remains preserved; new public accuracy reporting requires literal complete strings.


## Exact-replay limits

Notebook9 runs the portable multistage compact WATE software on your image/text data. Its budget64 dummy configuration is not the historical2048→512 recipe, and this17.9M family is not the newer1.687M multiscale reader. The source release bundles neither measured weights nor private benchmark data. Current corrected-policy PP, photometric tiny-reader and reference-training experiments are not implemented by this compact training command. Reproducing their exact steps still requires separately portable training adapters/configs, legitimate data with matching frozen splits/labels, the correct initial weights and launch/environment records. Dummy substitutions can establish software behavior, not the reported accuracy. No complete public-assets fresh replay of all research experiments has been demonstrated.
