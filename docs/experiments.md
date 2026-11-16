# Experiment record

## Questions and conclusions

We tested whether a compact OCR model could approach Florence-2's layout/generalization capability with lower latency and memory. Two tasks exposed different weaknesses: plate transcription and price tags with varying character sizes, superscript-like fractions, currency and non-uniform placement.

The strongest qualified WATERec operating points are RS/512 for plates and R/2048 for prices. They are separately trained models. They should not be described as a single universal checkpoint, the smallest model tested, or a demonstrated 1 ms solution for every task.

The plate model scored 2,475/2,693 exact and 880/984 on reviewed unseen images. The price model scored 153/206 exact, 180/206 strict numeric and 132/144 on the clear AI-reviewed numeric subset. Florence's corresponding final price counts were 148/176/133. That supports a strong speed/accuracy tradeoff, not uniform superiority on every metric or country. PARSeq was stronger than WATERec on the particular US subset of the foreign-plate comparison.

## What the difficult scenarios taught us

- **Small fractions matter.** A model can read ordinary words and still discard price cents. Whole-string numeric accuracy is therefore reported alongside exact text and normalized edit similarity. No metric silently removes a decimal or extracts a convenient substring.
- **Layout is more than character recognition.** Two-line plates and price digits at different vertical positions test output ordering. WATERec retains 2D patch information and uses an autoregressive character reader; task-specific annotations teach the desired sequence without boxes.
- **Unseen images require their own test.** Strong in-distribution accuracy does not establish different-country or capture-condition performance. Original public pretraining exposure is unknown; our duplicate checks do not prove global pretraining novelty.
- **Rotation correction is a pipeline feature.** A separately trained orientation head substantially improved quarter-turn results but slightly harmed upright accuracy. Native and corrected pipelines, model versions and decoder choices must remain separate in tables.
- **Increasing resolution is not free.** Larger budgets can preserve small details while increasing attention cost. A model trained at one budget can lose accuracy at another. Resolution changes require measurement, not inherited claims.
- **Parameter count is not process VRAM.** CUDA context, profiles, scratch space, activations and caches contribute to measured GPU memory. Host RAM, model weights and driver-accounted VRAM are distinct quantities.

See [the capability matrix](capabilities.md) for measured, qualitative and untested cases.

## Training and evaluation boundaries

Only images and expected transcription strings are supplied for OCR training. We do not require box annotations. Generic upstream weights provide initialization. Internal training holdouts select epochs; external evaluation cohorts are not used as the training loader or refit pool.

The strong price reference fitted 676 images, selected on an internal 144-image holdout, and was later qualified on 206 separate images. Its 20-epoch recipe used 860 updates, effective batch 16, three decoder-warmup epochs, encoder/decoder learning rates 1e-5/5e-5, AdamW weight decay 0.01, clipping 5, seed 15001, contrast augmentation 0.8–1.2 and mean-initialized SPACE. Joint training used checkpointing/Flash SDPA with explicitly constrained mask handling; evaluation retained the original FP32 attention path. The final Florence price reference used 820 training images, so training-pool equality is not claimed for that final comparison. The final price qualification was post-development rather than a prospective blind trial.

The portable training workflow preserves the model and image/text interface but uses configurable conservative batches and the vendored native attention path. It is not a byte-for-byte replay of the specialized batch-16 research harness. Users should record their data/split hashes, software, seeds, augmentation, precision, training policy and checkpoint selection. The dummy data reproduces the interface, not the private experiment.

Some price annotations came from model-generated labels. We retained raw-label metrics and separately reviewed numeric subsets rather than pretending all labels were human-certified. The full-cohort and reviewed-subset denominators remain explicit.

## Unsuccessful and experimental routes

Our small custom CCT/AR model (about 5.51M parameters) finished at 891/1,167 plate exact on an internal development split and 79/144 price exact with plain training. Response distillation increased the price result to 80/144, well below WATERec's 113/144 internal reference. These are different cohorts from the final tables above; the small model was not promoted as a replacement or given inherited TensorRT timings.

Width-only expansion of an earlier small reader did not recover complex-price accuracy. Removing WATERec encoder layers also lost accuracy. A rank-128 MLP compression retained two generic training words but lost decimal fractions on two of four price training examples. Those tiny probes are diagnostic evidence, not validation accuracy or a proof that all compression methods fail.

The experimental WATERec speculative decoder adds a 1,386,624-parameter draft to the unchanged OCR target. It is an optional faster-decoding mode, not a smaller WATERec architecture or PARSeq Fast Tiny. Draft/target checkpoint identities are bound together. Actual accuracy and numerical differences must be rechecked for each pair.

No Nano/Small/Medium WATERec family with demonstrated equivalent accuracy is released here. Future exploratory architectures should remain outside the supported model-options table until trained and validated.

## Timing interpretation

RGB-to-text includes preprocessing, transfer, encoder/decoder execution and output decoding. File-to-text additionally includes local file reading/image decoding. Network, detection, model loading, export and first-use compilation are excluded. Reference measurements used an RTX 5090, batch 1 and five warmups; exact cohorts/software and source evidence identifiers are in [reference_metrics.json](../results/reference_metrics.json).

GPU memory is a sampled per-process NVML peak. A model's weight file size is not that measurement. Training peaks must not be placed in an inference-memory column. Batch throughput must report batch size and be measured separately from batch-1 latency. The public static-batch-one runtime cannot inherit throughput numbers from separate research batch engines.

## Reproduce and reference

Use the numbered notebooks to exercise data, training, native inference, export, engine building and evaluation with fictional examples. Replace them with a separate licensed dataset for scientific replication. Cite the repository revision and upstream WATERec source; retain the dataset denominator, profile, precision and timing boundary when quoting a result.

Related experiment repositories: [Florence-2](https://github.com/algofly-oss/vllm-ocr-training), [PARSeq](https://github.com/algofly-oss/parseq-ocr-training), [CCT-S](https://github.com/algofly-oss/license-plate-ocr-training).
