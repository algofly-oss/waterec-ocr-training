# Compact OCR research update

This records trained experiments beyond the portable notebook options. A separate [experimental compact training/export/runtime workflow](compact_workflow.md) is now available; the measured reference weights are **not bundled**, and it is not a new standard R/RS pretrained option. The standard WATERec R/RS workflows remain the reproducible public interface. No Nano/Small/Medium family with equivalent accuracy has been established.

## Latest qualified point: restart2

The latest fixed terminal keeps the same **17,913,953-parameter** architecture: aspect-preserved R 2048 input (typically 128×256), selection of 512 patches after six encoder blocks, six remaining blocks and two AR decoder layers. Both new engines were rebuilt; the old decoder was not reused.

| Latest research model | Raw exact /206 | Numeric /206 | AI-reviewed numeric /144 | Mean normalized Levenshtein similarity | RGB / encoded mean ms | GPU / current host MiB |
|---|---:|---:|---:|---:|---:|---:|
| Token selection restart2, standalone public runtime v3 |161 (78.16%)|187 (90.78%)|132 (91.67%)|90.18%|1.356083 /1.620985|708 GPU;576.5 RGB /545.3 encoded host|
| Token selection restart 2, earlier private runtime |161 (78.16%)|187 (90.78%)|132 (91.67%)|90.18%|1.362 /1.620|708 GPU; 581.1 RGB /549.7 encoded host|

Raw exact also checks currency, punctuation and spacing. **These metrics are different:** raw exact is 78.16%; 90.78% is numeric equivalence of the full price value, including its decimal fraction, and 90.18% is mean normalized Levenshtein similarity (CER 13.61%). AI 144 is an uncertified AI-assisted subset of 206, not human ground truth or internal DEV 144. Florence remains one AI-reviewed numeric case higher (133/144), so this is not noninferior on every accuracy metric.

After the original full 820-image head stage (one epoch / 26 updates) and joint stage (four epochs / 208 updates), restart 2 used a fresh three-group AdamW optimizer for two prospectively fixed epochs / 104 updates / 1,640 exposures. All 820 images were TRAIN; no checkpoint was selected using the new 206 outputs. The 206 cohort had already been observed in research, so this is post-development evidence, not a blind test. Training remains image + expected text only.

Compared with the original custom terminal, 16 raw strings change: six raw gains and four losses, three numeric gains and two losses, and two AI-reference gains and two losses. This is **not a lossless model update**. Runtime output parity is instead exact against the new terminal's own native output: 206 replay strings and 1,236 timed strings. The old intermediate 0.05 numeric failures remain failures. A separate through-EOS output-fidelity policy checks exact tokens/strings, sufficient 2E&lt; argmax-margin, pixels, masks, finite values and legal selection from the same GPU scores; it does not claim numerical, routing or calibration equivalence.

The fast path caches geometry uploads and retains the queued validator. Fault flags 1/2/4 must be rejected before text returns; standalone encode consumers must explicitly call `validate_encoded`. Five warmups and three complete 206-image passes per fresh RGB/encoded process produced the table's RTX 5090 static batch 1 results. Sampled GPU memory includes CUDA context. Current host RSS includes libraries/metadata and the RGB cache where applicable; OS high-water accounting is separate, not minimum RAM. No edge, batch or rotation qualification is inherited. Reference weights remain unbundled. The architecture is available through the separate experimental CLI and notebook 9; standard R/RS model options are unchanged. Demonstration training does not inherit this checkpoint’s accuracy.

Hashes and exact values are appended as `latest_restart2` in [the aggregate JSON](../results/compact_research.json); the original top-level record and older results below remain historical.

## Earlier trained image-only token selection

The custom price reader has **17,913,953 parameters**. A recovered rank-128 WATERec encoder processes the full aspect-preserved image for six blocks. A learned 50,177-parameter scorer retains 512 patches in raster order, with their original two-dimensional coordinates, for the remaining six blocks and two-layer autoregressive decoder. Typical input is 128×256 with a 2,048-patch budget; smaller valid grids are padded and masked. It is not a detector-and-crop pipeline.

Training supplies images and expected text only. Teacher attention derived from the training transcription supervises the patch scorer; inference uses only the image. There are no human bounding-box annotations. Hard selection itself does not differentiate through indices, so the scorer has an explicit attention-distillation loss.

The proper 676 FIT / 144 DEV experiment selected joint epoch 4 and reached 115 raw, 126 numeric and 49/54 reviewed numeric on DEV, versus the paired full-token control 115/122/47. A prospective full 820 refit then used a fresh scorer for one epoch (26 updates) and joint training for four epochs (208 updates). Its terminal checkpoint was fixed before evaluating the separate 206 images. The former 144 DEV images are training data in this final refit.

| Price reader | Parameters | Raw exact /206 | Numeric /206 | AI-reviewed numeric /144 | RGB / encoded ms | GPU MiB |
|---|---:|---:|---:|---:|---:|---:|
| Custom token selection | 17.914M | 159 | 186 | 132 | 1.422 /1.695 | 708 |
| Recovered rank128, full2048 tokens | 17.864M | 159 | 182 | 130 | 1.622 /1.898 | 714 |
| Full WATERec R2048 reference | 26.121M | 153 | 180 | 132 | 1.684 /1.990 | 716 |
| Florence-2 reference | 231.414M | 148 | 176 | 133 | 5.359 /5.689 | 1178 |

The custom model preserves the recovered parent's raw accuracy and improves numeric results. It remains one AI-reviewed case below Florence and has not reached the approximately 1 ms target. These checkpoints have different training histories; the table is not a controlled parameter-count ablation. Numeric comparison retains the decimal value; raw exact also retains currency and punctuation. AI144 is an AI-assisted subset of206, not the internal DEV144 or certified human truth.

## TensorRT evidence and limits

Both encoder and decoder were rebuilt from the new weights. Batch1 RGB-to-text includes resize, normalization, packing, upload, selection, encoder, all cached greedy decoder steps and text decoding. Encoded-to-text additionally includes image decoding. Measurements use RTX5090, five warmups and three 206-image passes in fresh processes. All206 replay strings and all 1,236 timed strings match native output. Compilation, loading, detection and network are excluded.

The original strict 0.05 intermediate/logit tolerance **failed on seven prefixes across three FIT cases** and remains recorded as a failure. A separate output-fidelity qualification preserved all 12 FIT strings and 68 active-through-EOS tokens; all 68 visited prefixes satisfied a sufficient native argmax-margin bound. This supports the measured text-output parity, not numerical, calibration, routing or universal equivalence. No weights, precision policy or checkpoint were selected using 206 results.

GPU 708 MiB is sampled driver-accounted process usage including CUDA context, weights, scratch and activations. Current host RSS in the RGB timing process was 580.6 MiB with 206 images preloaded. The two engine files total 39,601,976 bytes; disk size is not runtime RAM. None of these establishes minimum RAM or latency on an edge device.

Scope is static batch 1, input budget 2048, selected memory 512 and contiguous valid-prefix grids 64–2048. Dynamic batching, arbitrary sparse padding and rotation correction are not qualified for this checkpoint. A separately trained orientation pipeline for the older recovered reader must not be assigned to this model.

## What did not preserve accuracy

Head-only token selection lost DEV accuracy before joint recovery. Learned 2×2 latent merging and late key/value mean pooling lost difficult fraction information. A no-fit local-attention-window probe also broke ordinary price sequences; it is not a qualified optimization. Profiling the qualified selector found about 72% of completed wall time in encoding, so decoder-only acceleration cannot remove most of its cost.

[Machine-readable aggregates](../results/compact_research.json) contain checkpoint and evidence hashes without private images, labels or weights. See [experiment history](experiments.md) for full-model resolution alternatives and [capabilities](capabilities.md) for scenario limitations.

An additional matched five-epoch training test moved selection from block 6 to block 4. Its selected internal DEV result was **111/144 raw, 123/144 numeric and 48/54 reviewed**, versus the freshly trained control **119/127/48** and prior source **115/126/49**. Both arms used identical proper-source initialization and 215 updates. It failed the prospective accuracy gate; it was not exported or tested on the final 206. Passing four initial training examples did not predict held-out accuracy.

## Same-weight runtime trials

Caching geometry uploads measured **1.397 ms RGB / 1.656 ms encoded**, with selection validation retained. An explicitly unchecked mode that also skips per-frame selection validation measured **1.332 / 1.620 ms**. Both retained 708 MiB process GPU usage, matched all 206 native strings and passed all combined 2,472 timed output checks. Weights, engines and precision were unchanged. These adapters are research code, not selectable notebook runtime options.

These are single experiments per mode; small differences do not establish statistical significance or a clock-controlled speedup. Current RGB host RSS was 581.5 MiB in both trials; the separate encoded processes were about 550 MiB. An identical inherited OS high-watermark field is retained in private evidence and is not presented as minimum RAM. The geometry-change FIT check includes a return to the original shape. Neither mode reaches 1 ms or adds edge, dynamic-batch or rotation qualification.

## Public workflow availability

[The dedicated workflow](compact_workflow.md) now provides `waterec-compact train/export/build/infer/evaluate/benchmark`, with typed checkpoint-family guards. It is multistage distillation, not the one-step Florence recipe, and supplied-teacher provenance remains the caller's responsibility. The charset is limited, automatic resume is absent, and the one-epoch defaults are software demonstrations. The current standalone runtime v3 preserves all 206 native strings and 1,236 timed strings for the exact restart2 checkpoint at 1.356083 ms RGB /1.620985 ms encoded and 708 MiB. Earlier private/cache rows above are historical snapshots, not replaced evidence or causal speedup controls. See [machine-readable workflow scope](../results/public_compact_workflow.json). Native budget flexibility is not arbitrary TensorRT support; tested profiles and selection precision bounds apply.
