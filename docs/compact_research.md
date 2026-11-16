# Compact OCR research update

This records trained experiments beyond the portable notebook options. The custom checkpoint and exporter are **not yet packaged as a supported model variant**. The standard WATERec R/RS workflows remain the reproducible public interface. No Nano/Small/Medium family with equivalent accuracy has been established.

## Trained image-only token selection

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
