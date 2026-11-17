# OCR model comparison

Measured operating points, updated **17 November 2026**. These are separate trained models and runtimes, not a controlled architecture ranking. Parameters are learned parameters; GPU memory is the sampled process peak, not model size or minimum host RAM.

## License plates

Exact means the complete output string matches the reference. Input dimensions are height × width; WATERec uses aspect-preserving 4×4 patches.

| Model | Parameters | Input | Validation exact /2,693 | Reviewed unseen exact /984 | RGB → text mean, ms | GPU, MiB |
|---|---:|---|---:|---:|---:|---:|
| WATERec RS | 26.121M | ≤512 patches | 2,475 (91.90%) | 880 (89.43%) | 1.256 | 694 |
| WATERec speculative¹ | 27.507M | ≤512 patches | 2,476 (91.94%) | 879 (89.33%) | 1.026 | 698 |
| WATERec RS rank128 (research) | 17.863M | ≤512 patches | 2,468 (91.65%) | 849 (86.28%) | 1.325 | 694 |
| Florence-2 optimized | 231.414M | 224×224 | 2,381 (88.41%) | 857 (87.09%) | 5.066 | 1130 |
| PP-OCRv6-derived medium | 15.580M | 48×320 | Not verified | Not verified | 1.157 | 654 |
| PARSeq Fast Tiny² | 6.049M | 48×192 | Not verified | Not verified | 0.686 | 682 |
| CCT-S³ | 0.978M | 64×128 | Not verified | Not verified | 0.447 | 742 |
| WATERec RS FP8 MLP (opt-in)⁴ | 26.121M | ≤512 patches | 2,472 (91.79%) | 880 (89.43%) | 1.323 | 584 |

The rank128 ALPR row uses **raw exact: 2,468/2,693**, not its separately reported canonicalized count of **2,469/2,693**. Canonicalization can change a match; the table does not mix those metrics.

1. Speculative decoding includes its draft parameters. These scores belong to that engine; it is **not lossless** against ordinary FP16 decoding.
2. Fast Tiny’s 2,378/2,693 and 715/984 are historical nonliteral/normalized aggregates, not verified literal counts. The currently preserved executable/reference replay verifies **203/256** only; the full-cohort figures are not current replay qualification.
3. CCT-S’s historical 2,315/2,693 has not been certified as literal; the original normalization implementation is unavailable for a complete identity proof. The raw files behind historical 644/984 have been deleted. Neither count is presented as literal accuracy. Its B1 timing retains the original **max-batch-32 allocation and output-copy policy**; the price row uses a separately measured static-B1 price model.
PP’s historical 2,310/2,693 and 571/984 were normalized before their predictions were saved, so the original literal counts cannot be reconstructed from those files. A new literal replay is needed. **Not verified** means no certified literal count, not zero accuracy.

4. FP8 MLP is hardware-specific and opt-in. It loses three validation matches against ordinary FP16; reduced memory does not establish better accuracy or speed.

## Price tags

**Only literal-string exact accuracy is used here.** Every character, currency symbol, decimal separator, space and its position in the sequence must match the expected text. A numerically equivalent amount with different text is a failure.

| Model | Parameters | Input | Literal exact /206 | RGB → text mean, ms | GPU, MiB |
|---|---:|---|---:|---:|---:|
| WATERec R, 2048 | 26.121M | 128×256; 2,048 patches | 153 (74.27%) | 1.684 | 716 |
| WATERec R, 1024 | 26.121M | 88×180; 990 valid patches | 154 (74.76%) | 1.090 | 700 |
| WATERec R, 512 | 26.121M | 64×128; 512 patches | 149 (72.33%) | 0.822 | 688 |
| Compact WATERec restart2, port B1⁵ | 17.914M | 2,048 → 512 selected patches | 161 (78.16%) | 1.390 | 708 |
| WATERec R rank128 (research) | 17.864M | 2,048 patches | 159 (77.18%) | 1.622 | 714 |
| Florence-2 optimized | 231.414M | 384×384 | 148 (71.84%) | 5.359 | 1178 |
| PP-OCRv6-derived medium⁶ | 15.580M | 128×320 | 161 (78.16%) | 1.183 | 654 |
| PARSeq Fast Tiny | 6.049M | 48×192 | 112 (54.37%) | 0.889 | 688 |
| CCT-S price model | 0.978M | 64×128 | 100 (48.54%) | 0.578 | 540 |

5. Compact restart2 has **17,913,953** parameters. This table uses the separately qualified **ported static-B1** runtime: **1.389778 ms /708 MiB**. The earlier default runtime’s **1.356083 ms /708 MiB** is a different run/configuration, not an interchangeable timing. The same 206 strings were qualified, but native/TensorRT identity is not a universal guarantee on arbitrary inputs or batch capacities.
6. PP price uses the newer capacity-1 allocation experiment. Its older **1.124618 ms /670 MiB** measurement belongs to a separate historical harness; do not combine the old latency with the new memory.

## How to read the comparison

- **Hardware/runtime:** RTX 5090, TensorRT, batch 1. Florence’s **5.066 ms ALPR /5.359 ms price** are actual TensorRT vision/shared-decoder measurements with beam 5, parallel beam ranking and exact GPU preprocessing—not native PyTorch timings. WATERec uses cached autoregressive decoding; PP uses CTC; CCT-S and Fast Tiny have their own output policies.
- **Timing boundary:** predecoded RGB → preprocessing/transfer → inference/decoding → completed host text. Model loading, engine building, file/JPEG decoding, detection and network transfer are excluded. File/encoded-input latency is higher and reported separately in the [reference measurements](../results/reference_metrics.json).
- **Different experiments:** training pools, adaptation histories, image budgets and timing cohorts differ. Plate timings generally use 256 crops, while price timings use 206; warmup/repeat policies and historical measurements are not all identical. These are measured operating points, not causal speedup or statistical significance claims. The full WATERec R2048 price reference used 676 fitting images; Florence’s final price reference used 820.
- **Qualification scope:** saved native/TensorRT parity is tied to each tested checkpoint, runtime and cohort. It does not establish universal equality, rotation invariance, arbitrary-script coverage, or equal out-of-distribution capability. The 206 price set was already observed during the broader project; it is not a blind discovery test. Historical artifacts retain their original metrics; they are not rewritten here. Current selection and reporting use literal-string exact accuracy only.

## Remaining research target

The primary goal is **one joint, image-only checkpoint with approximately 1M learned parameters and accuracy close to WATERec on both ALPR and difficult price tags**, prioritizing accuracy and speed over memory alone. The 1M size is an ideal, not a hard cutoff; modestly larger models remain candidates when measured accuracy and speed justify them. Inference must not require a task selector or task prompt. **That result has not been achieved here.** CCT-S supplies the approximately 1M size/speed reference, but its measured accuracy gap remains.

Easy few-shot fine-tuning is also an **unmet goal**: measure fixed **50/100/250-example curves per task**, reporting one-time pretraining separately from the data and compute a user needs for adaptation. The intended interface is a single command with images and raw expected text, with literal-string exact accuracy used for selection and reporting. Neither this usability goal nor few-shot accuracy is established by the current 17.9M multistage workflow.

Training must require only **an image and its raw expected output text**: no bounding boxes, character positions, task prompts or additional user annotations. Task-specific models are optional ablations, not a substitute for the primary joint-checkpoint goal. Learned internal attention and masks are model computations, not supplied annotations. The larger compact research models and the fictional notebook examples are not evidence that the 1M target has been met.

See [capabilities](capabilities.md), [experimental compact workflow](compact_workflow.md), and [compact research history](compact_research.md) for scenario and deployment limits.
