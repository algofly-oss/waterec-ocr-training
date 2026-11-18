# OCR model comparison

Measured operating points, updated **17 November 2026**. These are separate trained models and runtimes, not a controlled architecture ranking. Parameters are learned parameters; GPU memory is the sampled process peak, not model size or minimum host RAM.

## Current corrected-reference development results

The newer price experiments below use **676 fitting images and 144 development images**, with labels audited under the **full amount plus visible currency** convention. These are development-selected results, not final-test accuracy. All 144 images remain in the denominator, including 45 with unresolved visual ambiguity. They must not be mixed with the historical 206-image results below. Input sizes are height × width; each task uses its own checkpoint.

| Model / task | Learned parameters | Input H×W | Literal exact | RGB → text mean, ms (B1) | Encoded bytes → text mean, ms (B1) | GPU, MiB |
|---|---:|---|---:|---:|---:|---:|
| WATERec R2048 / price | 26.121M | 128×256 on this cohort | 106/144 (73.61%) | 1.967 | 2.204 | 716 |
| WATERec R1024 / price — FP32 encoder, FP16 decoder | 26.121M | 88×180; 990 valid tokens | 98/144 (68.06%) | 2.859 | 3.067 | 668 |
| PP-OCRv6-derived medium / price — CTC-semantic retention ‡ | 15.580M | 128×320, stretch | 104/144 (72.22%) | 0.884 | 1.087 | 654 |
| PP-OCRv6-derived medium / price — global FP32 control | 15.580M | 128×320, stretch | 104/144 (72.22%) | 1.413 | 1.688 | 588 |
| Florence-2 / price — native Torch BF16 | 231.414M | 384×384 | 105/144 (72.92%) | 35.366 | 35.527 | 1768 |
| Compact WATERec / price † | 17.914M | 128×256; 2,048 → 512 selected tokens | 102/144 (70.83%) | 1.593 | 1.816 | 708 |
| Custom multiscale reader, photometric replay / price | 1.687M | Bound 128×256 | 101/144 (70.14%) | 1.545 | 1.737 | 552 |
| Custom multiscale reader, context distillation / price | 1.687M | Bound 128×256 | 98/144 (68.06%) | 1.539 | 1.768 | 552 |
| Custom multiscale reader, synthetic-layout stage / price | 1.687M | Bound 128×256 | 97/144 (67.36%) | 1.513 | 1.747 | 552 |
| CCT / price | 0.978M | 64×128, stretch | 66/144 (45.83%) | 0.746 | 0.937 | 542 |
| Custom multiscale reader / ALPR | 1.687M | Bound 128×256 | 929/1,167 (79.61%) | 1.755 | 2.001 | 552 |
| Custom matched control / ALPR — own FP32 TRT | 1.687M | Bound 128×256 | 933/1,167 (79.95%) | 1.833 | 2.063 | 552 |
| Clean CCT / ALPR — 38 classes | 0.978M | 64×128, stretch | 953/1,167 (81.66%) | 0.846 | 1.032 | 542 |
| WATERec RS512 / ALPR — FP32 control | 26.121M | 64×128; 512 patches | 1,075/1,167 (92.12%) | 2.201 | 2.418 | 652 |
| WATERec RS512 / ALPR — mixed candidate, strict retention failed § | 26.121M | 64×128; 512 patches | 1,075/1,167 (92.12%) | 1.265 | 1.442 | 686 |
| PP-OCRv6-derived medium / ALPR — earlier checkpoint, FP32 | 15.580M | 48×320, stretch | 983/1,167 (84.23%) | 1.224 | 1.390 | 576 |
| PP-OCRv6-derived medium / ALPR48 — new empirical point ¶ | 15.580M | 48×320, stretch | 1,001/1,167 (85.78%) | 0.760 | 0.939 | 650 |
| PP-OCRv6-derived medium / ALPR128 — new empirical point ¶ | 15.580M | 128×320, stretch | 1,027/1,167 (88.00%) | 0.964 | 1.094 | 654 |
| Clean Florence-2 / ALPR — native BF16 | 231.414M | 224×224 | 812/1,167 (69.58%) | 34.966 | 34.947 | 1696 |

Current measured timings use RTX 5090, batch 1, three passes over the full corresponding development cohort and eight warmup images, with separate processes for RGB and encoded input. RGB timing includes preprocessing, transfer, encoder, complete decoding and host text. Encoded timing additionally includes image-byte decoding. File I/O, initialization and warmup are excluded. GPU memory is the sampled process peak including runtime overhead; it is not a host-RAM requirement or weight-file size. These measurements do not establish edge-device performance or higher-batch throughput.

The custom reader uses TensorRT learned operations with CUDA greedy decoding and CUDA Graphs. All listed custom checkpoints reproduced all native strings, token sequences and end-of-text positions on their respective cohorts. The separate clean CCT ALPR engine reproduced all 1,167 native strings and slot-token sequences; it has 977,623 learned parameters and 38 classes, including the literal quote, rather than the public 37-class plate alphabet. The synthetic-stage custom reader’s price gain over the same model without that stage is **one image (97 versus 96)**; this is not established statistical superiority. WATERec's one-image lead over Florence likewise does not establish superiority.

† Compact WATERec is an **empirical output-retention result**, not a strict numerical-parity qualification. Its runtime retained 102 correct outputs with no previously correct outputs lost, but changed one already-incorrect string and failed the strict numerical tolerance. The qualification failure remains part of the result.

Florence's current row is **native PyTorch, not TensorRT**: FP32 stored parameters with **BF16 autocast computation**, beam 5 and a 32-new-token cap. Its full encoder runs for every timed image, and all 144 saved strings and token sequences were reproduced. Current-checkpoint TensorRT timing remains unmeasured for Florence; the historical optimized Florence timings below are not assigned to this row. Comparing native Torch with TensorRT does not isolate architecture speed. Training histories, selection budgets, precision and input policies also differ. The clean CCT ALPR run completed all 20 epochs and selected epoch19: 953/1,167 literal exact, terminal940/1,167. Its own runtime is measured above; no older plate or price timing is inherited.

A same-source R1024 WATE trial selected epoch4 at 98/144, eight fewer matches than R2048’s106/144. Its separately qualified FP32-encoder/FP16-decoder runtime costs2.859 ms RGB /3.067 ms encoded,668 MiB. The first FP16-encoder conversion lost one previously correct output (97/144) and remains a failed conversion. R1024 saves48 MiB but is slower and less accurate than R2048, so is not promoted. PP’s current104 checkpoint has two measured operating points above. ‡ The faster mixed engine preserves all144 literal strings and canonical CTC output-token sequences, but changes two internal blank/duplicate frame alignments. Its original strict numerical and frame gates remain failed; this is explicitly CTC-semantic retention, not numerical/frame equivalence. The global FP32 control passes its frozen numeric and frame/raw checks. Neither inherits historical timings. Training-process peaks are not inference memory. Two other bounded experiments did not improve the chosen operating point. CCT at 128×256 (1.150M parameters) selected **65/144 (45.14%)**, versus 66/144 at 64×128; microbatching and dropout RNG partition also differ, so this is not an isolated resolution ablation. Its 3,072 MiB measurement was **training memory**, not inference memory; no inference cost is assigned. WATERec's separately qualified B1/B2/B4/B8 batch experiment retained all 106 correct native outputs, but B8 took **417.14 images/s,2.397 ms/image,1,630 MiB** in the later CUDA Graph4 trial for RGB, slower per image than the optimized static-B1 row; it is not promoted. These results do not rule out other resolution or batching implementations.

The photometric-replay custom checkpoint scores **101/144 (70.14%)**, selected at epoch0: its preceding32,000-exposure stage already used676 real FIT images, and20 further adaptation epochs did not improve it (terminal91). Versus the matched original selected95 it gains14 and loses8 cases; versus ContextKD98 it gains11 and loses8. This is a single development-selected trial, not established superiority. Its own four-engine FP32 TensorRT/Graph4 B1 deployment retains all144 native strings/tokens/EOS and measures1.545 ms RGB /1.737 ms encoded,552 MiB. Similar timings to98 do not establish a speed win; no prior timing or batch qualification is inherited.

The current prototypes have **not** reached the requested accuracy/size/speed combination. The fixed-checkpoint evaluations below expose a substantial external-photo gap. Broader rotation/OOD and few-shot validation remain necessary before recommending a replacement.

The new matched-control933 checkpoint has its own FP32 TensorRT/Graph4 measurement:1.833 ms RGB /2.063 ms encoded,552 MiB, all1,167 native strings/tokens/EOS retained. It has no own heldout/batch result; none of the older929 checkpoint’s results is inherited. Clean Florence224 now has its own native BF16 measurement above, retaining812/1,167 and all native tokens; it is not TensorRT or the user-supplied strong checkpoint. Its encoded mean being0.019 ms lower is measurement variability, not an encoding advantage.

## Backend, throughput and host-memory companion

These are the same operating points above. **Sequential B1 images/s =1,000 / RGB mean milliseconds**: a descriptive completion rate from the recorded single-request runs, not a concurrent-service or batched-throughput claim. Current host RSS is the process after the RGB run, including libraries and the predecoded144- or1,167-image cache; it is **not minimum RAM**. NM is unmeasured. Mixed precision means FP16 is allowed where the builder chooses it; requested FP32 operations do not certify every internal accumulator.

| Operating point | Backend / precision | Sequential B1 images/s | Current host RSS, MiB | Timed cohort × passes |
|---|---|---:|---:|---|
| WATE R2048 / price | TensorRT mixed; FP16 allowed | 508.4 | 546.3 | 144 ×3 |
| WATE R1024 / price | TensorRT FP32 encoder / mixed decoder | 349.7 | 533.8 | 144 ×3 |
| PP104 mixed / price | TensorRT mixed; CTC-semantic scope | 1131.3 | 527.1 | 144 ×3 |
| PP104 FP32 / price | TensorRT FP32, TF32 off | 707.9 | 515.5 | 144 ×3 |
| Florence105 / price | Native Torch: FP32 weights, BF16 autocast | 28.3 | 2353.7 | 144 ×3 |
| Compact WATE102 / price | TensorRT mixed; empirical retention | 627.8 | 564.3 | 144 ×3 |
| Custom101 / price | TensorRT FP32 + CUDA greedy/Graph4 | 647.2 | 578.7 | 144 ×3 |
| Custom98 / price | TensorRT FP32 + CUDA greedy/Graph4 | 649.7 | 581.4 | 144 ×3 |
| Custom97 / price | TensorRT FP32 + CUDA greedy/Graph4 | 660.8 | 578.5 | 144 ×3 |
| CCT66 / price | TensorRT FP32, TF32 off | 1340.0 | 500.6 | 144 ×3 |
| Custom933 matched control / ALPR | TensorRT FP32 + CUDA greedy/Graph4 | 545.6 | 890.7 | 1167 ×3 |
| Custom929 / ALPR | TensorRT FP32 + CUDA greedy/Graph4 | 569.8 | 891.6 | 1167 ×3 |
| CCT953 / ALPR | TensorRT FP32, TF32 off | 1181.8 | 811.1 | 1167 ×3 |
| WATE1075 FP32 / ALPR | TensorRT FP32, centered LayerNorm | 454.4 | 844.6 | 1167 ×3 |
| WATE1075 mixed / ALPR | TensorRT mixed; LN FLOAT requests | 790.3 | 856.1 | 1167 ×3 |
| PP983 earlier / ALPR | TensorRT FP32, TF32 off | 816.9 | 822.8 | 1167 ×3 |
| PP1001 new48 / ALPR | TensorRT mixed; empirical DEV only | 1315.9 | 835.1 | 1167 ×3 |
| PP1027 new128 / ALPR | TensorRT mixed; empirical DEV only | 1036.9 | 840.3 | 1167 ×3 |
| Clean Florence812 / ALPR | Native Torch: FP32 weights, BF16 autocast | 28.6 | 2503.4 | 1167 ×3 |

Price/native-Torch versus TensorRT and differing training/input policies are not isolated architecture comparisons. Strict-failure and empirical-retention caveats above remain part of each row.

## Native quarter-turn reading: current price DEV144

These are **native pipelines on the same current 144 development images**, not TensorRT rotation results, ALPR results or untouched final tests. Images were turned clockwise (CW) before each model’s ordinary preprocessing. No orientation corrector, angle oracle, extra crop or GT-conditioned output choice was used. All six upright runs reproduced their own 144 saved strings and token sequences before rotation scoring. All 144 images, including 45 unresolved labels, remain. Four views of these images are not 576 independent photos.

| Native checkpoint | Upright exact /144 | CW90° exact /144 | CW180° exact /144 | CW270° exact /144 |
|---|---:|---:|---:|---:|
| WATERec R2048 | 106 | 26 | 22 | 66 |
| Compact WATERec | 102 | 20 | 16 | 29 |
| Custom reader, measured 97 checkpoint | 97 | 1 | 4 | 1 |
| Custom reader, native 98 candidate | 98 | 0 | 3 | 2 |
| CCT64 price research | 66 | 0 | 0 | 0 |
| Florence-2 current price | 105 | 45 | 4 | 49 |

**Preprocessing is a confound.** WATE’s 128×256 upright input becomes 256×128 after a quarter-turn, preserving area. Both custom readers use a 128×256 landscape bound: quarter-turns become 128×64, one-quarter of upright pixel area. CCT stretches every input to 64×128; Florence uses 384×384. All 180° cases retain upright geometry. These are whole-pipeline results, not an isolated encoder-invariance or architecture-causality test. The current price Florence checkpoint is not the user’s original ALPR Florence model. Strong intrinsic rotation robustness, especially for the tiny price models, remains unmet.

The 1.687M context-distilled checkpoint scores **98/144 (68.06%)**, versus97/144 for the prior synthetic-stage checkpoint: **six exact gains and five losses** in one development-selected trial, not established superiority. Its own upright TensorRT engines now preserve all144 native strings/tokens/EOS and measure1.539 ms RGB /1.768 ms encoded,552 MiB. The97 checkpoint remains a distinct measured point at1.513 /1.747 ms. No speed improvement over97 is claimed. The rotation table above is still native-only, not TensorRT rotation qualification.

An **optional learned orientation pipeline** (1,962,662 learned parameters, including a275,524-parameter head) gives **97/97/97/98 of144** atCW0/90/180/270, compared with the default98/0/3/2. It loses one previously correct upright output. Its own TensorRT pipeline reproduces the576 native views; all-angle RGB mean is **1.961 ms,560 MiB**, with upright mean1.920 ms. Encoded lossless-PNG mean2.734 ms is not a matched JPEG comparison. The head was selected on these same144 DEV images, not a blind rotation test; the default upright workflow is unchanged.

Variable-shape support is not scale-invariant accuracy: the same98 checkpoint gives **98,88,89,48,9 exact matches of144** at bounds128×256,96×192,160×320,192×384,256×512 respectively. These inference-only trials do not rule out higher-resolution training; no speed claim or default change follows.

A separate20-epoch matched training pair starting from the original98 checkpoint also did not favor160×320: the128 control selected98/144 at epoch0 (terminal95), while160 selected95/144 at epoch16 (terminal93), with13 gains and16 losses versus128. It improves over that frozen checkpoint’s160-bound89/144, but does not beat the128 control; it is not promoted.

The actual101 checkpoint also has a separately qualified **fixed128×256 batch1/2/4/8** TensorRT path, retaining all144 native strings/tokens/EOS at every capacity. At batch8, RGB is **8.373 ms per batch,1.047 ms per image amortized,955.5 images/s,634 MiB**; encoded is10.028 ms per batch. This is throughput, not a1 ms single-request latency. Keep the separate static-B1 **1.545 ms,552 MiB** point for single requests; the batch path atB1 is slower/larger (1.877 ms,618 MiB). There is no mixed-resolution or OOD qualification.

PP104 also has a separately audited **dynamic batch 1/2/4/8**, fixed 128×320 path. It retains 104/144 literal outputs at every capacity, with batch-8 RGB **4.689 ms/batch, 0.586 ms/image amortized, 1,706 images/s, 700 MiB**. This is throughput, not 0.586 ms single-request latency. Qualification is CTC-semantic only: all capacities fail the original strict logit gate, and three DEV frame sequences change while literal strings stay identical. Current host RSS was not captured; initialization high-water memory is not minimum RAM. No heldout batch or rotation qualification is inherited.

## Few-shot development findings

Measured few-shot development curves for the synthetic-stage reader are **77/144 (53.47%) with50 real FIT images,83/144 (57.64%) with100, and87/144 (60.42%) with250**, versus97/144 (67.36%) for its676-image control. These nested single-seed runs each used13,520 task exposures and144 DEV labels to select among21 checkpoints. They also inherit public pretraining,384,000 natural-image exposures from20,000 unique crops, and32,000 synthetic exposures. They are not cold50-label adaptation, do not establish a simple user workflow, and have no own deployment timings.

¶ These **new PP ALPR weights** have their own B1 measurements on all1,167 DEV images, three passes/eight warmups. They are **empirical DEV accuracy/cost points, not native-output or numerical equivalence**: both retain their native exact counts, but48 changes two strings (one correctness gain and one loss), and128 changes two already-incorrect strings (no correctness gain/loss). Original strict native-output and logit gates remain failed. Separate full-FP32 controls retain every native DEV string/frame:48 costs1.203 ms RGB /1.451 ms encoded,576 MiB;128 costs1.596 /1.738 ms,588 MiB. Those are separate operating points, not the faster mixed timings. **Their own heldout results are now reported below, separately for native and empirical mixed TensorRT**; the older983-checkpoint results must not be assigned to them. The48 run is a declared non-bitexact continuation after interruption, with discarded work recorded;128 is a fresh five-epoch arm. This is not proof of a pure resolution effect.

## Current fixed-checkpoint heldout results

Weights and input/decoding policies were fixed before these runs. **No model, precision, reference or policy was changed using the results.** These cohorts have historical project exposure: exact current FIT/DEV image-disjointness is verified, but upstream pretraining and near-duplicate overlap are unknown. They are not newly blind tests. Only aggregate results are used here; no example-driven tuning follows.

### ALPR: own current checkpoints

| Fixed checkpoint | DEV exact /1,167 | Heldout exact /2,693 | Reviewed exact /984 | Backend retention |
|---|---:|---:|---:|---|
| WATERec RS512 | 1,075 | 2,476 (91.94%) | 855 (86.89%) | All native/TRT strings and tokens identical |
| PP-OCRv6-derived medium | 983 | 2,321 (86.19%) | 606 (61.59%) | All native/TRT strings and CTC frames identical |
| CCT, clean 38-class checkpoint | 953 | 2,226 (82.66%) | 628 (63.82%) | All native/TRT strings and slot tokens identical |
| Custom multiscale reader | 929 | 2,180 (80.95%) | 438 (44.51%) | All native/TRT strings, tokens and EOS identical |
| New matched custom control | 933 | Unmeasured | Unmeasured | Own DEV retention only; no inherited929 results |
| Clean Florence-2, 224-square decoder-only recipe | 812 | 2,002 (74.34%) | 797 (81.00%) | Native BF16 only; current TRT unmeasured |
| New PP ALPR48 checkpoint | 1,001 | Native2,341; mixed TRT2,339 | Native624; mixed TRT626 | Empirical; strict native-output identity failed |
| New PP ALPR128 checkpoint | 1,027 | Native2,367; mixed TRT2,368 | Native533; mixed TRT534 | Empirical; strict native-output identity failed |

For the new PP48 checkpoint, native→mixed TensorRT changes exact correctness by **one gain/three losses on2,693** and **three gains/one loss on984**;9 and11 raw strings change respectively. PP128 has **one gain/no loss on each cohort**, with6 and12 raw strings changing. These are fixed empirical operating points, not prediction-equivalent conversions. The separate strict-FP32 TensorRT controls were qualified on DEV; these new heldout TRT counts belong to the mixed engines, not those controls. No weights, thresholds, precision policy or references were changed using heldout outputs.

**PP128 improves DEV and the2,693 cohort over PP48, but worsens reviewed984**: native533 versus624 and mixed534 versus626. This substantial tradeoff prevents a blanket recommendation. The48 numerical-continuation lineage, different training trajectories, historical cohort exposure and reviewed-label uncertainty remain relevant; this is not an isolated causal resolution experiment or final-guided checkpoint selection. Runtime costs above remain separately measured on DEV1167, not these heldout operators.

These are the checkpoints in the current DEV table, **not the earlier models in the historical table below**. WATE's ancestry includes 2,048 synthetic fitting inputs and historical selection; PP membership is reconstructed from source/summary evidence rather than a contemporaneous per-update journal. The clean Florence run used generic initialization and decoder-only adaptation; it is not the user's strong supplied Florence checkpoint or a family maximum. The reviewed984 labels are assistant-vision references, not human certification. Costs above come from separate full-DEV measurements, not timed heldout inference.

§ The **same WATE checkpoint** has a separate mixed-precision candidate with243 LayerNorm operations requested in FP32, measured at **1.265 ms RGB /1.442 ms encoded,686 MiB** on DEV1167; all1075 correct native DEV outputs are retained. On2693 it gives **2477 exact (91.98%) versus2476 native**, with one accuracy gain and no losses, but only2692/2693 raw/token sequences are identical. Reviewed984 remains855/984 with all outputs retained. **The original strict output-retention gate fails and stays failed.** This records a faster measured accuracy-nonregression operating point, not identical predictions, certified internal accumulation, universal equivalence, automatic adoption or a final-guided policy change. The full-FP32 control remains distinct at2.201 ms/652 MiB.

### Price: corrected206 and external photos

| Fixed price checkpoint / backend | Literal exact /206 | Confirmed exact /6 scorable external photos | Possible exact across all9 external photos |
|---|---:|---:|---:|
| WATE106 native / TRT | 163 (79.13%) | 1 | 1–4 of9 |
| PP104 native / mixed TRT / FP32 TRT | 163 (79.13%) | 0 | 0–3 of9 |
| Custom101 native / TRT | 152 (73.79%) | 1 | 1–4 of9 |
| CCT66 native / TRT | 91 (44.17%) | 0 | 0–3 of9 |
| Florence105 native BF16 | 159 (77.18%) | 0 | 0–3 of9 |

The **full206 result remains primary**. A supplemental split uses the **pre-existing**, prediction-independent AI label-review mask:138 marked resolved under the review policy and68 provisional retained. Applicable native/TRT streams have the same stratum counts:

| Fixed price checkpoint | Full206 exact (primary) | AI-review resolved exact /138 | Provisional retained exact /68 |
|---|---:|---:|---:|
| WATE106 | 163 | 125 | 38 |
| PP104 | 163 | 123 | 40 |
| Custom101 | 152 | 116 | 36 |
| CCT66 | 91 | 79 | 12 |
| Florence105 | 159 | 119 | 40 |

Each stratum total and correct count sums to the unchanged primary result. **AI-review resolved is not independently human-verified ground truth**; this descriptive split does not justify excluding provisional references or selecting models using final results.

All206 references remain, including provisional ambiguous labels. **All9 external images remain in the cohort, but3 have no resolvable full-string reference.** They are null/unscorable, not automatic errors; the six-reference result is secondary and the bounds retain the full9-image uncertainty. References are AI visually audited, not human-certified. This small, difficult photo cohort does not establish broad OOD performance.

WATE, custom, CCT and PP's FP32 control reproduce every native raw output on both cohorts. Mixed PP preserves205/206 price strings: one already-incorrect output changes, leaving163 correct unchanged;202/206 internal CTC frame sequences match. On external9 its raw outputs all match, with8/9 frame sequences equal. Equal exact counts do not imply universal backend equivalence; previous strict mixed-engine qualification failures remain recorded. Current timings remain the separate DEV144 measurements, not a final-cohort benchmark. No final result is used to choose a new checkpoint or backend.

## Completed matched pretraining study and generic few-shot limits

A matched1.687M custom-reader study increased natural engineering REPORT exact from979/2,000 to1,000/2,000 with a CTC pretraining objective, but **did not improve either downstream task**. This REPORT cohort was used for engineering, not an untouched final test.

| Matched arm | Natural REPORT exact /2,000 | ALPR selected /1,167 (terminal) | Price selected /144 (terminal) |
|---|---:|---:|---:|
| Control | 979 | 933, epoch17 (917) | 98, epoch15 (96) |
| CTC objective | 1,000 | 927, epoch19 (899) | 92, epoch7 (88) |

Both arms used the same actual training inputs/order/pixels/seeds:9823 ALPR FIT images,6140 updates/196460 exposures; price used676 FIT images with32,000 stage exposures plus13,520 clean-adaptation exposures. Selected CTC versus control gains/losses were57/63 on ALPR and8/14 on price. This is a preserved negative result from one pair, not a universal architecture conclusion. Only control933 has its own measured runtime above; the other new weights have unmeasured runtime/heldout performance. Do not transfer costs from historical929 or other98-point checkpoints.

The completed official-generic PP few-shot study is **nonmonotonic**, with no104-task warm start. Results include the negative100-image run rather than omitting it:

| Real FIT labels | PP15.580M selected /144 (epoch; terminal) | Custom1.687M selected /144 (epoch; terminal) |
|---|---:|---:|
| 50 | 66 (epoch8;64) | 77 (epoch18;71) |
| 100 | 11 (epoch0;0) | 83 (epoch14;76) |
| 250 | 90 (epoch9;83) | 87 (epoch14;79) |

Every cell used860 updates/13,520 exposures and144 additional DEV annotations for checkpoint selection, not N total annotations. The nested FIT order and DEV references match, but pretraining, parameters, objective, input policy and compute differ: PP starts from official generic weights; custom also inherits384k local natural and32k synthetic exposures. This is not a controlled architecture or equal-prior comparison. PP100 selected the unupdated initializer; updated epochs peaked at1/144 (epochs4 and7); the terminal scored0/144. It was not persistently blank, and audits found no concrete saved-data/recipe/codec bug. Overfitting or optimization sensitivity is a hypothesis, not an established cause. These few-shot weights have no own runtime/heldout measurements. General-purpose few-shot reliability and the approximately1M/WATE-like goal remain unmet.

## Historical license-plate measurements

Exact means the complete output string matches the reference. Input dimensions are height × width; WATERec uses aspect-preserving 4×4 patches.

| Model | Parameters | Input | Validation exact /2,693 | Reviewed unseen exact /984 | RGB → text mean, ms | GPU, MiB |
|---|---:|---|---:|---:|---:|---:|
| WATERec RS | 26.121M | ≤512 patches | 2,475 (91.90%) | 880 (89.43%) | 1.256 | 694 |
| WATERec speculative¹ | 27.507M | ≤512 patches | 2,476 (91.94%) | 879 (89.33%) | 1.026 | 698 |
| WATERec RS rank128 (research) | 17.863M | ≤512 patches | 2,468 (91.65%) | 849 (86.28%) | 1.325 | 694 |
| Florence-2 optimized | 231.414M | 224×224 | 2,381 (88.41%) | 857 (87.09%) | 5.066 | 1130 |
| PP-OCRv6-derived medium | 15.580M | 48×320 | 2,310 (85.78%) | 571 (58.03%) | 1.157 | 654 |
| PARSeq Fast Tiny² | 6.049M | 48×192 | 2,377 (88.27%) | 715 (72.66%) | 0.686 | 682 |
| CCT-S³ | 0.978M | 64×128 | 2,315 (85.96%) | 644 (65.45%) | 0.447 | 742 |
| WATERec RS FP8 MLP (opt-in)⁴ | 26.121M | ≤512 patches | 2,472 (91.79%) | 880 (89.43%) | 1.323 | 584 |

The rank128 ALPR row uses **literal exact: 2,468/2,693**. Older transformed-label summaries are not used as literal scores.

1. Speculative decoding includes its draft parameters. These scores belong to that engine; it is **not lossless** against ordinary FP16 decoding.
2. Fast Tiny’s fresh literal replay verifies **2,377/2,693 and 715/984**. Its historical validation aggregate was one match higher. The historical timing cohort remains 256 crops; this replay is an accuracy check, not a new latency measurement.
3. CCT-S’s fresh literal replay verifies **2,315/2,693 and 644/984** using the current engine and restored original image bytes. Its B1 timing retains the original **max-batch-32 allocation and output-copy policy**; the price row uses a separately measured static-B1 price model.
PP’s fresh literal replay verifies **2,310/2,693 and 571/984**. The raw CTC strings were saved before scoring; runtime trimming changed no outputs on these cohorts. See the [fresh replay receipts](../results/literal_alpr_replay_2026_11_17.json).

4. FP8 MLP is hardware-specific and opt-in. It loses three validation matches against ordinary FP16; reduced memory does not establish better accuracy or speed.

## Historical price-tag measurements

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

## Recent negative optimization studies

These are development/engineering diagnostics, not replacement operating points. The primary accuracy/cost tables above remain unchanged.

- **PP natural replay:** matched ALPR exposure gave control 999/1167 versus replay 984/1167 (18 gains, 33 losses), while natural engineering REPORT improved 32/2000 to 106/2000 (75 gains, 1 loss). Replay added 6,140 natural exposures to 49,115 ALPR exposures per arm, so compute was not matched. Task-preservation failed; no promotion or new runtime/heldout result.
- **Florence current 105 precision controls:** the FP16 candidate gave 103/144 (0 gains, 2 losses; 141 raw/token/EOS outputs retained), BF16-encoder-flagged variant 104/144 (0/1; 143 retained), FP32 shared-head variant 105/144 (1/1; 142 retained), and global FP32 encoder 103/144 (0/2;141 retained). All 144 prepared pixel inputs matched. Each retained 11/12 FIT outputs including all 8 real FIT images, but failed the unchanged numerical gate (0/372 finite comparisons passed). Equal aggregate accuracy in the shared-head variant did not satisfy the predeclared no-loss rule. The BF16 flag did not mean uniform BF16: FP16 depthwise operations remained.
- **Native precision control:** disabling native BF16 autocast alone gave 103/144 versus 105, with 0 gains/2 losses. All 144 raw/token/EOS outputs matched the global-FP32-encoder TRT diagnostic. This is scoped evidence, not a rebased reference or universal conversion-equivalence claim.
- **Hybrid diagnostic:** native BF16 encoder plus existing TensorRT decoder, using a CPU bridge, failed the FIT retention gate (11/12 retained). Completed DEV retains all 144 raw/token/EOS outputs at 105/144, with 0 gains/0 losses; 144 saved features were checked finite and bound to matching pixels. This does not erase the FIT failure or qualify deployment timing. This hybrid is not end-to-end TensorRT.

No Florence diagnostic above qualified latency or lifecycle measurements; build/diagnostic memory peaks are not substituted for inference requirements. Existing native 105 latency belongs only to its separately measured BF16 pipeline. A definitive cause of the conversion differences was not established. Original failures and exact source/checkpoint/audit hashes remain in the research evidence index; no example-driven tuning or reference changes are implied.

## Remaining research target

**Completed negative evidence:** task and bounded natural PLM/control studies did not establish accuracy-preserving parallel decoding. The completed natural-CTC matched study is negative, as recorded below. General-purpose few-shot reliability and a simple fresh user workflow remain unproven. None of the reported operating points establishes the requested approximately1M-parameter, WATE-like accuracy/speed goal.

The primary goal is **compact, image-only OCR with approximately 1M learned parameters and accuracy close to WATERec on both ALPR and difficult price tags**, prioritizing accuracy and speed over memory alone. The 1M size is an ideal, not a hard cutoff; modestly larger models remain candidates when measured accuracy and speed justify them. Separate ALPR and price-tag checkpoints are allowed; each accepts an image without additional user annotations. **That result has not been achieved here.** CCT-S supplies the approximately 1M size/speed reference, but its measured accuracy gap remains.

Easy few-shot fine-tuning remains an **unmet usability and accuracy goal**. The price curves above are initial task-informed development measurements, not evidence of WATE-like accuracy or a cold few-shot workflow; ALPR curves remain unmeasured. Report one-time pretraining separately from user adaptation. The intended interface is a single command with images and raw expected text, with literal-string exact accuracy used for selection and reporting. Neither this usability goal nor few-shot accuracy is established by the current 17.9M multistage workflow.

Training must require only **an image and its raw expected output text**: no bounding boxes, character positions, task prompts or additional user annotations. Task-specific checkpoints are permitted; a single checkpoint that handles both tasks remains an additional research option. Learned internal attention and masks are model computations, not supplied annotations. The larger compact research models and the fictional notebook examples are not evidence that the 1M target has been met.

See [capabilities](capabilities.md), [experimental compact workflow](compact_workflow.md), and [compact research history](compact_research.md) for scenario and deployment limits.
