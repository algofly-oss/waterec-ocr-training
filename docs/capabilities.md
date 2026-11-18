# WATERec capabilities and reference comparisons

This repository provides WATE training and deployment workflows. The table separates current task results from optional pipeline experiments and qualitative observations.

## Scenario evidence

“Not measured” means this reference collection does not establish the capability. It is not a zero score. Architecture support, qualitative experience and measured accuracy are different kinds of evidence.

| Scenario | CCT-S plate model | WATERec references | Florence reference |
|---|---|---|---|
| Upright plate crops | Historical literal replay:2,315/2,693 validation and644/984 separate unseen plates. The timing cohort is a different256-image sample; this does not qualify clean new training. | ALPR: 2,475/2,693 exact; separate reviewed unseen cohort 880/984; extracted fields 332/443. These denominators must not be combined. | Historical optimized Florence:2,381/2,693 validation and857/984 separate unseen plates; different training histories. |
| Two-line plates | Whole crop is resized to 64×128; no separately qualified two-line subset. | Native 2D patches retain spatial arrangement, but no isolated two-line accuracy is reported here. | Layout capability should be evaluated on the actual two-line cohort; no figure supplied here. |
| Native 90°/180°/270° reading | Not measured. Fixed input shape does not provide rotation invariance. | An older native FP32 checkpoint scored 247/788, 159/788, 251/788 on clockwise rotations, without correction. This is a separate diagnostic, not the current final-engine result. | No matched current rotation qualification claimed. |
| Rotation with correction | No qualified correction stage in this repository. | Optional experimental orientation pipeline: 706/788, 706/788, 707/788 at clockwise 90°/180°/270°. Upright reviewed cohort: 879/984 versus 880/984 without correction. These are pipeline results, not intrinsic reader invariance. | No correction result supplied here. |
| Unseen countries and fonts | Qualitative out-of-distribution weaknesses have been observed; no country-stratified score is supplied. | The reviewed unseen plate result is evidence for that cohort, not every country or script. | No universal country coverage claim. |
| Price: tiny raised fractions, currency, unequal glyph sizes | This 37-class plate model is not the price model. Its alphabet cannot represent decimal punctuation or spaces. | Historical R2048 price:153/206 literal exact. Detail preservation still matters; not a guarantee for every raised fraction. | Same historical price references:148/206 literal exact. This does not establish universal equivalence. |
| Extra text, clutter, few-shot adaptation | No demonstrated few-shot selective-reading capability. Training on plate crops does not establish price-field selection. | Current full-data scores are not 50-shot results. Selection among distractors remains task/data dependent. | User-observed useful adaptation from roughly 50 examples with clutter is qualitative evidence here, not a matched 90% benchmark or a guarantee. |
| Arbitrary source resolution | Images are accepted and resized to fixed 64×128. This is not preservation of arbitrary-resolution detail. | Aspect-preserving patch-4 preprocessing uses a configurable token budget. Actual dimensions/token counts depend on the image; changing budget requires fresh accuracy and resource checks. | This comparison does not qualify arbitrary-resolution export behavior. |
| Batch throughput and memory | Historical B1: 0.4474 ms, 2,235 images/s; B32: 2.5137 ms/batch, 12,730 images/s; both sampled 742 MiB. | Qualified static B1: plate 1.2558 ms / 694 MiB; price R2048 1.6838 ms / 716 MiB. Separate dynamic-batch implementations must be measured independently. | No directly comparable timing supplied in this table. |

All listed timing uses predecoded RGB, including preprocessing, transfer, inference and text decoding, with the relevant engine/runtime. CCT timing used 256 reference crops, five warmups and three passes, a batch-32 engine profile and batch-32 allocations. WATE plate and price timings have different cohorts and profiles. These are not architecture-only or hardware-independent comparisons. See the repository's reference metadata for evidence hashes and exact scope. Dummy-data runs establish wiring only.

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

The 1.687M context-distilled checkpoint has98/144 literal exact, six gains and five losses versus97/144 in one development-selected trial. Its own upright TensorRT deployment is now qualified at1.539 ms RGB /1.768 ms encoded and552 MiB; the97 checkpoint’s1.513 /1.747 ms remains a separate measurement. This one-case gain does not establish superiority or a speed improvement. The rotation table is native-only. Current clean CCT ALPR is a separate38-class checkpoint with953/1,167 and own0.846 /1.032 ms,542 MiB; it is not a price-rotation result. Current WATE R1024 retains98/144 with its qualified FP32 encoder / FP16 decoder at2.859 ms RGB /3.067 ms encoded,668 MiB; R2048 is more accurate and faster.

## Choosing an experiment

For familiar, upright, tightly cropped alphanumeric plates with strict throughput constraints, CCT-S is a useful compact baseline. Its 15 parallel character slots avoid an autoregressive decode loop. Validate unfamiliar layouts and countries rather than assuming the speed result implies robustness.

For mixed glyph scales, currency and richer spatial layouts, the retained 2D WATE encoder plus autoregressive reader is the stronger measured price reference in this collection. Higher token budgets can retain smaller details but increase computation. PyTorch versus TensorRT is an execution choice, not the reason one model handles spatial layout better; exported precision, preprocessing and decoding must remain qualified.

Florence is a useful generative reference when selective reading and few-example adaptation matter. The user observation about clutter motivates testing, but does not substitute for a matched held-out evaluation. A larger generative model is not automatically the fastest deployment choice.

## Limits and model identities

- WATE R and RS are the same approximately 26.12M-parameter architecture with different pretrained initialization, not different size tiers. The price SPACE extension changes the classifier mapping and adds 768 parameters.
- FastTiny/PARSeq is a separate model family, not a smaller WATE variant. Unqualified custom 5.5M experiments are not public supported variants.
- The WATE export described here uses an ASCII character set and at most 25 decode steps including EOS. The default safe text length is 24 characters plus EOS; configuration accepting length 25 does not guarantee a terminating EOS within 25 steps. This is not universal Unicode OCR.
- CCT-S uses `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_`, with `_` as padding, and 15 slots. The generic upstream checkpoint's 10-slot/region head differs from the local 15-slot plate model.
- Orientation correction is optional experimental plate-pipeline evidence. No price-orientation qualification or automatic best-angle oracle is implied.
- Train/validation selection must precede final testing. Separate ALPR and price weights are allowed; passing one task does not qualify the other.


An **optional learned orientation pipeline** (1,962,662 learned parameters, including a275,524-parameter head) gives **97/97/97/98 of144** atCW0/90/180/270, compared with the default98/0/3/2. It loses one previously correct upright output. Its own TensorRT pipeline reproduces the576 native views; all-angle RGB mean is **1.961 ms,560 MiB**, with upright mean1.920 ms. Encoded lossless-PNG mean2.734 ms is not a matched JPEG comparison. The head was selected on these same144 DEV images, not a blind rotation test; the default upright workflow is unchanged.

Variable-shape support is not scale-invariant accuracy: the same98 checkpoint gives **98,88,89,48,9 exact matches of144** at bounds128×256,96×192,160×320,192×384,256×512 respectively. These inference-only trials do not rule out higher-resolution training; no speed claim or default change follows.

Measured few-shot development curves for the synthetic-stage reader are **77/144 (53.47%) with50 real FIT images,83/144 (57.64%) with100, and87/144 (60.42%) with250**, versus97/144 (67.36%) for its676-image control. These nested single-seed runs each used13,520 task exposures and144 DEV labels to select among21 checkpoints. They also inherit public pretraining,384,000 natural-image exposures from20,000 unique crops, and32,000 synthetic exposures. They are not cold50-label adaptation, do not establish a simple user workflow, and have no own deployment timings.


Current PPmedium104/144 has two qualified operating points: mixed TensorRT **0.884 ms RGB /1.087 ms encoded,654 MiB** preserves canonical CTC output tokens and literal strings while two internal frame alignments differ; original numeric/frame failures remain recorded. The global FP32 control retains frames and passes its frozen numeric gate at **1.413 /1.688 ms,588 MiB**. Neither is a universal equivalence claim. A newer1.687M photometric-replay checkpoint scores **101/144 (70.14%)**, selected before further adaptation; its own B1 FP32 TensorRT/Graph4 deployment preserves all144 native strings/tokens/EOS at1.545 ms RGB /1.737 ms encoded,552 MiB. Similar runtime to98 is not a speed-win claim. All are single-seed, development-selected results on the same144 images, not final-test evidence.


A separate20-epoch matched training pair starting from the original98 checkpoint also did not favor160×320: the128 control selected98/144 at epoch0 (terminal95), while160 selected95/144 at epoch16 (terminal93), with13 gains and16 losses versus128. It improves over that frozen checkpoint’s160-bound89/144, but does not beat the128 control; it is not promoted.

The actual101 checkpoint also has a separately qualified **fixed128×256 batch1/2/4/8** TensorRT path, retaining all144 native strings/tokens/EOS at every capacity. At batch8, RGB is **8.373 ms per batch,1.047 ms per image amortized,955.5 images/s,634 MiB**; encoded is10.028 ms per batch. This is throughput, not a1 ms single-request latency. Keep the separate static-B1 **1.545 ms,552 MiB** point for single requests; the batch path atB1 is slower/larger (1.877 ms,618 MiB). There is no mixed-resolution or OOD qualification.

## Current checkpoint vocabularies and output limits

These limits describe the **actual checkpoints in the current comparison**, not every upstream model or a guarantee of accuracy on every supported character.

| Current checkpoint | Emittable text and special tokens | Output limit |
|---|---|---|
| WATERec price106 | Printable ASCII plus space, including `$`; no `£`, `€`, `¥` or `₹` in this head. 96 output classes; EOS0, BOS96, PAD97. | 25 generated tokens including EOS;24 text characters leaves room for EOS. |
| WATERec ALPR1075 native reference | Original94 printable ASCII characters, without space or `£€¥₹`;95 output classes, EOS0, BOS95, PAD96. |25 generated tokens including EOS. |
| PPmedium price104 / ALPR983 | Restricted local ASCII letters, digits, punctuation and space; blank0 among96 classes. No `£`, `€`, `¥` or `₹`. This is not the full official PP vocabulary. | 40 CTC frames; at most40 emitted characters, with extra blank frames needed between repeated characters. |
| Custom price101 /98 /97 | Printable ASCII plus space and `£€¥₹`:99 characters and EOS,100 output classes; BOS100, PAD101. Other scripts remain outside this head. | 26 autoregressive queries, including EOS;25 text characters. |
| CCT price66 | Digits, uppercase A–Z, period, space and lowercase `d`/`h` only.41 classes; `_` is PAD40, not an emitted underscore. No `$£€¥₹`. | 15 parallel character slots; remove PAD only. |
| CCT ALPR953 | Digits, uppercase A–Z and literal quote;38 classes, PAD37. No space or currency symbols. Distinct from the public37-class default. |15 parallel slots. |
| Florence price105 | Pretrained51,289-token BPE vocabulary, broader than these restricted heads. Tokenizer coverage is not an any-script accuracy guarantee. |32 new BPE tokens, not32 characters; beam5. |

Unsupported reference characters or lengths do **not** justify filtering benchmark images, changing frozen heads or normalizing outputs. A new vocabulary requires a separately trained and qualified checkpoint. Both CTC and autoregressive models can learn punctuation and a requested output format from image/text examples; architecture alone does not guarantee layout reading, few-shot transfer or generic OCR. The custom reader reuses pretrained PaddleOCR visual components and a PARSeqTiny reader, with local multiscale modifications—it is not an entirely new model trained from scratch. Separate task checkpoints are used.


## Fixed-checkpoint heldout scope

Current price206 exact counts are WATE163, PP163, custom101152, Florence159 and CCT91. On9 external photos only6 full-string references are resolvable; WATE/custom each match1 of6, the others0, with3 unscorable images retained. This is a small uncertain, historically exposed cohort, not evidence of general-photo robustness. Mixed PP changes one already-wrong206 string; equal accuracy is not universal backend parity. Current ALPR checkpoints have separate native/TRT retention across2,693+984 images; clean Florence224 remains native-only. See the [full fixed-checkpoint results and caveats](model_comparison.md#current-fixed-checkpoint-heldout-results). No post-evaluation tuning or silent input-policy fallback is permitted.
