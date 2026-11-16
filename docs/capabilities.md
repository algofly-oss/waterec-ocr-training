# WATERec capabilities and reference comparisons

This repository provides WATE training and deployment workflows. The table separates current task results from optional pipeline experiments and qualitative observations.

## Scenario evidence

“Not measured” means this reference collection does not establish the capability. It is not a zero score. Architecture support, qualitative experience and measured accuracy are different kinds of evidence.

| Scenario | CCT-S plate model | WATERec references | Florence reference |
|---|---|---|---|
| Upright plate crops | Small fixed-slot recognizer; measured speed below. No full-test accuracy claim is attached to that timing cohort. | ALPR: 2,475/2,693 exact; separate reviewed unseen cohort 880/984; extracted fields 332/443. These denominators must not be combined. | No matched upright plate score included here. |
| Two-line plates | Whole crop is resized to 64×128; no separately qualified two-line subset. | Native 2D patches retain spatial arrangement, but no isolated two-line accuracy is reported here. | Layout capability should be evaluated on the actual two-line cohort; no figure supplied here. |
| Native 90°/180°/270° reading | Not measured. Fixed input shape does not provide rotation invariance. | An older native FP32 checkpoint scored 247/788, 159/788, 251/788 on clockwise rotations, without correction. This is a separate diagnostic, not the current final-engine result. | No matched current rotation qualification claimed. |
| Rotation with correction | No qualified correction stage in this repository. | Optional experimental orientation pipeline: 706/788, 706/788, 707/788 at clockwise 90°/180°/270°. Upright reviewed cohort: 879/984 versus 880/984 without correction. These are pipeline results, not intrinsic reader invariance. | No correction result supplied here. |
| Unseen countries and fonts | Qualitative out-of-distribution weaknesses have been observed; no country-stratified score is supplied. | The reviewed unseen plate result is evidence for that cohort, not every country or script. | No universal country coverage claim. |
| Price: tiny raised fractions, currency, unequal glyph sizes | This 37-class plate model is not the price model. Its alphabet cannot represent decimal punctuation or spaces. | R2048 price: 153/206 raw exact, 180/206 numeric, 132/144 on a separate AI-reviewed subset. Detail preservation still matters; not a guarantee for every raised fraction. | Same price references: 148/206 raw, 176/206 numeric, 133/144 reviewed. The reviewed scores are close; they do not establish universal equivalence. |
| Extra text, clutter, few-shot adaptation | No demonstrated few-shot selective-reading capability. Training on plate crops does not establish price-field selection. | Current full-data scores are not 50-shot results. Selection among distractors remains task/data dependent. | User-observed useful adaptation from roughly 50 examples with clutter is qualitative evidence here, not a matched 90% benchmark or a guarantee. |
| Arbitrary source resolution | Images are accepted and resized to fixed 64×128. This is not preservation of arbitrary-resolution detail. | Aspect-preserving patch-4 preprocessing uses a configurable token budget. Actual dimensions/token counts depend on the image; changing budget requires fresh accuracy and resource checks. | This comparison does not qualify arbitrary-resolution export behavior. |
| Batch throughput and memory | Historical B1: 0.4474 ms, 2,235 images/s; B32: 2.5137 ms/batch, 12,730 images/s; both sampled 742 MiB. | Qualified static B1: plate 1.2558 ms / 694 MiB; price R2048 1.6838 ms / 716 MiB. Separate dynamic-batch implementations must be measured independently. | No directly comparable timing supplied in this table. |

All listed timing uses predecoded RGB, including preprocessing, transfer, inference and text decoding, with the relevant engine/runtime. CCT timing used 256 reference crops, five warmups and three passes, a batch-32 engine profile and batch-32 allocations. WATE plate and price timings have different cohorts and profiles. These are not architecture-only or hardware-independent comparisons. See the repository's reference metadata for evidence hashes and exact scope. Dummy-data runs establish wiring only.

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
