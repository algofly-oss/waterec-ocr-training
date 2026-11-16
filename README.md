# WATERec OCR: experiments, benchmarks and notebooks

Our experiments with compact OCR for license plates and difficult price tags, with a notebook workflow for training and ONNX/TensorRT deployment. The organization follows our [Florence-2 repository](https://github.com/algofly-oss/vllm-ocr-training): numbered notebooks, Docker/JupyterLab, an optional text annotator, and example data.

**Start with the findings:** [scenario capabilities](docs/capabilities.md), [experiments and methodology](docs/experiments.md), and [machine-readable reference measurements](results/reference_metrics.json). The included fictional images exercise the workflow; they are not the benchmark data and do not reproduce the reported accuracy.

Related repositories: [PARSeq Base / Tiny / Fast Tiny](https://github.com/algofly-oss/parseq-ocr-training), [CCT-S plate OCR](https://github.com/algofly-oss/license-plate-ocr-training), and [Florence-2](https://github.com/algofly-oss/vllm-ocr-training).

For the measured price accuracy/latency/memory tradeoff at 512/1024/2048 patch budgets, see the [lower-resolution reference table](docs/experiments.md#price-resolution-measured-lower-budget-alternatives) and [scoped aggregate metadata](results/resolution_reference.json). The checkpoints have different training recipes; GPU-preprocessing timings do not automatically describe the default runtime.

## What we found

WATERec was our strongest compact accuracy/speed compromise on the tested plate and price datasets. CCT-S and PARSeq Fast Tiny are smaller/faster operating points; they should not be assumed to preserve WATERec's difficult-layout or unseen-image accuracy.

| Trained WATERec operating point | Parameters | Input policy | Evaluation | Accuracy | RGB → text mean | File → text mean | GPU memory |
|---|---:|---|---|---:|---:|---:|---:|
| RS, plates | 26,120,544 | Aspect-preserving, up to 512 4×4 patches | 2,693 plate crops | 2,475 / 2,693 exact (91.90%) | 1.256 ms | 2.260 ms | 694 MiB |
| Same plate model | same | same | 984 reviewed unseen crops | 880 / 984 exact (89.43%) | same timing cohort above | — | — |
| Same plate model | same | same | 443 Brazil/Europe/US crops | 332 / 443 exact (74.94%) | not measured on this cohort | — | — |
| R, prices, SPACE vocabulary | 26,121,312 | Up to 2,048 patches; these crops become 128×256 RGB | 206 price crops | 153 / 206 exact; 180 / 206 numeric | 1.684 ms | 1.990 ms | 716 MiB |

Timing: RTX 5090, TensorRT, batch 1, five warmups, fresh processes, setup/compilation excluded. These reference timings use the separately enabled GPU preprocessing path; the default CPU preprocessing path has different costs. Plate timing used 256 mixed plate/unseen crops; price timing used 206 crops. GPU memory is a sampled process peak including the CUDA context, not model-file size or host RAM. Training batch size and batch-1 inference latency are different measurements. Details, dates, hardware and evidence identifiers are in the reference JSON.

The price model achieved **132/144 on the clear AI-reviewed numeric subset**, versus Florence-2's 133/144; raw/numeric counts were 153/180 versus Florence's 148/176 on all 206 images. These are complementary metrics, not an unqualified win on every criterion. WATERec used 676 fitting images; the final Florence reference used 820. Reviewed labels are not independently certified ground truth.

## Model choices

| Choice | What changes | Starting configuration |
|---|---|---|
| WATERec RS, plate profile | RS initialization, 512-patch budget | `configs/waterec_alpr.yaml` |
| WATERec R, plate profile | R initialization; same architecture/size | `configs/waterec_r_alpr.yaml` |
| WATERec R, detailed price profile | SPACE output, 2,048-patch budget | `configs/waterec_price.yaml` |
| Alternate resolution profiles | 1,024-patch plates or 512-patch prices; validate separately | `configs/waterec_alpr_1024.yaml`, `configs/waterec_price_512.yaml` |
| Experimental speculative decoding | Extra 1,386,624-parameter draft bound to its OCR checkpoint; optional, plate/512 only | Notebook 8 |

R and RS are different pretrained checkpoints of the **same** 12-layer 384-wide encoder and two-layer autoregressive decoder. Changing resolution changes computation, not parameter count. There is no validated official “WATERec Fast Tiny” variant here: **Fast Tiny belongs to PARSeq**, documented in its separate repository. A trained 17.9M custom token-selection reader has now preserved the recovered price reader's raw accuracy, but it has not met every Florence accuracy/runtime target and is not yet a portable release variant. See the [compact research update](docs/compact_research.md) for measured results and explicit limits.

## Run like the Florence repository

With Docker, Docker Compose and NVIDIA Container Toolkit installed:

```bash
docker compose up --build
```

Open JupyterLab at **http://localhost:8780** (default local token `waterec-local`) and the optional text annotator at **http://localhost:8000**. Set `JUPYTER_TOKEN` in your local environment or `.env` to change it. Add your own annotator images to `annotator/static/images`; labels are same-stem text files. Docker services bind to localhost. Container execution needs a compatible NVIDIA driver; inspect the validation record for what was actually tested.

Or use a virtual environment:

```bash
python -m pip install -e '.[notebooks,onnx]'
# On the target NVIDIA/CUDA system, for engine building/inference:
python -m pip install -e '.[tensorrt]'
jupyter lab notebooks
```

## Notebook workflow

| Notebook | Purpose |
|---|---|
| [0. Model options](notebooks/0_model_options.ipynb) | Findings, R/RS, resolution and inference tradeoffs |
| [1. Dataset preparation](notebooks/1_dataset_preparation.ipynb) | Inspect fictional plate/price examples and validate image/text pairs |
| [2. Model training](notebooks/2_model_training.ipynb) | Training-only holdout selection; optional full refit |
| [3. Native inference](notebooks/3_inference.ipynb) | PyTorch prediction and evaluation |
| [4. ONNX export](notebooks/4_onnx_export.ipynb) | Dense-RoPE encoder/cross-KV and cached-decoder export |
| [5. ONNX inference](notebooks/5_onnx_inference.ipynb) | CPU ONNX Runtime inference and native-string comparison |
| [6. TensorRT inference](notebooks/6_trt_inference.ipynb) | Build target-specific engines and check predictions |
| [7. Evaluation and benchmark](notebooks/7_evaluation_and_benchmark.ipynb) | Accuracy, edit similarity, numeric prices and timing scopes |
| [8. Optional acceleration](notebooks/8_optional_acceleration.ipynb) | Explicit GPU preprocessing, orientation and speculative options |

Notebooks default to `configs/dummy_alpr.yaml`; set `WATEREC_CONFIG=configs/dummy_price.yaml` for the price examples. Dummy training is one epoch, CPU FP32 and decoder warmup. It downloads the selected SHA-verified upstream pretrained weights; no private fine-tuned weights are included. GPU/build/benchmark cells are explicit opt-ins. Saved notebook outputs are cleared.

## Your data and training

```text
dataset/
  train/example.png
  train/example.txt
  val/another.png
  val/another.txt
```

Supply the complete expected text as UTF-8. Training needs **images and text only**, with no bounding boxes. Price profiles preserve spaces and decimal punctuation. Plate profiles explicitly uppercase and remove whitespace. The supplied vocabulary covers printable ASCII, with SPACE added for prices. The public cached decoder has 25 prediction steps, reserving EOS; default labels are limited to 24 characters. Other scripts/symbols or longer outputs require model/export changes and new training.

```bash
waterec-ocr train --config configs/dummy_alpr.yaml
waterec-ocr infer --config configs/dummy_alpr.yaml --image dummy_data/plates/val/000.png
waterec-ocr export --config configs/dummy_alpr.yaml
waterec-ocr infer --backend onnx --config configs/dummy_alpr.yaml --image dummy_data/plates/val/000.png
waterec-ocr build --config configs/dummy_alpr.yaml
waterec-ocr infer --backend tensorrt --config configs/dummy_alpr.yaml --image dummy_data/plates/val/000.png
```

Use a new output directory when changing the data or model profile. Epoch selection uses an internal split of the training pool; external validation/OOD images remain for reporting. Training can optionally refit all training images for the selected epoch count. Starting presets are practical workflows, not a claim that their default batch/backend exactly repeats every research run; see the recipe notes.

## Resolution, batches and optional speed paths

Input resolution follows a native aspect-preserving 4×4 patch budget; it does not squeeze every crop to a fixed square. Configurations accept integer budgets ≥64, subject to memory and deployment checks. 512 and 2,048 are the measured reference operating points. Each budget requires matching exported assets and engines; an altered budget does not inherit a benchmark score.

Native PyTorch accepts image lists. The supplied ONNX and TensorRT recognizers use **static batch 1** and process lists sequentially. True batched research engines are not claimed as this default deployment. Optional GPU preprocessing preserves the Pillow resize/normalization policy but still decodes image files on the CPU. Orientation correction is a separate trained component, not native quarter-turn invariance. Experimental speculative decoding can exchange individual predictions through numerical differences and must be checked on your data. Both optional orientation/speculative workflows currently require the 512-token plate profile.

## Reproducibility and publication

See [experiments](docs/experiments.md) for scenario findings, unsuccessful approaches, source/selection boundaries and benchmark definitions. Public source includes fictional data, code, pinned upstream provenance and aggregate results; private benchmark images, annotation services, fine-tuned checkpoints, engines and credentials are excluded. TensorRT engines are GPU/software-specific and must be built locally. Dummy workflow checks establish software behavior, not OCR quality. Fresh check status is recorded separately in `results/validation_summary.json` when available.

Upstream: [OpenOCR-WATERec](https://github.com/YesianRohn/OpenOCR-WATERec). Source and modification notices are preserved under `waterec_ocr/_vendor/waterec`; optional resize-kernel terms are under `_native/waterec`. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
