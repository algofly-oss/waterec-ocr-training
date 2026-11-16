# Lean TensorRT deployment

The same package supports a Torch-free inference installation. It does not bundle
weights or engines, and this is not a claim of compatibility or performance on an
untested edge device.

```bash
# Run from a source checkout in a clean virtual environment:
python -m pip install -e '.[tensorrt]'
waterec-ocr infer --help
# To build engines from already exported ONNX graphs on the target system:
python -m pip install -e '.[build,tensorrt]'
```

The `tensorrt` extra installs the Python inference dependencies, not Torch, ONNX,
ONNX Runtime or Jupyter. The `build` extra adds ONNX graph inspection without
Torch. `build`, `build-speculative` and `build-orientation` use the same graph,
precision and output-barrier policies as before this packaging change. They
require existing exported graphs and metadata; producing those graphs still
requires the full export environment. Existing imports of the builders from the
export modules remain supported, but importing those export modules needs Torch.
The CLI selects the Torch-free builder directly.

Native PyTorch inference and training require `.[training]`. Existing
`.[notebooks,onnx]`, `requirements.txt` and the Docker notebook workflow retain
Torch and the complete training/export dependencies. The Docker image remains a
training image, not a lean deployment image.

## System setup and assets

Use a compatible NVIDIA driver, CUDA toolkit and TensorRT installation. The
runtime compiles its small greedy CUDA kernel through PyCUDA/NVCC even with CPU
preprocessing. Optional GPU preprocessing also compiles bundled C code and needs
a working C compiler (`CC` or `cc`). Python extras alone do not supply these
system tools. Compilation is cached; setup and compilation were excluded from
the published timing scopes.

Serialized TensorRT engines are GPU/software-specific. Build and qualify engines
for the actual target; do not assume an RTX 5090 engine works on Jetson or another
GPU. The current CUDA source chooses a virtual architecture capped at compute89;
that is not an architecture/package-availability or accuracy guarantee. No new
hardware measurements accompany this packaging change.

Supply your matching external YAML via `--config`, plus its matching deployment
assets. Repository example configs are available in a source checkout, but are
not packaged as wheel data. Relative paths inside a config resolve against the
package's parent directory, so use absolute asset paths for an installed wheel.
Check `waterec-ocr infer --help` for backend, speculative and GPU-preprocessing
options. The inference CLI loads only the selected backend; selecting native or
ONNX inference requires the corresponding extras. Evaluation and benchmarking
with the TensorRT backend also remain Torch-free.
