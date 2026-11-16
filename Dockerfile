FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-devel
RUN apt-get update && apt-get install --no-install-recommends -y build-essential fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
WORKDIR /workspace
COPY pyproject.toml LICENSE NOTICE ./
COPY waterec_ocr ./waterec_ocr
RUN pip install --no-cache-dir -e '.[notebooks,onnx,tensorrt]'
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace/notebooks
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8780", "--allow-root", "--no-browser"]
