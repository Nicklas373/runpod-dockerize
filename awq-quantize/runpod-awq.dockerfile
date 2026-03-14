# Use runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404 as base image
FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

# Configure image maintainer
LABEL maintainer="Nicklas373 <herlambangdicky5@gmail.com>"
LABEL version="1.3.0-PROD"
LABEL description="Docker container for Runpod, used for LLM Quantization with LLM Compressor (AWQ)"

# Configure environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0;12.0"

# VS Code Server
ENV PASSWORD=""
ENV CODE_SERVER_ARGS="--bind-addr 0.0.0.0:8080 --auth none"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    git-lfs \
    wget \
    ca-certificates \
    dumb-init \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Configure workspace
WORKDIR /workspace

# Install and upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Install specific Nvidia Nemotron packages
RUN pip install causal-conv1d==1.6.0 mamba-ssm==2.3.0 --no-build-isolation --no-cache-dir -v

# Install Python dependencies
RUN pip install accelerate datasets huggingface-hub hf-transfer llmcompressor transformers

# Install torchvision
RUN pip install pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130

# Debug Torch Version
RUN python3 - <<EOF
import torch
print("Torch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("TorchAudio version:", torchaudio.__version__)
EOF

# Debug Mamba SSM Version
RUN python3 - <<EOF
import mamba_ssm
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
print("Mamba SSM version:", mamba_ssm.__version__)
print("Selective Scan Function:", selective_scan_fn)
print("mamba CUDA OK")
EOF

# Create Offload Folder
RUN mkdir /workspace/offload_model

# Copy quantization scripts into the container
COPY model_consolidated.py /workspace/
COPY model_eval.py /workspace/
COPY model_perplexity.py /workspace/
COPY model_quantize.py /workspace/
COPY model_upload.py /workspace/

# Expose VS Code port
EXPOSE 8080

# Set entrypoint and default command
ENTRYPOINT ["dumb-init", "--"]
CMD ["bash", "-c", "code-server $CODE_SERVER_ARGS /workspace & python3 model_quantize.py --help && sleep infinity"]