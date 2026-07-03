# Use runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404 as base image
FROM runpod/pytorch:1.0.7-cu1300-torch291-ubuntu2404

# Configure image maintainer
LABEL maintainer="Nicklas373 <herlambangdicky5@gmail.com>"
LABEL version="1.4.6-PROD"
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
RUN python3 -m pip install --upgrade pip setuptools wheel numpy

# Install Python dependencies
RUN pip install accelerate datasets flash-linear-attention causal-conv1d huggingface-hub hf-transfer llmcompressor

# Re-structure compressed-tensors !
RUN pip uninstall compressed-tensors -y
RUN pip install git+https://github.com/vllm-project/compressed-tensors 

# Re-structure torch !
RUN pip uninstall torch torchvision torchaudio -y
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install latest transformers
RUN pip install --upgrade transformers

# Create Offload Folder
RUN mkdir /workspace/offload_model

# Copy quantization scripts into the container
COPY model_consolidated.py /workspace/
COPY model_eval.py /workspace/
COPY model_perplexity.py /workspace/
COPY model_quantize.py /workspace/
COPY model_upload.py /workspace/
COPY model_visual_remapping.py /workspace/

# Expose VS Code port
EXPOSE 8080

# Set entrypoint and default command
ENTRYPOINT ["dumb-init", "--"]
CMD ["bash", "-c", "code-server $CODE_SERVER_ARGS /workspace & python3 model_quantize.py --help && sleep infinity"]