# Use runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404 as base image
FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

# Configure image maintainer
LABEL maintainer="Nicklas373 <herlambangdicky5@gmail.com>"
LABEL version="1.1.0-PROD"
LABEL description="Docker container for Runpod, used for Comfy UI"

# Configure environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="8.6;8.9"
ENV UV_PYTHON_PREFERENCE=only-system
ENV UV_HTTP_TIMEOUT=1800
ENV EXT_PARALLEL=4
ENV NVCC_APPEND_FLAGS="--threads 8" 
ENV MAX_JOBS=12

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

# Install uv for python virtual environments
RUN curl -LsSf https://astral.sh/uv/install.sh | bash
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Configure workspace
WORKDIR /workspace

# Create virtual environment
RUN uv venv /workspace/.venv
ENV PATH="/workspace/.venv/bin:${PATH}"

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI -b master ComfyUI

# Copy custom requirements to workspace
COPY requirements.txt /workspace/

# Install other required Python packages
RUN uv pip install -r /workspace/ComfyUI/requirements.txt
RUN uv pip install -r /workspace/ComfyUI/manager_requirements.txt
RUN uv pip install -r /workspace/requirements.txt

# Install sage attention 2
RUN git clone https://github.com/thu-ml/SageAttention.git
RUN cd SageAttention && python3 setup.py install

# Reset sage attention variable
ENV EXT_PARALLEL=
ENV NVCC_APPEND_FLAGS=
ENV MAX_JOBS=

# Install ComfyUI Custom Nodes
RUN cd /workspace/ComfyUI/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper && \
    git clone https://github.com/Comfy-Org/ComfyUI-Manager && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts && \
    git clone https://github.com/chflame163/ComfyUI_LayerStyle && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    git clone https://github.com/yolain/ComfyUI-Easy-Use && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    git clone https://github.com/crystian/ComfyUI-Crystools && \
    git clone https://github.com/jags111/efficiency-nodes-comfyui && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/ssitu/ComfyUI_UltimateSDUpscale && \
    git clone https://github.com/kijai/ComfyUI-segment-anything-2 && \
    git clone https://github.com/Acly/comfyui-inpaint-nodes && \
    git clone https://github.com/welltop-cn/ComfyUI-TeaCache && \
    git clone https://github.com/cubiq/ComfyUI_essentials && \
    git clone https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation && \
    git clone https://github.com/Jonseed/ComfyUI-Detail-Daemon && \
    git clone https://github.com/chrisgoringe/cg-use-everywhere && \
    git clone https://github.com/TTPlanetPig/Comfyui_TTP_Toolset && \
    git clone https://github.com/ltdrdata/ComfyUI-Inspire-Pack && \
    git clone https://github.com/pollockjj/ComfyUI-MultiGPU && \
    git clone https://github.com/Acly/comfyui-tooling-nodes && \
    git clone https://github.com/kijai/ComfyUI-GIMM-VFI && \
    git clone https://github.com/raindrop313/ComfyUI-WanVideoStartEndFrames && \
    git clone https://github.com/shiimizu/ComfyUI_smZNodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Subpack && \
    git clone https://github.com/Smirnov75/ComfyUI-mxToolkit && \
    git clone https://github.com/spacepxl/ComfyUI-Image-Filters && \
    git clone https://github.com/lrzjason/Comfyui-In-Context-Lora-Utils && \
    git clone https://github.com/calcuis/gguf && \
    git clone https://github.com/jamesWalker55/comfyui-various && \
    git clone https://github.com/evanspearman/ComfyMath && \
    git clone https://github.com/alexopus/ComfyUI-Image-Saver && \
    git clone https://github.com/wallish77/wlsh_nodes && \
    git clone https://github.com/JPS-GER/ComfyUI_JPS-Nodes && \
    git clone https://github.com/chibiace/ComfyUI-Chibi-Nodes && \
    git clone https://github.com/Flow-two/ComfyUI-WanStartEndFramesNative && \
    git clone https://github.com/M1kep/ComfyLiterals && \
    git clone https://github.com/EeroHeikkinen/ComfyUI-eesahesNodes && \
    git clone https://github.com/Chaoses-Ib/ComfyUI_Ib_CustomNodes && \
    git clone https://github.com/ShmuelRonen/ComfyUI-WanVideoKsampler && \
    git clone https://github.com/kinfolk0117/ComfyUI_GradientDeepShrink && \
    git clone https://github.com/theUpsider/ComfyUI-Logic && \
    git clone https://github.com/kijai/ComfyUI-MMAudio && \
    git clone https://github.com/kijai/ComfyUI-WanAnimatePreprocess && \
    git clone https://github.com/vrgamegirl19/comfyui-vrgamedevgirl && \
    git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack && \
    git clone https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler && \
    git clone https://github.com/WASasquatch/was-node-suite-comfyui && \
    for d in */; do \
        if [ -f "$d/requirements.txt" ]; then \
            uv pip install -r "$d/requirements.txt"; \
        fi; \
    done

# Manage another requirement
RUN uv pip uninstall pynvml && uv pip install compel nvidia-ml-py

# Install code-server
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Copy sage attention and workflow to workspace
COPY init-sageattention.sh /workspace/
COPY Reasoning_Loops_v1_0.json /workspace/

# Create blank folder for SEEDVR2
RUN mkdir /workspace/ComfyUI/models

# Expose VS Code port
EXPOSE 8080

# Expose ComfyUI port
EXPOSE 8083

# Set entrypoint and default command
ENTRYPOINT ["dumb-init", "--"]
CMD ["bash", "-c", "code-server $CODE_SERVER_ARGS /workspace & python3 ComfyUI/main.py --use-sage-attention --listen 0.0.0.0 --port 8083"]