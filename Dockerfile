ARG CUDA_VERSION=12.4.1
# Switched from ubuntu20.04 to ubuntu22.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

# Change software sources and install basic tools and system dependencies
# Note: Ubuntu 22.04 natively uses Python 3.10, so deadsnakes PPA is removed
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git curl sudo ffmpeg fonts-noto wget \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    python3-pip \
    && python3 --version && python3 -m pip --version

# Clean apt cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

# Workaround for CUDA compatibility issues
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# Set working directory and clone repository
WORKDIR /app
RUN git clone https://github.com/Huanshere/VideoLingo.git .

# Install PyTorch and torchaudio matching CUDA 12.4 (using cu124 index)
RUN pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Clean up unnecessary files
RUN rm -rf .git

# Upgrade pip and install basic dependencies
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -e .

# Set CUDA-related environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Set CUDA architecture list
ARG TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

EXPOSE 8501

CMD ["streamlit", "run", "st.py"]