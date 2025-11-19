# QDK Azure Linux Base Image Dockerfile
# This Dockerfile creates base images for QDK (Quantum Development Kit) based on Azure Linux
# Supports CPU-only and GPU configurations across different architectures and Python versions

ARG AZURE_LINUX_VERSION=3.0
ARG IMAGE_TYPE=cpu
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.13
ARG ARCHITECTURE=amd64
ARG CUDA_ARCH=none
ARG GPU_TYPE=generic

# Base stage - Azure Linux core
FROM mcr.microsoft.com/azurelinux/base/core:${AZURE_LINUX_VERSION} AS base

ARG PYTHON_VERSION
ARG IMAGE_TYPE
ARG CUDA_VERSION
ARG ARCHITECTURE
ARG CUDA_ARCH
ARG GPU_TYPE

# Set timezone and locale to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Update system and install essential packages
RUN tdnf update -y && \
    tdnf install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    unzip \
    tar \
    gzip \
    vim \
    less \
    sudo \
    ca-certificates \
    pkg-config \
    # Python build dependencies
    gcc \
    g++ \
    gfortran \
    make \
    zlib-devel \
    openssl-devel \
    libffi-devel \
    sqlite-devel \
    bzip2-devel \
    xz-devel \
    readline-devel \
    ncurses-devel \
    # Scientific computing libraries
    openblas-devel \
    boost-devel \
    # System utilities
    which \
    procps \
    util-linux \
    findutils \
    && tdnf clean all

# Install Python from source for consistent version across architectures
RUN cd /tmp && \
    PYTHON_MAJOR_MINOR=$(echo ${PYTHON_VERSION} | cut -d. -f1-2) && \
    # Determine the latest patch version based on major.minor
    case "${PYTHON_MAJOR_MINOR}" in \
    "3.10") PYTHON_FULL_VERSION="3.10.15" ;; \
    "3.11") PYTHON_FULL_VERSION="3.11.11" ;; \
    "3.12") PYTHON_FULL_VERSION="3.12.8" ;; \
    "3.13") PYTHON_FULL_VERSION="3.13.1" ;; \
    *) PYTHON_FULL_VERSION="${PYTHON_VERSION}.0" ;; \
    esac && \
    echo "Installing Python ${PYTHON_FULL_VERSION}" && \
    wget https://www.python.org/ftp/python/${PYTHON_FULL_VERSION}/Python-${PYTHON_FULL_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_FULL_VERSION}.tgz && \
    cd Python-${PYTHON_FULL_VERSION} && \
    ./configure \
    --enable-optimizations \
    --with-ensurepip=install \
    --prefix=/usr/local \
    --enable-shared \
    --with-lto \
    --enable-ipv6 && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-${PYTHON_FULL_VERSION}* && \
    # Create symlinks for the major.minor version
    ln -sf /usr/local/bin/python${PYTHON_MAJOR_MINOR} /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python${PYTHON_MAJOR_MINOR} /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip${PYTHON_MAJOR_MINOR} /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip${PYTHON_MAJOR_MINOR} /usr/local/bin/pip && \
    # Update library cache
    echo "/usr/local/lib" > /etc/ld.so.conf.d/python.conf && \
    ldconfig && \
    echo "Python ${PYTHON_FULL_VERSION} installed successfully" && \
    /usr/local/bin/python --version && \
    /usr/local/bin/pip --version

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/bin:$PATH \
    CC=gcc \
    CXX=g++ \
    FC=gfortran

# Install pip
RUN python --version && pip --version && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

# Create standard pip config
RUN mkdir -p /etc/pip && \
    echo "[global]" > /etc/pip/pip.conf && \
    echo "index-url=https://pypi.org/simple" >> /etc/pip/pip.conf && \
    echo "trusted-host=pypi.org" >> /etc/pip/pip.conf

# Final stage - combine CPU and GPU logic in one stage
FROM base AS final

ARG CUDA_VERSION
ARG CUDA_ARCH
ARG ARCHITECTURE
ARG GPU_TYPE
ARG IMAGE_TYPE

# Install CUDA components only for GPU builds
RUN if [ "${IMAGE_TYPE}" = "gpu" ] && [ "${ARCHITECTURE}" = "amd64" ]; then \
    echo "Installing CUDA for GPU build" && \
    cd /tmp && \
    # Download and install CUDA repository package for Azure Linux
    wget -q https://developer.download.nvidia.com/compute/cuda/13.0.1/local_installers/cuda-repo-azl3-13-0-local-13.0.1_580.82.07-1.x86_64.rpm && \
    rpm -i cuda-repo-azl3-13-0-local-13.0.1_580.82.07-1.x86_64.rpm && \
    # Install Azure Linux extended repositories
    tdnf -y install azurelinux-repos-extended && \
    tdnf clean all && \
    # Install CUDA toolkit (exclude lm-sensors to avoid modprobe warnings)
    tdnf -y install cuda-toolkit-13-0 --exclude=lm-sensors && \
    # Install cuDNN (manual installation as it may not be in repos)
    CUDNN_VERSION=""9.13.1.26"" && \
    CUDNN_TAR="cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive.tar.xz" && \
    wget -q https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_TAR} && \
    tar -xf ${CUDNN_TAR} && \
    cp -P cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive/include/cudnn*.h /usr/local/cuda/include && \
    cp -P cudnn-linux-x86_64-${CUDNN_VERSION}_cuda13-archive/lib/libcudnn* /usr/local/cuda/lib64 && \
    chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn* && \
    # Set CUDA environment variables
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf && \
    ldconfig && \
    # Cleanup
    cd / && \
    rm -rf /tmp/cuda-repo-*.rpm /tmp/cudnn-* && \
    echo "CUDA 13.0 and cuDNN ${CUDNN_VERSION} installed successfully"; \
    else \
    echo "Skipping CUDA installation for CPU build or non-amd64 architecture"; \
    fi

# Set CUDA environment variables (will be no-op for CPU builds)
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_PATH=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH


# Add image labels
LABEL org.opencontainers.image.title="Azure Linux Base" \
    org.opencontainers.image.description="Base image based on Azure Linux with Python support" \
    org.opencontainers.image.vendor="Microsoft" \
    python.version="${PYTHON_VERSION}" \
    image.type="${IMAGE_TYPE}" \
    architecture="${ARCHITECTURE}" \
    cuda.version="${CUDA_VERSION}" \
    cuda.arch="${CUDA_ARCH}" \
    gpu.type="${GPU_TYPE}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; print(f'Azure Linux Base - Python {sys.version_info.major}.{sys.version_info.minor} - Ready')" || exit 1

# Set working directory
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
