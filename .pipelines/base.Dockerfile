# QDK/Chemistry Base Image Dockerfile
# This image contains all the scientific computing dependencies, libraries, and external packages
# This image is used as a base for both the development and runtime containers.

ARG BASE_IMAGE_X86_64
ARG BASE_IMAGE_AARCH64
ARG cuda_arch='90'
ARG march='native'
ARG build_shared='ON'
ARG SPDLOG_SOURCE_DIR
ARG EXCHCXX_SOURCE_DIR
ARG GAUXC_SOURCE_DIR
ARG TARGETARCH

# Main build image (removed multi-stage dependencies, now using artifacts)
# Architecture-specific base images
FROM ${BASE_IMAGE_X86_64} AS base-x86_64
FROM ${BASE_IMAGE_AARCH64} AS base-aarch64

# Select the appropriate base image based on target architecture
FROM base-${TARGETARCH:-x86_64}

ARG cuda_arch
ARG march

# Set timezone and locale to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies for Python build and scientific packages
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    ninja-build \
    cmake \
    pkg-config \
    wget \
    sudo \
    vim \
    less \
    make \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    uuid-dev \
    tk-dev \
    libghc-zlib-dev \
    libcurl4-gnutls-dev \
    libxml2-dev \
    libxslt1-dev \
    gettext \
    unzip \
    autoconf \
    libgtest-dev \
    libgmock-dev \
    libopenblas-dev \
    liblapack-dev \
    libboost-all-dev \
    libeigen3-dev \
    nlohmann-json3-dev \
    libfmt-dev \
    libecpint-dev \
    libcutensor-dev \
    libcutensor2 \
    apt-transport-https \
    ca-certificates \
    gnupg \
    curl \
    # Documentaion tools
    doxygen \
    # Other
    tar \
    gzip \
    zip \
    libicu-dev \
    jq \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Build and install Python 3.13 from official source
RUN cd /tmp && \
    wget https://www.python.org/ftp/python/3.13.7/Python-3.13.7.tgz && \
    tar -xzf Python-3.13.7.tgz && \
    cd Python-3.13.7 && \
    ./configure --enable-optimizations --with-ensurepip=install --prefix=/usr/local && \
    make -j${nproc} && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-3.13.7* && \
    ln -sf /usr/local/bin/python3.13 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.13 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.13 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.13 /usr/local/bin/pip && \
    echo "Python 3.13 installed successfully" && \
    /usr/local/bin/python --version && \
    /usr/local/bin/python3 --version

# Set environment variables including Python paths
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/usr/local/bin:/usr/local/lib/python3/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    CC=gcc \
    CXX=g++ \
    FC=gfortran \
    HIGHFIVE_SOURCE_DIR=${HIGHFIVE_SOURCE_DIR} \
    SPDLOG_SOURCE_DIR=${SPDLOG_SOURCE_DIR} \
    EXCHCXX_SOURCE_DIR=${EXCHCXX_SOURCE_DIR} \
    GAUXC_SOURCE_DIR=${GAUXC_SOURCE_DIR} \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python base packages including coverage tools
RUN python --version && pip --version && \
    pip install --no-cache-dir --upgrade pip setuptools wheel pre-commit

# Create standard pip config to avoid authentication prompts during build
RUN echo "[global]" > /etc/pip.conf && \
    echo "index-url=https://pypi.org/simple" >> /etc/pip.conf && \
    echo "trusted-host=pypi.org" >> /etc/pip.conf

# Pre-install Python scientific packages (QDK/Chemistry dependencies only)
RUN pip install --no-cache-dir \
    "pyscf>=2.9.0"

# Pre-install Pybind11
RUN pip install --no-cache-dir "pybind11[global]"

# Copy external dependencies and pipeline-cloned sources to /ext
COPY qatk/external/ /ext/
# Copy install scripts
COPY qatk/.pipelines/install-scripts/ /ext/install-scripts/

# Copy sources from pipeline-cloned locations to ext for consistent build paths
# These are cloned by the Azure DevOps pipeline and available at build context root
COPY spdlog/ /ext/spdlog/
COPY ExchCXX/ /ext/ExchCXX/
COPY GauXC/ /ext/GauXC/
# Copy libint2 downloaded and extracted by pipeline
COPY libint2/ /ext/libint2/
# Copy CMake source cloned by pipeline
COPY cmake/ /ext/cmake/
# Copy Git source cloned by pipeline
COPY git/ /ext/git/
# Copy HDF5 extracted by pipeline
COPY hdf5/ /ext/hdf5/

WORKDIR /ext

# Make all install scripts executable
RUN chmod +x /ext/install-scripts/*.sh

ARG cuda_arch
ARG march
ARG build_shared

# Install dependencies one by one
RUN cd /ext/install-scripts && \
    echo "=== Installing CMake 3.28.3 ===" && \
    ./install-cmake.sh /usr/local

# Build and install Git 2.51.0 from source
RUN cd /ext/git && \
    echo "=== Building Git 2.51.0 from source ===" && \
    make configure && \
    ./configure --prefix=/usr/local && \
    make all -j${nproc} && \
    make install && \
    /usr/local/bin/git --version

# Conditional CUDA-dependent installations
RUN cd /ext/install-scripts && \
    if [ "${cuda_arch}" = "none" ]; then \
    echo "=== CUDA disabled, removing CUDA-specific dependencies ===" && \
    rm -rf /ext/NVTX; \
    else \
    echo "=== CUDA enabled, installing CUDA-specific dependencies for architecture ${cuda_arch} ==="; \
    fi

RUN cd /ext/install-scripts && \
    echo "=== Installing HDF5 1.13.0 ===" && \
    ./install-hdf5.sh /usr/local Release

RUN cd /ext/install-scripts && \
    echo "=== Installing Libint2 ===" && \
    ./install-libint2.sh /usr/local Release ${cuda_arch} ${march} ${build_shared}

RUN cd /ext/install-scripts && \
    echo "=== Installing GauXC ===" && \
    ./install-gauxc.sh /usr/local Release ${cuda_arch} ${march} ${build_shared}

RUN cd /ext/install-scripts && \
    echo "=== Installing SPDLOG ===" && \
    ./install-spdlog.sh /usr/local  Release ${march} ${build_shared}

# Update environment for CMake
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH \
    PATH=/usr/local/bin:$PATH

# Clean up source code and build artifacts to reduce image size
RUN rm -rf /ext

# Set working directory to workspace root
WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
