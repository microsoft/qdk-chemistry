FROM ubuntu:24.04:latest

# Create a non-root user with sudo privileges
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install -y \
    libeigen3-dev \
    nlohmann-json3-dev \
    libhdf5-dev \
    libopenblas-dev \
    libboost-all-dev \
    libgtest-dev \
    libgmock-dev \
    libfmt-dev \
    libecpint-dev \
    ninja-build \
    gcovr \
    patchelf

# Create the non-root user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chsh -s /bin/bash $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user for subsequent commands
USER $USERNAME

# Setup bash prompt
RUN echo 'export PS1="\033[1;34m\]\u@prepq-dev:\w$ \[\033[0m\]"' >> ~/.bashrc
