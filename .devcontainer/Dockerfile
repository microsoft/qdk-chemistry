FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive

# Set locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    QDK_UARCH=native \
    LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH} \
    PATH=/usr/local/bin:${PATH}

# Create a non-root user with sudo privileges
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    curl \
    vim \
    wget \
    git \
    gnupg \
    build-essential \
    cmake \
    clang \
    pkg-config \
    python3 \
    python3-venv \
    python3-pip \
    python-is-python3 \
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
    patchelf \
    doxygen \
    graphviz \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install C++ dependencies
COPY .devcontainer/scripts/install_cpp_dependencies.sh /tmp/install_cpp_dependencies.sh
COPY cpp/manifest/qdk-chemistry/cgmanifest.json /tmp/cpp_cgmanifest.json
COPY external/macis/manifest/cgmanifest.json /tmp/macis_cgmanifest.json
RUN bash /tmp/install_cpp_dependencies.sh /tmp/cpp_cgmanifest.json /tmp/macis_cgmanifest.json && rm /tmp/install_cpp_dependencies.sh /tmp/cpp_cgmanifest.json /tmp/macis_cgmanifest.json

# Create/align the non-root user and group (safe for repeated builds)
RUN if getent group "$USER_GID" >/dev/null; then \
    EXISTING_GROUP=$(getent group "$USER_GID" | cut -d: -f1); \
    if [ "$EXISTING_GROUP" != "$USERNAME" ]; then \
    groupmod --new-name "$USERNAME" "$EXISTING_GROUP"; \
    fi; \
    else \
    groupadd --gid "$USER_GID" "$USERNAME"; \
    fi \
    && if id -u "$USERNAME" >/dev/null 2>&1; then \
    usermod --gid "$USER_GID" --home /home/$USERNAME --shell /bin/bash "$USERNAME"; \
    elif getent passwd "$USER_UID" >/dev/null; then \
    EXISTING_USER=$(getent passwd "$USER_UID" | cut -d: -f1); \
    usermod --login "$USERNAME" --uid "$USER_UID" --gid "$USER_GID" --home /home/$USERNAME --move-home --shell /bin/bash "$EXISTING_USER"; \
    else \
    useradd --uid "$USER_UID" --gid "$USER_GID" -m -s /bin/bash "$USERNAME"; \
    fi \
    && echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to non-root user for subsequent commands
USER $USERNAME



# Setup bash prompt
RUN echo 'export PS1="\[\033[1;34m\]\u@qdk-dev:\w$ \[\033[0m\]"' >> ~/.bashrc
