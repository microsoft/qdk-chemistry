ARG HOST_ARCH
ARG CUDA_ARCH
FROM quantumapps.azurecr.io/qatk-dev-${HOST_ARCH}-${CUDA_ARCH}:latest

# Create a non-root user with sudo privileges
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Configure pip to not use Azure feeds during container setup
# Will copy the project's pip config during postCreateCommand
RUN mkdir -p /tmp/pip-config-backup
COPY pip.conf* /tmp/pip-config-backup/
COPY pip.ini* /tmp/pip-config-backup/

# Create standard pip config to avoid authentication prompts during build
# The original config file will be restored in postCreateCommand
RUN echo "[global]" > /etc/pip.conf && \
    echo "index-url=https://pypi.org/simple" >> /etc/pip.conf && \
    echo "trusted-host=pypi.org" >> /etc/pip.conf

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
